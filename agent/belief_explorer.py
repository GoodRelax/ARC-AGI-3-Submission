"""BeliefExplorer — the integration 'brain' for partial-observability levels (spec 89 §4; R12).

Ties the three belief layers together into one turn loop and learns the rest from observation (NO
game-specific hardcoding, per PRINCIPLES):
  * WorldBelief (terrain) + ObjectTracker (objects/routes) are updated every frame and NEVER discarded.
  * The MOVE MODEL is learned online: issue an action, watch the controllable's displacement -> moves[a].
  * PASSABLE colours are learned: cells the controllable vacates reveal the floor; its own colours are
    walkable-for-it. (We never see under the avatar, so its footprint is recorded as the learned floor.)
  * The FOG colour is learned (detect_fog_colour) from the first few frames.
Action policy: first try each action once to learn the move model, then frontier-explore toward the fog;
if a goal cell is supplied/known, path to it. Pure module (numpy only) so it is fully unit-testable; the
framework FrameData/GameAction adapter is a thin shell (see belief_policy.py).
"""
from __future__ import annotations

import numpy as np

from agent.world_belief import WorldBelief
from agent.belief_objects import ObjectTracker, position
from agent.explore import frontier_explore_action, bfs_path_action, bfs_action_dist
from agent.fuel import FuelSensor


class BeliefExplorer:
    MOVE_MAX = 6                              # a single action moves <= this; bigger = teleport/spring (don't learn)

    def __init__(self, h=64, w=64, fog_warmup=3):
        self.belief = WorldBelief(h, w)
        self.objects = ObjectTracker()
        self.fuel = FuelSensor()              # learns the resource bar + refuel pickups (R12, spec 89 §4.4)
        self.moves: dict = {}                 # learned action_key -> (dr, dc)
        self.passable: set = set()            # learned passable colours
        self.walls: set = set()               # learned BLOCKING colours (a known move that no-op'd hit one)
        self.probed_cells: set = set()        # every cell the avatar has stood on (= markers already probed)
        self.refuel_margin = 3                # detour to refuel when moves_left <= dist + this safety buffer
        self.fog = None
        self._warmup = fog_warmup
        self._frames: list = []
        self.prev_tl = None
        self.prev_foot_cells: set = set()
        self.usage: dict = {}
        self._tried_here: set = set()         # actions tried since the avatar last MOVED (avoid no-op loops)
        self._flushed = False
        self._last_branch = None              # which _choose branch fired (diagnostics)
        self._dbg = {}                        # last refuel-detour state (diagnostics)
        self.t = -1

    # ------------------------------------------------------------------ one turn
    def act(self, grid, avatar_cells, action_space, last_action=None, goal_cells=None):
        """grid: 2-D int array (the settled frame). avatar_cells: iterable of (r,c) of the controllable
        block this frame. action_space: iterable of action keys. last_action: the key issued last turn (or
        None). goal_cells: known goal cells to path to (or None -> explore). Returns the chosen action key."""
        grid = np.asarray(grid, dtype=int)
        self.t += 1
        avatar_cells = set(map(tuple, avatar_cells))
        tl = position(avatar_cells)
        foot = [(r - tl[0], c - tl[1]) for r, c in sorted(avatar_cells)]
        avatar_colours = {int(grid[r, c]) for r, c in avatar_cells}

        # 1. learn the move model + passable floor from the last transition (works pre/post fog)
        moved = False
        if last_action is not None and self.prev_tl is not None:
            disp = (tl[0] - self.prev_tl[0], tl[1] - self.prev_tl[1])
            if disp != (0, 0):
                moved = True
                self._tried_here.clear()                            # moved -> earlier blocks may now clear
                # learn the move vector ONLY from a plausible single step: a fuel-0 respawn TELEPORT (back
                # to spawn) or a spring launch is large/diagonal and must NOT corrupt moves[last_action].
                if (disp[0] == 0 or disp[1] == 0) and max(abs(disp[0]), abs(disp[1])) <= self.MOVE_MAX:
                    self.moves[last_action] = disp
            for (r, c) in (self.prev_foot_cells - avatar_cells):    # cells the avatar just vacated
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    v = int(grid[r, c])
                    if self.fog is None or v != self.fog:
                        self.passable.add(v)                        # revealed true floor under the avatar
        self.passable |= avatar_colours                             # avatar's own cells walkable-for-it
        self.probed_cells |= avatar_cells                           # every cell stood on = marker probed
        # learn WALL colours: a KNOWN-move action that produced no displacement was blocked, so the
        # leading-edge cells the footprint would have entered are walls (keeps walls out of _markers).
        if (not moved) and last_action in self.moves and self.prev_tl is not None:
            dr, dc = self.moves[last_action]
            for (fr, fc) in self.prev_foot_cells:
                nr, nc = fr + dr, fc + dc
                if (nr, nc) in self.prev_foot_cells:
                    continue
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    v = int(grid[nr, nc])
                    if (self.fog is None or v != self.fog) and v not in self.passable:
                        self.walls.add(v)
        self.fuel.observe(grid, self.fog, moved)                    # learn the resource bar / refuel pickups

        # 2. fog detection (buffer frames until fog is known, then FLUSH them in — waste no frame).
        #    Robust signal: the avatar is always surrounded by revealed terrain, so FOG is the colour that
        #    stays FARTHEST from the avatar (floor/walls appear right next to it; fog only past the torch).
        self._frames.append((grid.copy(), frozenset(avatar_cells)))
        if self.fog is None and len(self._frames) >= self._warmup:
            self.fog = self._detect_fog()
            self.belief.fog_colour = self.fog
        if self.fog is not None:
            if not self._flushed:
                for g, ac in self._frames:
                    self.belief.update(g, position(ac))
                self._flushed = True
            else:
                self.belief.update(grid, tl)
            bg = set(self.passable) | {self.fog}
            obj_colours = {int(v) for v in np.unique(grid)} - bg    # non-background = transformers/markers
            if obj_colours:
                self.objects.update(grid, obj_colours)

        self.prev_tl = tl
        self.prev_foot_cells = avatar_cells

        # 4. choose an action
        action_space = list(action_space)
        pick = self._choose(tl, foot, action_space, goal_cells)
        self._tried_here.add(pick)
        self.usage[pick] = self.usage.get(pick, 0) + 1
        return pick

    def _detect_fog(self, near=2):
        """FOG = the most FREQUENT colour that stays AWAY from the avatar's body (Manhattan > `near`)
        across the warmup frames. Floor/walls appear adjacent to the avatar (excluded); only fog lives
        beyond the torch. Frequency then beats small far-off HUD bits (lives, a blinker). Robust to torch
        size (unlike a re-cover-frequency heuristic) and to far distractors (unlike pure farthest-colour)."""
        counts: dict = {}
        closest: dict = {}
        avc: set = set()
        for g, ac in self._frames:
            r0 = min(r for r, _ in ac); r1 = max(r for r, _ in ac)
            c0 = min(c for _, c in ac); c1 = max(c for _, c in ac)
            avc |= {int(g[r, c]) for r, c in ac}
            vals, cnts = np.unique(g, return_counts=True)
            for v, cnt in zip(vals.tolist(), cnts.tolist()):
                cells = np.argwhere(g == v)
                dr = np.clip(np.maximum(r0 - cells[:, 0], cells[:, 0] - r1), 0, None)
                dc = np.clip(np.maximum(c0 - cells[:, 1], cells[:, 1] - c1), 0, None)
                d = int((dr + dc).min())
                closest[v] = min(closest.get(v, 10 ** 9), d)
                counts[v] = counts.get(v, 0) + int(cnt)
        far = [v for v in counts if v not in avc and closest[v] > near]
        return max(far, key=counts.get) if far else None

    def _markers(self):
        """Believed PLAY-AREA cells that are neither learned-passable, fog, nor the fuel/refuel colour =
        candidate transformers / goal marks worth probing (R11). The HUD band (lives etc.) is excluded."""
        if self.fog is None:
            return set()
        bg = set(self.passable) | set(self.walls) | {self.fog}
        if self.fuel.bar_colour is not None:
            bg.add(self.fuel.bar_colour)
        m = self.belief.seen & ~np.isin(self.belief.map, list(bg))
        top = self.belief.h - self.fuel.hud_band
        rs, cs = np.where(m)
        return {(int(r), int(c)) for r, c in zip(rs.tolist(), cs.tolist()) if r < top}

    def _choose(self, tl, foot, action_space, goal_cells):
        # (a) learn the move model: try an unlearned action NOT already tried (and stuck) at this position.
        #     (Don't insist on learning ALL before moving — blocked no-ops here must not loop forever.)
        learn = [a for a in action_space if a not in self.moves and a not in self._tried_here]
        if learn:
            self._last_branch = "learn"
            return min(learn, key=lambda a: self.usage.get(a, 0))
        # (b) SURVIVAL first: detour to refuel when the fuel left only just covers the trip to the nearest
        #     pickup (else a fuel-0 costs a life + respawns the avatar at the start — the L7 death loop).
        if self.moves and self.fuel.ready:
            refuel = self.fuel.refuel_cells(self.belief)
            ml = self.fuel.moves_left()
            if refuel and ml is not None:
                hit = lambda t: any((t[0] + dr, t[1] + dc) in refuel for dr, dc in foot)
                a, dist = bfs_action_dist(self.belief, tl, self.moves, foot, self.passable, hit,
                                          optimistic_unknown=True, extra_passable=refuel)
                self._dbg = {"ml": ml, "refuel_dist": dist, "n_refuel": len(refuel)}
                if a is not None and dist is not None and ml <= dist + self.refuel_margin:
                    self._last_branch = "refuel"
                    return a
        # (c) a known goal: head toward it, optimistically through fog (its coords are known).
        if goal_cells:
            a = bfs_path_action(self.belief, tl, goal_cells, self.moves, foot, self.passable,
                                optimistic_unknown=True)
            if a is not None:
                self._last_branch = "goal"
                return a
        # (d) PROBE the nearest unprobed marker (transformer / goal candidate) — drives interaction so the
        #     belief learns what each object DOES (R11), instead of only mapping empty terrain.
        if self.moves:
            unprobed = self._markers() - self.probed_cells
            if unprobed:
                hit = lambda t: any((t[0] + dr, t[1] + dc) in unprobed for dr, dc in foot)
                a, _ = bfs_action_dist(self.belief, tl, self.moves, foot, self.passable, hit,
                                       optimistic_unknown=True, extra_passable=unprobed)
                if a is not None:
                    self._last_branch = "marker"
                    return a
        # (e) explore toward the fog frontier with the moves learned so far.
        a = frontier_explore_action(self.belief, tl, self.moves, foot, self.passable)
        if a is not None:
            self._last_branch = "frontier"
            return a
        # (f) fallback: any learned move (least-used) to relocate — a new position may unblock learning /
        #     reveal new frontier. If nothing learned yet, just take the least-used action.
        self._last_branch = "fallback"
        pool = list(self.moves) if self.moves else action_space
        return min(pool, key=lambda x: self.usage.get(x, 0))
