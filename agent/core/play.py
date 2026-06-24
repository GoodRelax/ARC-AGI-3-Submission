"""[agent/core] play -- the self-solve loop (choose_action / is_done) for the L1 slice.

Model-predictive: each turn it perceives, learns from the last transition (move effect,
walls, carried-pose interactions), reads the goal once an interaction reveals the carried
twin, plans with BFS over the AbstractSituation, and commits ONE move (MPC). Three behaviours fall
out of one policy:
  1. CALIBRATE -- learn each action's translation by trying the unknown ones;
  2. EXPLORE   -- with no way yet to change the carried state, navigate to the nearest
                  compact object to provoke an interaction (curiosity);
  3. SOLVE     -- once the goal is locked, BFS deliver+match and follow it.

No game literals: actions, colours, the avatar, the cross, the goal are all discovered.
"""

from __future__ import annotations

import numpy as np

from agent.controllable import find_controllable
from agent.core import attributes as A
from agent.core import goal as G
from agent.core import solver as Solve
from agent.core.perceive import perceive
from agent.core.world_model import WorldModel

ACTIONS = (1, 2, 3, 4)


def _obj_for(objs, cells):
    cs = set(cells)
    best, bov = None, 0
    for o in objs:
        ov = len(o.cells & cs)
        if ov > bov:
            best, bov = o, ov
    return best


def _name_action(delta):
    """Name an action by its LEARNED effect (the agent names things by what they do)."""
    dr, dc = delta
    if dr < 0:
        return "up"
    if dr > 0:
        return "down"
    if dc < 0:
        return "left"
    if dc > 0:
        return "right"
    return "none"


def _nearest_same_color(o, prev_objs):
    """The previous-frame object of the same dominant colour nearest ``o`` (frame-to-frame
    identity by colour + proximity) -- used to spot which object's POSE changed."""
    cands = [p for p in prev_objs if p.dom_color == o.dom_color]
    if not cands:
        return None
    return min(cands, key=lambda p: abs(p.centroid[0] - o.centroid[0])
              + abs(p.centroid[1] - o.centroid[1]))


class Agent:
    def __init__(self, verbose=False):
        self.model = WorldModel()
        self.verbose = verbose
        self.ctrl_state = None
        self.prev_objs = None
        self.prev_avatar_pos = None
        self.last_action = None
        self.offsets = None
        self.target = None              # the goal mark Obj (locked once discovered)
        self.target_pose = None
        self.container = frozenset()
        self.poked = []                 # object cell-sets already visited during curiosity
        self.won = False
        self._cycle = 0
        self._cal = -1                  # round-robin cursor for action calibration
        self._objs = []                 # last perception (for introspection)
        self._avatar = None

    # ------------------------------------------------------------------ public API
    def is_done(self, grid) -> bool:
        return self.won

    def choose_action(self, grid) -> int:
        grid = np.asarray(grid, dtype=int)
        objs = perceive(grid)
        cells, self.ctrl_state = find_controllable(grid, self.ctrl_state)
        avatar = _obj_for(objs, cells) if cells is not None else None
        avatar_pos = avatar.pos if avatar is not None else None
        if avatar is not None and self.offsets is None:
            self.offsets = frozenset((r - avatar.pos[0], c - avatar.pos[1]) for r, c in avatar.cells)
        self._objs, self._avatar = objs, avatar

        self._learn(grid, objs, avatar, avatar_pos)
        action = self._decide(grid, objs, avatar, avatar_pos)

        self.prev_objs = objs
        self.prev_avatar_pos = avatar_pos
        self.last_action = action
        return action

    # ------------------------------------------------------------------ learning
    def _learn(self, grid, objs, avatar, avatar_pos):
        if self.last_action is None or self.prev_avatar_pos is None or avatar_pos is None:
            return
        disp = (avatar_pos[0] - self.prev_avatar_pos[0], avatar_pos[1] - self.prev_avatar_pos[1])
        self.model.learn_move(self.last_action, disp)
        if disp == (0, 0):
            self._learn_wall_ahead(grid)
        elif avatar is not None:
            self.model.learn_passable(int(grid[r, c]) for r, c in avatar.cells)
        if avatar is not None:
            self._observe_carried(grid, objs, avatar)

    def _learn_wall_ahead(self, grid):
        d = self.model.move_delta.get(self.last_action)
        if d is None:
            return
        r0, c0 = self.prev_avatar_pos
        for dr, dc in self.offsets:
            if (dr + d[0], dc + d[1]) in self.offsets:
                continue                       # still inside the body -> not a leading edge
            rr, cc = r0 + dr + d[0], c0 + dc + d[1]
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                col = int(grid[rr, cc])
                if col not in self.model.passable:
                    self.model.learn_wall(col)

    def _observe_carried(self, grid, objs, avatar):
        if self.prev_objs is None:
            return
        for o in objs:
            po = _nearest_same_color(o, self.prev_objs)
            if po is not None and A.pose(po) != A.pose(o):
                trig = self._overlapped(avatar)         # the object now under the avatar = trigger
                self.model.learn_trigger(trig or frozenset(), A.pose(po), A.pose(o))
                if self.target is None:
                    self._lock_goal(grid, objs, o)
                return

    def _overlapped(self, avatar):
        foot = set(avatar.cells)
        for o in self.prev_objs:
            if o.dom_color != avatar.dom_color and (foot & o.cells):
                return o.cells
        return None

    def _lock_goal(self, grid, objs, carried_obj):
        target = G.twin_of(objs, carried_obj)
        if target is None:
            return
        self.target = target
        self.target_pose = A.pose(target)
        box = G.container_of(grid, target)
        if box:
            r0, c0, r1, c1 = box
            self.container = frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
        if self.verbose:
            print(f"   [goal locked] target@{target.pos} container={box}")

    # ------------------------------------------------------------------ deciding
    def _decide(self, grid, objs, avatar, avatar_pos):
        if avatar_pos is None:
            return ACTIONS[0]                              # move once to reveal the avatar
        # CALIBRATE: learn every action's translation, round-robin so the avatar stays near
        # spawn (trying one action repeatedly can strand it where that action is walled).
        if not self.model.moves_known():
            for _ in range(len(ACTIONS)):
                self._cal = (self._cal + 1) % len(ACTIONS)
                if ACTIONS[self._cal] not in self.model.move_delta:
                    return ACTIONS[self._cal]
        # SOLVE: goal locked -> BFS deliver+match
        if self.target is not None:
            carried_pose = self._carried_pose(objs)
            if carried_pose is not None:
                plan = Solve.bfs_solve(self.model, grid, self.offsets, avatar_pos,
                                       carried_pose, self.target_pose, self.container)
                if plan:
                    return plan[0]
        # EXPLORE: navigate to the nearest compact, un-poked object to provoke an interaction
        a = self._curiosity(grid, objs, avatar, avatar_pos)
        if a is not None:
            return a
        # fallback: cycle through actions
        self._cycle = (self._cycle + 1) % len(ACTIONS)
        return ACTIONS[self._cycle]

    def _carried_pose(self, objs):
        if self.target is None:
            return None
        cands = [o for o in objs if o.pos != self.target.pos
                 and o.dom_color == self.target.dom_color and A.shape(o) == A.shape(self.target)]
        return A.pose(cands[0]) if cands else None

    def _curiosity(self, grid, objs, avatar, avatar_pos):
        avatar_size = len(self.offsets)
        cands = [o for o in objs
                 if o is not avatar and o.size <= 2 * avatar_size and o.cells not in self.poked]
        cands.sort(key=lambda o: abs(o.centroid[0] - avatar.centroid[0])
                   + abs(o.centroid[1] - avatar.centroid[1]))
        for o in cands:
            path = Solve.bfs_to(self.model, grid, self.offsets, avatar_pos, set(o.cells))
            if path:
                if len(path) == 1:                        # about to reach it -> remember we tried it
                    self.poked.append(o.cells)
                return path[0]
            if path == []:                                # already overlapping; nothing happened -> skip it
                self.poked.append(o.cells)
        return None

    # ------------------------------------------------------------------ introspection
    def _role(self, o):
        if self._avatar is not None and o.pos == self._avatar.pos:
            return "controllable"
        for trig in self.model.triggers:
            if set(o.cells) & trig:
                return "interactor"
        if self.target is not None:
            if o.pos == self.target.pos:
                return "target"
            if o.dom_color == self.target.dom_color and A.shape(o) == A.shape(self.target):
                return "carried-state"
        return "unclassified"

    def vocabulary(self):
        """The agent's own grounded abstraction, by domain facet: object / world / goal.
        Each thing is NAMED by its function, each attribute is a separate axis."""
        objects = []
        for o in self._objs:
            objects.append({
                "role": self._role(o),
                "pos": o.pos,
                "color": A.color(o),
                "shape": "#%03d" % (abs(hash(A.shape(o))) % 1000),
                "orient": A.orientation_index(o),
                "size": A.size(o),
            })
        world = {
            "move_effects": {a: "%s %s" % (_name_action(d), d) for a, d in sorted(self.model.move_delta.items())},
            "wall_colors": sorted(self.model.walls),
            "interaction_rules": (
                ["footprint meets interactor -> carried-state orientation advances (%d transition learned)"
                 % len(self.model.pose_succ)] if self.model.triggers else []),
        }
        if self.target is not None:
            goal = ("deliver(controllable -> container) AND match(carried-state, target | color+shape+orientation); "
                    "target@%s orient=%s" % (self.target.pos, A.orientation_index(self.target)))
        else:
            goal = "not yet abduced (still discovering how to change a state)"
        return {"object": objects, "world": world, "goal": goal}


# =====================================================================================
# Play / decision behaviours cluster -- DiagnoseDivergence (CMP-30), DetectFutility observe
# side (CMP-32), ExploreWorld / ProbeObject (CMP-33 / CMP-34). Built ON TOP of the frozen
# siblings (world_model.classify_move_effect / MoveEffect / ModelWorld; goal.GoalPredicate;
# situation.AbstractSituation), operating purely over the abstract AbstractSituation, so it is
# game-literal-free (NFR-6) and deterministic (DP-10: stable string routes / canonical
# AbstractSituation identity / no RNG / no builtin ``hash()`` for stable identity). The L1 ``Agent``
# above stays the byte-for-byte concrete grid slice the framework + tools import; this layer
# adds the GENERAL decision behaviours over AbstractSituations and never touches the L1 ``Agent``.
#
# Canon (cite, never duplicate):
#   - _assets/gr-arc-3-sequence-diagnose.md (sheet 3)
#       The 4 divergence routes 1:1 with phases: locate the FIRST delta, consult WorldModel
#       ("is delta a known mechanism, which rule predicted it") and GoalPredicate ("did the
#       goal-test mismatch"), then route -- unknown object/interaction -> (1) EXPLORE /
#       a known rule mispredicted -> (2) WORLD / the goal-test was wrong -> (3) GOAL /
#       model right but path blocked-or-suboptimal -> (4) PLAN. First divergence localizes
#       the bug (R14); go back as far as the wrong layer, re-plan, drive on real again (MPC).
#   - _assets/gr-arc-3-terms.md
#       TERM-33 Outcome (the observed result the goal consult reads); TERM-17 EffectSignature
#       (an effect-unknown object is one with no learned InteractionRule yet); TERM-31
#       AbstractSituation (the «value» compared on canonical()); TERM-25 Profile / confidence.
#   - 04-specification SC-11 / SC-12 / SC-13 ; 05-test-strategy TS-11 / TS-12 / TS-13 ;
#     sequence choose-action v005 (#4 DetectFutility records the played MoveEffect; explore);
#     object-schema v003 §8 Q5 (fog: unknown -> a role=unknown object; keep coverage C1).
#
# DetectFutility here is the OBSERVE side of futility (the symmetric partner of
# solver.check_futility's PREDICT-side prune): every turn it classifies the ACTUALLY-played
# move's MoveEffect via classify_move_effect(prev, observed, goal.distance) and RECORDS it.
# =====================================================================================

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from agent.core.goal import GoalPredicate
from agent.core.situation import AbstractSituation, _canon_value
from agent.core.world_model import (
    InteractionRule,
    ModelWorld,
    MoveEffect,
    classify_move_effect,
)


# ------------------------------------------------------------------ DiagnoseDivergence (CMP-30)
# The four routing PHASES, 1:1 with the diagnose-sequence back-edges. Stable string labels
# (not a salted Enum) so equality / logging / serialization are deterministic across
# processes (DP-10). Each maps to exactly one phase the agent re-enters.
class Phase:
    """Namespace of the 4 divergence-routing phases (CMP-30; sequence-diagnose sheet 3).

    The routes are 1:1 with the phases the agent re-enters on a mispredicted move:
      * ``EXPLORE`` -- (1) an UNKNOWN mechanism: the observed change has no learned rule;
      * ``WORLD``   -- (2) a KNOWN rule MISPREDICTED: a rule fired but its effect was wrong;
      * ``GOAL``    -- (3) a GOAL-judgment error: the win/goal-test mismatched observation
        (a predicted win did not win, or a win happened off-model);
      * ``PLAN``    -- (4) model right but the PATH was blocked / suboptimal (re-plan only).
    """

    EXPLORE: str = "explore"
    WORLD: str = "world"
    GOAL: str = "goal"
    PLAN: str = "plan"

    ALL: tuple = (EXPLORE, WORLD, GOAL, PLAN)


@dataclass(frozen=True)
class RoutingVerdict:
    """The verdict :func:`diagnose_divergence` returns -- a «value» (CMP-30 output).

    Fields
    ------
    phase:
        The :class:`Phase` to re-enter (one of the 4 stable routes). This is the load-bearing
        field: the agent goes back as far as this layer and re-plans (MPC).
    delta:
        The FIRST diverging component between the armed prediction and the observation, as a
        small structural descriptor ``(kind, key)`` (e.g. ``("scalar", "remaining")`` or
        ``("relation", <desc>)`` or ``("role", <label>)`` or ``("goal", "win")``). It
        localizes the bug (R14); ``None`` iff no salient-state delta was found (a pure
        goal/plan divergence). NEVER a colour / coordinate / id (it keys on role labels /
        scalar names / relation descriptors only -- NFR-6).
    cause:
        A short, game-literal-free explanation token for logs (e.g. ``"unknown-mechanism"``).

    Frozen + primitive fields => deterministic equality (no RNG, no builtin-hash of mutable
    state). Two verdicts with the same phase/delta/cause compare equal."""

    phase: str
    delta: Optional[tuple] = None
    cause: str = ""

    def re_enter(self, router: "PhaseRouter") -> object:
        """Drive the agent back into the routed phase via ``router`` (returns whatever the
        phase handler returns). A convenience so a caller can do ``verdict.re_enter(router)``
        instead of ``router.route(verdict)``."""
        return router.route(self)


def _first_state_delta(predicted: AbstractSituation, observed: AbstractSituation) -> Optional[tuple]:
    """Locate the FIRST salient-state component where ``predicted`` and ``observed`` differ
    (R14: the earliest divergence localizes the bug). Deterministic scan order (DP-10):
    scalars by name, then relations by canonical order, then roles by label. Returns a
    ``(kind, key)`` descriptor or ``None`` if the salient AbstractSituations are identical.

    The descriptor keys ONLY on scalar NAMES / relation DESCRIPTORS / role LABELS -- never a
    colour / coordinate / id (NFR-6)."""
    if predicted.canonical() == observed.canonical():
        return None
    # 1) scalar gauges, by name (a Markov scalar mismatch is the cheapest, most local signal).
    # Compare on the CANONICAL value (same normaliser AbstractSituation.canonical() uses), so a scalar
    # that is canonically equal but not raw-equal (e.g. a list vs a tuple of equal content) is
    # NOT mis-reported as the divergence -- it must agree with the canonical short-circuit above
    # (R14 localizes the REAL first delta; DP-10 stays consistent with AbstractSituation identity).
    names = sorted(set(predicted.scalars.keys()) | set(observed.scalars.keys()))
    _MISSING = object()
    for name in names:
        pv = predicted.scalars.get(name, _MISSING)
        ov = observed.scalars.get(name, _MISSING)
        pc = pv if pv is _MISSING else _canon_value(pv)
        oc = ov if ov is _MISSING else _canon_value(ov)
        if pc != oc:
            return ("scalar", name)
    # 2) holding relations, in canonical (sorted) order.
    pred_rels = frozenset(tuple(r) for r in predicted.relations)
    obs_rels = frozenset(tuple(r) for r in observed.relations)
    diff_rels = pred_rels ^ obs_rels
    if diff_rels:
        first = sorted(diff_rels, key=repr)[0]
        return ("relation", first)
    # 3) role profiles, by label.
    labels = sorted(set(predicted.roles.keys()) | set(observed.roles.keys()))
    for label in labels:
        p = predicted.roles.get(label)
        o = observed.roles.get(label)
        if (p is None) != (o is None) or (p is not None and o is not None
                                          and _profile_key(p) != _profile_key(o)):
            return ("role", label)
    # Fallback: the canonical forms differ but no component above caught it (defensive) --
    # report a generic state delta so the caller still routes deterministically.
    return ("state", None)


def _profile_key(profile) -> tuple:
    """A deterministic, order-independent key for a Profile's entries (DP-10), CONSISTENT with
    ``AbstractSituation._canon_profile`` (it canonicalises each value the same way), so the role-layer
    delta scan agrees with the canonical short-circuit above -- a role is reported as differing
    only when ``AbstractSituation.canonical()`` would also see it differ (R14 / NFR-6). Entries are keyed
    by ``dim_id`` (unique), so the sort is total without comparing heterogeneous canonical values."""
    return tuple(sorted(
        ((dim_id, _canon_value(value), conf) for dim_id, (value, conf) in profile.entries.items()),
        key=lambda e: e[0],
    ))


def _rule_applies(world: Optional[ModelWorld], prior: Optional[AbstractSituation],
                  move: Optional[int]) -> bool:
    """True iff the world model has a learned mechanism (an :class:`InteractionRule`) that
    FIRES for ``(prior, move)`` -- the WorldModel consult of the diagnose sequence ("is delta
    a known mechanism, which rule predicted it"). A model with no applicable rule means the
    mechanism that produced the observed change is UNKNOWN (route to EXPLORE).

    Defensive: a ``None`` world or ``None`` prior is treated as "no known mechanism" (the
    agent cannot point at a rule), which routes an unexplained change to EXPLORE."""
    if world is None or prior is None:
        return False
    for rule in getattr(world, "rules", ()):  # type: ignore[attr-defined]
        if rule.applies(prior, move):
            return True
    return False


def diagnose_divergence(
    predicted: AbstractSituation,
    observed: AbstractSituation,
    world: Optional[ModelWorld] = None,
    goal: Optional[GoalPredicate] = None,
    prior: Optional[AbstractSituation] = None,
    move: Optional[int] = None,
    observed_win: Optional[bool] = None,
) -> RoutingVerdict:
    """DiagnoseDivergence (CMP-30; SC-11): localize the FIRST delta and route to the phase
    whose recognition was wrong (sequence-diagnose sheet 3).

    Runs ONLY on a divergence (a committed move's armed prediction met reality and they did
    NOT match -- the caller checks that). It locates the first diverging component, consults
    the world model (is the delta a known mechanism?) and the goal (did the goal-test
    mismatch?), and returns a :class:`RoutingVerdict` whose ``phase`` is 1:1 with the cause:

      (1) EXPLORE -- unknown object / interaction: the salient state changed but NO learned
          :class:`InteractionRule` fires for ``(prior, move)`` (the mechanism is unknown);
      (2) WORLD   -- a known rule mispredicted: the state changed AND a rule DID fire (so the
          mechanism is known but its predicted effect was wrong -- refine / re-type the rule).
          The over-prune case (futility predicted a move futile but it progressed) also lands
          here, as a MoveEffect rule error;
      (3) GOAL    -- a goal-judgment error: the salient state was predicted correctly
          (predicted == observed) yet the goal-test disagreed with the observed win signal
          (a predicted win did not win, or a win happened off-model);
      (4) PLAN    -- model right AND goal right, but the path was blocked / suboptimal: nothing
          about world or goal was wrong, so only the plan needs redoing.

    Parameters
    ----------
    predicted, observed:
        The armed predicted :class:`AbstractSituation` (for the committed move) and the observed one.
    world:
        The :class:`world_model.ModelWorld` consulted for "is the delta a known mechanism".
        Optional; absent => an unexplained change routes to EXPLORE.
    goal:
        The :class:`goal.GoalPredicate` consulted for "did the goal-test mismatch". Optional;
        absent => no goal divergence can be diagnosed (a no-state-delta case routes to PLAN).
    prior:
        The :class:`AbstractSituation` BEFORE the committed move -- needed to ask the model which rule
        (if any) fires for ``(prior, move)``. Optional; absent => no rule can be pointed at, so
        a state delta routes to EXPLORE.
    move:
        The committed GameMove id (opaque token) -- the move whose prediction diverged.
    observed_win:
        The env's ACTUAL win signal for the observed turn (the goal consult's ground truth,
        from the Outcome). When ``None`` it defaults to ``goal.test(observed)`` (the model's
        own read), so a state-correct turn yields PLAN unless an explicit win mismatch is
        supplied.

    Returns
    -------
    RoutingVerdict
        ``phase`` is one of :class:`Phase`; ``delta`` localizes the first divergence; ``cause``
        is a log token. Deterministic (DP-10): the scan order and the route decision use only
        canonical AbstractSituation content and rule applicability -- no RNG, no builtin ``hash()``.
    """
    delta = _first_state_delta(predicted, observed)

    if delta is not None:
        # The salient STATE diverged. Ask the world model whether the mechanism is known.
        if _rule_applies(world, prior, move):
            # A rule fired but its effect was wrong -> the model is wrong (route 2).
            return RoutingVerdict(phase=Phase.WORLD, delta=delta, cause="known-rule-mispredict")
        # No rule explains the observed change -> an unknown mechanism (route 1).
        return RoutingVerdict(phase=Phase.EXPLORE, delta=delta, cause="unknown-mechanism")

    # The salient STATE matched (the world model was right). The only thing left that can have
    # diverged is the GOAL judgment or the PLAN. Consult the goal.
    if goal is not None:
        predicted_win = bool(goal.test(predicted))
        actual_win = observed_win if observed_win is not None else bool(goal.test(observed))
        if predicted_win != actual_win:
            # The goal-test mismatched reality (predicted win did not win, OR won off-model).
            return RoutingVerdict(phase=Phase.GOAL, delta=("goal", "win"),
                                  cause="goal-judgment-error")

    # Model right AND goal right: nothing recognized wrong -> the PLAN was suboptimal (route 4).
    return RoutingVerdict(phase=Phase.PLAN, delta=None, cause="model-correct-bad-path")


# A phase handler: re-entering a phase is an opaque callback taking the verdict and doing
# whatever that phase does (re-plan, re-explore, re-abduce the goal, refine the model). The
# router never interprets the return value -- it just dispatches deterministically.
PhaseHandler = Callable[[RoutingVerdict], object]


@dataclass
class PhaseRouter:
    """Dispatch a :class:`RoutingVerdict` to the matching phase handler (CMP-30 re-entry).

    Holds one optional handler per :class:`Phase`. :meth:`route` calls EXACTLY the handler the
    verdict's ``phase`` names -- this is the "agent re-enters the routed phase" the spec asks
    for; a test spies re-entry by passing recording callbacks. Determinism (DP-10): the
    dispatch is a direct phase->handler lookup (no RNG, no ordering ambiguity). A missing
    handler for the routed phase is a no-op returning ``None`` (the caller may treat that as
    "phase not wired yet")."""

    explore: Optional[PhaseHandler] = None
    world: Optional[PhaseHandler] = None
    goal: Optional[PhaseHandler] = None
    plan: Optional[PhaseHandler] = None

    def _handler_for(self, phase: str) -> Optional[PhaseHandler]:
        return {
            Phase.EXPLORE: self.explore,
            Phase.WORLD: self.world,
            Phase.GOAL: self.goal,
            Phase.PLAN: self.plan,
        }.get(phase)

    def route(self, verdict: RoutingVerdict) -> object:
        """Invoke the handler for ``verdict.phase`` (the phase the agent re-enters). Returns
        the handler's result, or ``None`` when no handler is wired for that phase."""
        handler = self._handler_for(verdict.phase)
        if handler is None:
            return None
        return handler(verdict)


# ------------------------------------------------------------------ DetectFutility (CMP-32, observe)
@dataclass
class DetectFutility:
    """The OBSERVE side of futility (CMP-32; SC-12 record half) -- the symmetric partner of
    ``solver.check_futility``'s PREDICT-side prune.

    Each turn, after the move is PLAYED, :meth:`record` classifies the ACTUALLY-played move's
    :class:`world_model.MoveEffect` from the (prev AbstractSituation, observed AbstractSituation, goal-distance)
    triple via :func:`world_model.classify_move_effect` -- the EXACT same 3-way oracle the
    prune side uses -- and appends it to an ordered history:
      * ``invariant``   -- the observed AbstractSituation is IDENTICAL to the previous one;
      * ``no_progress`` -- the AbstractSituation changed but the goal-distance is UNCHANGED;
      * ``progress``    -- the goal-distance DECREASED.
    ``invariant`` and ``no_progress`` are *futile* (``MoveEffect.futile``); only ``progress``
    advances the goal. Holds the per-turn record (a list of MoveEffect labels) and small
    counters, so the agent can detect a futile streak and the over-prune divergence (a played
    move observed as ``progress`` that the predict side had pruned -- routed to WORLD).

    Stateful (it accumulates the history) but deterministic (DP-10): it only calls the injected
    goal-distance and compares canonical AbstractSituations -- no RNG, no builtin ``hash()``."""

    history: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=lambda: {e: 0 for e in MoveEffect.ALL})

    def record(
        self,
        prev: AbstractSituation,
        observed: AbstractSituation,
        goal_distance: Callable[[AbstractSituation], object],
    ) -> str:
        """Classify and RECORD the played move's MoveEffect for this turn; return the label.

        ``goal_distance`` is the injected SearchHeuristic ``AbstractSituation -> distance`` (API-05),
        identical to the prune side -- only its order relation is used, so the distance may be
        any comparable value. Appends the label to :attr:`history` and bumps :attr:`counts`."""
        effect = classify_move_effect(prev, observed, goal_distance)
        self.history.append(effect)
        self.counts[effect] = self.counts.get(effect, 0) + 1
        return effect

    def last(self) -> Optional[str]:
        """The most recently recorded MoveEffect label, or ``None`` before any turn."""
        return self.history[-1] if self.history else None

    def futile_run(self) -> int:
        """The length of the CURRENT trailing run of futile moves (``invariant`` /
        ``no_progress``). ``0`` iff the last recorded move made progress (or none recorded).
        Lets the agent decide it is thrashing and should re-plan / explore."""
        run = 0
        for effect in reversed(self.history):
            if MoveEffect.futile(effect):
                run += 1
            else:
                break
        return run

    def is_over_prune(self, predicted_futile: bool, observed_effect: str) -> bool:
        """True iff this turn is an OVER-PRUNE divergence (sequence-diagnose G1, v004): the
        PREDICT side judged the played move futile (``predicted_futile``) but it was OBSERVED
        as ``progress``. That is a MoveEffect rule error -> the caller routes to
        :data:`Phase.WORLD` and corrects the over-pruning (the safety valve that keeps futility
        from blocking a win)."""
        return predicted_futile and not MoveEffect.futile(observed_effect)


# ------------------------------------------------------------ ExploreWorld / ProbeObject (CMP-33/34)
# Default understanding thresholds (TS-13 fixture defaults; TERM-27-style tau): an object is
# UNDERSTOOD once its InteractionRule confidence reaches ``tau_rule`` OR ``k`` consecutive
# probes had expected == observed. Game-literal-free scalars (a confidence threshold and a
# repeat count), never a colour / coordinate.
DEFAULT_TAU_RULE: float = 0.8
DEFAULT_K_CONFIRM: int = 3


def _cell_key(cell) -> tuple:
    """A deterministic, hashable key for a frontier cell coordinate. Coordinates here are
    EPISTEMIC bookkeeping (which cells the agent has OBSERVED), NOT a goal/role key, so using
    them for the visited set does not introduce a game literal into any predicate (NFR-6): no
    GoalPredicate / EffectSignature ever consults them."""
    return tuple(cell)


def choose_frontier(
    frontier: Sequence[tuple],
    visited: "frozenset | set",
    epistemic_value: Optional[Callable[[tuple], object]] = None,
) -> Optional[tuple]:
    """ExploreWorld frontier selection (CMP-33; SC-13(a)): prefer an UNVISITED frontier cell
    (maximal epistemic value), deterministically.

    Among ``frontier`` cells, the UNVISITED ones (not in ``visited``) are preferred over any
    visited cell -- observing a never-seen cell has strictly higher epistemic value than
    re-observing a known one (fog: object-schema §8 Q5). Ties (and the all-visited fallback)
    break by ``epistemic_value`` descending, then by the cell's sort order -- a TOTAL
    deterministic order (DP-10), no RNG.

    Returns the chosen cell, or ``None`` for an empty frontier. ``epistemic_value`` defaults to
    a constant (so the order is purely unvisited-first then coordinate-sorted)."""
    if not frontier:
        return None
    visited_set = set(visited)
    value = epistemic_value if epistemic_value is not None else (lambda _c: 0)

    def _rank(cell: tuple) -> tuple:
        unvisited = _cell_key(cell) not in visited_set
        # unvisited first (True sorts after False, so negate); then higher value; then cell order.
        return (not unvisited, _neg_key(value(cell)), _cell_key(cell))

    return min(frontier, key=_rank)


def _neg_key(value: object) -> tuple:
    """A descending-order key for an epistemic value (higher value sorts FIRST). Wraps with the
    type name so heterogeneous-but-comparable values never raise on ``<`` (DP-10). For numeric
    values this negates; for everything else it falls back to repr (stable, deterministic)."""
    if isinstance(value, bool):
        return ("bool", not value)
    if isinstance(value, (int, float)):
        return ("num", -value)
    return ("other", repr(value))


@dataclass
class ProbeObject:
    """Track the active probing of ONE effect-unknown object and decide when it is UNDERSTOOD
    (CMP-34; SC-13(b)(c)).

    The agent probes an object whose interaction effect is not yet learned, updates the world
    model by EXPECTED-vs-OBSERVED, and STOPS once the object is understood. "Understood" =
    EITHER the object's :class:`world_model.InteractionRule` confidence has reached
    ``tau_rule`` OR ``expected == observed`` held for ``k`` CONSECUTIVE probes (a confirmation
    streak). The object id is a stable, opaque handle (e.g. its EffectSignature.stable_id or a
    role label) -- never a colour / coordinate (NFR-6).

    Stateful (it accumulates the confirmation streak + the learned confidence) but
    deterministic (DP-10): expected/observed are compared on :meth:`AbstractSituation.canonical` (or
    plain equality for scalar observations); no RNG, no builtin ``hash()``."""

    object_id: str
    tau_rule: float = DEFAULT_TAU_RULE
    k_confirm: int = DEFAULT_K_CONFIRM
    confidence: float = 0.0
    streak: int = 0
    probes: int = 0
    rule: Optional[InteractionRule] = None

    def observe(self, expected: object, observed: object,
                confidence: Optional[float] = None) -> bool:
        """Record one probe's outcome and update the model; return whether the object is now
        UNDERSTOOD (so the caller stops probing it).

        ``expected`` / ``observed`` are compared for equality (AbstractSituations compare on
        ``canonical()`` via their ``__eq__``; scalars/tuples compare directly). A MATCH extends
        the confirmation streak and (absent an explicit ``confidence``) nudges the learned
        confidence up; a MISMATCH resets the streak and lowers confidence -- the
        expected-vs-observed world-model update (SC-13(b)). An explicit ``confidence`` (e.g.
        the InteractionRule confidence the model now reports) overrides the nudge. Returns
        :meth:`understood`."""
        self.probes += 1
        match = _values_equal(expected, observed)
        if match:
            self.streak += 1
        else:
            self.streak = 0
        if confidence is not None:
            self.confidence = float(confidence)
        else:
            # Deterministic nudge toward/away from understanding (a fixed step, clamped).
            step = 0.34
            if match:
                self.confidence = min(1.0, self.confidence + step)
            else:
                self.confidence = max(0.0, self.confidence - step)
        return self.understood()

    def attach_rule(self, rule: InteractionRule) -> None:
        """Record the learned :class:`InteractionRule` for this object and sync the tracked
        confidence to the rule's (so :meth:`understood` can fire on the confidence threshold)."""
        self.rule = rule
        self.confidence = float(rule.confidence)

    def understood(self) -> bool:
        """True iff the object is UNDERSTOOD: its InteractionRule confidence has reached
        ``tau_rule`` OR ``expected == observed`` has held for ``k_confirm`` consecutive probes
        (either criterion stops further probing -- SC-13(c))."""
        return self.confidence >= self.tau_rule or self.streak >= self.k_confirm


@dataclass
class ExploreWorld:
    """Active observation under fog (CMP-33; SC-13): prefer unvisited frontier cells and probe
    effect-unknown objects until understood, stopping probes on understood objects.

    Holds the set of VISITED frontier cells (epistemic bookkeeping) and a per-object
    :class:`ProbeObject` registry keyed by the object's opaque id. :meth:`next_frontier` picks
    the next observation target (unvisited-first); :meth:`probe` records a probe outcome for an
    object and returns whether to KEEP probing it; :meth:`should_probe` answers "is this object
    still worth probing" so a spy can confirm no further probes once understood (SC-13(c)).

    Determinism (DP-10): frontier choice and probe bookkeeping use canonical identity / sorted
    order only -- no RNG, no builtin ``hash()`` for stable identity. Game-literal-free (NFR-6):
    object ids are opaque handles; frontier coordinates are epistemic-only and never enter a
    predicate."""

    tau_rule: float = DEFAULT_TAU_RULE
    k_confirm: int = DEFAULT_K_CONFIRM
    visited: set = field(default_factory=set)
    probers: Dict[str, ProbeObject] = field(default_factory=dict)

    # ---- frontier (epistemic) ----
    def mark_visited(self, cell: tuple) -> None:
        """Record that ``cell`` has been OBSERVED (so it is no longer an unvisited frontier)."""
        self.visited.add(_cell_key(cell))

    def next_frontier(
        self,
        frontier: Sequence[tuple],
        epistemic_value: Optional[Callable[[tuple], object]] = None,
    ) -> Optional[tuple]:
        """The next frontier cell to observe -- an UNVISITED one preferred (SC-13(a)). Delegates
        to :func:`choose_frontier` with the agent's visited set. Does NOT auto-mark the chosen
        cell visited (the caller marks it once the observation actually lands)."""
        return choose_frontier(frontier, self.visited, epistemic_value)

    # ---- probing (effect-unknown objects) ----
    def prober(self, object_id: str) -> ProbeObject:
        """Get (creating on first use) the :class:`ProbeObject` tracking ``object_id`` with this
        explorer's ``tau_rule`` / ``k_confirm`` defaults."""
        p = self.probers.get(object_id)
        if p is None:
            p = ProbeObject(object_id=object_id, tau_rule=self.tau_rule, k_confirm=self.k_confirm)
            self.probers[object_id] = p
        return p

    def should_probe(self, object_id: str) -> bool:
        """True iff ``object_id`` is still worth probing -- i.e. it is NOT yet understood. Once
        the object is understood this returns ``False`` forever (the agent does not waste moves
        re-touching a understood object -- SC-13(c)); a spy asserts no further probes follow."""
        p = self.probers.get(object_id)
        if p is None:
            return True                      # never probed -> effect unknown -> worth probing
        return not p.understood()

    def probe(
        self,
        object_id: str,
        expected: object,
        observed: object,
        confidence: Optional[float] = None,
    ) -> bool:
        """Probe ``object_id`` once: record expected-vs-observed (updating the world model) and
        return whether to KEEP probing it (``True`` until understood). After this call,
        :meth:`should_probe` reflects the new understanding state. Returns ``not understood``."""
        p = self.prober(object_id)
        understood = p.observe(expected, observed, confidence=confidence)
        return not understood

    def understood_objects(self) -> frozenset:
        """The set of object ids currently understood (for assertions / introspection)."""
        return frozenset(oid for oid, p in self.probers.items() if p.understood())


def _values_equal(a: object, b: object) -> bool:
    """Deterministic equality for probe expected-vs-observed comparison. :class:`AbstractSituation`
    compares on ``canonical()`` via its ``__eq__``; everything else uses ``==``. Never relies on
    builtin ``hash()`` (DP-10)."""
    if isinstance(a, AbstractSituation) and isinstance(b, AbstractSituation):
        return a.canonical() == b.canonical()
    return a == b
