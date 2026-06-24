"""Reactive frontier exploration over a WorldBelief (partial-observability levels; R12, spec 89 §4.1).

A fixed replay cannot explore fog — exploration must REACT to what each move reveals. This policy heads
the avatar toward the nearest position that would reveal UNKNOWN cells, treating unknown as *optimistically
passable* (so it walks into the fog to learn it). If an optimistic step turns out to hit a wall, the move
is a no-op, the belief learns the wall from the next frame, and the BFS routes around it next turn — the
loop is self-correcting. Pure module (no framework imports); the action vocabulary + footprint are passed
in, so it is game-agnostic (ls20 = 5x5 footprint, ±5 moves).
"""
from __future__ import annotations

from collections import deque


def _footprint(tl, footprint_offsets):
    return [(tl[0] + dr, tl[1] + dc) for dr, dc in footprint_offsets]


def frontier_explore_action(belief, avatar_tl, moves, footprint_offsets, passable_colours):
    """Return the action key (from `moves`) for the first step toward the nearest UNKNOWN-revealing
    position, or None if nothing new is reachable (fully explored, or walled off).

    moves: {action_key: (dr, dc)}.  footprint_offsets: list of (dr, dc) for the avatar's body.
    Unknown cells are treated as passable here (optimistic) so the search reaches into the fog."""
    def occupiable(tl):
        return all(belief.passable(c, passable_colours, optimistic_unknown=True)
                   for c in _footprint(tl, footprint_offsets))

    def reveals_unknown(tl):
        return any(belief.is_unknown(r, c) for r, c in _footprint(tl, footprint_offsets))

    seen = {avatar_tl}
    q = deque([(avatar_tl, None)])          # (tl, first_action_taken_to_get_into_this_branch)
    while q:
        tl, first = q.popleft()
        if first is not None and reveals_unknown(tl):
            return first                     # nearest position whose footprint touches fog → head there
        for a, (dr, dc) in moves.items():
            ntl = (tl[0] + dr, tl[1] + dc)
            if ntl in seen or not occupiable(ntl):
                continue
            seen.add(ntl)
            q.append((ntl, a if first is None else first))
    return None                              # nothing left to reveal that is reachable


def bfs_action_dist(belief, avatar_tl, moves, footprint_offsets, passable_colours, is_goal_tl,
                    optimistic_unknown=True, extra_passable=None):
    """General BFS returning (first_action, n_steps) to the nearest top-left where `is_goal_tl(tl)` holds,
    or (None, 0) if already satisfied, or (None, None) if unreachable. `extra_passable` = a cell-set the
    footprint may stand on regardless of learned colour (e.g. an unprobed marker / a refuel pickup the
    avatar hasn't learned to be walkable yet — treated optimistically, self-correcting like fog). Used by
    the refuel-detour and marker-probe (it needs the DISTANCE, not just the action, for the fuel budget)."""
    extra = extra_passable or set()

    def occupiable(tl):
        for dr, dc in footprint_offsets:
            cell = (tl[0] + dr, tl[1] + dc)
            if cell in extra:
                continue
            if not belief.passable(cell, passable_colours, optimistic_unknown=optimistic_unknown):
                return False
        return True

    if is_goal_tl(avatar_tl):
        return (None, 0)
    seen = {avatar_tl}
    q = deque([(avatar_tl, None, 0)])
    while q:
        tl, first, d = q.popleft()
        for a, (dr, dc) in moves.items():
            ntl = (tl[0] + dr, tl[1] + dc)
            if ntl in seen or not occupiable(ntl):
                continue
            nf = a if first is None else first
            if is_goal_tl(ntl):
                return (nf, d + 1)
            seen.add(ntl)
            q.append((ntl, nf, d + 1))
    return (None, None)


def bfs_path_action(belief, avatar_tl, goal_tls, moves, footprint_offsets, passable_colours,
                    optimistic_unknown=False):
    """Once a target is known (e.g. the goal cell, or a refuel box), step toward it on
    believed-passable terrain. Returns the first action of a shortest path to any cell in `goal_tls`,
    or None if unreachable on the current belief."""
    goal = set(goal_tls)
    if avatar_tl in goal:
        return None
    def occupiable(tl):
        return all(belief.passable(c, passable_colours, optimistic_unknown=optimistic_unknown)
                   for c in _footprint(tl, footprint_offsets))
    seen = {avatar_tl}
    q = deque([(avatar_tl, None)])
    while q:
        tl, first = q.popleft()
        for a, (dr, dc) in moves.items():
            ntl = (tl[0] + dr, tl[1] + dc)
            if ntl in seen or not occupiable(ntl):
                continue
            if ntl in goal:
                return a if first is None else first
            seen.add(ntl)
            q.append((ntl, a if first is None else first))
    return None
