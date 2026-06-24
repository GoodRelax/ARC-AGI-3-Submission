"""Object/role layer of the World Belief (spec 89 §2; R1/R8/R11).

The terrain layer (`world_belief.py`) accumulates static occupancy. This layer tracks the DYNAMIC content:
segment objects from each frame, give them stable identities across frames (never discard a track when it
re-fogs/occludes), and LEARN each object's motion model — crucially, **the period + route of a moving
object** (the L6 period-8 transformer finding, R8). Descriptive geometry names come from
`object-naming.md` §2; functional affordance names (effect on another object) need interaction probes and
are added by a later increment. Pure module (numpy only).
"""
from __future__ import annotations

import numpy as np


def cluster_by_proximity(cells, gap=2):
    """Group cells so any two within Chebyshev distance `gap` share a cluster (a glyph drawn with small
    gaps stays one object). Returns a list of frozensets."""
    cells = list(cells)
    clusters: list[set] = []
    for (r, c) in cells:
        hit = [i for i, cl in enumerate(clusters)
               if any(max(abs(r - rr), abs(c - cc)) <= gap for rr, cc in cl)]
        if not hit:
            clusters.append({(r, c)})
        else:
            merged = {(r, c)}
            for i in sorted(hit, reverse=True):
                merged |= clusters.pop(i)
            clusters.append(merged)
    return [frozenset(cl) for cl in clusters]


def position(cells):
    """Canonical integer position of an object = its bounding-box top-left (stable for tracking)."""
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return (min(rs), min(cs))


def shape_base(cells):
    """Minimal geometric base (object-naming.md §2 subset): dot/bar/box/rect/blob."""
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    h = max(rs) - min(rs) + 1
    w = max(cs) - min(cs) + 1
    n = len(cells)
    if n == 1:
        return "dot"
    if h == 1 or w == 1:
        return "bar" if n == max(h, w) else "blob"
    if n == h * w:
        return "box" if h == w else "rect"
    return "blob"


class Track:
    __slots__ = ("id", "colour", "cells", "history")

    def __init__(self, tid, colour, cells, t):
        self.id = tid
        self.colour = colour
        self.cells = cells
        self.history = [(t, position(cells), cells)]      # (t, bbox-TL, cells)

    def observe(self, cells, t):
        self.cells = cells
        self.history.append((t, position(cells), cells))

    def positions(self):
        return [p for _, p, _ in self.history]

    def is_moving(self):
        ps = self.positions()
        return len(set(ps)) > 1

    def _contiguous_tail(self):
        """The LONGEST run of consecutive-t observations (period detection needs no gaps; an occlusion
        gap must not shrink us to a short tail — the shape-changer was occluded 1 frame in L6)."""
        best, run = [], [self.history[0]]
        for h in self.history[1:]:
            if h[0] == run[-1][0] + 1:
                run.append(h)
            else:
                if len(run) > len(best):
                    best = run
                run = [h]
        if len(run) > len(best):
            best = run
        return [p for _, p, _ in best]

    def period(self, min_cycles=2):
        """Smallest P>=1 such that the (contiguous) position sequence repeats with period P, observed
        over >= `min_cycles` full cycles. Returns P or None (aperiodic / too little data / static)."""
        ps = self._contiguous_tail()
        n = len(ps)
        if n < 2 or len(set(ps)) == 1:
            return None
        for P in range(1, n // min_cycles + 1):
            if all(ps[i] == ps[i + P] for i in range(n - P)):
                return P
        return None

    def route(self):
        """For a periodic mover: {phase 0..P-1: bbox-TL position} over one cycle. Phase is relative to
        the start of the contiguous tail (absolute phase must be aligned to the game clock externally)."""
        P = self.period()
        if P is None:
            return None
        ps = self._contiguous_tail()
        return {phase: ps[phase] for phase in range(P)}

    def name(self):
        mv = "moving" if self.is_moving() else "static"
        if self.is_moving() and self.period() is not None:
            mv = f"periodic(P={self.period()})"
        return f"{mv} {shape_base(self.cells)} c{self.colour}"


class ObjectTracker:
    """Maintains stable object tracks across frames. NEVER deletes a track (re-fog/occlusion just stops
    extending its history) — consistent with the belief's never-discard rule."""

    def __init__(self, gap=2, match_dist=8):
        self.gap = gap
        self.match_dist = match_dist
        self.tracks: dict[int, Track] = {}
        self._next = 0
        self.t = -1

    def _segment(self, grid, object_colours):
        objs = []
        for col in object_colours:
            cells = [(int(r), int(c)) for r, c in zip(*np.where(grid == col))]
            for cl in cluster_by_proximity(cells, self.gap):
                objs.append((col, cl))
        return objs

    def update(self, grid, object_colours):
        """Segment objects of the given colours and match them to existing tracks (nearest bbox-TL within
        match_dist and same colour); spawn new tracks for unmatched detections."""
        grid = np.asarray(grid, dtype=int)
        self.t += 1
        dets = self._segment(grid, object_colours)
        used = set()
        for col, cells in dets:
            p = position(cells)
            best, bestd = None, self.match_dist + 1
            for tid, tr in self.tracks.items():
                if tid in used or tr.colour != col:
                    continue
                tp = tr.history[-1][1]
                d = abs(tp[0] - p[0]) + abs(tp[1] - p[1])
                if d < bestd:
                    best, bestd = tid, d
            if best is not None:
                self.tracks[best].observe(cells, self.t)
                used.add(best)
            else:
                self.tracks[self._next] = Track(self._next, col, cells, self.t)
                used.add(self._next)
                self._next += 1
        return self.tracks

    def movers(self):
        return [tr for tr in self.tracks.values() if tr.is_moving()]
