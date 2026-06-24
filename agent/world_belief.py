"""Persistent World Belief for PARTIAL-OBSERVABILITY levels (the ls20-L7 "torch/fog" finding; R12).

Principle (user, 2026-06-03): "ARC-AGI-3 = maximally use limited input to understand the level + its
goal; waste NO information; ORGANIZE it so it is immediately usable; CONSTANTLY update it." This module
is the spatial/terrain layer of that belief — the first-class, persistent state perception writes to and
planning reads from. It is the opposite of frame-local reasoning.

Design:
  * Localization is FREE here: the frame is a fixed 64x64 GLOBAL grid and the avatar is always rendered
    (never fogged), so the avatar's absolute (row,col) is directly observed. No SLAM pose estimation.
  * The map is a global occupancy of last-known colour per cell. UNKNOWN = never seen.
  * NEVER DISCARD: a cell that re-fogs (leaves the torch) KEEPS its last-known value — re-fogging does
    not overwrite knowledge. Only a fresh (lit) observation overwrites a cell.
  * CONSTANTLY UPDATE + revisable: every frame integrates; `last_seen_t` lets a consumer judge staleness
    (static terrain stays valid; time-varying objects belong in a separate model layer — R8 routes).
  * IMMEDIATELY USABLE: query `passable()`, `frontiers()` (the exploration boundary), `coverage()`.

Pure module (numpy only, no framework imports) so it is unit-testable and policy-agnostic.
"""
from __future__ import annotations

import numpy as np

UNKNOWN = -1


def detect_fog_colour(frames):
    """Infer the FOG colour (the 'unobserved' marker) from how the view changes as the avatar moves:
    fog = the colour cells most often REVERT TO across consecutive frames (the re-fogging trail dominates;
    the avatar's own short trail is negligible). Returns the colour int, or None if undetermined.
    `frames` = iterable of 2-D int arrays (>=2)."""
    frames = [np.asarray(f, dtype=int) for f in frames]
    if len(frames) < 2:
        return None
    became = {}
    for g0, g1 in zip(frames, frames[1:]):
        changed = g0 != g1
        vals, cnts = np.unique(g1[changed], return_counts=True)
        for v, c in zip(vals, cnts):
            became[int(v)] = became.get(int(v), 0) + int(c)
    if not became:
        return None
    return max(became, key=became.get)


class WorldBelief:
    def __init__(self, h: int = 64, w: int = 64, fog_colour: int | None = None):
        self.h, self.w = h, w
        self.fog_colour = fog_colour          # the 'unobserved' colour (e.g. 5 in L7); may be set later
        self.map = np.full((h, w), UNKNOWN, dtype=int)
        self.seen = np.zeros((h, w), dtype=bool)
        self.last_seen_t = np.full((h, w), -1, dtype=int)
        self.t = -1
        self.avatar_tl: tuple[int, int] | None = None

    # ------------------------------------------------------------------ update
    def update(self, grid, avatar_tl=None) -> int:
        """Integrate one observed frame. Lit (non-fog) cells overwrite the map; fogged cells are PRESERVED
        at their last-known value (never discarded). Returns the count of newly-revealed cells."""
        grid = np.asarray(grid, dtype=int)
        self.t += 1
        if avatar_tl is not None:
            self.avatar_tl = avatar_tl
        lit = np.ones_like(grid, dtype=bool) if self.fog_colour is None else (grid != self.fog_colour)
        newly = int((lit & ~self.seen).sum())
        self.map[lit] = grid[lit]
        self.seen |= lit
        self.last_seen_t[lit] = self.t
        return newly

    # ------------------------------------------------------------------ queries
    def coverage(self) -> int:
        return int(self.seen.sum())

    def is_unknown(self, r, c) -> bool:
        return not (0 <= r < self.h and 0 <= c < self.w) or not self.seen[r, c]

    def passable(self, cell, passable_colours, optimistic_unknown=True) -> bool:
        """Known-passable colour -> True; known wall -> False; UNKNOWN -> `optimistic_unknown`
        (True drives exploration INTO the fog; False is the conservative read)."""
        r, c = cell
        if not (0 <= r < self.h and 0 <= c < self.w):
            return False
        if not self.seen[r, c]:
            return optimistic_unknown
        return int(self.map[r, c]) in passable_colours

    def frontier_cells(self, passable_colours):
        """Cells the avatar's footprint could stand on whose neighbourhood touches UNKNOWN = where to go to
        reveal more. Returns a set of KNOWN-passable cells 4-adjacent to an UNKNOWN cell (the classic
        frontier). Planning navigates the avatar toward the nearest frontier to grow the map."""
        known_pass = self.seen & np.isin(self.map, list(passable_colours))
        unknown = ~self.seen
        fr = set()
        rs, cs = np.where(known_pass)
        for r, c in zip(rs.tolist(), cs.tolist()):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.h and 0 <= nc < self.w and unknown[nr, nc]:
                    fr.add((r, c))
                    break
        return fr

    def render(self, r0=0, c0=0, r1=None, c1=None) -> str:
        """ASCII of the believed map; '.' = UNKNOWN. For debugging / introspection."""
        r1 = self.h - 1 if r1 is None else r1
        c1 = self.w - 1 if c1 is None else c1
        hexd = "0123456789ABCDEF"
        out = []
        for r in range(r0, r1 + 1):
            out.append("".join(hexd[int(self.map[r, c])] if self.seen[r, c] else "."
                               for c in range(c0, c1 + 1)))
        return "\n".join(out)
