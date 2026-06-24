"""FuelSensor — learns a depleting RESOURCE BAR + its pickups from observation (R12, spec 89 §4.4).

The ls20-L7 finding: a torch/fog level where FUEL is the binding constraint — the avatar drains 4px/move
(cap 84 = 21 moves) and a fuel-0 costs a life + respawns it at the level start. A fuel-blind explorer dies
every ~21 moves and can never range far. This sensor gives the belief explorer the scalar it needs to
detour-and-refuel BEFORE dying, learning everything (no game literals):

  * the BAR COLOUR is locked the moment a long horizontal HUD run is seen to DEPLETE on a move — that
    rejects static terrain rows (never shrink) and the short lives/score bits (run too short).
  * `fuel` = current count of that colour in the HUD band; `drain` = the modal per-move decrement.
  * REFUEL PICKUPS share the bar colour — the HUD shows the resource, the pickups ARE the resource. So
    `refuel_cells(belief)` = believed cells of the bar colour ABOVE the HUD band (in the play area).

Pure module (numpy only); the brain owns one and feeds it each frame's full grid (HUD included).
"""
from __future__ import annotations

import numpy as np


def _longest_h_run(band: np.ndarray, colour: int) -> int:
    """Length of the longest horizontal run of `colour` in `band` (a bar reads as one long run; the
    lives/score bits and most terrain read short), so the bar is identified by shape, not cell count."""
    best = 0
    for row in band:
        run = 0
        for v in row:
            run = run + 1 if int(v) == colour else 0
            if run > best:
                best = run
    return best


class FuelSensor:
    def __init__(self, hud_band: int = 5, min_bar: int = 16):
        self.hud_band = hud_band          # bottom N rows treated as HUD
        self.min_bar = min_bar            # a bar must run at least this long (excludes terrain/lives bits)
        self.bar_colour: int | None = None
        self.fuel: int | None = None
        self.fuel_max: int = 0
        self._drains: dict = {}           # per-move decrement -> count (modal = the true drain)
        self._prev_counts: dict = {}
        self._prev_fuel: int | None = None
        self.ready = False

    def observe(self, grid, fog, moved: bool) -> None:
        """Integrate one frame. `fog` = the learned fog colour (or None); `moved` = the avatar changed
        position this turn (so a fuel drop is a real per-move drain, not a respawn/refill)."""
        grid = np.asarray(grid, dtype=int)
        h = grid.shape[0]
        band = grid[h - self.hud_band:, :]
        counts = {int(c): int((band == c).sum()) for c in np.unique(band)
                  if fog is None or int(c) != fog}
        if self.bar_colour is None:
            # lock a LONG-RUN colour the moment it DEPLETES on a move: a real resource bar shrinks; static
            # terrain rows and the short lives/score HUD bits do not qualify.
            if moved and self._prev_counts:
                for c, n in counts.items():
                    if (c in self._prev_counts and n < self._prev_counts[c]
                            and _longest_h_run(band, c) >= self.min_bar):
                        self.bar_colour = c
                        self._prev_fuel = self._prev_counts[c]   # so the locking drain still counts
                        break
            self._prev_counts = counts
            if self.bar_colour is None:
                return
        f = counts.get(self.bar_colour, 0)
        if moved and self._prev_fuel is not None and f < self._prev_fuel:
            d = self._prev_fuel - f
            self._drains[d] = self._drains.get(d, 0) + 1
        self.fuel = f
        self.fuel_max = max(self.fuel_max, f)
        self._prev_fuel = f
        self.ready = True

    @property
    def drain(self) -> int | None:
        return max(self._drains, key=self._drains.get) if self._drains else None

    def moves_left(self) -> float | None:
        d = self.drain
        if self.fuel is None or not d:
            return None
        return self.fuel / d

    def refuel_cells(self, belief) -> set:
        """Believed PLAY-AREA cells of the bar colour = refuel pickups (the HUD band is excluded; the
        avatar refuels by moving its footprint onto one)."""
        if self.bar_colour is None:
            return set()
        m = belief.seen & (belief.map == self.bar_colour)
        top = belief.h - self.hud_band
        rs, cs = np.where(m)
        return {(int(r), int(c)) for r, c in zip(rs.tolist(), cs.tolist()) if r < top}
