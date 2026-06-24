"""Locate the CONTROLLABLE object from raw frames, generally (no game literals) — needed to feed the
BeliefExplorer the avatar cells each turn (R12 wiring). The controllable = the non-background connected
component whose centroid MOVED since the previous frame (the thing our actions push around). Background =
the learned fog/floor (when known) plus the most-frequent colours (floor/wall/fog) as a cold-start
fallback. Pure module (numpy only). Validated on real ls20-L7 frames (`tools/ls20_l7_controllable.py`)."""
from __future__ import annotations

import numpy as np


def _components(mask):
    """4-connected components of a boolean mask -> list of cell-sets (pure flood fill)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                comp = []
                stack = [(r, c)]
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    comp.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                out.append(frozenset(comp))
    return out


def _centroid(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return (sum(rs) / len(rs), sum(cs) / len(cs))


def background_colours(grid, fog=None, passable=None, top=3):
    """Floor/wall/fog: the learned fog + passable colours, plus the `top` most frequent colours."""
    vals, cnts = np.unique(grid, return_counts=True)
    order = [int(v) for v, _ in sorted(zip(vals.tolist(), cnts.tolist()), key=lambda x: -x[1])]
    bg = set(order[:top])
    if fog is not None:
        bg.add(int(fog))
    if passable:
        bg |= set(int(x) for x in passable)
    return bg


def _norm(cells):
    minr = min(r for r, _ in cells)
    minc = min(c for _, c in cells)
    return frozenset((r - minr, c - minc) for r, c in cells), (minr, minc)


def find_controllable(grid, prev, fog=None, passable=None, max_step=30):
    """Return (controllable_cells | None, new_state). The controllable is the non-background component
    that underwent a RIGID TRANSLATION (same shape/size, shifted position) vs the previous frame — this
    cleanly distinguishes the avatar from things that merely CHANGE (a shrinking fuel bar, a blinking
    marker), which are not rigid movers. Among rigid movers, the largest (the avatar block) wins. On a
    static frame, re-locates the previous controllable by its shape. None until motion is first seen."""
    grid = np.asarray(grid, dtype=int)
    bg = background_colours(grid, fog, passable)
    comps = _components(~np.isin(grid, list(bg)))
    objs = [(*_norm(c), c) for c in comps]                # (shape, pos, cells)

    prev_shapes = (prev or {}).get("shapes", {})          # shape -> set of last-frame positions
    movers = []
    for shape, pos, cells in objs:
        prevpos = prev_shapes.get(shape)
        if prevpos and pos not in prevpos:
            d = min(abs(pos[0] - p[0]) + abs(pos[1] - p[1]) for p in prevpos)
            if 0 < d <= max_step:
                movers.append((len(cells), cells))        # rigid translation of an identical shape

    best = max(movers, key=lambda x: x[0])[1] if movers else None
    if best is None:
        cs = (prev or {}).get("ctrl_shape")               # static frame: re-find prev controllable by shape
        if cs is not None:
            for shape, pos, cells in objs:
                if shape == cs:
                    best = cells
                    break

    new = {"shapes": {}, "ctrl_shape": (prev or {}).get("ctrl_shape")}
    for shape, pos, cells in objs:
        new["shapes"].setdefault(shape, set()).add(pos)
    if best is not None:
        new["ctrl_shape"] = _norm(best)[0]
    return best, new
