"""[agent/core] perceive -- a Frame (64x64 grid) -> a list of multi-colour objects.

Objects are NON-BACKGROUND connected components (4-connected over the non-background
mask), so a multi-colour thing (e.g. an orange-capped blue avatar) is ONE object rather
than split per colour. Each object carries the raw material of the four foundational ARC
attributes (colour / shape / orientation / size) plus its position. Pure: numpy + stdlib,
no framework import, no game literals.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

GRID = 64


@dataclass(frozen=True)
class Obj:
    """One perceived object. ``colored`` carries the colour at every cell (the raw material
    for shape/orientation); ``dom_color`` is the most common colour; ``pos`` is the bbox
    top-left (the translation anchor)."""

    cells: frozenset            # {(r, c)}
    colored: frozenset          # {(r, c, color)}
    dom_color: int
    colors: frozenset           # set of colours present
    bbox: tuple                 # (r0, c0, r1, c1)
    size: int                   # len(cells)
    # Structural fields produced by the full object-schema v003 parser (laminar tree).
    # Defaulted so the L1-slice constructors stay valid and every cluster can rely on them.
    parts: tuple = ()           # recursive sub-objects (laminar: child.cells subset of cells); () = leaf
    is_field: bool = False      # background/field object (object-schema D1 / invariant C4)
    # Extra schema fields the object-schema v003 parser (agent/core/parse.py) fills in.
    # Defaulted so every existing Obj(...) call site (the L1 slice + 323 tests) stays valid.
    oid: int = -1               # stable parse-local identity (object-schema example-2 key); -1 = unassigned
    scale: int = 1              # integer render scale (object-schema scaled cue / a Dimension, never split)
    cue: str = ""               # winning cue name that explains this object (cheapest DescLen_cue)

    @property
    def pos(self) -> tuple:
        return (self.bbox[0], self.bbox[1])

    @property
    def centroid(self) -> tuple:
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return (sum(rs) / self.size, sum(cs) / self.size)


def background_colors(grid, learned=None, top: int = 3) -> set:
    """Floor/wall/void: the ``top`` most frequent colours, plus any learned passable colours.
    General (a frequency heuristic over the fixed board) -- no specific colour value baked in."""
    grid = np.asarray(grid, dtype=int)
    vals, cnts = np.unique(grid, return_counts=True)
    order = [int(v) for v, _ in sorted(zip(vals.tolist(), cnts.tolist()), key=lambda x: -x[1])]
    bg = set(order[:top])
    if learned:
        bg |= set(int(x) for x in learned)
    return bg


def perceive(grid, background=None, merge_gap: int = 2) -> list:
    """Segment ``grid`` into a list of :class:`Obj` (non-background components).

    Components of the SAME dominant colour within ``merge_gap`` Chebyshev cells are merged,
    so a mark rendered with small internal gaps reads as ONE object (general; not a game fact).
    """
    grid = np.asarray(grid, dtype=int)
    bg = background if background is not None else background_colors(grid)
    mask = ~np.isin(grid, list(bg))
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objs = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                objs.append(_build(grid, cells))
    if merge_gap > 0:
        objs = _merge_proximity(objs, merge_gap)
    return objs


def _close(a: Obj, b: Obj, gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox
    dr = max(0, ar0 - br1, br0 - ar1)
    dc = max(0, ac0 - bc1, bc0 - ac1)
    if max(dr, dc) > gap:
        return False
    return min(max(abs(r1 - r2), abs(c1 - c2)) for r1, c1 in a.cells for r2, c2 in b.cells) <= gap


def _merge_proximity(objs, gap: int) -> list:
    parent = list(range(len(objs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            if objs[i].dom_color == objs[j].dom_color and _close(objs[i], objs[j], gap):
                parent[find(i)] = find(j)
    groups: dict = {}
    for i in range(len(objs)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idxs in groups.values():
        if len(idxs) == 1:
            out.append(objs[idxs[0]])
            continue
        colored = set()
        for k in idxs:
            colored |= set(objs[k].colored)
        out.append(_build_from_colored(colored))
    return out


def _build_from_colored(colored) -> Obj:
    cells = [(r, c) for r, c, _ in colored]
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    cols = [col for _, _, col in colored]
    dom = Counter(cols).most_common(1)[0][0]
    return Obj(
        cells=frozenset(cells),
        colored=frozenset(colored),
        dom_color=dom,
        colors=frozenset(cols),
        bbox=(min(rs), min(cs), max(rs), max(cs)),
        size=len(cells),
    )


def _build(grid, cells) -> Obj:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    cols = [int(grid[r, c]) for r, c in cells]
    colored = frozenset((r, c, int(grid[r, c])) for r, c in cells)
    dom = Counter(cols).most_common(1)[0][0]
    return Obj(
        cells=frozenset(cells),
        colored=colored,
        dom_color=dom,
        colors=frozenset(cols),
        bbox=(min(rs), min(cs), max(rs), max(cs)),
        size=len(cells),
    )
