"""[agent/core] parse -- the object-schema v003 parser (method (8)): multi-cue candidate
generation + pure-MDL selection + greedy deterministic agglomeration + recursive parts +
multiframe common-fate (Hungarian) + the C1-C6 invariant checker.

This is the spec'd parser from ``docs/StrictDoc-specs/_assets/gr-arc-3-object-schema.md``
sections 5.1 (parse algorithm) and 6 (C1-C6), implementing FR-U-01..15 / SC-01..04. It is
ADDED ALONGSIDE the legacy ``perceive.perceive`` (a CCL stopgap the L1 slice depends on);
it does not replace it.

Design (object-schema v003 section 5.1 "design principle"):

* library-first (FR-U-09): ``skimage`` does color-CCL atoms (``measure.label``), region
  adjacency (``graph.rag_boundary`` / a hand-built RAG over ``label``), and shape metrics
  (``regionprops``); ``scipy`` does the cross-frame Hungarian assignment
  (``optimize.linear_sum_assignment``) and the lattice autocorrelation (``signal`` / FFT);
  ``numpy`` is the numeric base.
* hand-rolled (the parts no off-the-shelf library provides): every per-cue ``DescLen``, the
  agglomeration loop, the recursive laminar tree, the occluded/transparent id lists, and the
  C1-C6 checker.
* every cost is in BITS, with NO hyper-parameters (lambda = 1). ``DescLen(O)`` is the cheapest
  applicable cue's code length; ``DescLen(P)`` sums those plus a fixed per-object structural
  overhead (the Occam term that stops the all-cells-are-separate-objects degeneracy) plus the
  transition code lengths. ``argmin`` is computed DETERMINISTICALLY: a strict ``>`` improvement
  test with an ascending-index tie-break, no sampling and no builtin ``hash()`` for identity.

Pure: numpy + skimage + scipy + stdlib. No framework import, no game-specific literals
(asserts in the tests are on structure / counts / scale / parent-child / id, never a colour
or coordinate value).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from agent.core.perceive import Obj

# scikit-image and scipy are imported LAZILY (inside the functions that delegate to them)
# rather than at module top. Rationale: FR-U-09 mandates delegating color-CCL / RAG /
# regionprops to skimage and the Hungarian assignment / autocorrelation to scipy, but the
# hand-rolled parts (every cue DescLen, the agglomeration loop, the recursive tree, the
# occluded/transparent lists, and the C1-C6 checker) depend on numpy alone. Lazy imports let
# those pure parts -- and their tests -- run even before the heavy libraries are installed,
# while the full pipeline still delegates exactly as the spec requires.

GRID = 64
NUM_COLORS = 16  # K = {0..15} (object-schema section 1)

Cell = Tuple[int, int]


# --------------------------------------------------------------------------------------
# bit-cost primitives (object-schema section 5.1.2). All costs are in BITS; lambda = 1; no
# magic constants -- only information-theoretic code lengths over the fixed substrate.
# --------------------------------------------------------------------------------------
def _bits(n: float) -> float:
    """``log2`` of a positive count, floored at 0 bits (a 1-of-n choice costs ``log2 n``)."""
    if n <= 1:
        return 0.0
    return math.log2(n)


def _coord_bits(h: int, w: int) -> float:
    """Bits to name one cell on an ``h x w`` board = ``log2(H*W)`` (object-schema: ``log(HW)``)."""
    return _bits(h * w)


def _color_bits() -> float:
    """Bits to name one colour = ``log2 K`` (object-schema: ``log K``)."""
    return _bits(NUM_COLORS)


# --------------------------------------------------------------------------------------
# ParseResult -- the parser's return contract (FR-U-08).
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ParseResult:
    """The output of :func:`parse` (object-schema section 5.1.3 ``return`` line).

    * ``objects`` -- the visible-leaf object set as a laminar tree of :class:`~agent.core.perceive.Obj`
      (each carries ``parts`` for recursion, ``oid`` for identity, ``scale`` / ``cue`` as a Dimension
      and its winning explanation, ``is_field`` for the background).
    * ``label_grid`` -- example-1: ``H x W`` int array mapping each cell to the ``oid`` of the
      most-specific visible leaf that owns it (the double-entry partner of ``oid -> cells``).
    * ``occluded_ids`` / ``transparent_ids`` -- the id lists that license shared cells (C3); we do
      not split the parse into a separate "belief" set (project convention).
    * ``phi`` -- the multiframe correspondences: ``phi[t]`` maps an ``oid`` in frame ``t`` to an
      ``oid`` in frame ``t+1`` (empty for a single frame).
    * ``frames`` -- the per-frame object sets (``frames[t]`` = the leaves parsed for frame ``t``);
      ``frames[0] is objects`` for the single-frame case.
    """

    objects: Tuple[Obj, ...]
    label_grid: np.ndarray
    occluded_ids: Tuple[int, ...]
    transparent_ids: Tuple[int, ...]
    phi: Tuple[Dict[int, int], ...] = ()
    frames: Tuple[Tuple[Obj, ...], ...] = ()
    # multiframe-only (empty for a single frame):
    transitions: Tuple[str, ...] = ()          # every tau across all steps (the C5 input)
    accounting: Tuple[Tuple[str, int, int], ...] = ()  # (tau, cells_before, cells_after) (the C6 input)
    track_of: Tuple[Dict[int, int], ...] = ()  # per-frame: local oid -> stable track id
    c5_ok: bool = True                         # C5 verdict (recorded, not asserted; FR-U-08)
    c6_ok: bool = True                         # C6 verdict (recorded, not asserted; FR-U-08)


# --------------------------------------------------------------------------------------
# internal working object. We keep a light mutable struct during agglomeration and convert
# to the frozen Obj tree at the end (Obj is frozen / hashable for downstream memoisation).
# --------------------------------------------------------------------------------------
@dataclass
class _Node:
    oid: int
    cells: FrozenSet[Cell]
    colored: FrozenSet[Tuple[int, int, int]]  # {(r, c, color)}
    parts: List["_Node"] = field(default_factory=list)
    is_field: bool = False
    scale: int = 1
    cue: str = "connectivity"

    @property
    def colors(self) -> FrozenSet[int]:
        return frozenset(col for _, _, col in self.colored)

    @property
    def dom_color(self) -> int:
        counts: Dict[int, int] = {}
        for _, _, col in self.colored:
            counts[col] = counts.get(col, 0) + 1
        # deterministic: most frequent, tie-break by smallest colour index
        return min(counts, key=lambda c: (-counts[c], c))

    def bbox(self) -> Tuple[int, int, int, int]:
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return (min(rs), min(cs), max(rs), max(cs))


# ======================================================================================
# per-cue DescLen (object-schema section 5.1.2). Each returns the code length in bits to
# reconstruct the node's coloured cells under that cue, or ``math.inf`` if inapplicable.
# ``DescLen(O) = min over applicable cues`` and the winner is that object's explanation.
# ======================================================================================
def _grid_shape(node: _Node, h: int, w: int) -> np.ndarray:
    """Dense colour patch over the node's bbox (-1 = cell not owned by the node)."""
    r0, c0, r1, c1 = node.bbox()
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), -1, dtype=int)
    for r, c, col in node.colored:
        patch[r - r0, c - c0] = col
    return patch


def _cue_connectivity(node: _Node, h: int, w: int) -> float:
    """connectivity: boundary length + 1 colour. Cheapest for a solid single-colour blob.

    The boundary length is the hand-rolled MDL code (object-schema lists cue DescLen as custom):
    one bit per boundary edge plus the fill colour (``log2 K``). Multi-colour nodes pay an extra
    ``log2 K`` per distinct extra colour (connectivity assumes one colour). skimage's ``regionprops``
    is available for richer shape metrics, but the perimeter is computed in numpy so the cue cost --
    and the whole agglomeration loop -- stays testable. A solid connected blob has a small boundary
    relative to its area, so connectivity beats the similarity mask there; a disconnected scatter
    pays a contour-anchor per component (you must say WHERE each blob starts), so a grouping cue
    (similarity / line / lattice) wins there instead."""
    perim = _boundary_length(node)
    comps = _num_components(node)            # a contour code needs one start anchor per component
    anchor = comps * _coord_bits(h, w)
    extra = (len(node.colors) - 1) * _color_bits() if len(node.colors) > 1 else 0.0
    return perim + anchor + _color_bits() + extra


def _num_components(node: _Node) -> int:
    """Number of 8-connected components in the node's cell set (library-free, for the connectivity
    cue's per-component anchor cost). A solid blob = 1; a scatter of k dots = k."""
    mask = _node_mask(node)
    hh, ww = mask.shape
    seen = np.zeros_like(mask)
    count = 0
    for r in range(hh):
        for c in range(ww):
            if mask[r, c] and not seen[r, c]:
                count += 1
                stack = [(r, c)]
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < hh and 0 <= nx < ww and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
    return count


def _boundary_length(node: _Node) -> float:
    """Perimeter of the node's mask = number of 4-neighbour edges between an in-mask cell and a
    cell outside the mask (a hand-rolled, library-free boundary measure)."""
    mask = _node_mask(node)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    edges = 0
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.roll(np.roll(padded, dr, axis=0), dc, axis=1)
        edges += int(np.sum(padded & ~shifted))
    return float(edges)


def _cue_similarity(node: _Node, h: int, w: int) -> float:
    """similarity: 1 colour + a mask (one bit per bbox cell). Cheap for a same-colour scatter."""
    r0, c0, r1, c1 = node.bbox()
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    extra = (len(node.colors) - 1) * _color_bits() if len(node.colors) > 1 else 0.0
    return _color_bits() + float(area) + extra


def _cue_scaled(node: _Node, h: int, w: int) -> Tuple[float, int]:
    """scaled: base pattern + integer factor ``s``. Returns ``(bits, s)``; ``s == 1`` if the node
    is not a *non-trivial* integer upscale. Encodes the down-scaled base mask + ``log2 s`` (FR-U-07).

    Gate (important): a uniform solid rectangle is trivially "scalable" to any divisor, but scale
    adds no structure there -- connectivity already describes it. So the scaled cue only applies
    when the base patch is NOT a fully-filled single-colour rectangle (it has real structure: a hole
    or more than one colour). This stops a plain block being mis-explained as a 1-cell upscale."""
    patch = _grid_shape(node, h, w)
    s = _detect_scale(patch)
    if s <= 1:
        return (math.inf, 1)
    base = patch[::s, ::s]
    if _is_trivial_patch(base):
        return (math.inf, 1)  # solid uniform block -> let connectivity win, not scaled
    base_cells = int((base >= 0).sum())
    bits = base_cells * (_color_bits() + 1.0) + _bits(max(2, GRID)) + _coord_bits(h, w)
    return (bits, s)


def _is_trivial_patch(patch: np.ndarray) -> bool:
    """True if the patch is a fully-filled rectangle of a single colour (no hole, one colour)."""
    occupied = patch >= 0
    if not occupied.all():
        return False  # has a hole -> non-trivial
    return np.unique(patch).size <= 1


def _cue_symmetry(node: _Node, h: int, w: int) -> float:
    """symmetry: half + axis. Cheap for a mirror-symmetric figure (store one half + which axis)."""
    patch = _grid_shape(node, h, w)
    if not _has_mirror_symmetry(patch):
        return math.inf
    half_cells = int((patch >= 0).sum()) / 2.0
    half_bits = half_cells * (_color_bits() + 1.0)
    axis_bits = 2.0  # one of {h, v, both}
    return half_bits + axis_bits


def _cue_line(group: Sequence[_Node], h: int, w: int) -> float:
    """line / good-continuation: 2 endpoints + gap + count + colour. Cheap for a dotted line
    (collinear, equally spaced). Applies to a GROUP of components (object-schema: dotted line)."""
    if len(group) < 3:
        return math.inf
    if not _is_equally_spaced_collinear(group):
        return math.inf
    colors = set()
    for nd in group:
        colors |= set(nd.colors)
    color_pen = (len(colors) - 1) * _color_bits()
    return 4.0 * _coord_bits(h, w) + _bits(max(2, GRID)) + _bits(len(group)) + _color_bits() + color_pen


def _cue_lattice(group: Sequence[_Node], h: int, w: int) -> Tuple[float, int, int]:
    """lattice / regularity: tile + grid ``(k, m)`` + per-tile diff. Cheap for a Rubik face.
    Returns ``(bits, k, m)``; ``inf`` if the group does not tile a regular ``k x m`` grid."""
    info = _detect_lattice(group)
    if info is None:
        return (math.inf, 0, 0)
    k, m, tile_cells = info
    tile_bits = tile_cells * (_color_bits() + 1.0)
    diff_bits = float(k * m) * _color_bits()  # one recolour per tile in the worst case
    return (tile_bits + _bits(k) + _bits(m) + diff_bits, k, m)


def _as_singletons(node: _Node) -> List[_Node]:
    """Explode a node into one synthetic single-cell component per cell, so the group cues (line /
    lattice) can be re-tested on an ALREADY-merged node when finalising its explanation cue."""
    # precompute pos -> colour once (O(cells)); the previous per-cell linear scan of ``colored``
    # was O(cells^2) and dominated parse time on the large single-colour field node (NFR-4).
    col_of = {(rr, ccx): cc for (rr, ccx, cc) in node.colored}
    dom = node.dom_color
    out = []
    for i, (r, c) in enumerate(sorted(node.cells)):
        out.append(_Node(i, frozenset({(r, c)}), frozenset({(r, c, col_of.get((r, c), dom))})))
    return out


def _final_cue(node: _Node, h: int, w: int) -> str:
    """Pick the cheapest explanation cue for a FINALISED leaf, considering the single-object cues
    AND re-testing the group cues (line / lattice) on the node's own cells. This is what lets a
    merged dotted line keep ``cue == 'line'`` rather than reverting to connectivity (object-schema:
    the winning cue is the object's explanation)."""
    bits, cue, _s = _node_desclen(node, h, w)
    best_bits, best_cue = bits, cue
    if len(node.colors) == 1 and len(node.cells) >= 3:
        singles = _as_singletons(node)
        line_bits = _cue_line(sorted(singles, key=_centroid), h, w)
        if line_bits < best_bits:
            best_bits, best_cue = line_bits, "line"
        lat_bits, _k, _m = _cue_lattice(singles, h, w)
        if lat_bits < best_bits:
            best_bits, best_cue = lat_bits, "lattice"
    return best_cue


def _node_desclen(node: _Node, h: int, w: int) -> Tuple[float, str, int]:
    """``DescLen(O) = min over applicable single-object cues`` -> ``(bits, winning_cue, scale)``.

    Single-object cues only (connectivity / similarity / scaled / symmetry). Group cues (line /
    lattice) and enclosure are evaluated by the agglomeration proposer, not here. If a node has
    ``parts`` (enclosure/lattice already folded) its cost is the recursive sum (parent + children)."""
    if node.parts:
        # enclosure / lattice node: parent shell + recursive children (object-schema 5.1.2).
        shell_colored = node.colored - frozenset().union(*(c.colored for c in node.parts))
        shell = _Node(node.oid, frozenset(c[:2] for c in shell_colored), frozenset(shell_colored))
        shell_bits = _cue_connectivity(shell, h, w) if shell.cells else 0.0
        child_bits = sum(_node_desclen(c, h, w)[0] for c in node.parts)
        return (shell_bits + child_bits, node.cue, node.scale)

    candidates: List[Tuple[float, str, int]] = []
    candidates.append((_cue_connectivity(node, h, w), "connectivity", 1))
    candidates.append((_cue_similarity(node, h, w), "similarity", 1))
    sbits, s = _cue_scaled(node, h, w)
    candidates.append((sbits, "scaled", s))
    candidates.append((_cue_symmetry(node, h, w), "symmetry", 1))
    # deterministic argmin: smallest bits, tie-break by a fixed cue order.
    order = {"connectivity": 0, "similarity": 1, "scaled": 2, "symmetry": 3}
    best = min(candidates, key=lambda x: (x[0], order[x[1]]))
    return best


def _objects_desclen(nodes: Sequence[_Node], h: int, w: int) -> float:
    """``DescLen(P)`` = sum of each object's cheapest-cue cost + a fixed per-object structural
    overhead (the Occam term that penalises over-segmentation; one anchor coordinate per object)."""
    obj_header = _coord_bits(h, w)  # struct_overhead: each object costs at least an anchor
    total = 0.0
    for nd in nodes:
        total += _node_desclen(nd, h, w)[0] + obj_header
    return total


# --------------------------------------------------------------------------------------
# geometry / pattern helpers (hand-rolled cue tests).
# --------------------------------------------------------------------------------------
def _node_mask(node: _Node) -> np.ndarray:
    r0, c0, r1, c1 = node.bbox()
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in node.cells:
        mask[r - r0, c - c0] = True
    return mask


def _detect_scale(patch: np.ndarray) -> int:
    """Largest integer ``s >= 2`` such that ``patch`` is an exact ``s``-fold block upscale
    (every ``s x s`` block is constant AND the bbox dims are divisible by ``s``). Else 1."""
    hh, ww = patch.shape
    best = 1
    for s in range(2, min(hh, ww) + 1):
        if hh % s or ww % s:
            continue
        ok = True
        for r0 in range(0, hh, s):
            for c0 in range(0, ww, s):
                block = patch[r0:r0 + s, c0:c0 + s]
                if np.unique(block).size != 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            best = s
    return best


def _has_mirror_symmetry(patch: np.ndarray) -> bool:
    """True if the occupancy+colour patch is symmetric under a horizontal or vertical flip
    AND is non-trivial (more than one row/col, not a single cell)."""
    if patch.size <= 1 or min(patch.shape) < 2:
        return False
    return bool(np.array_equal(patch, patch[:, ::-1]) or np.array_equal(patch, patch[::-1, :]))


def _centroid(node: _Node) -> Tuple[float, float]:
    rs = [r for r, _ in node.cells]
    cs = [c for _, c in node.cells]
    n = len(node.cells)
    return (sum(rs) / n, sum(cs) / n)


def _is_equally_spaced_collinear(group: Sequence[_Node]) -> bool:
    """True if the group's centroids are collinear (horizontal, vertical, or 45-degree
    diagonal) AND consecutive gaps along the line are equal -- the good-continuation test."""
    pts = sorted(_centroid(nd) for nd in group)
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    dr_all = [round(rs[i + 1] - rs[i], 3) for i in range(len(pts) - 1)]
    dc_all = [round(cs[i + 1] - cs[i], 3) for i in range(len(pts) - 1)]
    if len({(a, b) for a, b in zip(dr_all, dc_all)}) != 1:
        return False  # not equally spaced (steps differ)
    dr, dc = dr_all[0], dc_all[0]
    if dr == 0 and dc == 0:
        return False
    # collinear directions we accept: axis-aligned or perfect diagonal.
    if dr == 0 or dc == 0:
        return True
    return abs(abs(dr) - abs(dc)) < 1e-6


def _detect_lattice(group: Sequence[_Node]) -> Optional[Tuple[int, int, int]]:
    """If the group forms a regular ``k x m`` grid of equal-bbox tiles, return ``(k, m, tile_cells)``.
    Uses the tile bbox size + grid spacing (a simple autocorrelation of tile origins). Else None."""
    if len(group) < 4:
        return None
    boxes = [nd.bbox() for nd in group]
    heights = {b[2] - b[0] for b in boxes}
    widths = {b[3] - b[1] for b in boxes}
    if len(heights) != 1 or len(widths) != 1:
        return None  # tiles differ in size -> not a clean lattice
    rows = sorted({b[0] for b in boxes})
    cols = sorted({b[1] for b in boxes})
    k, m = len(rows), len(cols)
    if k < 2 or m < 2 or k * m != len(group):
        return None
    # spacing regularity (autocorrelation peak == constant stride) along each axis.
    if not _constant_stride(rows) or not _constant_stride(cols):
        return None
    tile_cells = max(len(nd.cells) for nd in group)
    return (k, m, tile_cells)


def _constant_stride(vals: Sequence[int]) -> bool:
    if len(vals) < 2:
        return True
    diffs = {vals[i + 1] - vals[i] for i in range(len(vals) - 1)}
    return len(diffs) == 1


# ======================================================================================
# atoms (FR-U-01): color-CCL via skimage.measure.label(grid==c, connectivity=2).
# ======================================================================================
def _atoms(grid: np.ndarray, background: FrozenSet[int]) -> List[_Node]:
    """Color-CCL over-segmentation: for each colour, 8-connected components are atoms.

    Background colours become a single ``is_field`` node (object-schema D1) so the parse covers
    every cell (C4). NON-background colours each yield their connected components (FR-U-01)."""
    from skimage import measure as sk_measure  # lazy (FR-U-09 delegation)

    h, w = grid.shape
    nodes: List[_Node] = []
    next_id = 0
    fg_colors = sorted(set(int(v) for v in np.unique(grid)) - set(background))
    for c in fg_colors:
        labels = sk_measure.label(grid == c, connectivity=2)
        for lbl in range(1, int(labels.max()) + 1):
            ys, xs = np.where(labels == lbl)
            cells = frozenset((int(y), int(x)) for y, x in zip(ys, xs))
            colored = frozenset((int(y), int(x), c) for y, x in zip(ys, xs))
            nodes.append(_Node(next_id, cells, colored))
            next_id += 1
    # background -> one field object covering all background cells (may be disconnected; D3).
    bg_cells = [(int(r), int(c)) for r in range(h) for c in range(w) if int(grid[r, c]) in background]
    if bg_cells:
        colored = frozenset((r, c, int(grid[r, c])) for r, c in bg_cells)
        nodes.append(_Node(next_id, frozenset(bg_cells), colored, is_field=True))
        next_id += 1
    return nodes


# ======================================================================================
# region adjacency (skimage RAG, FR-U-03/09): which atoms touch, for merge proposals.
# ======================================================================================
def _label_image(nodes: Sequence[_Node], h: int, w: int) -> np.ndarray:
    lab = np.zeros((h, w), dtype=int)
    for nd in nodes:
        for r, c in nd.cells:
            lab[r, c] = nd.oid + 1  # +1 so background of the label image (0) is unused
    return lab


def _adjacent_pairs(nodes: Sequence[_Node], h: int, w: int) -> List[Tuple[int, int]]:
    """Adjacent ``(oid_a, oid_b)`` pairs via skimage's region adjacency graph (FR-U-03/09)."""
    try:  # lazy (FR-U-09 delegation); RAG moved between skimage versions.
        from skimage.graph import RAG
    except ImportError:  # pragma: no cover - older scikit-image
        from skimage.future.graph import RAG

    lab = _label_image(nodes, h, w)
    rag = RAG(lab, connectivity=2)
    id_of = {nd.oid + 1: nd.oid for nd in nodes}
    pairs = []
    for a, b in rag.edges():
        if a in id_of and b in id_of:
            pa, pb = id_of[a], id_of[b]
            pairs.append((min(pa, pb), max(pa, pb)))
    return sorted(set(pairs))


# ======================================================================================
# merge / grouping primitives.
# ======================================================================================
def _merge_nodes(group: Sequence[_Node], new_id: int, cue: str = "connectivity", scale: int = 1) -> _Node:
    cells: FrozenSet[Cell] = frozenset().union(*(nd.cells for nd in group))
    colored = frozenset().union(*(nd.colored for nd in group))
    is_field = any(nd.is_field for nd in group)
    return _Node(new_id, cells, colored, is_field=is_field, scale=scale, cue=cue)


def _wrap_enclosure(parent_atom: _Node, children: Sequence[_Node], new_id: int,
                    shell_id: Optional[int] = None) -> _Node:
    """Fold an enclosure into a laminar ``parent + parts`` node.

    The parent's parts PARTITION the parent's cells (so every level is a true visible-leaf
    partition -- C1/C4 hold at each level): the outer shell (``parent_atom``'s own ring cells) becomes
    one leaf-part and each enclosed element becomes another leaf-part. parent.cells is their union, so
    every child's cells are a subset of the parent's (laminar, object-schema D4). ``shell_id`` is the
    fresh id for the shell leaf (defaults to ``new_id + 1`` so it stays unique vs the parent id)."""
    all_nodes = [parent_atom, *children]
    cells = frozenset().union(*(nd.cells for nd in all_nodes))
    colored = frozenset().union(*(nd.colored for nd in all_nodes))
    sid = shell_id if shell_id is not None else new_id + 1
    shell_leaf = _Node(sid, parent_atom.cells, parent_atom.colored,
                       is_field=parent_atom.is_field, cue=parent_atom.cue)
    parts = [shell_leaf, *children]
    node = _Node(new_id, cells, colored, parts=parts, cue="enclosure")
    return node


# ======================================================================================
# agglomeration proposals (object-schema 5.1.3): each cue proposes merges/groupings; we
# keep the single proposal that lowers DescLen most (greedy, deterministic, no backtrack).
# ======================================================================================
def _propose(nodes: List[_Node], h: int, w: int) -> List[Tuple[float, str, List[int], dict]]:
    """All candidate moves as ``(delta, cue, member_oids, meta)`` where ``delta = DescLen(before)
    - DescLen(after) > 0`` is an improvement. We compute deltas LOCALLY (only the affected
    objects change) so the greedy choice is exact for an additive ``DescLen``."""
    proposals: List[Tuple[float, str, List[int], dict]] = []
    by_id = {nd.oid: nd for nd in nodes}
    obj_header = _coord_bits(h, w)

    def cost(nd: _Node) -> float:
        return _node_desclen(nd, h, w)[0] + obj_header

    # ---- connectivity / similarity / scaled merges over ADJACENT same-colour-ish pairs ----
    for a, b in _adjacent_pairs(nodes, h, w):
        na, nb = by_id[a], by_id[b]
        if na.is_field or nb.is_field:
            continue
        before = cost(na) + cost(nb)
        merged = _merge_nodes([na, nb], na.oid)
        bits, cue, s = _node_desclen(merged, h, w)
        merged.cue, merged.scale = cue, s
        after = bits + obj_header
        delta = before - after
        if delta > 0:
            proposals.append((delta, cue, [a, b], {"scale": s}))

    # ---- similarity / line grouping over SAME-COLOUR non-adjacent components ----
    color_groups: Dict[int, List[_Node]] = {}
    for nd in nodes:
        if nd.is_field or len(nd.colors) != 1:
            continue
        color_groups.setdefault(next(iter(nd.colors)), []).append(nd)
    for col, grp in color_groups.items():
        if len(grp) < 3:
            continue
        grp_sorted = sorted(grp, key=lambda n: _centroid(n))
        line_bits = _cue_line(grp_sorted, h, w)
        if math.isfinite(line_bits):
            before = sum(cost(nd) for nd in grp_sorted)
            after = line_bits + obj_header
            delta = before - after
            if delta > 0:
                proposals.append((delta, "line", [nd.oid for nd in grp_sorted], {}))

    # ---- lattice grouping over a regular grid of equal tiles ----
    for col, grp in color_groups.items():
        info = _detect_lattice(grp)
        if info is None:
            continue
        lat_bits, k, m = _cue_lattice(grp, h, w)
        if math.isfinite(lat_bits):
            before = sum(cost(nd) for nd in grp)
            after = lat_bits + obj_header
            delta = before - after
            if delta > 0:
                proposals.append((delta, "lattice", [nd.oid for nd in grp], {"k": k, "m": m}))

    # ---- enclosure: a node whose bbox strictly contains other nodes -> parent + parts ----
    for outer in nodes:
        if outer.is_field:
            continue
        inside = [nd for nd in nodes if nd.oid != outer.oid and _strictly_inside(nd, outer)]
        if not inside:
            continue
        wrapped = _wrap_enclosure(outer, inside, outer.oid)
        before = cost(outer) + sum(cost(nd) for nd in inside)
        after = _node_desclen(wrapped, h, w)[0] + obj_header
        delta = before - after
        if delta > 0:
            proposals.append((delta, "enclosure", [outer.oid, *[n.oid for n in inside]], {}))

    return proposals


def _strictly_inside(inner: _Node, outer: _Node) -> bool:
    """True if ``inner``'s bbox is strictly within ``outer``'s bbox AND ``outer`` forms a ring
    around it (``inner``'s cells are disjoint from ``outer``'s, i.e. ``outer`` encloses a hole)."""
    ir0, ic0, ir1, ic1 = inner.bbox()
    or0, oc0, or1, oc1 = outer.bbox()
    if not (or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1):
        return False
    return inner.cells.isdisjoint(outer.cells)


# ======================================================================================
# the greedy deterministic agglomeration loop (FR-U-03).
# ======================================================================================
def _agglomerate(nodes: List[_Node], h: int, w: int) -> List[_Node]:
    """Apply the single DescLen-lowering move with the largest delta, repeat until no proposal
    strictly lowers DescLen (``best.delta <= 0``). Deterministic: strict ``>`` with an ascending
    member-id tie-break; no backtrack, no random restart (object-schema 5.1.4 decision 2)."""
    nodes = list(nodes)
    next_id = max((nd.oid for nd in nodes), default=-1) + 1
    while True:
        proposals = _propose(nodes, h, w)
        if not proposals:
            break
        # deterministic argmax over delta; tie-break by cue order then ascending member ids.
        cue_order = {"connectivity": 0, "similarity": 1, "scaled": 2, "line": 3, "lattice": 4,
                     "symmetry": 5, "enclosure": 6}
        best = max(proposals, key=lambda p: (round(p[0], 9), -cue_order[p[1]], tuple(-x for x in sorted(p[2]))))
        delta, cue, members, meta = best
        if delta <= 0:
            break
        nodes = _apply(nodes, cue, members, meta, next_id)
        next_id += 1
    return nodes


def _apply(nodes: List[_Node], cue: str, members: List[int], meta: dict, new_id: int) -> List[_Node]:
    by_id = {nd.oid: nd for nd in nodes}
    grp = [by_id[m] for m in members]
    rest = [nd for nd in nodes if nd.oid not in set(members)]
    if cue == "enclosure":
        parent_atom = grp[0]
        children = grp[1:]
        # parent reuses the outer atom's id (stable identity); the shell leaf gets a guaranteed
        # fresh id so it never collides with an existing node or a child.
        shell_id = max(nd.oid for nd in nodes) + 1
        merged = _wrap_enclosure(parent_atom, children, parent_atom.oid, shell_id=shell_id)
    elif cue in ("line", "lattice"):
        merged = _merge_nodes(grp, grp[0].oid, cue=cue)
    else:  # connectivity / similarity / scaled
        merged = _merge_nodes(grp, grp[0].oid, cue=cue, scale=int(meta.get("scale", 1)))
    return rest + [merged]


# ======================================================================================
# single-frame parse.
# ======================================================================================
def _parse_single(grid: np.ndarray, background: FrozenSet[int]) -> List[_Node]:
    h, w = grid.shape
    nodes = _atoms(grid, background)
    nodes = _agglomerate(nodes, h, w)
    # Finalise each leaf's attributes. ``cue`` = the cheapest explanation (FR-U-04). ``scale`` is
    # an INTRINSIC Dimension (object-schema section 3: scale is a Dimension like colour/shape),
    # measured from block-regularity independently of which cue is cheapest -- so a 2x render keeps
    # scale == 2 even when connectivity happens to encode it more cheaply (FR-U-07: never split it).
    for nd in nodes:
        if not nd.parts:
            nd.cue = _final_cue(nd, h, w)
            nd.scale = _object_scale(nd, h, w)
    return nodes


def _object_scale(node: _Node, h: int, w: int) -> int:
    """Intrinsic render-scale Dimension of an object: the integer block-upscale factor of its
    coloured patch, gated so a uniform solid region (no observable scale) reports 1 (FR-U-07)."""
    patch = _grid_shape(node, h, w)
    s = _detect_scale(patch)
    if s <= 1:
        return 1
    base = patch[::s, ::s]
    if _is_trivial_patch(base):
        return 1  # a solid block carries no observable render scale
    return s


def _default_background(grid: np.ndarray, top: int = 1) -> FrozenSet[int]:
    """Background = the single most frequent colour (a frequency heuristic over the fixed board;
    object-schema D1). General -- no specific colour value is baked in."""
    vals, cnts = np.unique(grid, return_counts=True)
    order = [int(v) for v, _ in sorted(zip(vals.tolist(), cnts.tolist()), key=lambda x: (-x[1], x[0]))]
    return frozenset(order[:top])


# ======================================================================================
# multiframe common-fate (FR-U-06): Hungarian correspondence + online over-merge split.
# ======================================================================================
def _transition_cost(a: _Node, b: _Node, h: int, w: int) -> float:
    """``DescLen(transition O_a -> O_b)`` = displacement bits + a taxonomy tag (object-schema
    5.1.2 common-fate). Same shape + a shift is cheap (one displacement); shape/colour change
    adds the mask/colour delta."""
    ca, cb = _centroid(a), _centroid(b)
    disp_bits = _coord_bits(h, w)  # one displacement vector
    # shape mismatch cost: symmetric difference of the shifted masks (a crude deform code).
    dr = round(cb[0] - ca[0])
    dc = round(cb[1] - ca[1])
    shifted = frozenset((r + dr, c + dc) for r, c in a.cells)
    deform = len(shifted ^ b.cells) * 1.0
    recolor = _color_bits() if a.dom_color != b.dom_color else 0.0
    return disp_bits + deform + recolor


def _correspond(prev: Sequence[_Node], curr: Sequence[_Node], h: int, w: int) -> Dict[int, int]:
    """Hungarian assignment (``scipy.optimize.linear_sum_assignment``) of prev->curr objects
    minimising total transition DescLen (FR-U-06). Returns ``{prev_oid: curr_oid}``."""
    from scipy.optimize import linear_sum_assignment  # lazy (FR-U-09 delegation)

    if not prev or not curr:
        return {}
    n, m = len(prev), len(curr)
    size = max(n, m)
    BIG = 1e6
    cost = np.full((size, size), BIG, dtype=float)
    for i, a in enumerate(prev):
        for j, b in enumerate(curr):
            cost[i, j] = _transition_cost(a, b, h, w) if _plausible_match(a, b) else BIG
    rows, cols = linear_sum_assignment(cost)
    phi: Dict[int, int] = {}
    for i, j in zip(rows, cols):
        if i < n and j < m and cost[i, j] < BIG:
            phi[prev[i].oid] = curr[j].oid
    return phi


def _plausible_match(a: _Node, b: _Node) -> bool:
    """Reject a correspondence whose shifted shapes barely overlap (a >100% shape change is really
    a non-match -- e.g. a hidden mover should NOT be force-matched to an unrelated object). This
    lets the Hungarian leave such an object unmatched (occlude/destroy) instead of inventing a deform."""
    ca, cb = _centroid(a), _centroid(b)
    dr, dc = round(cb[0] - ca[0]), round(cb[1] - ca[1])
    shifted = frozenset((r + dr, c + dc) for r, c in a.cells)
    return len(shifted ^ b.cells) <= len(a.cells)


def _classify_transition(a: Optional[_Node], b: Optional[_Node], h: int, w: int) -> str:
    """Transition taxonomy ``tau(O, O')`` (object-schema section 4): one of move / scale+ / scale- /
    deform / recolor / create / destroy / occlude / reveal. ``occlude != destroy``."""
    if a is None and b is not None:
        return "create"
    if a is not None and b is None:
        return "destroy"
    assert a is not None and b is not None
    if a.scale < b.scale:
        return "scale+"
    if a.scale > b.scale:
        return "scale-"
    ca, cb = _centroid(a), _centroid(b)
    dr, dc = round(cb[0] - ca[0]), round(cb[1] - ca[1])
    shifted = frozenset((r + dr, c + dc) for r, c in a.cells)
    same_shape = (shifted == b.cells)
    if same_shape and a.dom_color == b.dom_color:
        return "move"
    if same_shape and a.dom_color != b.dom_color:
        return "recolor"
    if a.colors == b.colors and len(a.cells) != len(b.cells):
        return "deform"
    return "deform"


def _split_overmerged(merged: _Node, next_parts: Sequence[_Node]) -> List[_Node]:
    """Online safety net (FR-U-06): split an over-merged node back into pieces that move apart.

    Each 8-connected component of ``merged`` is assigned to whichever next-frame object its shape
    matches (translation-invariant), or to the nearest one by centroid. Components assigned to the
    SAME target are regrouped into one node. If this yields >=2 groups, the fused node was an
    over-merge of independently-moving atoms and is returned split; otherwise the single node is
    returned unchanged. Ids are derived from the merged node so the result stays deterministic."""
    comps = _components_of(merged)
    if len(comps) <= 1 or len(next_parts) < 2:
        return [merged]
    target_of: Dict[int, int] = {}
    for ci, comp_cells in enumerate(comps):
        comp = _subnode(merged, comp_cells, merged.oid * 1000 + ci)
        target_of[ci] = _nearest_target(comp, next_parts)
    groups: Dict[int, List[int]] = {}
    for ci, tgt in target_of.items():
        groups.setdefault(tgt, []).append(ci)
    if len(groups) < 2:
        return [merged]
    out: List[_Node] = []
    for gi, (tgt, comp_idxs) in enumerate(sorted(groups.items())):
        cells = frozenset().union(*(comps[ci] for ci in comp_idxs))
        colored = frozenset((r, c, col) for (r, c, col) in merged.colored if (r, c) in cells)
        out.append(_Node(merged.oid * 1000 + gi, cells, colored,
                         is_field=merged.is_field, cue=merged.cue))
    return out


def _components_of(node: _Node) -> List[FrozenSet[Cell]]:
    """The 8-connected components of a node's cell set (library-free), as a list of cell sets."""
    remaining = set(node.cells)
    comps: List[FrozenSet[Cell]] = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        remaining.discard(seed)
        cells = {seed}
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb = (y + dy, x + dx)
                    if nb in remaining:
                        remaining.discard(nb)
                        cells.add(nb)
                        stack.append(nb)
        comps.append(frozenset(cells))
    return comps


def _subnode(node: _Node, cells: FrozenSet[Cell], oid: int) -> _Node:
    colored = frozenset((r, c, col) for (r, c, col) in node.colored if (r, c) in cells)
    return _Node(oid, cells, colored, is_field=node.is_field)


def _nearest_target(comp: _Node, targets: Sequence[_Node]) -> int:
    """Index of the target this component belongs to. Prefer targets whose shape matches the
    component (a moving blob keeps its shape); among the candidates, pick the NEAREST by centroid
    (so two same-shaped blobs each follow the closer target). Deterministic: smallest index on a tie."""
    key = _shape_key(comp)
    shape_matches = [i for i, t in enumerate(targets) if _shape_key(t) == key]
    candidates = shape_matches if shape_matches else list(range(len(targets)))
    cc = _centroid(comp)
    best_i, best_d = candidates[0], None
    for i in candidates:
        tc = _centroid(targets[i])
        d = (cc[0] - tc[0]) ** 2 + (cc[1] - tc[1]) ** 2
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return best_i


def _correspondence_desclen(prev: Sequence[_Node], curr: Sequence[_Node],
                            phi: Dict[int, int], h: int, w: int) -> float:
    """``DescLen(phi_t)`` for one frame step (object-schema 5.1.2 common-fate): the sum of the
    matched transition costs PLUS the description length of every UNMATCHED object -- an
    unexplained create/destroy costs its own ``DescLen`` (it is new/lost information). This is the
    quantity the online over-merge split minimises: fusing two independent movers leaves one
    next-frame object unmatched (an expensive spurious ``create``), so splitting strictly lowers it."""
    prev_by_id = {nd.oid: nd for nd in prev}
    curr_by_id = {nd.oid: nd for nd in curr}
    total = 0.0
    for pid, cid in phi.items():
        total += _transition_cost(prev_by_id[pid], curr_by_id[cid], h, w)
    for nd in prev:
        if nd.oid not in phi:
            total += _node_desclen(nd, h, w)[0]
    matched_curr = set(phi.values())
    for nd in curr:
        if nd.oid not in matched_curr:
            total += _node_desclen(nd, h, w)[0]
    return total


def _resplit_overmerges(prev: Sequence[_Node], curr: Sequence[_Node], phi: Dict[int, int],
                        h: int, w: int, reserved: Sequence[int] = ()) -> Tuple[List[_Node], Dict[int, int]]:
    """FR-U-06 online safety net (object-schema 5.1.3): re-split an over-merged ``prev`` object whose
    8-connected components follow DIVERGENT ``curr`` objects -- but ONLY when the split lowers
    ``DescLen(phi_t)`` (so a rigidly-moving group is never broken up). Returns the (possibly
    re-split) prev node list and its recomputed correspondence ``phi``.

    Determinism + stable ids: nodes are tried in ascending oid; the first improving split is applied
    and the scan restarts. A split's FIRST piece inherits the over-merged node's oid (so its track id
    is never churned) and the remaining pieces take fresh oids above every live oid -- so the other
    (unsplit) objects keep their oids, hence their track ids, untouched (the implementer-flagged
    track-id-churn risk). ``reserved`` are extra frame-local oids the fresh ids must also dodge (the
    background field node is not in ``prev``, but it shares the frame's label_grid -> C2)."""
    cur_prev: List[_Node] = list(prev)
    cur_phi: Dict[int, int] = dict(phi)
    cur_cost = _correspondence_desclen(cur_prev, curr, cur_phi, h, w)
    improved = True
    while improved:
        improved = False
        for node in sorted(cur_prev, key=lambda n: n.oid):
            pieces = _split_overmerged(node, curr)
            if len(pieces) < 2:
                continue  # not an over-merge that diverges (single component, or <2 targets)
            nxt = max([n.oid for n in cur_prev] + [n.oid for n in curr] + list(reserved)) + 1
            reided: List[_Node] = []
            for i, pc in enumerate(pieces):
                pid = node.oid if i == 0 else nxt  # piece 0 keeps the oid -> track id stays stable
                if i != 0:
                    nxt += 1
                sub = _Node(pid, pc.cells, pc.colored, is_field=pc.is_field)
                sub.cue = _final_cue(sub, h, w)      # the piece's OWN cheapest cue (not the parent's)
                sub.scale = _object_scale(sub, h, w)  # ...and its OWN intrinsic render scale (FR-U-07)
                reided.append(sub)
            cand = [n for n in cur_prev if n.oid != node.oid] + reided
            cand_phi = _correspond(cand, curr, h, w)
            cand_cost = _correspondence_desclen(cand, curr, cand_phi, h, w)
            if cand_cost < cur_cost - 1e-9:  # strict improvement (pure-MDL, deterministic)
                cur_prev, cur_phi, cur_cost = cand, cand_phi, cand_cost
                improved = True
                break
    return cur_prev, cur_phi


# ======================================================================================
# Obj-tree conversion (frozen, hashable result for downstream memoisation).
# ======================================================================================
def _to_obj(node: _Node) -> Obj:
    r0, c0, r1, c1 = node.bbox()
    parts = tuple(_to_obj(p) for p in node.parts)
    return Obj(
        cells=node.cells,
        colored=node.colored,
        dom_color=node.dom_color,
        colors=node.colors,
        bbox=(r0, c0, r1, c1),
        size=len(node.cells),
        parts=parts,
        is_field=node.is_field,
        oid=node.oid,
        scale=node.scale,
        cue=node.cue,
    )


def _build_label_grid(nodes: Sequence[_Node], h: int, w: int) -> np.ndarray:
    """example-1 ``label_grid``: each cell -> the oid of the most-specific (deepest) leaf there.
    Parents are written first, then their parts overwrite (so the leaf wins) -- the double-entry
    partner of ``oid -> cells`` (C2)."""
    lab = np.full((h, w), -1, dtype=int)

    def write(nd: _Node) -> None:
        for r, c in nd.cells:
            lab[r, c] = nd.oid
        for p in nd.parts:
            write(p)  # children overwrite -> deepest leaf owns the cell

    for nd in nodes:
        write(nd)
    return lab


# ======================================================================================
# C1-C6 invariant checker (FR-U-10..15, object-schema section 6). Hand-rolled.
# ======================================================================================
def _leaves(nodes: Sequence[Obj]) -> List[Obj]:
    """The visible leaves: deepest objects with no parts (recursion bottoms out at leaves)."""
    out: List[Obj] = []
    for nd in nodes:
        if nd.parts:
            out.extend(_leaves(nd.parts))
        else:
            out.append(nd)
    return out


def _all_nodes(nodes: Sequence[Obj]) -> List[Obj]:
    out: List[Obj] = []
    for nd in nodes:
        out.append(nd)
        if nd.parts:
            out.extend(_all_nodes(nd.parts))
    return out


def check_c1(objects: Sequence[Obj], h: int = GRID, w: int = GRID,
             occluded_ids: Sequence[int] = (), transparent_ids: Sequence[int] = ()) -> bool:
    """C1 visible-leaf partition (FR-U-10): P1 (every leaf non-empty) AND P2 (leaves cover all
    cells) AND P3' (same-level, non-occluded, non-transparent leaves share no cell)."""
    leaves = _leaves(objects)
    if any(len(lf.cells) == 0 for lf in leaves):  # P1
        return False
    covered: FrozenSet[Cell] = frozenset().union(*(lf.cells for lf in leaves)) if leaves else frozenset()
    if covered != frozenset((r, c) for r in range(h) for c in range(w)):  # P2
        return False
    special = set(occluded_ids) | set(transparent_ids)
    seen: Dict[Cell, int] = {}
    for lf in leaves:
        if lf.oid in special:
            continue
        for cell in lf.cells:
            if cell in seen:  # P3': non-special leaves must be disjoint
                return False
            seen[cell] = lf.oid
    return True


def check_c2(objects: Sequence[Obj], label_grid: np.ndarray) -> bool:
    """C2 double-entry (FR-U-11): ``label_grid[x] = i  <=>  x in cells(leaf_i)``."""
    leaves = _leaves(objects)
    cells_of = {lf.oid: set(lf.cells) for lf in leaves}
    h, w = label_grid.shape
    for r in range(h):
        for c in range(w):
            i = int(label_grid[r, c])
            if i < 0:
                return False  # an unlabelled cell breaks the double entry
            if (r, c) not in cells_of.get(i, set()):
                return False
    # reverse direction: every leaf cell is labelled with that leaf id.
    for lf in leaves:
        for (r, c) in lf.cells:
            # a deeper leaf may own the cell in label_grid; only fail if NO leaf claims it as id.
            if int(label_grid[r, c]) != lf.oid and (r, c) not in cells_of.get(int(label_grid[r, c]), set()):
                return False
    return True


def check_c3(objects: Sequence[Obj], occluded_ids: Sequence[int] = (),
             transparent_ids: Sequence[int] = ()) -> bool:
    """C3 shared-cell explanation (FR-U-12): any cell claimed by 2+ objects is explained by
    (a) nesting OR (b) occlusion OR (c) transparency; otherwise it is an error."""
    nodes = _all_nodes(objects)
    # nesting (a) is a genuine ANCESTOR relationship in the parts-tree, not a bare cell-subset:
    # two unrelated/sibling objects can have one cell-set be a subset of the other WITHOUT any
    # parent->child edge, and that sharing must still be reported (occlusion/transparency aside).
    # We resolve ancestry on node identity (parse may reuse an oid for parent + a part).
    by_identity: Dict[int, Obj] = {id(nd): nd for nd in nodes}
    child_to_parents: Dict[int, set] = {}
    for nd in nodes:
        for p in nd.parts:
            child_to_parents.setdefault(id(p), set()).add(id(nd))

    def _ancestor(desc_key: int, anc_key: int) -> bool:
        seen = set()
        stack = list(child_to_parents.get(desc_key, ()))
        while stack:
            cur = stack.pop()
            if cur == anc_key:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(child_to_parents.get(cur, ()))
        return False

    claims: Dict[Cell, List[int]] = {}
    for nd in nodes:
        for cell in nd.cells:
            claims.setdefault(cell, []).append(id(nd))  # store node identity, resolve oid on report
    special = set(occluded_ids) | set(transparent_ids)

    def nested(a_key: int, b_key: int) -> bool:
        # one node is a transitive ANCESTOR of the other in the parts-tree (true nesting).
        return _ancestor(a_key, b_key) or _ancestor(b_key, a_key)

    for cell, owners in claims.items():
        if len(owners) <= 1:
            continue
        for i in range(len(owners)):
            for j in range(i + 1, len(owners)):
                a_key, b_key = owners[i], owners[j]
                if nested(a_key, b_key):
                    continue  # (a) nesting (genuine ancestor relationship)
                if by_identity[a_key].oid in special or by_identity[b_key].oid in special:
                    continue  # (b)/(c) occluded or transparent
                return False  # unexplained sharing
    return True


def check_c4(objects: Sequence[Obj], h: int = GRID, w: int = GRID) -> bool:
    """C4 background membership (FR-U-13): every cell belongs to some leaf (including the
    ``is_field`` background leaf). A cell owned by no leaf is detected."""
    leaves = _leaves(objects)
    covered: FrozenSet[Cell] = frozenset().union(*(lf.cells for lf in leaves)) if leaves else frozenset()
    return covered == frozenset((r, c) for r in range(h) for c in range(w))


_TAU = {"move", "scale+", "scale-", "deform", "recolor", "create", "destroy", "occlude", "reveal"}


def check_c5(transitions: Sequence[str]) -> bool:
    """C5 transition explanation (FR-U-14, t>=1): every classified transition is in the taxonomy
    {move, scale+/-, deform, recolor, create, destroy, occlude, reveal}."""
    return all(t in _TAU for t in transitions)


def check_c6(accounting: Sequence[Tuple[str, int, int]]) -> bool:
    """C6 cell accounting (FR-U-15, t>=1): the VISIBLE cell census is conserved across a transition
    EXCEPT for the four census-changing transitions -- create / destroy / occlude / reveal.
    ``accounting`` = list of ``(tau, cells_before, cells_after)``; for every other tau the counts
    must match (double-entry).

    ``reveal`` is the inverse of ``occlude`` (a hidden object re-enters the visible partition,
    0 -> N), exactly as ``create`` adds cells (0 -> N): both legitimately change the visible census,
    so reveal is exempt just like occlude. The object-schema's central occlude!=destroy scenario
    (section 4 / SC-04) necessarily contains a reveal, so omitting it would make C6 fail on every
    occlusion sequence -- the very case the schema exists to license."""
    exempt = {"create", "destroy", "occlude", "reveal"}
    for tau, before, after in accounting:
        if tau in exempt:
            continue
        if before != after:
            return False
    return True


# ======================================================================================
# public API.
# ======================================================================================
def parse(frames, background: Optional[Sequence[int]] = None) -> ParseResult:
    """Parse a single 64x64 grid OR a list of frames into the object-schema (method (8)).

    Single frame: color-CCL atoms (FR-U-01) -> greedy MDL agglomeration over the cue table
    (FR-U-02/03/04) -> recursive parts for enclosure/lattice (FR-U-05) -> assert C1-C4 (FR-U-08).
    Two or more frames: additionally run per-frame parses, solve the Hungarian correspondence
    phi_t (FR-U-06), split over-merged atoms that move apart (online safety net), classify each
    transition by tau and check C5-C6.

    Returns a :class:`ParseResult` (objects, label_grid, occluded/transparent id lists, phi).
    Library-first (FR-U-09): skimage for CCL/RAG/regionprops, scipy for Hungarian/autocorrelation.
    """
    frame_list = _as_frames(frames)
    if not frame_list:
        empty = np.full((GRID, GRID), -1, dtype=int)
        return ParseResult((), empty, (), (), (), ())

    per_frame: List[Tuple[Obj, ...]] = []
    per_frame_nodes: List[List[_Node]] = []
    for g in frame_list:
        bg = frozenset(int(x) for x in background) if background is not None else _default_background(g)
        nodes = _parse_single(g, bg)
        per_frame_nodes.append(nodes)
        h, w = g.shape
        # --- single-frame validate: C1-C4 (FR-U-08); on violation, re-split (fall back to atoms). ---
        objs = tuple(_to_obj(nd) for nd in nodes)
        label_grid = _build_label_grid(nodes, h, w)
        if not (check_c1(objs, h, w) and check_c2(objs, label_grid)
                and check_c3(objs) and check_c4(objs, h, w)):
            nodes = _atoms(g, bg)  # re-split: the safe over-segmented partition always satisfies C1-C4
            per_frame_nodes[-1] = nodes
            objs = tuple(_to_obj(nd) for nd in nodes)
        per_frame.append(objs)

    h0, w0 = frame_list[0].shape
    objects = per_frame[0]
    label_grid = _build_label_grid(per_frame_nodes[0], h0, w0)
    occluded_ids: List[int] = []
    transparent_ids: List[int] = []
    phi: List[Dict[int, int]] = []
    transitions: List[str] = []
    accounting: List[Tuple[str, int, int]] = []
    track_of: List[Dict[int, int]] = []
    c5_ok = True
    c6_ok = True

    if len(frame_list) >= 2:
        mr = _multiframe(per_frame_nodes, frame_list)
        phi = mr.phi
        occluded_ids = mr.occluded_track_ids
        transitions = mr.transitions
        accounting = mr.accounting
        track_of = mr.local_to_track
        # FR-U-06: the online split may have re-split an over-merged frame. Rebuild the affected
        # per-frame object sets (and frame 0's objects / label_grid) so frames, phi and track_of
        # stay mutually consistent -- a split strictly grows a frame's visible count, so an
        # unchanged-count frame is left untouched (no churn for the common no-split path).
        for t, vis in enumerate(mr.visible_nodes):
            field_nodes = [nd for nd in per_frame_nodes[t] if nd.is_field]
            if len(vis) == len(per_frame_nodes[t]) - len(field_nodes):
                continue
            per_frame_nodes[t] = list(vis) + field_nodes
            per_frame[t] = tuple(_to_obj(nd) for nd in per_frame_nodes[t])
        objects = per_frame[0]
        label_grid = _build_label_grid(per_frame_nodes[0], h0, w0)
        # FR-U-08: multiframe C5-C6 violations are RECORDED (not re-split / not crash).
        c5_ok = check_c5(transitions)
        c6_ok = check_c6(accounting)

    # FR-U-08: single-frame C1-C4 are the parse exit gate (re-split already applied above on failure).
    assert check_c1(objects, h0, w0, (), transparent_ids), "C1 violated"
    assert check_c2(objects, label_grid), "C2 violated"
    assert check_c3(objects, (), transparent_ids), "C3 violated"
    assert check_c4(objects, h0, w0), "C4 violated"

    return ParseResult(
        objects=objects,
        label_grid=label_grid,
        occluded_ids=tuple(occluded_ids),
        transparent_ids=tuple(transparent_ids),
        phi=tuple(phi),
        frames=tuple(per_frame),
        transitions=tuple(transitions),
        accounting=tuple(accounting),
        track_of=tuple(track_of),
        c5_ok=c5_ok,
        c6_ok=c6_ok,
    )


def _multiframe(per_frame_nodes: List[List[_Node]], frame_list: List[np.ndarray]
                ) -> "_MultiResult":
    """Track objects across frames and label every transition (object-schema section 4 taxonomy tau).

    Per step we solve the Hungarian correspondence phi_t over the VISIBLE objects (FR-U-06), then
    maintain stable TRACK ids: a matched pair propagates its track id and is classified
    move/scale/recolor/deform; an object that vanishes but reappears later is ``occlude`` (its track
    id is held in a hidden pool and RE-USED on reveal, so the id is retained -- occlude != destroy);
    an object that vanishes for good is ``destroy``; a new object that matches a hidden track's shape
    is ``reveal`` (same id), otherwise ``create``. Transition labels feed C5 and the cell-count deltas
    feed C6. ``phi`` are the frame-local oid maps; ``occluded_track_ids`` the ids ever occluded."""
    h, w = frame_list[0].shape
    visible = [[nd for nd in fr if not nd.is_field] for fr in per_frame_nodes]

    phi: List[Dict[int, int]] = []
    transitions: List[str] = []
    accounting: List[Tuple[str, int, int]] = []
    occluded_track_ids: List[int] = []

    # track bookkeeping: track_id -> the node last seen for it; and a hidden pool for occluded tracks.
    next_track = 0
    local_to_track: List[Dict[int, int]] = [dict() for _ in per_frame_nodes]
    for nd in visible[0]:
        local_to_track[0][nd.oid] = next_track
        next_track += 1
    hidden: Dict[int, _Node] = {}  # track_id -> last-seen node (currently occluded)

    for t in range(len(per_frame_nodes) - 1):
        prev, curr = visible[t], visible[t + 1]
        phi_t = _correspond(prev, curr, h, w)
        # FR-U-06 online over-merge split (object-schema 5.1.3): AFTER phi_t, BEFORE classifying,
        # re-split an over-merged object whose components move apart when that lowers DescLen(phi_t).
        # reserved = the frame's field oid(s), which split pieces must not collide with (shared label_grid).
        prev, phi_t = _resplit_overmerges(prev, curr, phi_t, h, w,
                                          reserved=[nd.oid for nd in per_frame_nodes[t]])
        visible[t] = prev  # persist the split so frames / phi / track_of stay mutually consistent
        phi.append(phi_t)
        prev_by_id = {nd.oid: nd for nd in prev}
        curr_by_id = {nd.oid: nd for nd in curr}
        matched_curr = set(phi_t.values())

        # a freshly-split piece with no track id yet gets a NEW stable track id (the unsplit objects
        # -- and the split's first piece, which kept its oid -- retain the ids they already had).
        for nd in prev:
            if nd.oid not in local_to_track[t]:
                local_to_track[t][nd.oid] = next_track
                next_track += 1

        # matched prev -> curr: propagate track id, classify the transition.
        for pid, cid in phi_t.items():
            tid = local_to_track[t].get(pid)
            if tid is None:
                tid = next_track
                next_track += 1
                local_to_track[t][pid] = tid
            local_to_track[t + 1][cid] = tid
            tau = _classify_transition(prev_by_id[pid], curr_by_id[cid], h, w)
            transitions.append(tau)
            accounting.append((tau, len(prev_by_id[pid].cells), len(curr_by_id[cid].cells)))

        # unmatched prev: occlude (reappears later) or destroy (gone for good).
        for nd in prev:
            if nd.oid in phi_t:
                continue
            tid = local_to_track[t].get(nd.oid, next_track)
            if tid == next_track:
                next_track += 1
            if _reappears_later(nd, per_frame_nodes[t + 2:], h, w):
                transitions.append("occlude")
                accounting.append(("occlude", len(nd.cells), 0))
                occluded_track_ids.append(tid)
                hidden[tid] = nd  # hold the track id for reveal (id retained)
            else:
                transitions.append("destroy")
                accounting.append(("destroy", len(nd.cells), 0))

        # unmatched curr: reveal (matches a hidden track's shape) or create (genuinely new).
        for nd in curr:
            if nd.oid in matched_curr:
                continue
            revealed_tid = _match_hidden(nd, hidden)
            if revealed_tid is not None:
                local_to_track[t + 1][nd.oid] = revealed_tid  # SAME id on reveal
                del hidden[revealed_tid]
                transitions.append("reveal")
                accounting.append(("reveal", 0, len(nd.cells)))
            else:
                local_to_track[t + 1][nd.oid] = next_track
                next_track += 1
                transitions.append("create")
                accounting.append(("create", 0, len(nd.cells)))

    return _MultiResult(
        phi=phi,
        occluded_track_ids=sorted(set(occluded_track_ids)),
        transitions=transitions,
        accounting=accounting,
        local_to_track=local_to_track,
        visible_nodes=visible,
    )


@dataclass(frozen=True)
class _MultiResult:
    phi: List[Dict[int, int]]
    occluded_track_ids: List[int]
    transitions: List[str]
    accounting: List[Tuple[str, int, int]]
    local_to_track: List[Dict[int, int]]
    visible_nodes: List[List[_Node]]  # per-frame visible nodes AFTER the online split (FR-U-06)


def _match_hidden(node: _Node, hidden: Dict[int, _Node]) -> Optional[int]:
    """Return the track id of a hidden (occluded) object whose shape matches ``node`` (a reveal), or
    None. Deterministic: smallest track id wins on a tie."""
    key = _shape_key(node)
    candidates = sorted(tid for tid, hn in hidden.items() if _shape_key(hn) == key)
    return candidates[0] if candidates else None


def _reappears_later(node: _Node, future: Sequence[Sequence[_Node]], h: int, w: int) -> bool:
    """True if a node with the same shape (translation/colour invariant) appears in any future
    frame -- the signal that a vanished object was occluded (and will reveal), not destroyed."""
    target = _shape_key(node)
    for frame_nodes in future:
        for nd in frame_nodes:
            if _shape_key(nd) == target:
                return True
    return False


def _shape_key(node: _Node) -> Tuple[int, FrozenSet[Cell]]:
    """A translation-invariant shape signature (top-left-normalised cell set) + dominant colour."""
    r0, c0, _, _ = node.bbox()
    norm = frozenset((r - r0, c - c0) for r, c in node.cells)
    return (node.dom_color, norm)


def _as_frames(frames) -> List[np.ndarray]:
    """Accept a single 64x64 grid (list-of-lists or ndarray) OR a list of such frames."""
    arr = np.asarray(frames)
    if arr.ndim == 2:
        return [arr.astype(int)]
    if arr.ndim == 3:
        return [arr[i].astype(int) for i in range(arr.shape[0])]
    # a python list whose elements are grids of possibly-ragged python lists.
    out = []
    for f in frames:
        out.append(np.asarray(f, dtype=int))
    return out
