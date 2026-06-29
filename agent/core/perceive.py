"""DivideFrame parser (agent.core.perceive) -- Stage A (single-frame MDL parse).

Realizes the parse algorithm (method-8 / CMP-23) of
``docs/StrictDoc-specs/_assets/gr-arc-3-object-schema.md`` and the FR-U-* unit
obligations of ``04-specification.sdoc`` §4.2/§4.3:

  * FR-U-01 -- atoms = color-CCL over-segmentation (skimage, connectivity=2).
  * FR-U-02 -- pure-MDL objective: ``P* = argmin DescLen(P)`` in BITS, lambda=1,
    no hyperparameters (object-schema §5.1.1).
  * FR-U-03 -- greedy deterministic one-pass agglomeration, stop on
    ``best.delta <= 0`` (object-schema §5.1.3 / §5.1.4 decision 2).
  * FR-U-04 -- cue code table: exactly TWO cues here (connectivity, similarity),
    each a single-color cost model (object-schema §5.1.2 rows 1-2).
  * FR-U-08 -- assert invariants at the parse exit (object-schema §6).
  * FR-U-09 -- library-first: ``skimage.measure.label`` for CCL; the cue DescLen,
    the agglomeration loop, the double-entry store and C1-C4 are self-built
    (object-schema §5.1.5).
  * FR-U-10..13 -- C1-C4 static invariants (object-schema §6 / §2).

Stage A merges ONLY spatially-adjacent same-colour atoms (a same-colour pair is
a merge candidate only if it shares a 4-neighbour boundary). DISJOINT same-colour
regions stay SEPARATE objects: "同色連結 is the weakest seed" (object-schema §5),
so a bare colour match across the board is not a merge reason. Merging disjoint
same-colour regions belongs to deferred cues -- dotted-line / regular scatter is
continuation / lattice, multi-colour nesting is enclosure + parts (D4), co-moving
disjoint is common-fate -- all Stage B/C per object-schema §5 cue table. Since
atoms are already 8-connected single-colour CCL components, adjacency-restricted
proposals leave the color-CCL components standing as the Stage A objects.

Stage A is SINGLE-FRAME. The multi-frame seams (common-fate / phi / occlusion /
transparency / enclosure recursion) are interface-only here and marked
``# SEAM (Stage B/C): ...``; their cues (scaled / line / lattice / symmetry /
enclosure / common-fate, object-schema §5.1.2 rows 3-9) are NOT registered yet.

All costs are BITS (log = log2). No RNG, no builtin ``hash()`` (DP-10): ids are
content-stable (sorted by min-cell), tie-breaks use the merged set's min-cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, log2
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import skimage.measure

from agent.core import registry
from agent.core.attributes import _bbox, _degree
from agent.core.model import GameObject

Cell = Tuple[int, int]


# --------------------------------------------------------------------------- #
# Parse -- the result value (object-schema §2: double-entry label_grid + id->cells)
# --------------------------------------------------------------------------- #

@dataclass
class Parse:
    """A single-frame parse result (object-schema §2 / §6 / FR-U-08).

    ``objects`` are the visible leaves as :class:`GameObject` (string ids
    ``obj_%04d`` so a lexical id sort agrees with the numeric id used in
    ``label_grid``); ``label_grid`` (cell -> int leaf id) and ``id_to_cells``
    (int leaf id -> cells) are the double-entry store (C2). ``field_ids`` is the
    is_field address (object-schema D1 / C4) carried as a set on the Parse
    (there is no is_field slot on GameObject).

    The remaining fields are SEAMS for later stages and are inert in Stage A:
    ``occluded_ids`` / ``transparent_ids`` (Stage B occlusion/transparency lists,
    object-schema §2 (b)/(c)) and ``phi`` (Stage C transition correspondence,
    object-schema §4).
    """

    objects: List[GameObject]
    label_grid: np.ndarray  # 2D int32, cell (row, col) -> leaf int id
    id_to_cells: Dict[int, FrozenSet[Cell]]
    occluded_ids: FrozenSet[int] = frozenset()  # SEAM (Stage B): empty in A
    transparent_ids: FrozenSet[int] = frozenset()  # SEAM (Stage B): empty in A
    phi: Optional[object] = None  # SEAM (Stage C): None in A
    field_ids: FrozenSet[int] = frozenset()  # is_field address (a set on Parse)


# --------------------------------------------------------------------------- #
# Internal mutable parse state (double-entry store kept consistent through merges)
# --------------------------------------------------------------------------- #

@dataclass
class _State:
    """The mutable agglomeration state: the double-entry store plus per-object
    color (single-color in Stage A) and the frame dimensions / palette size that
    the cue cost models read. ``id_to_cells`` and ``label_grid`` are updated
    ATOMICALLY on every merge (C2).

    Stage B adds the LAMINAR TREE (object-schema §2 D4 / §2 (a)): ``id_to_cells``
    / ``label_grid`` / ``id_to_color`` stay the DEEPEST-LEAF partition (the ground
    truth for C1 / C2 / C4 -- a true disjoint cover of C, UNCHANGED by wraps).
    ``id_to_children`` carries the nesting: a PARENT id maps to its (ordered)
    child ids; a leaf maps to ``()``. Parent ids are FRESH (allocated above every
    leaf id) so they NEVER appear in ``label_grid`` and never collide with a leaf.
    ``id_to_parent`` is the inverse (child id -> parent id) for ancestor lookups
    (C3 exception (a)). A parent's cells / color set are DERIVED on demand from
    its leaf descendants (:func:`_descendant_leaves`), never stored on the leaf
    partition (the ``centroid`` pattern -- single source of truth)."""

    label_grid: np.ndarray
    id_to_cells: Dict[int, FrozenSet[Cell]]
    id_to_color: Dict[int, FrozenSet[int]]  # color set per object (Stage A: size 1)
    board_height: int
    board_width: int
    palette_size: int
    # Stage B laminar tree (object-schema §2 D4). Default empty = Stage A flat.
    # ``id_to_children[parent]`` lists ALL child ids; the FIRST ``id_to_own_count``
    # of them form the parent's OWN footprint (the single-colour ring PLUS any
    # interior FIELD cells absorbed into it), the REMAINING ones are the interior
    # PARTS (the differently-coloured nested objects, object-schema §2 D2).
    id_to_children: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    id_to_own_count: Dict[int, int] = field(default_factory=dict)
    id_to_parent: Dict[int, int] = field(default_factory=dict)
    # ``id_to_kind[parent]`` tags how a parent is COSTED / surfaced:
    #   "enclosure" (Stage B1, object-schema §2 D4) = parent ring + nested PARTS;
    #   "line"      (Stage B2, object-schema §5.1.2 line/good-continuation) = ONE
    #               flat object whose children are collinear same-colour fragments
    #               (all "own", no parts -- a dotted line is one thing, not nested);
    #   "lattice"   (Stage B3, object-schema §5.1.2 regularity) = a PARENT + k
    #               identical regularly-spaced tile PARTS (the count/regularity is
    #               kept by surfacing the tiles as parts, unlike the flat line).
    # Absent = "enclosure" (the B1 default).
    id_to_kind: Dict[int, str] = field(default_factory=dict)
    next_id: int = -1  # next fresh parent id; lazily seeded from the leaves

    @property
    def struct_overhead_unit(self) -> float:
        return _struct_overhead_unit(self.board_height, self.board_width)

    def _seed_next_id(self) -> None:
        if self.next_id < 0:
            existing = list(self.id_to_cells) + list(self.id_to_children)
            self.next_id = (max(existing) + 1) if existing else 0

    def alloc_id(self) -> int:
        """Allocate a fresh id above every leaf / parent id (DP-10: monotone,
        content-independent -- a parent id is never reused and never a leaf)."""
        self._seed_next_id()
        new_id = self.next_id
        self.next_id += 1
        return new_id


# --------------------------------------------------------------------------- #
# laminar tree helpers (object-schema §2 D4 / §2 (a)). The leaf partition is the
# ground truth; a parent's cells / colors are DERIVED from its leaf descendants.
# --------------------------------------------------------------------------- #

def _is_leaf(state: _State, node_id: int) -> bool:
    """A node is a leaf iff it has no children (and thus owns cells in the leaf
    partition). Parent ids never appear in ``id_to_cells`` / ``label_grid``."""
    return not state.id_to_children.get(node_id)


def _own_ids(state: _State, parent_id: int) -> Tuple[int, ...]:
    """The leading children that form the parent's OWN footprint = the
    single-colour ring leaf plus any interior FIELD cells absorbed into it (these
    are NOT surfaced as parts)."""
    children = state.id_to_children[parent_id]
    return children[: state.id_to_own_count[parent_id]]


def _part_ids(state: _State, parent_id: int) -> Tuple[int, ...]:
    """The interior PARTS of a parent = the differently-coloured nested objects
    (object-schema §2 D2). These recurse and are surfaced under
    ``GameObject.parts``."""
    children = state.id_to_children[parent_id]
    return children[state.id_to_own_count[parent_id]:]


def _descendant_leaves(state: _State, node_id: int) -> Tuple[int, ...]:
    """The leaf ids under ``node_id`` (``node_id`` itself if it is a leaf), in a
    deterministic DFS order (children visited in stored order). The laminar tree
    is acyclic by construction, so no visited guard is needed here."""
    if _is_leaf(state, node_id):
        return (node_id,)
    out: List[int] = []
    for child in state.id_to_children[node_id]:
        out.extend(_descendant_leaves(state, child))
    return tuple(out)


def _node_cells(state: _State, node_id: int) -> FrozenSet[Cell]:
    """The cells of ``node_id`` = union of its leaf descendants' cells (per
    object-schema §2 (a): a parent CONTAINS its children's cells)."""
    cells: FrozenSet[Cell] = frozenset()
    for leaf in _descendant_leaves(state, node_id):
        cells = cells | state.id_to_cells[leaf]
    return cells


def _node_colors(state: _State, node_id: int) -> FrozenSet[int]:
    """The color set of ``node_id`` = union of its leaf descendants' colors (per
    object-schema §2 D2: a parent is multi-color = ring color + child colors)."""
    colors: FrozenSet[int] = frozenset()
    for leaf in _descendant_leaves(state, node_id):
        colors = colors | state.id_to_color[leaf]
    return colors


def _top_level_ids(state: _State) -> List[int]:
    """The roots of the laminar forest (objects with no parent), sorted by their
    min-cell (content-stable, DP-10). These are the objects surfaced at the top of
    ``parse.objects``; their descendants live under ``GameObject.parts``."""
    roots = [
        node_id
        for node_id in list(state.id_to_cells) + list(state.id_to_children)
        if node_id not in state.id_to_parent
    ]
    return sorted(roots, key=lambda nid: min(_node_cells(state, nid)))


def _ancestors(state: _State, node_id: int) -> FrozenSet[int]:
    """The proper ancestors of ``node_id`` (walking ``id_to_parent`` to the
    root). Used by the C3 nesting exception (a)."""
    out: List[int] = []
    cur = state.id_to_parent.get(node_id)
    while cur is not None:
        out.append(cur)
        cur = state.id_to_parent.get(cur)
    return frozenset(out)


# --------------------------------------------------------------------------- #
# cue cost models (object-schema §5.1.2; BITS, log=log2, lambda=1, no hyperparams)
# --------------------------------------------------------------------------- #
# A cue is a COST MODEL applicable only to a SINGLE-COLOR object. A multi-color
# object has NO applicable cue, so its DescLen is +inf (deferring cross-color
# enclosure/recursion to Stage B). In Stage A merges only ever pair same-color
# objects, so multi-color objects never actually form; the guard keeps the cost
# model correct regardless.


def boundary_len(cells: FrozenSet[Cell]) -> int:
    """Exposed 4-neighbour unit edges = ``sum(4 - _degree(cell, cells))``
    (object-schema §5.1.2 connectivity code: the boundary length)."""
    return sum(4 - _degree(cell, cells) for cell in cells)


def _palette_bits(palette_size: int) -> float:
    """``log2(palette_size)`` bits to name one color out of the frame palette;
    0.0 when the palette has at most one color (guard log2(<=1))."""
    if palette_size <= 1:
        return 0.0
    return log2(palette_size)


def _is_single_color(color_set: FrozenSet[int]) -> bool:
    return len(color_set) == 1


@registry.cue("cue_connectivity")
def cue_connectivity(cells: FrozenSet[Cell], palette_size: int) -> float:
    """connectivity cue cost (object-schema §5.1.2 row 1): encode the BOUNDARY
    plus ONE color = ``boundary_len + palette_bits``. Cheap for a single-color
    CONNECTED blob (small boundary)."""
    return boundary_len(cells) + _palette_bits(palette_size)


@registry.cue("cue_similarity")
def cue_similarity(cells: FrozenSet[Cell], palette_size: int) -> float:
    """similarity cue cost (object-schema §5.1.2 row 2): encode ONE color plus a
    raster MASK over the bbox = ``palette_bits + mask_bits`` where
    ``mask_bits = bbox_height * bbox_width`` (1 bit per bbox cell). Cheap for a
    same-color DISCONNECTED scatter whose bbox is small relative to the boundary
    a connectivity code would pay."""
    _, _, _, _, bbox_height, bbox_width = _bbox(cells)
    mask_bits = bbox_height * bbox_width
    return _palette_bits(palette_size) + mask_bits


@registry.cue("cue_enclosure")
def cue_enclosure(
    ring_cells: FrozenSet[Cell],
    palette_size: int,
    child_desclens: Tuple[float, ...] = (),
) -> float:
    """enclosure cue cost (object-schema §5.1.2 row 7 / §5 cue table line ~143:
    ``parent bits + Sigma child DescLen``). The parent RING is a single-color blob
    (the frame's colour) so its OWN cost reuses the connectivity model
    (boundary + 1 colour); the children carry their own (possibly nested) DescLen
    and are SUMMED in. Recursive: a child DescLen may itself be an enclosure cost.

    ``ring_cells`` are the parent's OWN cells (the leaf that became the ring after
    its interior children were folded out -- NOT the full descendant union). The
    cue is the single multi-colour-licensing code: a parent + differently-coloured
    parts (object-schema §2 D2) instead of an inf multi-colour blob."""
    return cue_connectivity(ring_cells, palette_size) + sum(child_desclens)


def _log2_count(n: int) -> float:
    """``log2(n)`` bits to encode a non-negative integer count / gap, guarding
    log2(<=1) -> 0.0 (a single value needs no bits)."""
    return log2(n) if n > 1 else 0.0


@registry.cue("cue_continuation")
def cue_continuation(
    fragment_count: int,
    gap: int,
    fragment_bits: float,
    board_height: int,
    board_width: int,
    palette_size: int,
) -> float:
    """line / good-continuation cue cost (object-schema §5.1.2 line row / §5 cue
    table line ~140): encode a dotted / evenly-spaced collinear run as
    ``2 endpoints + spacing + count + colour + one fragment template``:

        2 * (log2(board_height) + log2(board_width))   the two endpoints' addresses
        + log2(gap)                                    the constant spacing
        + log2(fragment_count)                         how many fragments
        + palette_bits                                 the (single) colour
        + fragment_bits                                ONE fragment's shape template

    ``fragment_bits`` is the connectivity cost of a SINGLE representative fragment
    (the repeated tile); the line code pays for the template ONCE rather than N
    times (that is what makes it beat N separate similarity codes). All terms are
    bits; no hyperparameter (object-schema §5.1.1 lambda=1)."""
    endpoint_bits = 2.0 * (
        _log2_count(board_height) + _log2_count(board_width)
    )
    return (
        endpoint_bits
        + _log2_count(gap)
        + _log2_count(fragment_count)
        + _palette_bits(palette_size)
        + fragment_bits
    )


@registry.cue("cue_lattice")
def cue_lattice(
    tile_count: int,
    period: int,
    tile_bits: float,
    residual_bits: float,
    board_height: int,
    board_width: int,
    palette_size: int,
) -> float:
    """lattice / regularity cue cost (object-schema §5.1.2 lattice row / §5 cue
    table line ~141: ``tile bits + log k + log m + Sigma diff``). Encode a regular
    repeat of identical tiles as ONE tile template + the lattice descriptor:

        tile_bits                                      ONE tile's shape template
        + 2 * (log2(board_height) + log2(board_width)) the lattice origin (anchor)
        + log2(tile_count)        (= log k)            how many tiles
        + log2(period)            (= log m)            the period vector magnitude
        + palette_bits                                 the (single) colour
        + residual_bits           (= Sigma diff)       per-tile deviation (0 if exact)

    Unlike the line code (which surfaces ONE flat object), the lattice surfaces a
    PARENT + k identical tile PARTS (object-schema §2 D4 / §5.1.3): the tiles are
    preserved as parts so the count / regularity is kept. All terms are bits; no
    hyperparameter (object-schema §5.1.1 lambda=1)."""
    anchor_bits = 2.0 * (_log2_count(board_height) + _log2_count(board_width))
    return (
        tile_bits
        + anchor_bits
        + _log2_count(tile_count)
        + _log2_count(period)
        + _palette_bits(palette_size)
        + residual_bits
    )


def register_cues() -> None:
    """Re-arm the parse cues (idempotent). The autouse test fixture wipes the
    registry, so tests call this to re-register (mirrors
    ``attributes.register_detectors``). Stage A: connectivity + similarity;
    Stage B1 adds enclosure (object-schema §5.1.2 row 7); Stage B2 adds
    continuation (the line / good-continuation row); Stage B3 adds lattice (the
    regularity row)."""
    for impl_key, fn in (
        ("cue_connectivity", cue_connectivity),
        ("cue_similarity", cue_similarity),
        ("cue_enclosure", cue_enclosure),
        ("cue_continuation", cue_continuation),
        ("cue_lattice", cue_lattice),
    ):
        if not registry.is_registered(impl_key):
            registry.cue(impl_key)(fn)


# SEAM (Stage B3+): the deferred cues are NOT registered yet --
#   scaled / symmetry (object-schema §5.1.2 rows 3, 6) and
# SEAM (Stage C): common-fate (object-schema §5.1.2 row 9, multi-frame only).


# --------------------------------------------------------------------------- #
# DescLen (object-schema §5.1.1)
# --------------------------------------------------------------------------- #

def _struct_overhead_unit(board_height: int, board_width: int) -> float:
    """Per-object fixed bbox-address cost (object-schema §5.1.1 degeneration
    penalty): ``2*log2(board_height) + 2*log2(board_width)`` -- the bits to place
    a bbox on the board (top-row + left-col + height + width). A board constant,
    the SAME for every object; guards log2(1) -> 0.0. This P-level term is what
    punishes the all-cells-separate degeneration (it makes merging cheaper).

    ☆☆ (Claude proposal, user-approved 2026-06-28): object-schema §5.1.1 names
    struct_overhead ("tree / object-count / occupancy structural bits") but pins
    no formula; the bbox-address form here fills that gap with no free parameter
    (it scales with board size, not a magic constant). See the handoff §4 flag."""
    bits_height = log2(board_height) if board_height > 1 else 0.0
    bits_width = log2(board_width) if board_width > 1 else 0.0
    return 2.0 * bits_height + 2.0 * bits_width


def _desclen_object(
    cells: FrozenSet[Cell], color_set: FrozenSet[int], palette_size: int
) -> float:
    """``DescLen(O) = min over applicable cues of cue_cost(O)`` (object-schema
    §5.1.1 / §5.1.2). A cue applies only if O is single-color; a multi-color
    object has no applicable cue, so its DescLen is ``inf`` (defers cross-color
    enclosure/recursion to Stage B)."""
    if not _is_single_color(color_set):
        return inf
    return min(
        cue_connectivity(cells, palette_size),
        cue_similarity(cells, palette_size),
    )


def _desclen_node(state: _State, node_id: int) -> float:
    """``DescLen`` of a laminar-tree node (object-schema §5.1.1 / §5.1.2).

    A LEAF is costed by :func:`_desclen_object` (min over the single-colour cues).
    A "line" PARENT (Stage B2) is costed by the continuation code (endpoints + gap
    + count + colour + one fragment template). A "lattice" PARENT (Stage B3) is
    costed by the lattice code (one tile template + period + count). An "enclosure"
    PARENT (Stage B1) is costed by the recursive enclosure code (object-schema
    §5.1.2 row 7): its OWN ring cost plus the sum of its interior children's
    DescLen. The struct_overhead term is added by the caller (one unit per surfaced
    top-level node)."""
    if _is_leaf(state, node_id):
        return _desclen_object(
            state.id_to_cells[node_id],
            state.id_to_color[node_id],
            state.palette_size,
        )
    kind = state.id_to_kind.get(node_id)
    if kind == "line":
        return _line_desclen(state, state.id_to_children[node_id])
    if kind == "lattice":
        return _lattice_desclen(state, state.id_to_children[node_id])
    ring_cells = frozenset().union(
        *(_node_cells(state, o) for o in _own_ids(state, node_id))
    )
    child_desclens = tuple(
        _desclen_node(state, c) for c in _part_ids(state, node_id)
    )
    return cue_enclosure(ring_cells, state.palette_size, child_desclens)


def _total_desclen(state: _State) -> float:
    """``DescLen(P) = sum_O DescLen(O) + N * struct_overhead_unit`` (object-schema
    §5.1.1; the phi transition term is 0 in Stage A). ``N`` = the number of
    TOP-LEVEL objects, NOT the number of tree nodes: struct_overhead is the
    object-COUNT penalty, and nesting collapses a ring + interior into ONE
    top-level object (its internal structure is paid for INSIDE the recursive
    enclosure DescLen = parent bits + Sigma child DescLen, with no extra per-child
    overhead -- the parent's bbox locates its children). With no parents this
    equals the Stage A leaf form. Provided as a helper for the determinism test."""
    roots = _top_level_ids(state)
    per_object = sum(_desclen_node(state, r) for r in roots)
    return per_object + len(roots) * state.struct_overhead_unit


# --------------------------------------------------------------------------- #
# atoms (FR-U-01; object-schema §5.1.3 / §5.1.4 decision 1)
# --------------------------------------------------------------------------- #

def _atoms(grid: np.ndarray) -> List[Tuple[FrozenSet[Cell], int]]:
    """Color-CCL over-segmentation (FR-U-01): for each color ``c`` in the frame,
    ``skimage.measure.label(grid == c, connectivity=2)`` (8-connectivity per the
    object-schema §5.1.3 pseudocode); each component is one single-color atom.
    Returns ``[(cells, color), ...]`` (order does not matter -- ids are assigned
    by min-cell later)."""
    atoms: List[Tuple[FrozenSet[Cell], int]] = []
    for color in np.unique(grid):
        # FR-U-09: skimage is the CCL source (self-built code never relabels).
        labels = skimage.measure.label(grid == color, connectivity=2)
        for component_id in range(1, int(labels.max()) + 1):
            rows, cols = np.where(labels == component_id)
            cells = frozenset(
                (int(row), int(col))
                for row, col in zip(rows.tolist(), cols.tolist())
            )
            atoms.append((cells, int(color)))
    return atoms


def _initial_state(grid: np.ndarray) -> _State:
    """Build the double-entry store from atoms. ids 0,1,2,... are assigned by
    sorting atoms on ``min(cells)`` ascending (content-stable, DP-10: disjoint
    atoms have unique min-cells, a strict total order)."""
    atoms = _atoms(grid)
    atoms_sorted = sorted(atoms, key=lambda atom: min(atom[0]))

    board_height, board_width = int(grid.shape[0]), int(grid.shape[1])
    palette_size = int(len(np.unique(grid)))

    label_grid = np.full((board_height, board_width), -1, dtype=np.int32)
    id_to_cells: Dict[int, FrozenSet[Cell]] = {}
    id_to_color: Dict[int, FrozenSet[int]] = {}
    for new_id, (cells, color) in enumerate(atoms_sorted):
        id_to_cells[new_id] = cells
        id_to_color[new_id] = frozenset({color})
        for (row, col) in cells:
            label_grid[row, col] = new_id

    return _State(
        label_grid=label_grid,
        id_to_cells=id_to_cells,
        id_to_color=id_to_color,
        board_height=board_height,
        board_width=board_width,
        palette_size=palette_size,
    )


# --------------------------------------------------------------------------- #
# merge + proposal generation (object-schema §5.1.3)
# --------------------------------------------------------------------------- #

def _are_4_adjacent(cells_a: FrozenSet[Cell], cells_b: FrozenSet[Cell]) -> bool:
    """True if some cell of ``a`` is 4-adjacent to some cell of ``b``."""
    neighbours = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for (row, col) in cells_a:
        for (dr, dc) in neighbours:
            if (row + dr, col + dc) in cells_b:
                return True
    return False


def _proposals(state: _State) -> List[Tuple[int, int]]:
    """Same-color merge proposals as unordered ``(i, j)`` pairs with ``i < j``,
    generated in a deterministic order (object ids ascending). A pair is a
    candidate ONLY IF the two objects are SPATIALLY ADJACENT (share a 4-neighbour
    boundary) AND have the same color. Stage A therefore merges only spatially-
    adjacent same-color atoms; DISJOINT same-color regions stay SEPARATE objects.

    Why adjacency-restricted (object-schema §5.1.2 / §5 cue table): "同色連結 is
    the WEAKEST seed" -- a bare colour match across the board is NOT a merge
    reason. Merging spatially DISJOINT same-color regions is the job of deferred
    Stage B/C cues, not of Stage A's similarity cost model:
      * dotted-line / regular scatter -> continuation / lattice (Stage B),
      * multi-colour nesting          -> enclosure + parts, D4 (Stage B),
      * co-moving disjoint groups      -> common-fate (Stage C).
    None of those are registered here, so a disjoint same-color scatter must NOT
    be agglomerated by Stage A.

    Cost model unchanged: cue_connectivity and cue_similarity remain the
    min-over-cue DescLen for a proposed pair (object-schema §5.1.2). Only the
    PROPOSAL set is restricted to adjacency -- the cues still pick the cheaper
    explanation for the pairs that ARE adjacent.

    Atoms are already 8-connected single-color CCL components (FR-U-01), so any
    same-color 8-adjacent cells are ALREADY one atom; restricting proposals to
    4-adjacency means same-color atoms that touch only diagonally can still be
    proposed, while disjoint same-color regions are never proposed. The result is
    the color-CCL components standing as the Stage A objects.
    """
    ids = sorted(state.id_to_cells)
    pairs: List[Tuple[int, int]] = []
    for a_index in range(len(ids)):
        for b_index in range(a_index + 1, len(ids)):
            i, j = ids[a_index], ids[b_index]
            if state.id_to_color[i] != state.id_to_color[j]:
                continue  # Stage A merges only same-color objects
            if not _are_4_adjacent(state.id_to_cells[i], state.id_to_cells[j]):
                continue  # disjoint same-color regions stay SEPARATE (weakest seed)
            pairs.append((i, j))
    return pairs


def _apply_merge(state: _State, i: int, j: int) -> None:
    """Merge object ``j`` into the SMALLEST id (``min(i, j)`` -- a merged object
    INHERITS the smallest constituent id, DP-10) and update BOTH stores
    atomically (C2)."""
    keep, drop = (i, j) if i < j else (j, i)
    merged_cells = state.id_to_cells[keep] | state.id_to_cells[drop]
    state.id_to_cells[keep] = merged_cells
    state.id_to_color[keep] = state.id_to_color[keep] | state.id_to_color[drop]
    for (row, col) in state.id_to_cells[drop]:
        state.label_grid[row, col] = keep
    del state.id_to_cells[drop]
    del state.id_to_color[drop]


def _merge_delta(state: _State, i: int, j: int) -> float:
    """LOCAL DescLen reduction of merging ``i`` and ``j``: the two objects'
    DescLen plus ONE struct_overhead_unit (one fewer object) MINUS the merged
    object's DescLen. ``delta > 0`` means the merge lowers total DescLen
    (object-schema §5.1.1 / §5.1.3)."""
    cells_i, cells_j = state.id_to_cells[i], state.id_to_cells[j]
    color_i, color_j = state.id_to_color[i], state.id_to_color[j]
    before = (
        _desclen_object(cells_i, color_i, state.palette_size)
        + _desclen_object(cells_j, color_j, state.palette_size)
        + state.struct_overhead_unit
    )
    merged_cells = cells_i | cells_j
    merged_color = color_i | color_j
    after = _desclen_object(merged_cells, merged_color, state.palette_size)
    return before - after


def _min_cell_of_merge(state: _State, i: int, j: int) -> Cell:
    return min(state.id_to_cells[i] | state.id_to_cells[j])


def _agglomerate(state: _State) -> None:
    """Greedy deterministic one-pass agglomeration (FR-U-02/03; object-schema
    §5.1.3). Each round picks the argmax-delta merge; ties broken by the
    lexicographically SMALLEST merged-set min-cell. Stops when no proposal has
    ``delta > 0`` (local minimum, ``best.delta <= 0``). No backtrack / restart.

    TIE-BREAK (SC-10 / DP-10): the key is ``(delta, neg_lex(min_cell))`` so a
    larger delta wins and, on a tie, the smaller merged min-cell wins. This is a
    STRICT TOTAL ORDER because distinct merged cell sets within a partition have
    DISTINCT min-cells (the cells are disjoint), so no two live proposals can
    share a min-cell -- RNG-free and hash-free."""
    while True:
        best_key: Optional[Tuple[float, Tuple[int, int]]] = None
        best_pair: Optional[Tuple[int, int]] = None
        for (i, j) in _proposals(state):
            delta = _merge_delta(state, i, j)
            min_row, min_col = _min_cell_of_merge(state, i, j)
            # neg_lex: smaller (row, col) -> larger key, so it wins the tie.
            key = (delta, (-min_row, -min_col))
            if best_key is None or key > best_key:
                best_key = key
                best_pair = (i, j)
        if best_pair is None or best_key[0] <= 0:
            break  # local minimum: no merge lowers DescLen
        _apply_merge(state, best_pair[0], best_pair[1])


# --------------------------------------------------------------------------- #
# is_field (object-schema D1 / C4; user-decided = a set on Parse)
# --------------------------------------------------------------------------- #

def _background_color(grid: np.ndarray) -> Optional[int]:
    """The frame's MODAL colour = the background / field colour (☆☆ rule,
    deterministic, no game literal). ``None`` for an empty frame. Single source of
    the background colour used by both :func:`_field_ids` and the enclosure pass
    (the substrate the rings sit on)."""
    if grid.size == 0:
        return None
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _field_ids(grid: np.ndarray, state: _State) -> FrozenSet[int]:
    """The is_field address (☆☆ rule, deterministic, no game literal): the
    background color is the frame's MODAL color; the field leaf is the leaf with
    the LARGEST cell count whose (single) color == background, ties broken by
    smallest min-cell. Returns ``frozenset({that leaf id})`` (object-schema D1)."""
    if grid.size == 0 or not state.id_to_cells:
        return frozenset()  # empty frame has no field leaf (total-function guard)
    background = _background_color(grid)

    candidates = [
        leaf_id
        for leaf_id, color_set in state.id_to_color.items()
        if background in color_set
    ]
    if not candidates:
        return frozenset()
    # largest cell count, ties -> smallest min-cell. Negate the count so the max
    # by (-count, min_cell) is the largest count then smallest min-cell.
    field_leaf = min(
        candidates,
        key=lambda leaf_id: (
            -len(state.id_to_cells[leaf_id]),
            min(state.id_to_cells[leaf_id]),
        ),
    )
    return frozenset({field_leaf})


# --------------------------------------------------------------------------- #
# invariant asserts (FR-U-08/10..13; object-schema §6 C1-C4). Stage A: C1-C4.
# C5/C6 need >=2 frames -> seam (defined, called only if prev is not None).
# --------------------------------------------------------------------------- #

def _all_grid_cells(label_grid: np.ndarray) -> FrozenSet[Cell]:
    height, width = label_grid.shape
    return frozenset(
        (row, col) for row in range(height) for col in range(width)
    )


def _assert_c1(state: _State) -> None:
    """C1 (FR-U-10): the leaves are a true partition of the visible cells --
    each non-empty (P1), pairwise disjoint (P3'), and their union == all grid
    cells (P2)."""
    seen: Dict[Cell, int] = {}
    for leaf_id, cells in state.id_to_cells.items():
        if not cells:
            raise AssertionError(f"C1(P1): leaf {leaf_id} has no cells")
        for cell in cells:
            if cell in seen:
                raise AssertionError(
                    f"C1(P3'): cell {cell} shared by leaves {seen[cell]} and {leaf_id}"
                )
            seen[cell] = leaf_id
    union = frozenset(seen)
    all_cells = _all_grid_cells(state.label_grid)
    if union != all_cells:
        raise AssertionError(
            f"C1(P2): leaves do not cover the frame "
            f"(missing {sorted(all_cells - union)[:8]})"
        )


def _assert_c2(state: _State) -> None:
    """C2 (FR-U-11): the double-entry store is consistent both directions --
    ``label_grid[row, col] == i  <=>  (row, col) in id_to_cells[i]``."""
    # forward: every id_to_cells entry matches label_grid.
    for leaf_id, cells in state.id_to_cells.items():
        for (row, col) in cells:
            actual = int(state.label_grid[row, col])
            if actual != leaf_id:
                raise AssertionError(
                    f"C2: id_to_cells[{leaf_id}] has {(row, col)} but "
                    f"label_grid says {actual}"
                )
    # backward: every label_grid entry matches id_to_cells.
    height, width = state.label_grid.shape
    for row in range(height):
        for col in range(width):
            leaf_id = int(state.label_grid[row, col])
            if (row, col) not in state.id_to_cells.get(leaf_id, frozenset()):
                raise AssertionError(
                    f"C2: label_grid[{row},{col}]={leaf_id} but "
                    f"{(row, col)} not in id_to_cells[{leaf_id}]"
                )


def _assert_c3(state: _State) -> None:
    """C3 (FR-U-12): a cell claimed by two objects must be explained by nesting,
    occlusion or transparency (object-schema §2 (a)/(b)/(c)).

    The LEAF partition itself never shares a cell (that is C1(P3')). The sharing
    C3 governs is between TREE NODES at different levels: a parent shares every
    cell with its descendants by construction (object-schema §2 (a) 入れ子). Stage
    B1 permits exactly that nesting exception -- a cell may be claimed by two nodes
    iff one is an ANCESTOR of the other. Exceptions (b) occluded / (c) transparent
    stay Stage-B2/C seams (occluded_ids / transparent_ids are still empty), so any
    OTHER overlap (two nodes with no ancestor relation) is a violation.

    Implementation: walk every node (leaves and parents) and record, per cell, the
    set of claimant nodes; a pair of claimants on one cell is allowed only if one
    is in the other's ancestor set."""
    nodes = list(state.id_to_cells) + list(state.id_to_children)
    cell_claims: Dict[Cell, List[int]] = {}
    for node_id in nodes:
        for cell in _node_cells(state, node_id):
            cell_claims.setdefault(cell, []).append(node_id)
    for cell, claimants in cell_claims.items():
        for a_index in range(len(claimants)):
            for b_index in range(a_index + 1, len(claimants)):
                a, b = claimants[a_index], claimants[b_index]
                if a in _ancestors(state, b) or b in _ancestors(state, a):
                    continue  # exception (a) 入れ子: one node nests the other
                raise AssertionError(
                    f"C3: cell {cell} shared by nodes {a} and {b} with no "
                    f"nesting/occlusion/transparency to explain it"
                )


def _assert_c4(state: _State, field_ids: FrozenSet[int]) -> None:
    """C4 (FR-U-13): every cell (including the is_field leaf) belongs to exactly
    one leaf -- the union of all id_to_cells (field leaf included) == the full
    grid cell set (object-schema §2 D1 / §6 C4)."""
    union = frozenset().union(*state.id_to_cells.values()) if state.id_to_cells else frozenset()
    all_cells = _all_grid_cells(state.label_grid)
    if union != all_cells:
        raise AssertionError(
            f"C4: not every cell belongs to a leaf "
            f"(missing {sorted(all_cells - union)[:8]})"
        )
    # the field leaf, if any, must be a real leaf participating in the cover.
    for field_leaf in field_ids:
        if field_leaf not in state.id_to_cells:
            raise AssertionError(f"C4: field leaf {field_leaf} is not a leaf")


# SEAM (Stage B/C): t >= 1 invariants -- defined but only called if prev is not None.
def _assert_c5(prev: object, state: _State) -> None:
    """C5 (FR-U-14, t>=1): every corresponded object's transition tau(O, phi(O))
    is a known taxonomy member (object-schema §6 C5). SEAM (Stage C): no logic in
    Stage A; reached only when ``prev is not None``."""
    raise NotImplementedError("C5 is a Stage C seam (needs >=2 frames)")


def _assert_c6(prev: object, state: _State) -> None:
    """C6 (FR-U-15, t>=1): cells are conserved across a transition except
    create / destroy / occlude (double-entry accounting, object-schema §6 C6).
    SEAM (Stage C): no logic in Stage A; reached only when ``prev is not None``."""
    raise NotImplementedError("C6 is a Stage C seam (needs >=2 frames)")


# --------------------------------------------------------------------------- #
# Stage B/C correspondence + recursion seams (interface only, NO logic)
# --------------------------------------------------------------------------- #

def _correspond(prev: object, curr: object) -> object:
    """SEAM (Stage C): cross-frame object correspondence (common-fate). The
    transition cost matrix ``cost[i][j] = DescLen(O_i -> O_j)`` is solved with
    ``scipy.optimize.linear_sum_assignment`` (Hungarian) -- object-schema §5.1.3
    online stage / FR-U-06. Not implemented in Stage A."""
    raise NotImplementedError("correspondence is a Stage C seam")


def _interior_of(state: _State, ring_id: int) -> FrozenSet[Cell]:
    """The hole(s) of leaf ``ring_id`` by HOLE-TOPOLOGY, not bbox (object-schema
    §2 enclosure): flood-fill the EXTERIOR from the ring's bbox border over all
    NON-ring cells; the ring's bbox cells NOT reached are its interior. A solid
    wall has an empty interior (its bbox border touches everything), so it encloses
    nothing -- which is exactly why a frame is a parent but a filled block is not.

    Purely topological -- no colour / coordinate literal. Returns the frozenset of
    interior cells (possibly empty)."""
    ring_cells = state.id_to_cells[ring_id]
    r0, r1, c0, c1, _, _ = _bbox(ring_cells)
    # Exterior reachability over the bbox-with-one-cell halo (so the border is
    # always reachable from outside): BFS over non-ring cells from the halo frame.
    in_box = lambda r, c: (r0 - 1) <= r <= (r1 + 1) and (c0 - 1) <= c <= (c1 + 1)
    exterior: set = set()
    # seed = the halo ring (the frame one cell outside the bbox), all non-ring.
    frontier: List[Cell] = []
    for r in range(r0 - 1, r1 + 2):
        for c in (c0 - 1, c1 + 1):
            if in_box(r, c) and (r, c) not in ring_cells and (r, c) not in exterior:
                exterior.add((r, c))
                frontier.append((r, c))
    for c in range(c0 - 1, c1 + 2):
        for r in (r0 - 1, r1 + 1):
            if in_box(r, c) and (r, c) not in ring_cells and (r, c) not in exterior:
                exterior.add((r, c))
                frontier.append((r, c))
    neighbours = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while frontier:
        r, c = frontier.pop()
        for dr, dc in neighbours:
            nr, nc = r + dr, c + dc
            if not in_box(nr, nc):
                continue
            if (nr, nc) in ring_cells or (nr, nc) in exterior:
                continue
            exterior.add((nr, nc))
            frontier.append((nr, nc))
    interior = frozenset(
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in ring_cells and (r, c) not in exterior
    )
    return interior


def _encloses(state: _State, ring_id: int, inner_id: int) -> bool:
    """``ring_id`` encloses ``inner_id`` iff every cell of ``inner_id`` lies in
    ``ring_id``'s topological interior (the hole), AND ``inner_id`` is not the
    ring itself. Hole-topology test (object-schema §2), not bbox containment."""
    if ring_id == inner_id:
        return False
    interior = _interior_of(state, ring_id)
    if not interior:
        return False
    return _node_cells(state, inner_id) <= interior


def _wrap_children(
    state: _State,
    own_ids: Tuple[int, ...],
    part_ids: Tuple[int, ...],
) -> int:
    """Fold an accepted enclosure into a PARENT node (object-schema §5.1.3 /
    FR-U-05). Allocates a fresh parent id whose children are ``own_ids +
    part_ids``: ``own_ids`` (the single-colour frame leaf, plus any interior
    FIELD cells absorbed into the parent footprint) form the parent's own ring;
    ``part_ids`` (the differently-coloured nested objects, object-schema §2 D2)
    are the interior PARTS that recurse. Re-parents every child to the new parent.
    The LEAF partition (``id_to_cells`` / ``label_grid`` / ``id_to_color``) is
    UNTOUCHED -- nesting shares cells, it does not move them (C1 / C2 / C4 stay
    exact). Returns the new parent id.

    Determinism: the caller supplies a deterministic child order; the parent id is
    freshly allocated (monotone, never reused)."""
    parent_id = state.alloc_id()
    children = tuple(own_ids) + tuple(part_ids)
    state.id_to_children[parent_id] = children
    state.id_to_own_count[parent_id] = len(own_ids)
    for child in children:
        state.id_to_parent[child] = parent_id
    return parent_id


# --------------------------------------------------------------------------- #
# enclosure structural pass (object-schema §5.1.3: after agglomeration, group
# the settled leaves into parent + parts by the enclosure cue, recursing into the
# interior; pure-MDL acceptance, deterministic).
# --------------------------------------------------------------------------- #

def _is_field_colored(state: _State, node_id: int, field_color: Optional[int]) -> bool:
    """True iff ``node_id`` is a single leaf whose colour is the field/background
    colour. Such an interior leaf is absorbed into a parent's footprint (it is
    background filler inside the ring), never surfaced as a part, and is never a
    ring candidate (the substrate is not a container, object-schema §2 D1)."""
    if field_color is None or not _is_leaf(state, node_id):
        return False
    return state.id_to_color[node_id] == frozenset({field_color})


def _ring_candidates(
    state: _State, candidate_ids: Tuple[int, ...], field_color: Optional[int]
) -> List[int]:
    """The leaves in ``candidate_ids`` that COULD be rings: a non-field single
    leaf with a non-empty hole-topology interior, sorted by min-cell (DP-10)."""
    return sorted(
        (
            rid
            for rid in candidate_ids
            if _is_leaf(state, rid)
            and not _is_field_colored(state, rid, field_color)
            and _interior_of(state, rid)
        ),
        key=lambda rid: min(state.id_to_cells[rid]),
    )


def _enclosure_pass(
    state: _State,
    candidate_ids: Tuple[int, ...],
    field_color: Optional[int],
) -> Tuple[int, ...]:
    """Deterministic enclosure grouping over ``candidate_ids`` (the roots of one
    region). Returns the surviving ROOTS after folding (parents replace their
    wrapped constituents). Two phases keep it laminar AND side-effect-free during
    scoring (object-schema §5.1.3 / §2 D4):

      PHASE 1 (structure, no mutation): from the containment partial order assign
        each candidate ring its DIRECT interior = the candidates it encloses that
        no INTERVENING candidate ring also encloses. This is laminar by hole
        topology (an interior cell lies on exactly one ring chain).
      PHASE 2 (acceptance, bottom-up): process rings INNERMOST-first (a ring whose
        direct interior contains no other ring goes first) so each wrap sees its
        children already folded. Wrap iff pure MDL accepts (``_wrap_gain > 0``);
        on reject the ring's would-be children stay as siblings at this level.

    Bottom-up + direct-interior assignment means the recursion is implicit (a ring
    nests already-formed sub-parents) -- no speculative mutation during scoring."""
    candidate_ids = tuple(candidate_ids)
    rings = _ring_candidates(state, candidate_ids, field_color)
    if not rings:
        return candidate_ids

    # PHASE 1: direct interior per ring (drop constituents an intervening ring also
    # encloses). encloses() is a partial order under hole-topology containment.
    encloses_set = {
        rid: {
            cid for cid in candidate_ids
            if cid != rid and _encloses(state, rid, cid)
        }
        for rid in rings
    }
    direct: Dict[int, List[int]] = {}
    for rid in rings:
        inner = encloses_set[rid]
        other_rings = [r for r in rings if r != rid and r in inner]
        direct[rid] = [
            cid
            for cid in inner
            if not any(cid in encloses_set[r] for r in other_rings)
        ]

    # depth = how many rings enclose this ring (deeper rings have more ancestors);
    # process DEEPEST first so a ring's children are already folded when it wraps.
    depth = {
        rid: sum(1 for other in rings if rid in encloses_set[other])
        for rid in rings
    }
    order = sorted(rings, key=lambda rid: (-depth[rid], min(state.id_to_cells[rid])))

    # PHASE 2: bottom-up MDL acceptance. ``current_root`` maps an original
    # constituent id to the root of the subtree it now belongs to (a wrapped child
    # is reached through its parent).
    current_root: Dict[int, int] = {}

    def root_of(node_id: int) -> int:
        seen = node_id
        while seen in current_root:
            seen = current_root[seen]
        return seen

    survivors = set(candidate_ids)
    for rid in order:
        child_roots = []
        for cid in direct[rid]:
            r = root_of(cid)
            if r in survivors:
                child_roots.append(r)
        child_roots = sorted(set(child_roots), key=lambda c: min(_node_cells(state, c)))
        own = (rid,) + tuple(
            c for c in child_roots if _is_field_colored(state, c, field_color)
        )
        parts = tuple(
            c for c in child_roots if not _is_field_colored(state, c, field_color)
        )
        if not parts:
            continue  # nothing but background inside -> not a real enclosure
        if _wrap_gain(state, own, parts) <= 0.0:
            continue  # pure-MDL reject: the wrap does not lower DescLen
        parent_id = _wrap_children(state, own, parts)
        for c in own[1:]:
            current_root[c] = parent_id
        for c in parts:
            current_root[c] = parent_id
        current_root[rid] = parent_id
        survivors -= ({rid} | set(own[1:]) | set(parts))
        survivors.add(parent_id)

    return tuple(sorted(survivors, key=lambda nid: min(_node_cells(state, nid))))


def _wrap_gain(
    state: _State, own_ids: Tuple[int, ...], part_ids: Tuple[int, ...]
) -> float:
    """The MDL bits SAVED by folding ``own_ids`` (ring + absorbed interior field)
    plus ``part_ids`` into one parent vs. leaving every constituent a separate
    top-level object (object-schema §5.1.3 / §5.1.2 row 7). Positive = the wrap
    lowers total DescLen. struct_overhead is the object-COUNT penalty, so:

      before = Sigma_{c in own U parts} [DescLen(c) + overhead]
               (each constituent is its OWN top-level object)
      after  = [ring connectivity bits + Sigma_{c in parts} DescLen(c)] + overhead
               (ONE composite object; the enclosure code pays for the interior
                structure, no extra per-part overhead -- the parent bbox locates
                them).

    The overhead saving (one fewer top-level object per folded constituent) is
    what makes a real ring+interior wrap win; absorbing the interior field cells
    into the ring footprint also removes their separate connectivity cost."""
    overhead = state.struct_overhead_unit
    constituents = tuple(own_ids) + tuple(part_ids)
    before = sum(_desclen_node(state, c) + overhead for c in constituents)

    ring_cells = frozenset().union(*(_node_cells(state, o) for o in own_ids))
    child_desclens = tuple(_desclen_node(state, c) for c in part_ids)
    after = cue_enclosure(ring_cells, state.palette_size, child_desclens) + overhead
    return before - after


# --------------------------------------------------------------------------- #
# continuation structural pass (Stage B2; object-schema §5.1.2 line /
# good-continuation, §5.1.3). Binds DISJOINT, collinear, evenly-spaced same-colour
# fragments into ONE FLAT line object across the gaps that Stage A connectivity
# leaves separate (so diagonals + dotted lines group without changing atom
# connectivity). Deterministic, integer-exact, pure MDL.
# --------------------------------------------------------------------------- #

def _scaled_centroid(state: _State, node_id: int) -> Tuple[int, int]:
    """The fragment centroid scaled by its cell count: ``(sum_row, sum_col)``.
    INTEGER-EXACT (no float, DP-10 / GEOM-1 avoidance). Collinearity / even spacing
    are tested on these scaled points; this is exact whenever the grouped fragments
    share a cell count (uniform scale preserves collinearity + equal deltas)."""
    cells = _node_cells(state, node_id)
    sum_row = sum(r for (r, _) in cells)
    sum_col = sum(c for (_, c) in cells)
    return (sum_row, sum_col)


def _run_cells(state: _State, run: Tuple[int, ...]) -> FrozenSet[Cell]:
    """The union of all cells of the fragments in ``run`` (its full footprint)."""
    cells: FrozenSet[Cell] = frozenset()
    for f in run:
        cells = cells | _node_cells(state, f)
    return cells


def _transpose_invariant_key(cells: FrozenSet[Cell]) -> Tuple[Tuple[int, int], ...]:
    """A canonical cell-set signature that is INVARIANT to a row<->col swap: the
    lexicographically smaller of the sorted cell tuple and the sorted TRANSPOSED
    cell tuple (DP-10, integer-exact, no float / RNG). Two runs that are exact
    transpose-pairs (a horizontal arm and the matching vertical arm of an
    L-corner) map to the SAME key -- this is what makes the structural-pass
    tie-break choose the same geometric content in a grid G and its transpose G^T,
    and lets a genuinely symmetric (unbreakable) tie be detected and declined."""
    forward = tuple(sorted(cells))
    flipped = tuple(sorted((c, r) for (r, c) in cells))
    return min(forward, flipped)


def _pick_best_run(
    state: _State,
    scored: List[Tuple[float, Tuple[int, ...]]],
) -> Optional[Tuple[int, ...]]:
    """Pick the run to fold this round from ``scored`` = ``[(gain, run), ...]`` of
    POSITIVE-gain candidates, transpose-invariantly (HIGH-1 fix). Returns the chosen
    run, or ``None`` if nothing should be folded (no candidates, or only a genuinely
    symmetric unbreakable tie remains).

    Rule:
      (a) NON-symmetric ties -> a transpose-INVARIANT key orders the runs: larger
          gain wins, ties broken by the SMALLER :func:`_transpose_invariant_key`.
          Because that key is identical for a run in G and its image in G^T, the
          same geometric content is chosen in both, so parse(G^T) == transpose(
          parse(G)).
      (b) Genuinely symmetric tie -> two runs with EQUAL gain and EQUAL
          transpose-invariant key that SHARE a fragment are an exact transpose-pair
          (e.g. the two arms of an L-corner); no single pick is transpose-invariant,
          so we DECLINE every run entangled in such a conflict this round (declining
          is itself invariant, and an ambiguous L has no MDL-preferred orientation).
          Remaining unconflicted runs are still eligible.

    Determinism: keyed only on (gain, invariant-key); no min-cell (which is NOT a
    strict total order across runs -- two runs can share a min-cell), no RNG / hash.
    """
    if not scored:
        return None
    keyed = [
        (gain, _transpose_invariant_key(_run_cells(state, run)), run)
        for (gain, run) in scored
    ]
    # identify runs entangled in a symmetric unbreakable tie: equal (gain, inv-key)
    # AND sharing >=1 fragment with another such run.
    blocked: set = set()  # indices into keyed
    for i in range(len(keyed)):
        for j in range(i + 1, len(keyed)):
            same_gain = keyed[i][0] == keyed[j][0]
            same_key = keyed[i][1] == keyed[j][1]
            shares = set(keyed[i][2]) & set(keyed[j][2])
            if same_gain and same_key and shares:
                blocked.add(i)
                blocked.add(j)
    eligible = [keyed[k] for k in range(len(keyed)) if k not in blocked]
    if not eligible:
        return None  # only a symmetric unbreakable tie remains -> decline (b)
    # (a) larger gain wins; tie -> smaller transpose-invariant key. Use ``min`` on
    # ``(-gain, inv_key)`` so the SMALLER key is selected without negating the
    # variable-length key tuple (which would mis-order under Python's prefix rule).
    best = min(eligible, key=lambda entry: (-entry[0], entry[1]))
    return best[2]


def _line_fragment_runs(state: _State, ids: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """All MAXIMAL arithmetic progressions (>=3) of fragments -- dotted / evenly-
    spaced collinear lines -- found integer-exactly. ``ids`` must be SAME colour +
    SAME cell count (so the scaled centroids share a uniform scale; collinearity +
    even spacing are then exact on the scaled points).

    A progression = fragments whose scaled centroids step by a single CONSTANT
    vector ``delta`` (this captures BOTH collinearity = constant direction AND even
    spacing = constant magnitude, orientation-agnostic -> diagonals included). For
    each candidate ``delta`` (the gap between an ordered pair) a progression is
    grown ONLY from its START (a fragment with no predecessor at ``centroid -
    delta``), so each maximal run is emitted exactly once. Returned sorted by start
    centroid then delta (deterministic, no float / RNG).

    Note: a fragment may appear in runs of DIFFERENT directions (e.g. a diamond
    corner shared by two edges); the caller's greedy pass picks one run at a time
    by MDL gain, so overlaps resolve deterministically."""
    if len(ids) < 3:
        return []
    point_of = {nid: _scaled_centroid(state, nid) for nid in ids}
    # a centroid may be shared by >1 fragment only if two fragments coincide, which
    # cannot happen for disjoint leaves -> the map is a bijection on live points.
    by_point: Dict[Tuple[int, int], int] = {}
    for nid in sorted(ids, key=lambda n: (point_of[n], n)):
        by_point.setdefault(point_of[nid], nid)
    ordered = sorted(ids, key=lambda nid: (point_of[nid], nid))

    seen_runs: set = set()
    runs: List[Tuple[int, ...]] = []
    for a_index in range(len(ordered)):
        a = ordered[a_index]
        pa = point_of[a]
        for b_index in range(a_index + 1, len(ordered)):
            b = ordered[b_index]
            pb = point_of[b]
            delta = (pb[0] - pa[0], pb[1] - pa[1])
            if delta == (0, 0):
                continue
            # only START a progression where ``a`` has no predecessor at pa-delta.
            prev = (pa[0] - delta[0], pa[1] - delta[1])
            if prev in by_point:
                continue
            # grow the progression from ``a`` stepping by ``delta``.
            chain = [a]
            cur = pa
            while True:
                nxt = (cur[0] + delta[0], cur[1] + delta[1])
                if nxt not in by_point:
                    break
                chain.append(by_point[nxt])
                cur = nxt
            if len(chain) >= 3:
                run = tuple(chain)
                if run not in seen_runs:
                    seen_runs.add(run)
                    runs.append(run)
    runs.sort(key=lambda run: (point_of[run[0]], point_of[run[1]]))
    return runs


def _line_desclen(state: _State, fragment_ids: Tuple[int, ...]) -> float:
    """``DescLen`` of a line object = the continuation code over its fragments
    (object-schema §5.1.2 line row). ``gap`` = the Chebyshev magnitude of the
    constant scaled-centroid step; ``fragment_bits`` = the connectivity cost of one
    representative fragment (the repeated tile, paid ONCE)."""
    ordered = sorted(
        fragment_ids, key=lambda nid: (_scaled_centroid(state, nid), nid)
    )
    p0 = _scaled_centroid(state, ordered[0])
    p1 = _scaled_centroid(state, ordered[1])
    gap = max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))
    fragment_bits = cue_connectivity(
        _node_cells(state, ordered[0]), state.palette_size
    )
    return cue_continuation(
        fragment_count=len(ordered),
        gap=gap,
        fragment_bits=fragment_bits,
        board_height=state.board_height,
        board_width=state.board_width,
        palette_size=state.palette_size,
    )


def _line_gain(state: _State, fragment_ids: Tuple[int, ...]) -> float:
    """The MDL bits SAVED by binding ``fragment_ids`` into ONE line object vs.
    leaving each a separate top-level object (object-schema §5.1.3). Positive = the
    grouping lowers total DescLen. struct_overhead is the object-COUNT penalty:

      before = Sigma_{f} [DescLen(f) + overhead]   (each fragment its own object)
      after  = line_desclen(fragments) + overhead  (ONE flat line object)

    The N->1 overhead collapse plus paying for ONE fragment template (not N) is
    what makes a real dotted line win; a non-collinear scatter never reaches this
    function (it forms no constant-delta run)."""
    overhead = state.struct_overhead_unit
    before = sum(_desclen_node(state, f) + overhead for f in fragment_ids)
    after = _line_desclen(state, fragment_ids) + overhead
    return before - after


def _make_line(state: _State, fragment_ids: Tuple[int, ...]) -> int:
    """Fold collinear fragments into ONE FLAT "line" object (object-schema §5.1.2
    good-continuation). Allocates a fresh parent whose children are ALL the
    fragments (all "own", no parts -- a dotted line is one thing, NOT a nested
    parent+parts). The leaf partition is UNTOUCHED (nesting shares cells; the line
    object's cells = the union of its fragments). Returns the new node id."""
    ordered = tuple(
        sorted(fragment_ids, key=lambda nid: (_scaled_centroid(state, nid), nid))
    )
    line_id = state.alloc_id()
    state.id_to_children[line_id] = ordered
    state.id_to_own_count[line_id] = len(ordered)  # all own -> _build flat (no parts)
    state.id_to_kind[line_id] = "line"
    for f in ordered:
        state.id_to_parent[f] = line_id
    return line_id


def _continuation_pass(
    state: _State,
    candidate_ids: Tuple[int, ...],
    field_color: Optional[int],
) -> Tuple[int, ...]:
    """Greedy deterministic continuation grouping over ``candidate_ids`` (the
    surviving top-level roots after the enclosure pass). Returns the new roots.

    Only single-leaf candidates participate (a dotted line binds atomic fragments,
    not already-composed objects -- composed objects are left to common-fate).
    FIELD-coloured leaves are EXCLUDED (FIX 2): the background plane must not be
    bound into a line, or ``is_field`` would point at a leaf buried below top level
    (the enclosure pass excludes field leaves the same way). Fragments are grouped
    by (colour, cell-count) so the scaled centroids share a uniform scale; within
    each group, maximal constant-delta runs (>=3) are the line candidates. The run
    to fold is chosen by :func:`_pick_best_run` (transpose-invariant key; a
    genuinely symmetric L-corner tie is declined), folded each round until none
    remain.

    SEAM (later): same-colour, uniform-size fragments only. Multi-colour dotted
    lines and mixed-size runs are deferred (object-schema §5 cue table: those lean
    on lattice / common-fate)."""
    candidate_ids = tuple(candidate_ids)
    while True:
        leaves = [c for c in candidate_ids if _is_leaf(state, c)]
        # group by (single colour, cell count); only single-colour, NON-field leaves.
        groups: Dict[Tuple[int, int], List[int]] = {}
        for lid in leaves:
            colors = state.id_to_color[lid]
            if len(colors) != 1:
                continue
            if _is_field_colored(state, lid, field_color):
                continue  # FIX 2: never bind the background/field plane into a line
            key = (next(iter(colors)), len(state.id_to_cells[lid]))
            groups.setdefault(key, []).append(lid)

        scored: List[Tuple[float, Tuple[int, ...]]] = []
        for _, members in sorted(groups.items()):
            for run in _line_fragment_runs(state, tuple(members)):
                gain = _line_gain(state, run)
                if gain > 0.0:  # pure-MDL reject below
                    scored.append((gain, run))
        best_run = _pick_best_run(state, scored)
        if best_run is None:
            break
        line_id = _make_line(state, best_run)
        consumed = set(best_run)
        candidate_ids = tuple(
            c for c in candidate_ids if c not in consumed
        ) + (line_id,)
    return candidate_ids


# --------------------------------------------------------------------------- #
# lattice / regularity structural pass (Stage B3; object-schema §5.1.2 lattice,
# §5.1.3 / FR-U-05). Binds N IDENTICAL-SHAPE, disjoint, regularly-spaced leaves
# into ONE PARENT + N tile PARTS (a structured repeat -- the count/regularity is
# preserved by surfacing the tiles, object-schema §2 D4 / §5 rubik precedent).
# Deterministic, integer-exact, pure MDL.
# --------------------------------------------------------------------------- #

def _shape_key(state: _State, node_id: int) -> Tuple[Tuple[int, int], ...]:
    """A translation-invariant shape signature: the cell offsets relative to the
    leaf's min-cell, sorted (integer-exact, DP-10). Two leaves are SHAPE-IDENTICAL
    iff their shape keys are equal (same footprint up to translation)."""
    cells = _node_cells(state, node_id)
    r0 = min(r for (r, _) in cells)
    c0 = min(c for (_, c) in cells)
    return tuple(sorted((r - r0, c - c0) for (r, c) in cells))


def _lattice_runs(state: _State, ids: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """All MAXIMAL 1-D arithmetic progressions (>=3) of tiles by their MIN-CELL --
    a regular row / column / diagonal of identical tiles. ``ids`` must be SAME
    colour + SAME shape (so the min-cells step by a constant tile period). Mirrors
    :func:`_line_fragment_runs` but on min-cells (the tile anchors) instead of
    scaled centroids -- exact, integer, each maximal run emitted once.

    SEAM (Stage B3 -> later): 1-D lattices only (a single period vector). A full 2-D
    grid (e.g. a rubik face: two independent periods) is NOT folded here -- it would
    decompose into row/column 1-D runs; the 2-D parent is deferred (object-schema §5
    rubik) along with multi-colour lattices."""
    if len(ids) < 3:
        return []
    point_of = {nid: min(_node_cells(state, nid)) for nid in ids}
    by_point: Dict[Tuple[int, int], int] = {}
    for nid in sorted(ids, key=lambda n: (point_of[n], n)):
        by_point.setdefault(point_of[nid], nid)
    ordered = sorted(ids, key=lambda nid: (point_of[nid], nid))

    seen_runs: set = set()
    runs: List[Tuple[int, ...]] = []
    for a_index in range(len(ordered)):
        a = ordered[a_index]
        pa = point_of[a]
        for b_index in range(a_index + 1, len(ordered)):
            pb = point_of[ordered[b_index]]
            delta = (pb[0] - pa[0], pb[1] - pa[1])
            if delta == (0, 0):
                continue
            prev = (pa[0] - delta[0], pa[1] - delta[1])
            if prev in by_point:
                continue  # only start a progression at its head
            chain = [a]
            cur = pa
            while True:
                nxt = (cur[0] + delta[0], cur[1] + delta[1])
                if nxt not in by_point:
                    break
                chain.append(by_point[nxt])
                cur = nxt
            if len(chain) >= 3:
                run = tuple(chain)
                if run not in seen_runs:
                    seen_runs.add(run)
                    runs.append(run)
    runs.sort(key=lambda run: (point_of[run[0]], point_of[run[1]]))
    return runs


def _lattice_desclen(state: _State, tile_ids: Tuple[int, ...]) -> float:
    """``DescLen`` of a lattice parent = the lattice code over its tiles
    (object-schema §5.1.2 lattice row). ``period`` = the Chebyshev magnitude of the
    constant min-cell step; ``tile_bits`` = one tile's connectivity template (paid
    ONCE); ``residual_bits`` = 0 (the tiles are shape-identical, no per-tile diff)."""
    ordered = sorted(tile_ids, key=lambda nid: (min(_node_cells(state, nid)), nid))
    p0 = min(_node_cells(state, ordered[0]))
    p1 = min(_node_cells(state, ordered[1]))
    period = max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))
    tile_bits = cue_connectivity(_node_cells(state, ordered[0]), state.palette_size)
    return cue_lattice(
        tile_count=len(ordered),
        period=period,
        tile_bits=tile_bits,
        residual_bits=0.0,
        board_height=state.board_height,
        board_width=state.board_width,
        palette_size=state.palette_size,
    )


def _lattice_gain(state: _State, tile_ids: Tuple[int, ...]) -> float:
    """The MDL bits SAVED by folding ``tile_ids`` into ONE lattice parent + tile
    parts vs. leaving each tile a separate top-level object (object-schema §5.1.3).
    Positive = the grouping lowers total DescLen. struct_overhead is the object-
    COUNT penalty:

      before = Sigma_{t} [DescLen(t) + overhead]   (each tile its own object)
      after  = lattice_desclen(tiles) + overhead   (ONE parent; the tiles are parts
                                                     paid for INSIDE the lattice code)

    The N->1 overhead collapse plus paying ONE tile template (not N) is what makes a
    real lattice win; an irregularly-spaced set never reaches this function (it
    forms no constant-period progression)."""
    overhead = state.struct_overhead_unit
    before = sum(_desclen_node(state, t) + overhead for t in tile_ids)
    after = _lattice_desclen(state, tile_ids) + overhead
    return before - after


def _make_lattice(state: _State, tile_ids: Tuple[int, ...]) -> int:
    """Fold identical regularly-spaced tiles into a PARENT + tile PARTS node
    (kind="lattice", object-schema §5.1.2 / §2 D4). Reuses :func:`_wrap_children`
    with NO own footprint (``own_ids = ()``) so every tile is a surfaced PART (the
    count / regularity is preserved, unlike the flat line). The leaf partition is
    UNTOUCHED. Returns the new parent id."""
    ordered = tuple(
        sorted(tile_ids, key=lambda nid: (min(_node_cells(state, nid)), nid))
    )
    parent_id = _wrap_children(state, (), ordered)
    state.id_to_kind[parent_id] = "lattice"
    return parent_id


def _lattice_pass(
    state: _State,
    candidate_ids: Tuple[int, ...],
    field_color: Optional[int],
) -> Tuple[int, ...]:
    """Greedy deterministic lattice grouping over ``candidate_ids`` (the surviving
    top-level roots, BEFORE enclosure/continuation). Returns the new roots.

    Only MULTI-CELL single-leaf candidates participate: a lattice repeats a SHAPE
    (>=2 cells), so single-pixel dotted runs fall through to the continuation cue
    (the documented precedence -- lattice = repeated tiles, continuation = loose
    collinear points). FIELD-coloured leaves are EXCLUDED (FIX 2): the background
    plane must not be bound into a lattice, or ``is_field`` would point at a buried
    leaf (the enclosure pass excludes field leaves the same way). Tiles are grouped
    by (colour, shape-key); within each group, maximal constant-period 1-D
    progressions (>=3) are the lattice candidates. The run to fold is chosen by
    :func:`_pick_best_run` (transpose-invariant key; a genuinely symmetric L tie is
    declined), folded each round until none remain.

    SEAM (later): same-colour, identical-shape, 1-D only. Multi-colour lattices and
    full 2-D grids (rubik) are deferred (object-schema §5)."""
    candidate_ids = tuple(candidate_ids)
    while True:
        leaves = [
            c for c in candidate_ids
            if _is_leaf(state, c) and len(state.id_to_cells[c]) >= 2
        ]
        # group by (single colour, shape-key); single-colour, NON-field leaves only.
        groups: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], List[int]] = {}
        for lid in leaves:
            colors = state.id_to_color[lid]
            if len(colors) != 1:
                continue
            if _is_field_colored(state, lid, field_color):
                continue  # FIX 2: never bind the background/field plane into a lattice
            key = (next(iter(colors)), _shape_key(state, lid))
            groups.setdefault(key, []).append(lid)

        scored: List[Tuple[float, Tuple[int, ...]]] = []
        for _, members in sorted(groups.items()):
            for run in _lattice_runs(state, tuple(members)):
                gain = _lattice_gain(state, run)
                if gain > 0.0:  # pure-MDL reject below
                    scored.append((gain, run))
        best_run = _pick_best_run(state, scored)
        if best_run is None:
            break
        lattice_id = _make_lattice(state, best_run)
        consumed = set(best_run)
        candidate_ids = tuple(
            c for c in candidate_ids if c not in consumed
        ) + (lattice_id,)
    return candidate_ids


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def divide_frame(grid, prev=None) -> Parse:
    """Parse a single frame into a visible-leaf partition (DivideFrame / CMP-23).

    ``grid`` is a numpy 2D array or ``list[list[int]]`` of color indices
    (normalized with ``np.asarray``). ``prev`` is a SEAM (Stage C): the previous
    parse for cross-frame common-fate -- ACCEPTED and IGNORED in Stage A (the
    single-frame path takes no transition into account; C5/C6 stay un-checked).

    Pipeline (object-schema §5.1.3): atoms (color-CCL, FR-U-01) -> greedy MDL
    agglomeration (FR-U-02/03) -> structural passes (FR-U-05): enclosure (B1) ->
    lattice (B3) -> continuation (B2) -> is_field (D1) -> assert C1-C4
    (FR-U-08/10..13).
    """
    grid = np.asarray(grid)
    register_cues()  # idempotent; ensures the cues are resolvable via registry

    state = _initial_state(grid)
    _agglomerate(state)

    # Structural passes (deterministic, idempotent) over the surviving top-level
    # leaves, in a fixed PRECEDENCE order (object-schema §5.1.3). The order is by
    # SIGNAL SPECIFICITY -- the most specific (rarest, hardest to fire by chance)
    # pattern binds first, so a general cue cannot steal its constituents:
    #   B3 lattice    : N identical MULTI-cell tiles at a single constant period ->
    #                   parent + k tile PARTS (a structured repeat; the very specific
    #                   "same shape AND regular period" signal binds FIRST, so e.g.
    #                   ls20's three identical life squares group BEFORE the
    #                   surrounding wall's enclosure could absorb two of them);
    #   B1 enclosure  : rings + interiors -> parent + nested parts (a lattice parent
    #                   formed above can itself be nested as interior);
    #   B2 continuation: loose 1-px collinear/evenly-spaced fragments -> ONE flat
    #                   line (the least specific; runs last on what is left).
    # Lattice is restricted to MULTI-cell tiles, so a 1-px dotted line stays a (flat)
    # continuation object while a repeat of a SHAPE becomes a lattice parent. Each
    # pass leaves the leaf partition UNTOUCHED (nesting shares cells), so the is_field
    # address (identified before the passes) survives every wrap.
    field_ids = _field_ids(grid, state)
    field_color = _background_color(grid)
    _lattice_pass(state, tuple(_top_level_ids(state)), field_color)
    _enclosure_pass(state, tuple(_top_level_ids(state)), field_color)
    _continuation_pass(state, tuple(_top_level_ids(state)), field_color)

    # exit gate: C1-C4 (single frame). C5/C6 only when a prev frame is supplied.
    # Stage-A simplification of FR-U-08: a violation RAISES (a correct parse never
    # trips it). The spec's prescribed disposition -- re-parse on C1-C4, record on
    # C5-C6 -- is a Stage B/C seam.
    _assert_c1(state)
    _assert_c2(state)
    _assert_c3(state)
    _assert_c4(state, field_ids)
    if prev is not None:
        # SEAM (Stage C): never taken in Stage A's single-frame contract.
        _assert_c5(prev, state)
        _assert_c6(prev, state)

    objects = _build_objects(state)
    return Parse(
        objects=objects,
        label_grid=state.label_grid,
        id_to_cells=dict(state.id_to_cells),
        field_ids=field_ids,
    )


def _build_object(state: _State, node_id: int) -> GameObject:
    """Materialize one laminar-tree node as a :class:`GameObject` (recursive).

    ``cells`` = the node's full cell union (object-schema §2 (a): a parent CONTAINS
    its descendants' cells). For a parent, ``parts`` = the interior children
    (children[1:] -- children[0] is the ring leaf, folded into the parent's own
    footprint, not surfaced as a separate part). A leaf has no parts. ids are
    ``obj_%04d`` of the leaf/parent id (a lexical sort agrees with the numeric id;
    parent ids sort after leaf ids since they are allocated above them)."""
    cells = _node_cells(state, node_id)
    if _is_leaf(state, node_id):
        return GameObject(id=f"obj_{node_id:04d}", cells=cells)
    parts = [_build_object(state, c) for c in _part_ids(state, node_id)]
    return GameObject(id=f"obj_{node_id:04d}", cells=cells, parts=parts)


def _build_objects(state: _State) -> List[GameObject]:
    """Materialize the TOP-LEVEL objects as :class:`GameObject` with ``obj_%04d``
    ids (so a lexical id sort agrees with the numeric id that
    ``situation._frame_objects`` relies on). Stage A leaves are flat; Stage B1
    surfaces enclosure PARENTS at top level, their children under ``parts``
    (object-schema §2 D4 laminar tree)."""
    return [_build_object(state, node_id) for node_id in _top_level_ids(state)]
