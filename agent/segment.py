"""[Entity] Pure domain services: grid -> ObjectSet segmentation + node hashing.

This module is the **functional core** (NFR-008): `segment()` and `node_hash()`
are pure functions of their `(grid, hud_mask)` inputs with no I/O and no global
mutation. The only history-dependent helper is `detect_hud()` (FR-008a), which is
*explicitly* impure — it reads recent frame history to build a mask, then hands
that mask to the pure core as an argument so the purity claim stays intact.

Hard rule (task / spec 3.1): this Entity module MUST NOT import the framework
(`agents` / `arcengine`). It depends only on numpy + stdlib so it unit-tests with
numpy alone. Connected-component labeling is a PURE union-find/BFS implementation
(no scipy dependency) per the task constraints.

Why object-level hashing (ADR-003): raw-pixel novelty explodes the state space and
is sensitive to cosmetic HUD changes. We hash a canonical, order-independent
signature of the HUD-masked object configuration instead, so two frames with the
same masked layout collapse to one StateNode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

# The grid contract (C-5 / 20-api-and-data.md §0): 64x64, integer colors 0..15.
GRID_SIZE = 64
# Sentinel color used for masked-out (HUD) cells. It is OUTSIDE the legal 0..15
# range on purpose so a masked region can never be confused with a real color and
# never forms a spurious "object" that survives masking. Masked cells are dropped
# from segmentation entirely (see `segment`).
MASKED = -1


@dataclass(frozen=True, slots=True)
class GridObject:
    """A connected region of equal-color cells (Glossary 1.8; schema §4.2).

    Frozen + slots: it is an immutable value object (Immutability principle), so it
    is safe to share across the graph and to hash.
    """

    color: int  # 0..15
    cells: frozenset[tuple[int, int]]  # (row, col), each in 0..63
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    centroid: tuple[float, float]  # (row_mean, col_mean)
    size: int  # == len(cells)
    shape_hash: int  # translation-invariant hash of cell offsets


@dataclass(frozen=True, slots=True)
class ObjectSet:
    """The full segmentation of one frame (schema §4.2).

    `signature` is the canonical, order-independent hash used as the StateNode
    identity (FR-009). Two frames with the same masked object configuration map to
    the same signature regardless of segmentation/iteration order.
    """

    objects: tuple[GridObject, ...]  # order-normalized (see `_normalize`)
    signature: int  # canonical hash of the masked configuration


def _shape_hash(cells: Iterable[tuple[int, int]]) -> int:
    """Translation-invariant hash of a cell set (FR-007).

    We subtract the top-left of the bounding box from every cell so the hash
    depends on the SHAPE, not the position. The offsets are sorted so iteration
    order does not matter. Color is intentionally NOT mixed in here — `shape_hash`
    describes geometry only; color is carried separately on `GridObject`.
    """
    cell_list = list(cells)
    min_r = min(r for r, _ in cell_list)
    min_c = min(c for _, c in cell_list)
    offsets = tuple(sorted((r - min_r, c - min_c) for r, c in cell_list))
    return hash(offsets)


def _build_object(color: int, cells: list[tuple[int, int]]) -> GridObject:
    """Construct a GridObject from a color and its connected cell list (FR-007)."""
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    bbox = (min(rows), min(cols), max(rows), max(cols))
    size = len(cells)
    # Centroid is a float mean (schema §4.2). Pure arithmetic, no numpy needed.
    centroid = (sum(rows) / size, sum(cols) / size)
    return GridObject(
        color=color,
        cells=frozenset(cells),
        bbox=bbox,
        centroid=centroid,
        size=size,
        shape_hash=_shape_hash(cells),
    )


def _label_components(grid: np.ndarray) -> list[GridObject]:
    """4-connected connected-component labeling via iterative BFS (FR-006).

    PURE implementation (numpy + stdlib only) — no scipy, per task constraints.
    Each component is a maximal set of 4-adjacent cells sharing one color. Cells
    equal to `MASKED` are skipped, so HUD cells never form objects.

    Iterative (explicit stack) rather than recursive: a single-color 64x64 grid is
    4096 cells, which would blow Python's recursion limit. Why BFS over union-find:
    same O(N) cost here, but it lets us collect each component's cells directly.
    """
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[GridObject] = []

    for r0 in range(h):
        for c0 in range(w):
            if visited[r0, c0]:
                continue
            color = int(grid[r0, c0])
            if color == MASKED:
                visited[r0, c0] = True  # never a real object; mark and move on
                continue

            # Flood-fill this component with an explicit stack (imperative shell of
            # an otherwise pure function — the mutation is local, not global).
            component: list[tuple[int, int]] = []
            stack = [(r0, c0)]
            visited[r0, c0] = True
            while stack:
                r, c = stack.pop()
                component.append((r, c))
                # 4-connectivity: up/down/left/right only (FR-006).
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            objects.append(_build_object(color, component))

    return objects


def _normalize(objects: list[GridObject]) -> tuple[GridObject, ...]:
    """Order-normalize objects so the ObjectSet is canonical (FR-009).

    Sort key is fully determined by object content (color, position, size, shape)
    so identical configurations always yield the identical tuple order — required
    for a stable, order-independent signature and for NFR-005 determinism.
    """
    return tuple(
        sorted(
            objects,
            key=lambda o: (o.color, o.bbox, o.size, o.shape_hash),
        )
    )


def _signature(objects: tuple[GridObject, ...]) -> int:
    """Canonical, order-independent hash of the masked configuration (FR-009).

    Built from each object's *position-bearing* identity (color + bbox + shape),
    because where an object sits is part of the state, not just its shape. The
    input tuple is already normalized, so this is deterministic.
    """
    parts = tuple((o.color, o.bbox, o.size, o.shape_hash) for o in objects)
    return hash(parts)


def segment(grid: np.ndarray, hud_mask: np.ndarray | None = None) -> ObjectSet:
    """Segment ONE 64x64 grid into a canonical ObjectSet (FR-006..FR-009).

    PURE function (NFR-008): output depends only on `grid` and `hud_mask`. The
    caller is responsible for reducing a multi-grid frame to a single grid first
    (FR-020, see `latest_grid`) and for supplying the HUD mask (FR-008a).

    `hud_mask` is a (64,64) bool array where True marks HUD cells to ignore. We
    overwrite masked cells with `MASKED` on a COPY (never mutate the caller's grid)
    so HUD changes do not affect the object set or its signature (FR-008).
    """
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError(f"segment expects a 2D grid, got shape {arr.shape!r}")

    if hud_mask is not None:
        if hud_mask.shape != arr.shape:
            raise ValueError(
                f"hud_mask shape {hud_mask.shape!r} != grid shape {arr.shape!r}"
            )
        arr = arr.copy()  # copy-on-mask: keep `segment` pure w.r.t. its input
        arr[hud_mask] = MASKED

    objects = _normalize(_label_components(arr))
    return ObjectSet(objects=objects, signature=_signature(objects))


def node_hash(obj_set: ObjectSet) -> int:
    """The StateNode identity == the ObjectSet signature (schema §4.2, FR-009).

    Pure passthrough kept as a named function so the policy/tests refer to the
    domain concept ("node hash") rather than an attribute, and so a future phase
    can change the hashing rule in one place.
    """
    return obj_set.signature


def latest_grid(frame: list[list[list[int]]]) -> np.ndarray:
    """Reduce a multi-grid `frame` (int[][][]) to the latest grid (FR-020).

    Phase A reduction policy: `frame` may carry one OR MORE 64x64 grids; we use the
    last one (`frame[-1]`) as the single grid for segmentation. This reducer lives
    at the call site (not inside `segment`) so a later phase can revise the
    multi-grid handling without touching the pure core.
    """
    if not frame:
        raise ValueError("frame is empty; cannot reduce to a grid")
    return np.asarray(frame[-1], dtype=int)


def detect_hud(
    history: list[np.ndarray] | None,
    current: np.ndarray,
    *,
    change_threshold: float = 0.5,
    edge_band: int = 3,
) -> np.ndarray | None:
    """Heuristic, HISTORY-DEPENDENT HUD detection (FR-008a, L-3).

    This is the ONE impure helper (excluded from the NFR-008 purity claim by
    design). It returns a (64,64) bool mask where True = HUD, computed from how
    often each cell's value changed across recent frames. The mask is the UNION of:

      (a) **edge-band cells** — any cell that changed in >= 1 observed transition AND
          lies within `edge_band` cells of any border (top/bottom/left/right). This
          encodes the GENERAL UI heuristic "a volatile strip hugging the screen edge
          is HUD" (status bars, score/timer rows). It is intentionally NOT tied to a
          specific game, row, or column — only to the geometry of the screen border.
      (b) **interior high-frequency cells** — cells whose change-rate exceeds
          `change_threshold`, anywhere on the grid. This keeps catching volatile
          in-scene counters that flicker every frame even when not at an edge.

    Edge-band masking only requires a cell to have changed at least once, because a
    volatile edge strip is rarely fully synchronized; the interior rule keeps the
    stricter frequency bar so a moving play-area object is not mistaken for HUD.

    Returns None when there is not enough history to decide (fewer than 2 distinct
    frames) AND therefore nothing to mask. A None mask means "no masking" — `segment`
    then hashes the raw layout, the graceful fallback for L-3 (we accept some hash
    noise rather than guess). Keeping the mask as an explicit return value (not hidden
    state) is what lets `segment`/`node_hash` stay pure.
    """
    frames: list[np.ndarray] = list(history or [])
    frames.append(np.asarray(current))
    if len(frames) < 2:
        return None

    h, w = frames[0].shape
    changes = np.zeros((h, w), dtype=np.float64)
    transitions = 0
    for prev, nxt in zip(frames, frames[1:]):
        if prev.shape != (h, w) or nxt.shape != (h, w):
            # Shape drift across history — bail to no-masking rather than crash.
            return None
        changes += (prev != nxt).astype(np.float64)
        transitions += 1

    if transitions == 0:
        return None
    change_rate = changes / transitions

    # (b) interior high-frequency cells: volatile anywhere on the grid.
    interior = change_rate > change_threshold

    # (a) edge-band cells: changed at least once AND within `edge_band` of a border.
    changed_at_all = changes > 0
    border = np.zeros((h, w), dtype=bool)
    if edge_band > 0:
        b = min(edge_band, h, w)
        border[:b, :] = True
        border[h - b :, :] = True
        border[:, :b] = True
        border[:, w - b :] = True
    edge = changed_at_all & border

    mask = interior | edge
    if not mask.any():
        return None  # nothing volatile enough; treat as "no HUD detected"
    return mask
