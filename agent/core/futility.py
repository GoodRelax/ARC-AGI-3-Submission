"""Wasted-move (futility) prevention — gauge-masked board recurrence.

Realizes the canonical futility components (``docs/StrictDoc-specs/_assets/
gr-arc-3-components.json``):

    MoveEffect{invariant|no_progress|progress}   («enumeration»; world_model.MOVE_EFFECTS)
    DetectFutility   (observe side, FR-C-9): classify last move's MoveEffect, record it
    CheckFutility    (predict side, FR-C-9): prune a (state, move) known to be non-progress
    WorkingMemory    (store):                visited masked-board hashes + (mhash, move) -> effect
    move_budget(HUD) (domain v031):          the receding gauge masked out of the board identity

This is the TwelveForms Lv2 (Contrast) realization: record past boards, MASK
the budget-gauge region, and if the new (gauge-masked) board RECURS the move that
produced it was wasted (no_progress) — so avoid that (state, move) pair.

The "gauge-excluded board" is the board identity used here (the user's definition):
a colour whose cell-count series is MONOTONE NON-INCREASING with at least one strict
decrease over the recent window is a GAUGE colour (a receding HUD bar / budget line);
its cells are masked to a fixed sentinel before hashing, so a turn that only shortened
the gauge does not look like a fresh board.

UNION MASK (the core of the gauge exclusion): :func:`detect_gauges` returns the gauge
colour's cells in the CURRENT grid only, but the caller (``search_agent._update_futility``)
ACCUMULATES the UNION of those cells across the history window. A receding bar shrinks, so
the cells it vacates revert to background and would otherwise differ between two consecutive
boards — making a gauge-tick look like a fresh board (futility would never fire). Masking the
UNION (the bar's MAXIMAL footprint) instead leaves the masked board UNCHANGED across a shrink,
so a back-and-forth / gauge-shrinking turn correctly reads as ``invariant`` / ``no_progress``.
The gauge colour is fixed per game and the avatar oscillates in a DIFFERENT colour, so it never
enters the gauge-colour union. A background-guard (see :func:`detect_gauges`) excludes any
candidate whose footprint would exceed a board fraction (it is the shrinking BACKGROUND, not a
HUD bar) so the union can never grow to the whole board.

Determinism discipline (DP-10): NO RNG, NO builtin ``hash()``. The board identity is a
``blake2b`` digest (the same backend style as ``agent.core.situation``). Gauge detection
is a pure function of the count history (sorted, integer comparisons only).

DEATH GUARD (move_budget==0 is OVER, not futile): a move that drives the budget gauge to
zero is TERMINAL (Outcome.state == "over"), and futility must NEVER suppress it. In this
MVP the masked board hash DROPS the gauge colour entirely, so the budget is not even part
of the board identity — there is nothing here to make a near-death move "recur". Terminal
(OVER) is decided by the engine / ``agent.core.world_model`` (gauge value <= 0), NOT by this
module. See that module's ``_terminal_outcome``.

DEFERRED (honest MVP scope): DIGIT / BINARY gauge decoding — a numeric HUD counter that
recedes by re-drawing different glyphs is NOT a monotone single-colour cell-count series, so
we only mask the gauge COLOUR's cells (a line / edge / panel) and defer digit decoding. And
the ``progress`` MoveEffect (which needs a goal / milestone) is NOT produced here: this MVP
only learns ``invariant`` (frame unchanged) and ``no_progress`` (board recurred) — both of
which need NO goal — so it works on a goal-less baseline. A ``progress`` verdict is left to a
goal-aware DetectFutility (CMP-31) later.
"""

from __future__ import annotations

import hashlib
from typing import Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

import numpy as np

from agent.core.world_model import MOVE_EFFECTS, futile

# The sentinel a gauge cell is set to before hashing (a fixed, out-of-palette
# value so masking is unambiguous and never collides with a real colour 0..15).
_GAUGE_SENTINEL = -1


# --------------------------------------------------------------------------- #
# DetectGauge — find the receding HUD colours (move_budget / health bars).
# --------------------------------------------------------------------------- #

# A candidate gauge colour whose CURRENT footprint exceeds this fraction of the board is
# treated as the (shrinking) BACKGROUND, not a HUD bar, and is EXCLUDED — otherwise its
# accumulated union mask could swallow the whole board and futility would mask everything.
_GAUGE_MAX_BOARD_FRACTION = 0.25


def detect_gauges(
    color_count_history: List[Mapping[int, int]],
    grid: np.ndarray,
    k: int = 3,
) -> FrozenSet[Tuple[int, int]]:
    """The cells (``(row, col)``) of every GAUGE colour in the CURRENT ``grid``.

    NOTE on the UNION mask: this returns only the CURRENT grid's gauge-colour cells. The
    caller (``search_agent._update_futility``) ACCUMULATES the UNION of these results across
    the history window so the bar's MAXIMAL footprint is always masked (a shrink then leaves
    the masked board unchanged — see the module docstring). Per-call this is the live frame's
    cells; the union is the caller's job.

    ``color_count_history`` is a list (OLDEST..NEWEST) of ``{color: count}`` maps over
    recent turns. A colour is a GAUGE colour iff, over the last ``k`` TRANSITIONS of its
    count series, the series is MONOTONE NON-INCREASING (each step ``<=`` the previous) AND
    has at least one STRICT decrease (a receding line; a flat series is NOT a gauge, which
    ignores idle animation). Multiple gauge colours are supported (the union of their cells
    is returned).

    BACKGROUND-GUARD: a candidate gauge colour whose CURRENT footprint exceeds
    :data:`_GAUGE_MAX_BOARD_FRACTION` of the board (default 25%) is the shrinking BACKGROUND,
    not a HUD bar, and is EXCLUDED. Without this, the accumulated union for a background-like
    colour could cover the whole board and make every board identical (futility would mask
    everything). A real budget bar / health line is a small fraction of the board, so the
    guard never drops a true gauge.

    A colour absent from a frame contributes count 0 for that frame (it receded to nothing),
    so a bar that empties still reads as monotone non-increasing.

    Returns ``frozenset`` of ``(row, col)`` for ``grid[row, col] == gauge_colour`` over ALL
    gauge colours; the empty set when there is no gauge colour (a "gentle" game — masking
    nothing, so pure board-duplication detection still works).

    Deterministic (DP-10): integer comparisons over a sorted colour set; no RNG, no hash.
    """
    arr = np.asarray(grid)
    # Need at least k transitions -> k + 1 frames of history.
    if len(color_count_history) < k + 1:
        return frozenset()

    window = color_count_history[-(k + 1):]  # k + 1 frames == k transitions
    # Every colour that appears anywhere in the window is a candidate.
    candidates: Set[int] = set()
    for frame_counts in window:
        candidates.update(frame_counts.keys())

    gauge_colors: Set[int] = set()
    for color in sorted(candidates):  # sorted -> deterministic order
        series = [int(frame_counts.get(color, 0)) for frame_counts in window]
        if _is_receding(series):
            gauge_colors.add(color)

    if not gauge_colors:
        return frozenset()

    board_cells = int(arr.size)
    max_cells = board_cells * _GAUGE_MAX_BOARD_FRACTION
    cells: Set[Tuple[int, int]] = set()
    for color in sorted(gauge_colors):
        rows, cols = np.where(arr == color)
        # BACKGROUND-GUARD: skip a colour that fills > _GAUGE_MAX_BOARD_FRACTION of the
        # board this frame (it is the receding BACKGROUND, not a HUD bar).
        if rows.size > max_cells:
            continue
        for row, col in zip(rows.tolist(), cols.tolist()):
            cells.add((int(row), int(col)))
    return frozenset(cells)


def _is_receding(series: List[int]) -> bool:
    """True iff ``series`` is MONOTONE NON-INCREASING with >= 1 STRICT decrease.

    A flat or rising series is NOT receding (so a constant colour and an oscillating colour
    are both rejected). Pure integer comparison (DP-10)."""
    strict_decrease = False
    for prev, cur in zip(series, series[1:]):
        if cur > prev:
            return False  # rose somewhere -> not monotone non-increasing
        if cur < prev:
            strict_decrease = True
    return strict_decrease


# --------------------------------------------------------------------------- #
# Masked board identity (the "gauge-excluded board").
# --------------------------------------------------------------------------- #

def masked_board_hash(
    grid: np.ndarray, gauge_cells: FrozenSet[Tuple[int, int]]
) -> bytes:
    """The gauge-excluded board identity = ``blake2b`` digest of the grid with every
    ``gauge_cells`` cell set to :data:`_GAUGE_SENTINEL`.

    Copies the grid (never mutates the caller's), masks the gauge region to the fixed
    sentinel, then hashes ``arr.tobytes()``. Two boards that differ ONLY inside the gauge
    region produce the SAME digest (the gauge is excluded from identity); two boards that
    differ OUTSIDE it produce DIFFERENT digests.

    Deterministic (DP-10) and CROSS-PLATFORM stable: the grid is cast to a FIXED ``np.int64``
    dtype in C-order UNCONDITIONALLY (regardless of platform default int width or whether a
    gauge is present), so ``.tobytes()`` yields the same buffer everywhere — a uint8 indexed
    array, an int32 array and an int64 array of the same values all hash identically, and the
    signed dtype keeps the sentinel (-1) from wrapping to a real colour. ``blake2b`` (same
    backend style as ``agent.core.situation``), no RNG, no builtin ``hash()``.
    """
    # Fixed dtype + C-order UNCONDITIONALLY: makes the byte buffer (and thus the digest)
    # identical across platforms and independent of the input dtype / gauge presence.
    # ``copy=True`` so we NEVER alias / mutate the caller's grid (ascontiguousarray would
    # return the input unchanged when it is already C-order int64).
    arr = np.array(grid, dtype=np.int64, order="C", copy=True)
    if gauge_cells:
        rows = [r for (r, _c) in gauge_cells]
        cols = [c for (_r, c) in gauge_cells]
        arr[rows, cols] = _GAUGE_SENTINEL
    return hashlib.blake2b(arr.tobytes()).digest()


# --------------------------------------------------------------------------- #
# WorkingMemory (store) + DetectFutility / CheckFutility (use-cases).
# --------------------------------------------------------------------------- #

class WorkingMemory:
    """The futility store (canon WorkingMemory) + the DetectFutility / CheckFutility
    use-cases (FR-C-9).

    Holds:
      * ``visited``: ``set[bytes]`` — masked-board hashes seen so far.
      * ``futile``:  ``dict[(masked_hash, move_id), MoveEffect]`` — the learned effect of
        playing ``move_id`` from the board whose identity is ``masked_hash``. The effect label
        is one of :data:`agent.core.world_model.MOVE_EFFECTS` (``invariant`` / ``no_progress``
        / ``progress``).

    DetectFutility (:meth:`record`) writes the effect of the move that was JUST played;
    CheckFutility (:meth:`is_futile`) reads it back to prune a known non-progress (state, move).
    The store is per-game scratch (reset between games; kept across levels / RESET within a
    game), exactly as the canon WorkingMemory remark says — it is NOT a domain component.

    DEATH GUARD: a move reaching budget==0 is TERMINAL (OVER), not futile. This store never
    sees the budget (the masked hash drops the gauge colour), and OVER is decided by the
    engine / world_model — never suppress a terminal move via futility. See module docstring.
    """

    def __init__(self) -> None:
        self.visited: Set[bytes] = set()
        self.futile: Dict[Tuple[bytes, int], str] = {}

    # -- DetectFutility (observe side, FR-C-9) ----------------------------- #

    def record(
        self,
        prev_mhash: bytes,
        move_id: int,
        prev_visited_before: bool,
        cur_mhash: bytes,
    ) -> str:
        """DetectFutility: classify and store the effect of the move just played.

        ``prev_mhash`` is the board identity BEFORE the move, ``cur_mhash`` AFTER it, and
        ``prev_visited_before`` is whether ``cur_mhash`` had ALREADY been visited at the time
        the move was committed (so a board that recurs is recognised as a revisit). The
        classification mirrors the canon derivation:

          * ``invariant``   — ``cur_mhash == prev_mhash`` (the board did not change at all);
          * ``no_progress`` — the board changed but the new board was SEEN BEFORE (a revisit /
            cycle — wasted, since we have been here already);
          * ``progress``    — the board changed to a board never seen before.

        Stores ``futile[(prev_mhash, move_id)] = effect`` and returns the effect.

        NOTE (MVP): ``progress`` here only means "a NEW board" (a weak, goal-less proxy); a
        goal-aware milestone test (CMP-31) is deferred. CheckFutility only ever PRUNES the
        non-progress labels, so this weak ``progress`` is safe — it merely declines to prune.
        """
        if cur_mhash == prev_mhash:
            effect = "invariant"
        elif prev_visited_before:
            effect = "no_progress"
        else:
            effect = "progress"
        assert effect in MOVE_EFFECTS  # canon vocabulary (world_model.MOVE_EFFECTS)
        self.futile[(prev_mhash, move_id)] = effect
        return effect

    def note_visit(self, mhash: bytes) -> None:
        """Add a masked board hash to :attr:`visited` (idempotent)."""
        self.visited.add(mhash)

    def has_visited(self, mhash: bytes) -> bool:
        """Whether ``mhash`` is already in :attr:`visited` (the revisit oracle)."""
        return mhash in self.visited

    # -- CheckFutility (predict side, FR-C-9) ------------------------------ #

    def is_futile(self, cur_mhash: bytes, move_id: int) -> bool:
        """CheckFutility: True iff ``(cur_mhash, move_id)`` is KNOWN to be non-progress
        (its learned MoveEffect is ``invariant`` or ``no_progress`` — both wasted).

        Unknown pairs are NOT futile (False) — an untried move always gets its chance.
        Uses :func:`agent.core.world_model.futile` so the non-progress predicate stays a
        single source of truth (TS-12).

        HONESTY (no_progress prune vs canon "no_progress gets one try"): this predicate
        HARD-PRUNES BOTH ``invariant`` AND ``no_progress`` (``futile`` returns True for each).
        That is still consistent with the canon rule that ``no_progress`` gets one try, because
        DetectFutility (:meth:`record`) only labels a (state, move) ``no_progress`` AFTER that
        move was ALREADY played once and observed to revisit a board (a RETROACTIVE
        classification — see :meth:`record`). So by the time :meth:`is_futile` can prune it, the
        one try has already happened; the record ordering, not a special case here, enforces the
        "try once" rule."""
        effect = self.futile.get((cur_mhash, move_id))
        if effect is None:
            return False
        return futile(effect)

    def effect_of(self, mhash: bytes, move_id: int) -> Optional[str]:
        """The learned MoveEffect for ``(mhash, move_id)``, or ``None`` if untried."""
        return self.futile.get((mhash, move_id))
