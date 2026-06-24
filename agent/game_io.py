"""[agent] GameIO -- the Adapter / CA boundary (CMP-37, API-06).

The single I/O path between the ARC framework and our pure core:

- ``read_frame``    : framework per-turn response -> (objects, frames). Perceives the
                      SETTLED grid (the last frame) yet PRESERVES the full multiframe
                      sequence so downstream common-fate / tracking sees every frame
                      (SC-19 / TS-17 -- never truncate to the last grid).
- ``validate_action``: a chosen move (int or ``GameAction``) is translated/validated
                      against that turn's legal ``available_actions`` (SC-16 / TS-16).
                      ACTION6 carries click coordinates ``x, y in [0, 63]``; ACTION7
                      (Undo) is a CHARGED action (it still costs +1 scored move, no
                      refund -- surfaced as ``scored=True``).
- ``emit_turn``     : append one ``TurnRecord`` (trace-schema v1.0) as one JSONL line,
                      validating the required keys are present (the (2) emit the later
                      integration controller calls with its working-memory snapshot).

This module MAY import the framework ``arcengine`` (it is the Adapter boundary), unlike
``agent/core``. Pure-deterministic (DP-10): no randomness, no game literals (NFR-6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple, Union

from arcengine import GameAction

from agent.core.perceive import Obj, perceive

# ARC action-space constants (API-01). These are the framework's own enum ids, not
# game-specific literals: id 6 = the only Complex (click) action, id 7 = Undo.
_RESET_ID = GameAction.RESET.value          # 0
_CLICK_ID = GameAction.ACTION6.value         # 6 -- Complex: requires coordinates
_UNDO_ID = GameAction.ACTION7.value          # 7 -- charged (scored, no refund)
_COORD_MIN = 0
_COORD_MAX = 63

# Required top-level keys of a TurnRecord (trace-schema v1.0 §2). Canonical class names,
# no abbreviations (trace-schema naming rule). ``capture_ref`` decouples the trace from
# raw frame storage (DIP); the integration controller fills these from working memory.
TURN_RECORD_KEYS: Tuple[str, ...] = (
    "turn",
    "level",
    "capture_ref",
    "game_objects",
    "situation",
    "world_model",
    "goal_predicate",
    "game_plan",
    "game_move",
    "move_effect",
    "verbalize",
)


@dataclass(frozen=True)
class ActionResult:
    """The validated, framework-ready encoding of one chosen move (output of
    ``validate_action``). Guaranteed: ``id in available_actions``.

    ``action``  -- the resolved ``GameAction`` enum member, with ``action_data`` already
                   populated for a click (so it is ready to hand to the framework).
    ``id``      -- the raw GameAction id (matches ``available_actions`` / capture).
    ``coords``  -- ``(x, y)`` for a click (ACTION6), else ``None``.
    ``scored``  -- True iff this move costs a scored move this turn. id 1..7 each cost +1
                   (ACTION7/Undo included -- no refund); only RESET (id 0) is free here
                   (soft/full split is decided by the engine via ``full_reset``).
    """

    action: GameAction
    id: int
    coords: Union[Tuple[int, int], None] = None
    scored: bool = True


def _frames_of(frame_response: Any) -> List[Any]:
    """Pull the full frame sequence off a framework response (or a duck-typed stand-in).

    ``FrameData.frame`` is a list of 64x64 grids; ``FrameDataRaw.frame`` is a list of
    ndarrays. We accept any object exposing ``.frame`` (a sequence) so tests need not
    build a real ``FrameData``. The sequence is returned AS-IS (not truncated): the whole
    Move-between animation must reach perceive / tracking (SC-19).
    """
    frames = getattr(frame_response, "frame", None)
    if frames is None:
        raise ValueError("frame_response has no .frame sequence")
    return list(frames)


def available_action_ids(frame_response: Any) -> List[int]:
    """The legal action ids for this turn (``available_actions``), as a list of ints."""
    avail = getattr(frame_response, "available_actions", None)
    if avail is None:
        raise ValueError("frame_response has no .available_actions")
    return [int(a) for a in avail]


def read_frame(frame_response: Any) -> Tuple[List[Obj], List[Any]]:
    """Frame -> object boundary (API-06 input; SC-19 / TS-17).

    Perceive the SETTLED grid (the LAST frame in the sequence -- the state the turn
    settles into) and return it alongside the FULL frame list. We do NOT truncate to the
    last grid: the complete multiframe sequence is handed back so downstream multiframe
    common-fate (FR-C-1) and tracking (FR-C-2) get every frame.

    Returns ``(objects, frames)`` where ``objects`` is ``perceive`` of the settled grid
    and ``frames`` preserves every frame of the response (``len(frames) ==`` response
    frame count). An empty frame sequence yields ``([], [])`` (graceful, no crash).
    """
    frames = _frames_of(frame_response)
    if not frames:
        return [], []
    settled = frames[-1]
    objects = perceive(settled)
    return objects, frames


def validate_action(
    action: Union[int, GameAction],
    available_actions: Sequence[int],
    coords: Union[Tuple[int, int], None] = None,
) -> ActionResult:
    """Move -> action boundary (API-06 output; SC-16 / TS-16).

    Translate ``action`` (a raw id or a ``GameAction``) into a framework-ready
    ``ActionResult`` and GUARANTEE the emitted id is an element of ``available_actions``.

    - ACTION6 (click) requires ``coords = (x, y)`` with ``x, y in [0, 63]``; the
      coordinates are validated (via the engine's pydantic model) and carried on the
      returned ``GameAction.action_data``.
    - ACTION7 (Undo) is a CHARGED action: it still costs +1 scored move (no refund), so
      ``scored=True``. Every id 1..7 is scored; only RESET (id 0) is free here.

    Raises ``ValueError`` if the action is not in ``available_actions``, or if a click is
    missing / has out-of-range coordinates.
    """
    legal = [int(a) for a in available_actions]
    action_id = action.value if isinstance(action, GameAction) else int(action)

    if action_id not in legal:
        raise ValueError(
            f"action id {action_id} is not in available_actions {legal}"
        )

    ga = GameAction.from_id(action_id)

    out_coords: Union[Tuple[int, int], None] = None
    if action_id == _CLICK_ID:
        if coords is None:
            raise ValueError("ACTION6 (click) requires coords=(x, y)")
        x, y = int(coords[0]), int(coords[1])
        if not (_COORD_MIN <= x <= _COORD_MAX and _COORD_MIN <= y <= _COORD_MAX):
            raise ValueError(
                f"ACTION6 coords ({x}, {y}) out of range "
                f"[{_COORD_MIN}, {_COORD_MAX}]"
            )
        # set_data re-validates against the engine's ComplexAction model (ge=0, le=63)
        # and (re)binds the coordinates onto the shared enum member each call, so prior
        # coordinates never leak (determinism, DP-10).
        ga.set_data({"x": x, "y": y})
        out_coords = (x, y)

    # RHAE charging (API-01): ids 1..7 each cost +1 (ACTION7/Undo included -- no refund);
    # RESET (id 0) is not charged here (engine decides soft/full via full_reset).
    scored = action_id != _RESET_ID

    return ActionResult(action=ga, id=action_id, coords=out_coords, scored=scored)


def missing_turn_record_keys(record: dict) -> List[str]:
    """The required trace-schema keys (TURN_RECORD_KEYS) absent from ``record``."""
    return [k for k in TURN_RECORD_KEYS if k not in record]


def emit_turn(record: dict, path: Any) -> None:
    """(2) emit: append one ``TurnRecord`` to ``path`` as one JSONL line (trace-schema
    v1.0). Validate that every required key (``TURN_RECORD_KEYS``) is present first.

    One JSON value per line, UTF-8, no trailing comma -- append-as-you-go (the optional
    header is written once at file open by the caller, not here). The integration
    controller calls this with its per-turn working-memory snapshot.

    Raises ``ValueError`` if a required key is missing.
    """
    missing = missing_turn_record_keys(record)
    if missing:
        raise ValueError(f"TurnRecord is missing required keys: {missing}")
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
