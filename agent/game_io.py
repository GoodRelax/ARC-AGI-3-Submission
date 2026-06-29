"""GameIO (CMP-37) -- FrameData <-> grid + GameAction adapter (the IO port leaf).

A clean, dependency-light translation layer between the engine's ``FrameData``
(a stack of 2D indexed grids + an ``available_actions`` id list) and the agent
core (which reads a single 2D ``numpy`` board and emits a ``GameAction``). It owns
NO game logic and NO perception -- it only adapts shapes (Clean Architecture: the
adapter at the boundary). Realizes the GameIO output-interface obligations of
FR-C-12 (SC-16 / SC-19): the action a turn returns is a member of
``available_actions``, ACTION6 coordinates are clamped into ``0..63``, and the
Move-between animation frames are surfaced intact (not discarded).

Determinism (DP-10): every function is a pure shape translation -- no RNG, no
builtin ``hash()``. Names are spelled out (``action_id`` / ``height`` / ``width``,
never ``a`` / ``h`` / ``w``) per the project naming rule.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Union

import numpy as np

from arcengine import GameAction

# The engine board is a fixed 64x64 indexed grid (colours 0..15); the empty-frame
# guard returns this shape so a caller never has to special-case an empty board.
_BOARD_HEIGHT = 64
_BOARD_WIDTH = 64

# ComplexAction (ACTION6) clamps x / y to this inclusive range (pydantic
# ``ge=0, le=63`` on arcengine.ComplexAction); we clamp BEFORE set_data so a
# coordinate derived from a coarse cycle can never raise a ValidationError.
_COORD_MIN = 0
_COORD_MAX = 63


def frame_to_grid(latest_frame: Any) -> np.ndarray:
    """The settled board as a 2D ``int`` ``numpy`` array (the last animation layer).

    ``FrameData.frame`` is a STACK of 2D indexed grids; the settled board is the
    last layer (``frame[-1]``). TOTAL function: an empty frame (``frame == []`` /
    ``is_empty()``) yields a ``64x64`` zero board so perception always receives a
    well-formed grid rather than crashing on a cold-start / malformed turn.
    """
    frame_stack = getattr(latest_frame, "frame", None)
    if not frame_stack:
        return np.zeros((_BOARD_HEIGHT, _BOARD_WIDTH), dtype=int)
    return np.asarray(frame_stack[-1], dtype=int)


def grids_of(latest_frame: Any) -> List[np.ndarray]:
    """Every animation layer of the frame as a list of 2D ``int`` arrays.

    The Move-between multi-frame sequence (SC-19 / FR-C-12 latter half): the
    inter-move animation must be passed through, not collapsed to the settled
    board. An empty frame yields ``[]`` (no layers). The settled board is the
    last element (mirrors ``frame_to_grid``).
    """
    frame_stack = getattr(latest_frame, "frame", None)
    if not frame_stack:
        return []
    return [np.asarray(layer, dtype=int) for layer in frame_stack]


def _clamp(value: int) -> int:
    """Clamp an integer coordinate into the ComplexAction range ``0..63``."""
    return max(_COORD_MIN, min(_COORD_MAX, int(value)))


def move_to_action(
    action_id: int,
    *,
    x: Optional[int] = None,
    y: Optional[int] = None,
    game_id: str = "",
    reasoning: Optional[Union[dict, str]] = None,
) -> GameAction:
    """Build a ready-to-submit :class:`GameAction` from a raw action id.

    For a COMPLEX action (ACTION6 / click) the ``x`` / ``y`` coordinates are
    CLAMPED into ``0..63`` and written via ``set_data`` together with the
    ``game_id`` (pydantic enforces the range, so the clamp is load-bearing -- an
    out-of-range coordinate would otherwise raise). For a simple action the
    coordinates are ignored. ``reasoning`` (a dict or string) is attached on the
    enum instance when given (the framework reads it off the enum, not the data).

    ENGINE CLICK CONTRACT (verified, butterfly-critical): the framework submits a
    click as ``data = action.action_data.model_dump()`` (``agents/agent.py``
    do_action_request), so the wire dict reaching the game's ``step()`` is the
    ComplexAction dump -- it ALWAYS carries ``x``, ``y`` AND ``game_id``. Some
    offline sims read the coordinate strictly as ``self.action.data["x"]`` (e.g.
    tn36) rather than ``.get("x", 0)`` (e.g. vc33/r11l/s5i5); a click missing the
    ``x`` key raises ``KeyError: 'x'`` deep in the engine, OUTSIDE the agent's
    try/except. Because ``set_data`` populates a ComplexAction (whose model always
    has ``x``/``y``), the dump can never omit those keys -- so the agent's click is
    accepted by BOTH the strict and lenient sims. Do NOT bypass this by building a
    raw dict that could drop ``x``/``y``.
    """
    action = GameAction.from_id(int(action_id))
    if action.is_complex():
        clamped_x = _clamp(x if x is not None else 0)
        clamped_y = _clamp(y if y is not None else 0)
        action.set_data({"x": clamped_x, "y": clamped_y, "game_id": game_id})
    else:
        action.set_data({"game_id": game_id})
    if reasoning is not None:
        action.reasoning = reasoning
    return action


# --------------------------------------------------------------------------- #
# Button-name <-> GameAction adapter (the LLM move-proposer surface, API-04).
#
# The move proposer never sees engine action ids (ACTIONn). It chooses among
# named buttons (inputs.buttons[].name) or a screen click (inputs.click). The
# name<->id map lives HERE in the classical adapter so the model surface carries
# NO 'ACTIONn'. The name is GAME-AGNOSTIC and stable: a simple action with id N
# is named "aN" (NO game-specific directional naming -- direction lives only in
# the grounded geometric `effect`, per the frozen contract). A complex action
# (ACTION6) is the click branch, not a button. Pure, deterministic (DP-10).
# --------------------------------------------------------------------------- #

_BUTTON_PREFIX = "a"


def button_name_for_action(action_id: int) -> str:
    """The stable button NAME for a simple (non-complex) action id: ``"aN"`` for
    ACTION``N`` (e.g. ``a1`` for ACTION1). Game-agnostic -- no directional naming.
    Deterministic. (A complex/ACTION6 id is a click, not a button; callers gate
    on :func:`arcengine.GameAction.is_complex` before naming.)"""
    return "%s%d" % (_BUTTON_PREFIX, int(action_id))


def action_for_button_name(name: str) -> Optional[int]:
    """Inverse of :func:`button_name_for_action`: the engine action id for a
    button name, or ``None`` if it does not match the ``"aN"`` form (so an
    out-of-vocabulary name from the model is a clean decline, never a crash).
    Deterministic; no RNG/hash."""
    if not isinstance(name, str):
        return None
    text = name.strip().lower()
    if not text.startswith(_BUTTON_PREFIX):
        return None
    suffix = text[len(_BUTTON_PREFIX):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def validate_action(action_id: int, latest_frame: Any) -> bool:
    """Whether ``action_id`` is currently legal (a member of the frame's
    ``available_actions``). SC-16 / FR-C-12: the action a turn returns MUST be a
    member of the available set."""
    available = getattr(latest_frame, "available_actions", None) or []
    return int(action_id) in available


def reset_action() -> GameAction:
    """The RESET action (id 0) -- the safe cold-start / fallback move."""
    return GameAction.RESET


# --------------------------------------------------------------------------- #
# TRACE serialization (CMP-37 / FR-C-12) -- the paired decision trace.
#
# A TurnRecord (trace-schema.md v1.0) is the per-turn working-memory snapshot the
# Result Inspector consumes alongside the raw capture. These are PURE shape
# translators (DP-10: no RNG, no builtin ``hash()``, no IO): they build a plain
# dict from VALUES the orchestrator hands in. This module owns serialization (the
# CA adapter at the boundary) and must NOT import ``agent.core`` nor reach into a
# situation / world -- the orchestrator (OurSearchAgent) OWNS the data and passes
# it in. The orchestrator does the file IO (mirrors ``_emit_capture``).
#
# HONEST GAP seams (legible, not faked): a per-object role is filled from the
# projected situation (the controllable object surfaces as role == "controllable";
# objects the projector did not bucket carry null -- e.g. the field plane),
# goal_predicate is null (GAP-3 no real goal SELECTION), move_effect is
# null (GAP-4 / F2 -- no WorldModel.classify, no record buffering), game_plan.moves
# is [] (GAP-1 descriptor-only solver), and a profile ``value`` is the
# Characteristic MAGNITUDE (a 1-D degree, not a colour index -- the inspector's
# colour swatch stays dark by design).
# --------------------------------------------------------------------------- #


def _centroid_of(cells: Any) -> Optional[List[float]]:
    """The cells-derived ``[row, col]`` mean (the ``centroid`` pattern), or
    ``None`` for an empty footprint. Deterministic; a pure value translation."""
    pts = list(cells or [])
    if not pts:
        return None
    rows = sum(int(r) for (r, _c) in pts)
    cols = sum(int(c) for (_r, c) in pts)
    n = len(pts)
    return [round(rows / n, 2), round(cols / n, 2)]


def _profile_dict(profile: Any) -> dict:
    """Render a :class:`agent.core.model.Profile` to the trace ``profile`` map:
    ``word_id -> {"value": magnitude, "confidence": confidence}``.

    ``value`` is the Characteristic MAGNITUDE (HONEST GAP / F5): a 1-D graded
    degree, NOT a colour index -- so the inspector's colour swatch on a ``color``
    row reads ``transparent`` (dark) by design. Read defensively so a bare object
    without ``.characteristics`` yields ``{}`` rather than raising."""
    chars = getattr(profile, "characteristics", None) or []
    out: dict = {}
    for char in chars:
        out[char.word_id] = {
            "value": char.magnitude,
            "confidence": char.confidence,
        }
    return out


def _game_object_dict(obj: Any, role: Optional[str] = None) -> dict:
    """Serialize one perceived object to a trace ``GameObject`` (object-schema
    v003): ``cells`` as sorted ``[row, col]`` pairs (the REAL outline, not a
    bbox), an empty ``parts`` (GAP: recursive division not emitted), the
    magnitude-valued ``profile``, and the projected ``role`` (the role bucket this
    object's handle joined in the AbstractSituation, or ``None`` if it was not
    bucketed -- e.g. the field plane). The role carries the controllable
    assignment; there is no top-level controllable/avatar field by contract."""
    cells = sorted([int(r), int(c)] for (r, c) in obj.cells)
    return {
        "id": obj.id,
        "cells": cells,
        "parts": [],
        "profile": _profile_dict(getattr(obj, "profile", None)),
        "role": role,
    }


def _situation_signature(situation: Any) -> str:
    """A deterministic, faithful projection of ``situation.objects`` (the
    ``role -> frozenset[ObjectRef]`` map): ``role@[handle, ...]`` joined by ``;``,
    roles sorted, handles sorted. NOT role-laden prose -- a mechanical render of
    the real abstract state. Empty / shapeless situations project to ``""``."""
    objects = getattr(situation, "objects", None) or {}
    parts = []
    for role in sorted(objects):
        handles = sorted(getattr(ref, "handle", "") for ref in objects[role])
        parts.append("%s@%s" % (role, handles))
    return ";".join(parts)


def _object_verbalize(obj: Any, color_id: Optional[int], shape_base_word: str) -> str:
    """One HONEST per-object NL line built from REAL primitives: the dominant
    colour id (or ``?``), the shape base, and the cells-derived centroid. No
    invented semantics / no role claim -- purely mechanical (verbalization v001,
    GAP-aware). e.g. ``"color 4 box at (3.0, 3.0)"``."""
    color_word = "color %d" % color_id if color_id is not None else "color ?"
    centroid = _centroid_of(obj.cells)
    where = "(%s, %s)" % (centroid[0], centroid[1]) if centroid else "(?, ?)"
    return "%s %s at %s" % (color_word, shape_base_word, where)


def trace_header(
    *,
    schema_version: str = "1.0",
    game: str = "",
    capture_file: str = "",
    agent: str = "oursearchagent",
) -> dict:
    """The optional first trace line (NO ``turn`` key -- readers treat a
    turn-less line as the header). Pure value -> dict."""
    return {
        "schema_version": schema_version,
        "game": game,
        "capture_file": capture_file,
        "agent": agent,
    }


def turn_record(
    *,
    turn: int,
    level: Any,
    capture_file: str,
    capture_row: Optional[int],
    game_objects: List[Any],
    object_color_ids: Mapping[str, Optional[int]],
    object_shape_bases: Mapping[str, str],
    situation: Any,
    situation_hash: Any,
    rule_count: int,
    goal_predicate: Any,
    game_plan: Optional[dict],
    move_id: int,
    move_name: str,
    grid_height: int,
    grid_width: int,
    object_roles: Optional[Mapping[str, str]] = None,
) -> dict:
    """Build ONE TurnRecord dict (trace-schema.md v1.0) from VALUES the
    orchestrator owns. PURE (DP-10): no IO, no ``agent.core`` import, no reach into
    a live situation/world beyond the read-only accessors used here.

    The orchestrator supplies pre-computed per-object colour ids
    (``object_color_ids: obj.id -> dominant colour or None``) and shape bases
    (``object_shape_bases: obj.id -> shape_base``) because computing them needs the
    grid + ``attributes`` (which live in ``agent.core`` -- off-limits to this
    adapter). ``situation_hash`` is reused from the capture (already computed there)
    to avoid re-hashing.

    ``object_roles`` (GAP-2/GAP-3): a ``handle -> role`` map from the projected
    AbstractSituation (the orchestrator inverts ``situation.objects``). Each
    GameObject's ``role`` field is filled from it (``None`` when the object was not
    bucketed -- e.g. the field plane). The controllable object surfaces HERE, as
    its per-object ``role == "controllable"``; the trace-schema contract has NO
    top-level controllable/avatar field, so the per-object role is the single
    source of truth (it also appears in ``situation.signature``).

    HONEST GAP nulls (legible seams, NOT bugs):
      * ``goal_predicate`` = passed-through (null in MVP -- GAP-3, no goal SELECTION).
      * ``world_model.changed`` = false, summary cites GAP-4 (no induction).
      * ``game_plan.moves`` = [] (GAP-1: descriptor-only solver).
      * ``move_effect`` = null (F2 / GAP-4: no WorldModel.classify, no buffering --
        the effect of turn T is only observable at T+1, which MVP does not wire)."""
    roles = object_roles or {}
    objects_out = [_game_object_dict(o, roles.get(o.id)) for o in game_objects]
    verbalize_objects = [
        _object_verbalize(
            o,
            object_color_ids.get(o.id),
            object_shape_bases.get(o.id, "blob"),
        )
        for o in game_objects
    ]
    sig = _situation_signature(situation)
    return {
        "turn": int(turn),
        "level": level,
        "capture_ref": {"file": capture_file, "row": capture_row},
        "game_objects": objects_out,
        "situation": {"hash": str(situation_hash), "signature": sig},
        "world_model": {
            "rule_count": int(rule_count),
            "changed": False,
            "summary": "no rules learned (induction deferred -- GAP-4)",
        },
        "goal_predicate": goal_predicate,
        "game_plan": game_plan,
        "game_move": {"id": int(move_id), "name": move_name},
        "move_effect": None,
        "verbalize": {
            "world": "%d objects on a %dx%d field; dynamics not yet learned (GAP-4)"
            % (len(game_objects), grid_height, grid_width),
            "goal": "no goal selected (GAP-3)",
            "objects": verbalize_objects,
        },
    }
