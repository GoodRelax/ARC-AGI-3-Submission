"""OurSearchAgent -- the MVP-0 integration-spine orchestrator (CMP-19 play loop).

This is the single concrete :class:`agents.agent.Agent` our code registers. It
wires the v14 core end-to-end so the agent BOOTS and PLAYS LEGAL MOVES on the real
engine while emitting inspectable captures:

    frame -> GameIO.frame_to_grid -> divide_frame (perceive) -> detect_features
          -> StateAbstraction.project -> Goal+SelectSolver (capture only)
          -> OBSERVABLE BASELINE action -> WorldModel.predict (armed) -> capture

MVP-0 scope. The v14 solver is DESCRIPTOR-ONLY (it names a Solver family, it does
NOT produce a move -- GAP-1), so the action this agent commits is an OBSERVABLE
BASELINE (a deterministic legal-action cycle = curiosity), NOT a real solver
output. It proves the spine and feeds the inspector; it does not need to win. The
goal/solver leg runs only to enrich the capture (and to exercise the wiring); its
``plan.chosen`` is recorded but never drives the action.

Determinism (DP-10): the committed action is a pure function of
``(available_actions, action_counter)`` -- no RNG, no builtin ``hash()``. Every
internal stage is wrapped so ANY failure falls back to a safe legal action; an
Agent must never crash the competition loop.

DEFERRED seams (NOT mistaken for complete): the canonical choose-action sequence
also has a MODEL-WORLD refine leg and a DIAGNOSE leg (sequence sheet
``gr-arc-3-sequence-choose-action.md`` steps 2 / 6 -- GAP-4). MVP-0 arms a
prediction but never checks it back, induces no rules, and runs no diagnosis;
goal SELECTION (GAP-3) is a placeholder ("first active pattern"). These are
legible seams for a later session, not implemented behaviour.

Import note: do NOT import this module standalone before ``agents`` -- that
cycles (``agents/__init__.py`` re-imports this module). Importing ``agents``
first (the live path; ``main.py`` does this) loads ``agents.agent`` before line
30, so ``from agents.agent import Agent`` resolves. Tests import via ``agents``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

import numpy as np

from arcengine import GameAction, GameState

from agents.agent import Agent  # the SUBMODULE (loaded before __init__.py line 30)

from agent import game_io  # adapter (CA outer ring), not agent.core
from agent.asset_loader import AssetLoader
from agent.core import attributes, futility, transform
from agent.core.attributes import FeatureContext, detect_features
from agent.core.futility import WorkingMemory
from agent.core.goal import register_operators
from agent.core.llm import NullGenerator, consult
from agent.core import llm_prompt
from agent.core.model import Characteristic, Goal
from agent.core.observe import (
    DataflowLog,
    identify_goal,
    lexicon_growth,
    summarize_world,
    verbalize_goal,
    verbalize_objects,
    verbalize_world,
)
from agent.core.perceive import divide_frame
from agent.core.roles import classify as classify_roles
from agent.core.situation import (
    ObjectTracker,
    StateAbstraction,
    default_role_of,
)
from agent.core.solver import SelectSolver, SolverContext, SolverLibrary
from agent.core.world_model import (
    Observation,
    WorldModel,
    affordance_evidence,
    footprint_shift,
)

logger = logging.getLogger(__name__)

# Cold-start / terminal states that demand a RESET rather than a play move.
_COLD_START_STATES = (GameState.NOT_PLAYED, GameState.GAME_OVER)

# The role enum allowed by input-observation-schema.json objects[].role. A
# projected role label outside this set is surfaced as 'other' (the schema's
# catch-all) so the observation never carries an off-contract role. (The role
# vocabulary itself is the agent's; this only constrains the model-facing surface.)
_SCHEMA_ROLES = frozenset(
    {
        "controllable",
        "target",
        "field",
        "hazard",
        "reference",
        "carried-state",
        "interactor",
        "ref",
        "other",
    }
)


def _looks_like_int(token: str) -> bool:
    """True iff ``token`` is a base-10 integer literal (optional leading sign).
    Used by the LLM move parser to tell a numeric action id from a decline token
    WITHOUT a noisy ``int()`` ValueError (CM-4(b)). Deterministic; no RNG."""
    if not token:
        return False
    body = token[1:] if token[0] in "+-" else token
    return body.isdigit()


def _coerce_int(value: Any) -> Optional[int]:
    """Coerce ``value`` to an int, or ``None`` if it cannot be (a non-numeric
    string, ``None``, a bool, or other junk). Quiet: returns ``None`` instead of
    raising, so a malformed LLM coordinate becomes a clean decline rather than a
    logged ValueError. Deterministic (DP-10)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        token = value.strip()
        return int(token) if _looks_like_int(token.lower()) else None
    return None


class OurSearchAgent(Agent):
    """The MVP-0 play-loop orchestrator (see the module docstring)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Arm the dispatch registries. The autouse test fixture wipes the registry
        # between tests, so re-arm explicitly here (construction happens after the
        # wipe): feature detectors (attributes), operator evaluators (goal), and the
        # transform algebra xf_* (transform is NOT auto-run at import -- transform.py
        # note). perceive re-arms its own cues inside divide_frame each call.
        attributes.register_detectors()
        register_operators()
        transform.register_transforms()

        # Load the builtin asset registries. assert_complete is a hard FK gate; a
        # gap must NOT crash the agent (load-bearing backstop) -- log and continue.
        self.assets = AssetLoader().load()
        try:
            AssetLoader().assert_complete(self.assets)
        except Exception as exc:  # noqa: BLE001 - never crash the agent on asset gaps
            logger.warning("AssetLoader.assert_complete failed (continuing): %r", exc)

        # Construct once and carry across turns (tracker keeps frame-to-frame
        # identity; the projector / world model / solver selector are stateless reads).
        self.tracker = ObjectTracker()
        # Role classification (AnalogizeRoles, FR-R-1): a situation-aware classify
        # pre-pass replaces the old per-object role_of override. The motion-grounded
        # controllable pick (_identify_controllable, FR-168) stays the real decision;
        # it now sets the `controllable` Characteristic on the picked object's Profile
        # (see _mark_controllable) so the canonical recognized_by = has(controllable)
        # predicate path tests true and AnalogizeRoles assigns `controllable`
        # canonically (field via has(static)/has(is_field); target via inside(self,
        # box)). Any object no role predicate claims falls through to default_role_of
        # (so non-classified bucketing is identical to a default StateAbstraction --
        # no regression).
        self.abstraction = StateAbstraction(
            self.tracker, classify=self._classify_roles
        )
        self.world = WorldModel(abstraction=self.abstraction, tracking=self.tracker)
        self.library = SolverLibrary.from_assets(self.assets)
        self.select = SelectSolver(self.library, self.assets)
        self.generator = NullGenerator()

        # LOCAL-LLM move proposer (API-04, OFF by default). self._llm is the
        # ConstrainedGenerator the per-turn move proposer consults BEFORE the
        # classical baseline. Default (ARC_LLM unset) -> NullGenerator -> the
        # proposer always declines -> the existing baseline path runs unchanged
        # (byte-identical to today; this is how the regression suite stays green).
        # When ARC_LLM is truthy we instantiate the offline Qwen backend from
        # ARC_LLM_MODEL; a load/import failure degrades to NullGenerator (never
        # crashes the agent — load-bearing backstop). _llm_budget bounds the number
        # of consults per game (wall-clock guard); _llm_consults counts usage and
        # _last_move_source surfaces LLM-vs-baseline in capture/trace.
        self._llm = self._build_llm()
        self._llm_budget = self._read_llm_budget()
        self._llm_consults = 0
        # Wall-clock guards (the count budget above is a cheap proxy; these are the
        # true TIME bound). Purpose is solely to never exceed the 12 h notebook cap --
        # a timeout loses the WHOLE submission (worse than the classical floor), while
        # LLM latency is free for RHAE scoring. Three layers: per-game cumulative LLM
        # seconds, a per-game ceiling, and a notebook-GLOBAL deadline (see _llm_active).
        self._llm_time_spent = 0.0  # cumulative consult seconds THIS game (process)
        self._llm_per_game_s = self._read_llm_env_float("ARC_LLM_GAME_S", 480.0)
        self._llm_deadline_s = self._read_llm_env_float("ARC_LLM_DEADLINE_S", 32400.0)
        # Notebook-start wall epoch for the GLOBAL deadline. The kernel writes
        # ARC_LLM_DEADLINE_EPOCH=<time.time()> ONCE before the run so elapsed = now -
        # epoch is notebook-global across the per-game agent processes; unset -> this
        # process's start (a safe per-process fallback).
        self._llm_epoch = self._read_llm_epoch()
        self._last_move_source = "baseline"
        # (x, y) carried by the most recent ACCEPTED LLM complex-action proposal this
        # turn (else None -> the baseline coverage lattice is used). Reset each turn.
        self._llm_click_xy: Optional[Tuple[int, int]] = None
        # The one-line `note` the proposer wrote LAST turn (the contract's rolling
        # scratchpad). Stored here and echoed back verbatim as observation.last_note
        # next turn (pure passthrough, not interpreted). None on the first turn.
        self._last_note: Optional[str] = None
        # Lazily-loaded, cached prompt template + full output schema (parsed once;
        # pure file IO, torch-free). None until the first live consult; a load
        # failure leaves them None and the proposer declines (classical fallback).
        self._prompt_tpl: Optional[Any] = None
        self._out_schema: Optional[Dict[str, Any]] = None
        # Cache: word-category -> frozenset of word ids (lexicon-derived, for the
        # observation's color/shape/flag classification). Game-agnostic.
        self._word_category_cache: Dict[str, frozenset] = {}
        # Bounded rolling history of recently PLAYED moves for the observation's
        # `recent_actions` (anti-no-op: the prompt's "never repeat a control that did
        # nothing" rule needs it). Each entry: {action: "aN"|"click", changed: bool,
        # [x, y]}. The just-played move's `changed` is finalized at the TOP of the
        # next turn (when the result board is visible). Independent of the futility
        # toggle (uses a plain unmasked board signature). Deterministic.
        self._recent_actions: List[Dict[str, Any]] = []
        # The previous turn's full (unmasked) board signature + the move that
        # produced this turn's frame (action id, optional click x/y), used to set
        # `changed` for the recent_actions log. None before the first played move.
        self._prev_board_sig: Optional[bytes] = None
        self._pending_recent: Optional[Dict[str, Any]] = None
        # The cause of the most recent LLM move DECLINE (set in _dispose_llm_move),
        # surfaced in the ARC_LLM_DEBUG coordination log so a Kaggle run is
        # diagnosable: 'validation' / 'no legal mapping' / 'low-confidence
        # non-exploratory' / 'low-confidence no-op repeat'. None when the last
        # disposition adopted a move (or no consult ran).
        self._llm_decline_reason: Optional[str] = None

        # MVP-0 scaffolding: prev situation feeds the next project(); the other two
        # are ARMED for a future DIAGNOSE/refine consumer (GAP-4) and intentionally
        # unread in MVP-0. armed_prediction holds world.predict's (situation,
        # outcome) pair, not a bare prediction.
        self.prev: Optional[Any] = None
        self.armed_prediction: Optional[Any] = None
        self.last_action_id: Optional[int] = None

        # GAP-2/GAP-3 controllable identification: a per-handle ordered track history
        # (handle -> [(action_id, Observation)]) feeds world_model.affordance_evidence
        # so the controllable object (the avatar that TRANSLATES in response to the
        # agent's moves) is found by translate_support and marked "controllable".
        # _controllable_id is the current pick (None until enough motion is observed).
        self._track_history: Dict[str, List[Tuple[int, Observation]]] = {}
        self._controllable_id: Optional[str] = None
        # Per-handle Affordance evidence computed this turn (handle -> Affordance),
        # fed into detect_features so behaviour modifiers (movable / static) are
        # honest. Empty until enough motion is observed.
        self._affordance_evidence: Dict[str, Any] = {}
        # handles seen in the immediately-previous turn (used to drop a stale
        # controllable that has been absent for two consecutive turns -> re-identify).
        self._prev_turn_handles: frozenset = frozenset()

        # WASTED-MOVE (futility) prevention (FR-C-9 / CMP-31 / CMP-32 -- the TwelveForms
        # Lv2 Contrast). Record past boards with the receding budget-gauge region MASKED;
        # if a (gauge-masked) board RECURS, the move that produced it was wasted -> prune
        # that (state, move). DetectFutility records the effect of last turn's move
        # retroactively; CheckFutility prunes a known non-progress (state, move) when
        # choosing. Gauge detection reads a bounded {color: count} window.
        self._color_count_history: list = []
        self._working_memory = WorkingMemory()
        self._prev_mhash: Optional[bytes] = None
        self._last_action_for_futility: Optional[int] = None
        self._gauge_cells: frozenset = frozenset()
        # Per-action try counts (deterministic stuck-breaker: least-recently-tried, no RNG).
        self._action_try_counts: Dict[int, int] = {}
        # Toggle (default ON): no agent/settings.py exists, so use an env var. Unset or
        # "1"/"on"/"true"/"yes" -> ON; "0"/"off"/"false"/"no" -> OFF. When OFF the futility
        # behaviour is byte-identical to the pre-feature baseline (DP-20: solvable with the
        # guardrail off).
        self._futility_on = self._read_futility_toggle()

        # GREEDY NAVIGATE (GAP-1, the "actually solve" classical policy). Toggle
        # (default ON): when a controllable AND a target object are recognized, choose
        # the legal action whose GROUNDED (row, col) effect most reduces the Manhattan
        # distance controllable->target (deterministic; futility-respecting). OFF
        # reproduces the round-robin baseline byte-identically. Navigate falls back to
        # the baseline (returns None) whenever a target is absent or no grounded action
        # reduces distance, so it never blocks the exploration that grounds effects.
        self._navigate_on = self._read_navigate_toggle()

        # CLICK-NAVIGATE (GAP-1 for click-driven games). Gated under the SAME
        # ARC_NAVIGATE toggle as directional navigate (OFF -> byte-identical to today,
        # no click-navigate). When a click action (ACTION6) is legal AND a `target`-role
        # object exists, the classical side clicks a real TARGET CENTROID rather than
        # leaving the click coordinate to the 1.5B LLM (which emits garbage coords). It
        # is deterministic AND exploratory across turns: a per-game ordered list of
        # distinct target centroids (sorted area-asc, row, col -> rarest/smallest first)
        # is round-robined on no-progress (the masked board hash did not change), so it
        # tries each marked target until one wins, without any RNG.
        #
        # _click_candidate_idx is the current round-robin position; _click_last_mhash +
        # _click_last_centroid remember the state we clicked FROM last turn so the next
        # turn can detect no-progress and advance. All reset per game (one agent =
        # one game; see Agent.__init__ per-game construction).
        self._click_candidate_idx: int = 0
        self._click_last_mhash: Optional[bytes] = None
        self._click_last_centroid: Optional[Tuple[int, int]] = None
        # Target centroids found NON-PROGRESSING while stuck at the current board state
        # (a coordinate-aware futility set, since the action-id futility memory cannot
        # distinguish two clicks). Cleared whenever the masked board hash changes.
        self._click_futile_here: set = set()
        # (x=col, y=row) carried by the most recent ACCEPTED classical click-navigate
        # move this turn (else None -> the baseline coverage lattice is used). Parallel
        # to _llm_click_xy but for the classical leg. Reset each turn.
        self._classical_click_xy: Optional[Tuple[int, int]] = None
        # No-progress streak (consecutive unchanged masked boards) -- gates
        # navigate / click-navigate so they only fire when the round-robin is STUCK.
        self._no_progress_streak: int = 0
        self._prev_mhash: Optional[bytes] = None
        # No-LEVEL-progress streak: turns since levels_completed last increased. On a
        # CLICK game the masked board churns every click (so _no_progress_streak keeps
        # resetting) yet no level is won -- this catches that "busy but not winning"
        # stall so the LLM may take over once blind click-coverage has exhausted its
        # targets. STALL turns env-tunable (ARC_LLM_STALL, default 30).
        self._no_level_progress_streak: int = 0
        self._prev_levels: int = -1
        try:
            self._LLM_STALL_TURNS: int = int(os.environ.get("ARC_LLM_STALL", "30"))
        except (TypeError, ValueError):
            self._LLM_STALL_TURNS = 30
        # Turns of no board change before navigate/click-navigate may deviate from
        # the round-robin (conservative: brief stalls don't trigger wrong-target aiming).
        # Env-tunable (ARC_AIM_STUCK) for offline config sweeps; default 6.
        try:
            self._AIM_STUCK_TURNS: int = int(os.environ.get("ARC_AIM_STUCK", "6"))
        except (TypeError, ValueError):
            self._AIM_STUCK_TURNS = 6

        # GOAL-MARKER detector (the single unblock for distinct-cell-goal games).
        # Toggle (default ON, ★★): when ON, a non-field object whose dominant colour
        # is RARE on the board is stamped `marked`, firing the has(marked) arm of the
        # roles.tsv `target` recognizer so NAVIGATE + the LLM observation get a target.
        # OFF -> marked never set -> `target` only fires on the box-enclosure arm
        # (byte-identical to today). See _read_marked_toggle for the ★★ trade-off.
        self._marked_on = self._read_marked_toggle()

        # Capture writer: a line-buffered JSONL sink, opened once, gated on
        # ARC_INTROSPECT (a path). Fully inert (None) when the env var is unset.
        self._capture_writer = None
        introspect_path = os.getenv("ARC_INTROSPECT")
        if introspect_path:
            try:
                self._capture_writer = open(
                    introspect_path, "a", buffering=1, encoding="utf-8"
                )
            except Exception as exc:  # noqa: BLE001 - capture is optional, never fatal
                logger.warning("ARC_INTROSPECT open failed (no capture): %r", exc)
                self._capture_writer = None

        # Trace writer (CMP-37 / FR-C-12): the PAIRED decision trace -- one
        # TurnRecord per played turn, written in LOCKSTEP with the capture, gated
        # on a SEPARATE env var ARC_TRACE (additive / non-breaking: ARC_INTROSPECT
        # stays the capture path exactly as-is; both inert when unset). The capture
        # basename is derived from ARC_INTROSPECT so a TurnRecord's capture_ref can
        # join back to the raw frame.
        self._trace_writer = None
        self._trace_header_written = False
        # capture_row counts only the capture LINES actually written (RESET turns
        # write NO capture line); _emit_capture stamps _last_capture_row with the
        # row it just wrote so _emit_trace references the right line (NOT the
        # action_counter -- those diverge as soon as a RESET is skipped).
        self._capture_row = 0
        self._last_capture_row = None
        self._capture_basename = (
            os.path.basename(introspect_path) if introspect_path else ""
        )
        trace_path = os.getenv("ARC_TRACE")
        if trace_path:
            try:
                self._trace_writer = open(
                    trace_path, "a", buffering=1, encoding="utf-8"
                )
            except Exception as exc:  # noqa: BLE001 - trace is optional, never fatal
                logger.warning("ARC_TRACE open failed (no trace): %r", exc)
                self._trace_writer = None

        # OBSERVABILITY (toggle-able; default OFF = byte-identical to today). When
        # ARC_DATAFLOW is truthy, the per-turn observability record (the inter-
        # component dataflow log, the Lexicon-growth snapshot, and the canonical
        # World/object/Goal verbalization) is folded into the ARC_INTROSPECT
        # capture under distinct overlay keys (so the inspector can render it).
        # The goal-id + world-summary it surfaces are ALSO read by the LLM briefing
        # when the proposer is live -- but they are computed for the log/briefing
        # ONLY and NEVER change the committed baseline move (inert-safe). OFF
        # (default) -> no goal-id / world-summary / verbalization is computed and
        # the capture carries no observability overlay (byte-identical).
        self._observe_on = self._read_observe_toggle()
        # Per-turn read artifacts (goal-id + world-summary), recomputed each turn
        # and consumed by BOTH the briefing and the observability overlay. None
        # when observability is OFF and the LLM proposer is not live.
        self._goal_id: Optional[Any] = None
        self._world_summary: Optional[Dict[str, Any]] = None
        # The previous turn's Lexicon word-id snapshot (for the growth delta).
        self._prev_lexicon_words: tuple = ()
        # The per-turn dataflow accumulator (cleared at the top of each turn).
        self._dataflow = DataflowLog()

    @staticmethod
    def _read_observe_toggle() -> bool:
        """Read the ARC_DATAFLOW observability toggle (default OFF). Unset or a
        falsy token ("" / "0" / "off" / "false" / "no") -> OFF (byte-identical to
        today); any other value -> ON. OFF is the scored-Kaggle-run default
        (wall-clock lean); ON is for local / practice runs."""
        raw = os.getenv("ARC_DATAFLOW")
        if raw is None:
            return False
        token = raw.strip().lower()
        if token in ("", "0", "off", "false", "no"):
            return False
        return True

    @staticmethod
    def _read_futility_toggle() -> bool:
        """Read the ARC_FUTILITY toggle (default ON). Unset or a truthy token
        ("1"/"on"/"true"/"yes") -> ON; a falsy token ("0"/"off"/"false"/"no") -> OFF.
        Any other value falls back to ON (fail-safe: the guardrail is on by default)."""
        raw = os.getenv("ARC_FUTILITY")
        if raw is None:
            return True
        token = raw.strip().lower()
        if token in ("0", "off", "false", "no"):
            return False
        return True

    @staticmethod
    def _read_navigate_toggle() -> bool:
        """Read the ARC_NAVIGATE toggle (default ON). Unset or a truthy token
        ("1"/"on"/"true"/"yes") -> ON; a falsy token ("0"/"off"/"false"/"no") -> OFF.
        Any other value falls back to ON (fail-safe).

        ON (default) is the GAP-1 solving improvement: when a controllable AND a
        target are both recognized AND at least one legal action's effect is grounded
        and distance-reducing, the agent MOVES PURPOSEFULLY toward the target instead
        of cycling. OFF reproduces the pre-navigate baseline (round-robin curiosity)
        BYTE-IDENTICALLY (for A/B + safety). Even when ON, navigate returns None (and
        the baseline runs) until effects are grounded and a target is present, so the
        early-game exploration that grounds those effects is unchanged."""
        raw = os.getenv("ARC_NAVIGATE")
        if raw is None:
            return True
        token = raw.strip().lower()
        if token in ("0", "off", "false", "no"):
            return False
        return True

    @staticmethod
    def _read_marked_toggle() -> bool:
        """Read the ARC_MARKED toggle (default ON). Unset or a truthy token
        ("1"/"on"/"true"/"yes") -> ON; a falsy token ("0"/"off"/"false"/"no") -> OFF.
        Any other value falls back to ON (fail-safe).

        ON (default, ★★) wires the GENERALIZING goal-marker detector: a non-field
        object whose dominant colour is RARE on the board (its colour covers only a
        small fraction of the non-field cells, OR appears in <= 2 connected
        components) is stamped ``marked`` (FeatureContext.marked), so the
        has(marked) arm of the roles.tsv `target` recognizer fires -> NAVIGATE and
        the LLM observation get a target to aim at (the single unblock for solving
        distinct-cell-goal games). OFF reproduces the pre-marked baseline
        BYTE-IDENTICALLY: marked is never set, so `target` only fires on the
        box-enclosure arm exactly as today. The risk ON carries is false-positive
        targets (a HUD/legend glyph is also small + rare); the conservative
        rare-colour threshold (a board-derived fraction, no game literal) bounds it,
        and target out-ranks the referent roles so a real goal still wins."""
        raw = os.getenv("ARC_MARKED")
        if raw is None:
            return True
        token = raw.strip().lower()
        if token in ("0", "off", "false", "no"):
            return False
        return True

    # -- LLM move-proposer wiring (API-04, OFF by default) ------------------ #

    # Default per-game consult budget (wall-clock guard): a scored kernel has a
    # ~12h global cap and propose ~2.25s/turn, so consulting every turn would time
    # out. Bound it; ARC_LLM_BUDGET overrides. Modest by design.
    _DEFAULT_LLM_BUDGET = 200

    @staticmethod
    def _llm_toggle_on() -> bool:
        """Whether ARC_LLM is truthy (the master switch for the move proposer).
        Unset / a falsy token ("" / "0" / "off" / "false" / "no") -> OFF (default);
        any other value -> ON. OFF is byte-identical to today (NullGenerator)."""
        raw = os.getenv("ARC_LLM")
        if raw is None:
            return False
        token = raw.strip().lower()
        if token in ("", "0", "off", "false", "no"):
            return False
        return True

    def _build_llm(self):
        """Construct the move-proposer generator. ARC_LLM truthy -> the offline
        Qwen backend from ARC_LLM_MODEL (lazy/guarded — a failed import/construct
        falls back to NullGenerator, never crashes the agent). Else NullGenerator
        (the default; the proposer always declines)."""
        if not self._llm_toggle_on():
            return NullGenerator()
        model_path = os.getenv("ARC_LLM_MODEL")
        if not model_path:
            logger.warning("ARC_LLM set but ARC_LLM_MODEL unset -> NullGenerator")
            return NullGenerator()
        try:
            from agent.core.llm_qwen import QwenGenerator  # lazy: torch only on load

            return QwenGenerator(model_path)
        except Exception as exc:  # noqa: BLE001 - never crash the agent on LLM wiring
            logger.warning("LLM backend construct failed -> NullGenerator: %r", exc)
            return NullGenerator()

    def _read_llm_budget(self) -> int:
        """Per-game consult budget from ARC_LLM_BUDGET (default
        :data:`_DEFAULT_LLM_BUDGET`). A non-int / negative value falls back to the
        default (fail-safe). Deterministic."""
        raw = os.getenv("ARC_LLM_BUDGET")
        if raw is None:
            return self._DEFAULT_LLM_BUDGET
        try:
            value = int(raw.strip())
        except (ValueError, AttributeError):
            return self._DEFAULT_LLM_BUDGET
        return value if value >= 0 else self._DEFAULT_LLM_BUDGET

    @staticmethod
    def _read_llm_env_float(name: str, default: float) -> float:
        """A positive float from env ``name`` (default ``default``); a non-float /
        non-positive value falls back to the default (fail-safe). Deterministic."""
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = float(raw.strip())
        except (ValueError, AttributeError):
            return default
        return value if value > 0 else default

    @staticmethod
    def _read_llm_epoch() -> float:
        """The notebook-start wall epoch for the GLOBAL LLM deadline. Reads
        ``ARC_LLM_DEADLINE_EPOCH`` (a ``time.time()`` float the kernel writes ONCE
        before the run) so elapsed = ``now - epoch`` is notebook-global across the
        per-game agent processes; unset / unparseable -> this process's start (a safe
        per-process fallback that still bounds time, just per game)."""
        raw = os.getenv("ARC_LLM_DEADLINE_EPOCH")
        if raw is not None:
            try:
                return float(raw.strip())
            except (ValueError, AttributeError):
                pass
        return time.time()

    def _llm_active(self) -> bool:
        """Whether the move proposer is live this turn: the toggle is on, a real
        backend is wired (not NullGenerator), AND all three wall-clock guards hold --
        the per-game consult COUNT, the per-game cumulative LLM TIME, and the
        notebook-GLOBAL deadline. Past ANY guard the LLM goes silent and the classical
        baseline finishes the run: a graceful degrade to the classical floor, never a
        12 h timeout (which would forfeit the whole submission)."""
        return (
            self._llm_toggle_on()
            and not isinstance(self._llm, NullGenerator)
            and self._llm_consults < self._llm_budget
            and self._llm_time_spent < self._llm_per_game_s
            and (time.time() - self._llm_epoch) < self._llm_deadline_s
        )

    # -- Agent interface ---------------------------------------------------- #

    def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
        """Done iff the latest frame reports a WIN (the only terminal we stop on;
        a GAME_OVER is handled by the cold-start RESET so the loop can replay)."""
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: List[Any], latest_frame: Any) -> GameAction:
        """Perceive -> (capture-only solve) -> commit an OBSERVABLE BASELINE action.

        The whole perception/solve body is wrapped (load-bearing backstop): ANY
        internal error falls back to a safe legal action and is logged -- it is
        NEVER propagated (an Agent must not crash the competition loop)."""
        try:
            return self._choose_action_inner(frames, latest_frame)
        except Exception as exc:  # noqa: BLE001 - never propagate into the engine loop
            logger.warning("choose_action fell back to a safe action: %r", exc)
            return self._safe_fallback(latest_frame)

    def cleanup(self, scorecard: Any = None) -> None:
        """Close the capture sink (opened once in ``__init__``) so a Swarm / long
        run does not leak a file handle, then defer to the base cleanup. Guarded:
        cleanup must never raise."""
        try:
            if self._capture_writer is not None:
                self._capture_writer.close()
                self._capture_writer = None
        except Exception as exc:  # noqa: BLE001 - cleanup must never raise
            logger.debug("capture writer close skipped: %r", exc)
        try:
            if self._trace_writer is not None:
                self._trace_writer.close()
                self._trace_writer = None
        except Exception as exc:  # noqa: BLE001 - cleanup must never raise
            logger.debug("trace writer close skipped: %r", exc)
        super().cleanup(scorecard)

    # -- internals ---------------------------------------------------------- #

    def _choose_action_inner(self, frames: List[Any], latest_frame: Any) -> GameAction:
        # 1. Cold-start guard: replay from the top on a not-played / over / empty frame.
        if latest_frame.state in _COLD_START_STATES or latest_frame.is_empty():
            self.last_action_id = GameAction.RESET.value
            return GameAction.RESET

        # Reset the per-turn observability accumulator (inert when OFF; the record
        # calls below are cheap no-op-ish appends, gated on _observe_on).
        self._dataflow.clear()

        # 2. Perceive the settled board.
        grid = game_io.frame_to_grid(latest_frame)
        # LLM proposer recent_actions bookkeeping: finalize the previous move's
        # `changed` flag now that its result board is visible. Gated on the live
        # proposer so the OFF path touches no extra state (byte-identical to today).
        if self._llm_active():
            self._update_recent_actions(grid)
        parse = divide_frame(grid)
        self._df(
            "perceive",
            "divide_frame",
            "grid %dx%d" % (grid.shape[0], grid.shape[1]),
            "%d object(s)" % len(parse.objects),
        )

        # 3. GAP-2/GAP-3: identify the controllable object (the avatar the agent
        #    steers) from the per-object motion trajectory FIRST, so the per-object
        #    Affordance evidence it computes is available to detect_features below
        #    (an object that translates reads as `movable`, not `static`, so the
        #    field predicate has(static) does not over-fire on the avatar). Load-
        #    bearing backstop: never break the loop.
        try:
            self._identify_controllable(parse, grid)
        except Exception as exc:  # noqa: BLE001 - controllable id is a non-fatal overlay
            logger.debug("controllable identification skipped: %r", exc)

        # 3b. DetectFeatures per object, feeding the per-object Affordance evidence
        #     (from 3) so the behaviour modifiers (movable / static / ...) are honest.
        #     GOAL-MARKER (ARC_MARKED, default ON): compute the board-context rare-
        #     colour `marked` set ONCE over the whole frame, then relay it per object
        #     (mirrors is_field). Empty set when OFF -> marked never stamped (byte-
        #     identical baseline).
        marked_ids = self._marked_object_ids(parse, grid) if self._marked_on else frozenset()
        for obj in parse.objects:
            ctx = FeatureContext(
                cells=obj.cells,
                color_counts=self._color_counts(obj.cells, grid),
                is_field=self._is_field(obj, parse),
                affordance=self._affordance_evidence.get(obj.id),
                marked=obj.id in marked_ids,
            )
            obj.profile = detect_features(ctx, self.assets.lexicon)

        # 3c. Stamp the grounded controllable pick onto its Profile so the canonical
        #     recognized_by = has(controllable) predicate tests true and AnalogizeRoles
        #     assigns `controllable` (the motion pick stays the decision; FR-168).
        self._mark_controllable(parse)

        # 4. Project the abstract situation (carry tracker + prev across turns).
        #    StateAbstraction runs AnalogizeRoles (self._classify_roles) over the
        #    salient objects to bucket roles (FR-R-1).
        situation = self.abstraction.project(parse, self.prev)
        self._df(
            "project",
            "StateAbstraction.project",
            "%d object(s)" % len(parse.objects),
            "situation: %d role(s) %s"
            % (
                len(getattr(situation, "objects", {}) or {}),
                sorted(getattr(situation, "objects", {}) or {}),
            ),
        )

        # 4c. CONNECT the unwired read components (INERT-SAFE): goal identification
        #     (察する = goal 同定) + the WorldModel summary. Computed when
        #     observability is ON or the LLM proposer is live -- they feed the
        #     observability overlay + the LLM briefing ONLY and NEVER change the
        #     committed baseline move. Guarded: a non-fatal read overlay.
        self._goal_id = None
        self._world_summary = None
        if self._observe_on or self._llm_active():
            self._compute_read_components(situation)

        # 4b. WASTED-MOVE (futility) -- DetectFutility (retroactive) + the masked board
        #     identity for THIS turn. Gauge-masked so a turn that only shortened the
        #     budget bar does not look like a fresh board. Whole leg gated on the toggle
        #     and guarded (never break the loop); cur_mhash drives CheckFutility below.
        cur_mhash = self._update_futility(grid)

        # 5. Goal + solver -- CAPTURE RICHNESS ONLY (not used by the baseline action).
        #    Guarded so a malformed pattern / predicate never aborts the turn.
        plan = None
        try:
            plan = self._solve_for_capture(situation, latest_frame)
        except Exception as exc:  # noqa: BLE001 - the solve leg is non-load-bearing here
            logger.debug("capture-only solve skipped: %r", exc)
            plan = None

        # 6. CLASSICAL ACTION. Default = OBSERVABLE BASELINE (deterministic legal-action
        #    cycle), augmented by CheckFutility (prune known non-progress (state, move)).
        #    When ARC_NAVIGATE is on, a GREEDY NAVIGATE (GAP-1) gets FIRST refusal: if a
        #    controllable AND a target are recognized AND a legal action's grounded effect
        #    reduces the controllable->target distance, that distance-reducing move WINS
        #    over the round-robin (the "actually solve" step). Navigate returns None when
        #    there is no target / no grounded distance-reducing action, so the baseline
        #    still explores to ground effects (and the OFF path is byte-identical).
        baseline_id = self._choose_baseline_with_futility(latest_frame, cur_mhash)
        legal_nonreset = [a for a in (latest_frame.available_actions or []) if a != 0]
        classical_source = "baseline"
        classical_id = baseline_id
        self._classical_click_xy = None
        nav_id = None
        click_nav = None
        # NO-PROGRESS STREAK: consecutive turns the masked board has NOT changed
        # (the blind round-robin is stuck). navigate / click-navigate -- which
        # DEVIATE from the productive round-robin (the cycle empirically solves
        # e.g. cd82/sp80 by blind coverage; targeting wrong markers BREAKS those) --
        # only fire once the baseline is genuinely STUCK. So a game the cycle solves
        # keeps its winning cycle, and only a stalled game falls through to aiming.
        if cur_mhash is not None and cur_mhash == self._prev_mhash:
            self._no_progress_streak += 1
        else:
            self._no_progress_streak = 0
        self._prev_mhash = cur_mhash
        # No-level-progress streak (click-stall signal; see __init__).
        cur_levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
        if cur_levels > self._prev_levels:
            self._no_level_progress_streak = 0
        else:
            self._no_level_progress_streak += 1
        self._prev_levels = cur_levels
        aim_stuck = self._no_progress_streak >= self._AIM_STUCK_TURNS
        if self._navigate_on and aim_stuck:
            try:
                nav_id = self._navigate_move(
                    situation, parse, latest_frame, legal_nonreset, cur_mhash
                )
            except Exception as exc:  # noqa: BLE001 - navigate is non-load-bearing
                logger.debug("navigate skipped: %r", exc)
                nav_id = None
        if nav_id is not None:
            classical_id = nav_id
            classical_source = "navigate"
        elif self._navigate_on and aim_stuck:
            # CLICK-NAVIGATE: directional navigate had nothing (no controllable+target
            # grounded move). If a click action is legal AND a target exists, click a
            # real target centroid (the classical side, not the LLM, decides the coord).
            try:
                click_nav = self._click_navigate_move(situation, latest_frame, cur_mhash)
            except Exception as exc:  # noqa: BLE001 - click-navigate is non-load-bearing
                logger.debug("click-navigate skipped: %r", exc)
                click_nav = None
            if click_nav is not None:
                classical_id, self._classical_click_xy = click_nav
                classical_source = "navigate"

        # 6b. LOCAL-LLM move proposer (API-04, OFF by default). Consulted BEFORE
        #     committing the classical move, only when it can plausibly help (a target role
        #     exists OR the baseline is stuck/futile this turn) AND the per-game budget
        #     is unspent. A valid legal proposal WINS; a decline / invalid / exhausted
        #     budget falls through to the classical move (navigate or the load-bearing
        #     baseline path). OFF (default) -> NullGenerator declines -> this is a pure
        #     no-op and the committed action is the classical move. self._last_move_source
        #     records which leg won (surfaced in capture/trace).
        self._last_move_source = classical_source
        # Y-team investigation hook (ARC_DUMP_OBS=<path>): dump the move-proposer
        # observation each turn for offline briefing analysis -- runs on the classical
        # path too (no torch), gated, never load-bearing. Default-unset = byte-identical.
        _dump_path = os.environ.get("ARC_DUMP_OBS")
        if _dump_path:
            try:
                _obs_dump = self.build_observation(situation, parse, latest_frame)
                _legal_d = [a for a in (latest_frame.available_actions or []) if a != 0]
                _per = self._action_displacements(_legal_d)
                _cid = self._controllable_id
                _hist = self._track_history.get(_cid, []) if _cid else []
                _obs_dump["_diag"] = {
                    "last_action": (int(self.last_action_id)
                                    if self.last_action_id is not None else None),
                    "ctrl_id": _cid,
                    "ctrl_hist_len": len(_hist),
                    "ctrl_footprint_size": (len(_hist[-1][1].cells) if _hist else None),
                    "per_action_shifts": {
                        str(a): sorted([list(s) for s in shifts])
                        for a, shifts in sorted(_per.items())
                    },
                }
                with open(_dump_path, "a", encoding="utf-8") as _fh:
                    _fh.write(json.dumps(_obs_dump, ensure_ascii=False) + "\n")
            except Exception as exc:  # noqa: BLE001 - debug overlay, never fatal
                logger.debug("ARC_DUMP_OBS dump skipped: %r", exc)
        llm_move = self._propose_llm_move(
            situation, parse, latest_frame, classical_id, cur_mhash
        )
        # A classical CLICK-NAVIGATE move aims at a REAL target centroid the classical
        # side computed; it beats the LLM's coordinate (a 1.5B emits degenerate click
        # coords -- empirically (0,0)/(0,31)). So when click-navigate produced a target
        # click, it WINS over the LLM. Otherwise the LLM proposal wins as before. (OFF
        # path unchanged: _classical_click_xy is only set when ARC_NAVIGATE is on.)
        if self._classical_click_xy is not None:
            chosen_id = classical_id  # _last_move_source stays the classical "navigate"
        elif llm_move is not None:
            chosen_id = llm_move
            self._last_move_source = "llm"
        else:
            chosen_id = classical_id
        self._df(
            "action",
            self._last_move_source,
            "baseline=%s navigate=%s legal=%s"
            % (baseline_id, nav_id, sorted(legal_nonreset)),
            "chosen=%s" % chosen_id,
        )

        if chosen_id is None:
            self.last_action_id = GameAction.RESET.value
            self._prev_mhash = cur_mhash
            self._last_action_for_futility = None
            return GameAction.RESET

        solver_id = plan.chosen.id if (plan is not None and plan.chosen) else None
        x = y = None
        if GameAction.from_id(chosen_id).is_complex():
            # An LLM complex move carries its own (x, y); a classical click-navigate
            # move carries the target centroid it chose; a baseline complex move uses
            # the deterministic coverage lattice.
            if self._last_move_source == "llm" and self._llm_click_xy is not None:
                x, y = self._llm_click_xy
            elif (
                self._last_move_source == "navigate"
                and self._classical_click_xy is not None
            ):
                x, y = self._classical_click_xy
            else:
                x, y = self._baseline_click_xy(grid)
        # Arm the recent_actions log for the move we are committing (its `changed`
        # is resolved at the top of next turn). Gated on the live proposer so the
        # OFF path stays byte-identical.
        if self._llm_active():
            click_xy = (x, y) if (x is not None and y is not None) else None
            self._arm_recent_action(chosen_id, click_xy)
        action = game_io.move_to_action(
            chosen_id,
            x=x,
            y=y,
            game_id=latest_frame.game_id,
            reasoning={
                "policy": "baseline-curiosity",
                "solver": solver_id,
                "move_source": self._last_move_source,
            },
        )

        # 7. Arm a prediction for the chosen move (near-inert in MVP; never fatal).
        try:
            self.armed_prediction = self.world.predict(situation, chosen_id)
        except Exception as exc:  # noqa: BLE001 - predict is near-inert; never fatal
            logger.debug("world.predict skipped: %r", exc)
            self.armed_prediction = None

        # 8. Emit one capture line (inert if ARC_INTROSPECT unset); advance state.
        self._emit_capture(grid, parse, situation, chosen_id, latest_frame, solver_id)
        # 8b. Emit the PAIRED TurnRecord in LOCKSTEP (inert if ARC_TRACE unset).
        self._emit_trace(grid, parse, situation, plan, chosen_id, latest_frame)
        self.prev = situation
        self.last_action_id = chosen_id
        # Futility bookkeeping: remember THIS turn's masked board + the move we played,
        # so next turn's DetectFutility can classify this move retroactively. Inert when
        # the toggle is OFF (cur_mhash is None then).
        self._prev_mhash = cur_mhash
        self._last_action_for_futility = chosen_id
        return action

    # -- wasted-move (futility) prevention ---------------------------------- #

    _COLOR_HISTORY_WINDOW = 12  # bounded {color: count} window for gauge detection.

    def _update_futility(self, grid: Any) -> Optional[bytes]:
        """Push this frame's ``{color: count}``, recompute the gauge cells, derive the
        masked board hash, run DetectFutility (retroactively classify last turn's move),
        then note this board as visited. Returns ``cur_mhash`` (``None`` when the toggle
        is OFF or on any internal failure -- the whole leg is guarded so it can NEVER
        break the play loop)."""
        if not self._futility_on:
            return None
        try:
            colors, counts = np.unique(np.asarray(grid), return_counts=True)
            self._color_count_history.append(
                {int(c): int(n) for c, n in zip(colors.tolist(), counts.tolist())}
            )
            if len(self._color_count_history) > self._COLOR_HISTORY_WINDOW:
                del self._color_count_history[
                    : len(self._color_count_history) - self._COLOR_HISTORY_WINDOW
                ]
            # UNION-MASK accumulation (FIX 1): mask the bar's MAXIMAL footprint, not just the
            # cells lit THIS frame. A receding gauge vacates cells that revert to background;
            # masking only the current cells would leave that vacated cell differing between two
            # consecutive boards, so a gauge-tick would look like a fresh board and futility would
            # never fire. Accumulate the UNION of detect_gauges results across turns: the gauge
            # colour is fixed per game so its union is the bar's max region, and the avatar
            # oscillates in a DIFFERENT colour so it never enters the union. The background-guard
            # inside detect_gauges keeps a shrinking BACKGROUND colour out of the union, so it can
            # never grow to the whole board. Reset only at construction (self._gauge_cells = ...).
            self._gauge_cells = self._gauge_cells | futility.detect_gauges(
                self._color_count_history, grid
            )
            cur_mhash = futility.masked_board_hash(grid, self._gauge_cells)

            # DetectFutility (retroactive): classify the move played LAST turn now that we
            # see its result. Compare cur_mhash to prev_mhash and to the visited set BEFORE
            # adding cur_mhash (so a board recurring counts as a revisit = no_progress).
            if (
                self._prev_mhash is not None
                and self._last_action_for_futility is not None
            ):
                visited_before = self._working_memory.has_visited(cur_mhash)
                self._working_memory.record(
                    self._prev_mhash,
                    self._last_action_for_futility,
                    visited_before,
                    cur_mhash,
                )
            self._working_memory.note_visit(cur_mhash)
            return cur_mhash
        except Exception as exc:  # noqa: BLE001 - futility must never break the loop
            logger.debug("futility update skipped: %r", exc)
            return None

    def _choose_baseline_with_futility(
        self, latest_frame: Any, cur_mhash: Optional[bytes]
    ) -> Optional[int]:
        """CheckFutility-augmented action selection.

        When the toggle is OFF (or there is no masked hash this turn), this is
        BYTE-IDENTICAL to :meth:`_baseline_action_id` (the pre-feature baseline). When ON,
        it drops legal non-RESET actions known-futile from ``cur_mhash`` and cycles the
        SURVIVORS by ``action_counter`` (same deterministic round-robin). If EVERY legal
        move is known-futile -> STUCK-BREAKER: the least-recently-tried legal action
        (deterministic per-action try counts; ascending-id tie-break; no RNG)."""
        if not self._futility_on or cur_mhash is None:
            return self._baseline_action_id(latest_frame)

        legal = [a for a in (latest_frame.available_actions or []) if a != 0]
        if not legal:
            return None

        survivors = [
            a for a in legal
            if not self._working_memory.is_futile(cur_mhash, a)
        ]
        if survivors:
            chosen = survivors[self.action_counter % len(survivors)]
        else:
            # STUCK-BREAKER: every legal move is known-futile here. Pick the
            # least-recently-tried legal action (min try-count, then smallest id).
            chosen = min(
                legal,
                key=lambda a: (self._action_try_counts.get(a, 0), a),
            )
        self._action_try_counts[chosen] = self._action_try_counts.get(chosen, 0) + 1
        return chosen

    # -- LLM move proposer (API-04, OFF by default) ------------------------- #

    # Max objects emitted into the observation -- keeps the prompt compact +
    # bounded regardless of board complexity (the frozen contract caps objects ~6).
    _LLM_OBSERVATION_MAX_OBJECTS = 6
    # Output token budget. The ENFORCED proposal (goal_prediction + move) is a small
    # JSON object (~60-100 tokens); 128 covers it with headroom while ~halving the
    # generation time vs the old 256 -- a direct wall-clock win on the 12 h cap (the
    # advisory proposals/note channels may truncate, which only drops the advisory,
    # never the move).
    _LLM_MAX_NEW_TOKENS = 128
    # PREFILL guard: max chars of the rendered system+user prompt a consult will
    # prefill. max_time bounds generation but NOT the prefill forward pass, so a large
    # prompt is bounded HERE instead. ~4 chars/token => ~40k chars ~= ~10k tokens,
    # ~4x the de-bloated normal (~10k chars): a backstop -- the structural caps
    # (objects <=6, trimmed vocab) are the primary bound.
    _LLM_MAX_PROMPT_CHARS = 40000

    def _propose_llm_move(
        self,
        situation: Any,
        parse: Any,
        latest_frame: Any,
        baseline_id: Optional[int],
        cur_mhash: Optional[bytes],
    ) -> Optional[int]:
        """Consult the LLM for the next action, returning a VALID legal action id or
        ``None`` (decline / invalid / not-consulted -> the caller uses the baseline).

        The PROVEN stage-2 path (``docs/llm-prompt-design/run_proposer_test.py``),
        wired: build an input-observation INSTANCE, render the system+user prompt,
        narrow the OUTPUT schema to THIS turn's buttons/click/templates, consult the
        constrained generator, validate the move/goal against the observation, and
        DISPOSE per :meth:`_dispose_llm_move` (low confidence -> do NOT adopt the
        goal -> explore/baseline; else map the button name / click {row,col} to a
        legal engine action).

        Bounding policy (deterministic, documented):
          * Live only when ARC_LLM is truthy AND a real backend is wired AND the
            per-game consult budget is unspent (:meth:`_llm_active`).
          * Even then consults ONLY when it can plausibly help: a ``target`` role
            exists OR the baseline is stuck/futile (every legal non-RESET move
            known-futile from ``cur_mhash``). Otherwise declines WITHOUT spending
            budget.
          * Each actual consult decrements the budget (counts even on decline).

        A returned id is a legal NON-RESET action; a click carries in-range
        ``x``/``y`` on ``self._llm_click_xy``. OFF (default) -> NullGenerator ->
        ``consult`` returns ``None`` -> byte-identical to today."""
        self._llm_click_xy = None
        if not self._llm_active():
            return None
        if not self._llm_should_consult(situation, latest_frame, baseline_id, cur_mhash):
            return None

        legal = frozenset(
            a for a in (latest_frame.available_actions or []) if a != 0
        )
        if not legal:
            return None

        # Lazily load + cache the prompt template + full output schema (pure file
        # IO, torch-free). A load failure -> decline (classical fallback), never crash.
        if self._prompt_tpl is None or self._out_schema is None:
            try:
                self._prompt_tpl = llm_prompt.load_prompt()
                self._out_schema = llm_prompt.load_output_schema()
            except Exception as exc:  # noqa: BLE001 - missing/garbled prompt -> fallback
                logger.warning("LLM prompt load failed -> classical fallback: %r", exc)
                return None

        observation = self.build_observation(situation, parse, latest_frame)
        messages = llm_prompt.render_messages(self._prompt_tpl, observation)
        # PREFILL guard (max_time caps generation, not the prefill over a big prompt):
        # the prompt is already structurally bounded, so this only fires if something
        # unexpectedly inflates it -> DECLINE to classical rather than pay an unbounded
        # prefill on the 12 h clock.
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        if prompt_chars > self._LLM_MAX_PROMPT_CHARS:
            logger.debug(
                "LLM consult skipped: prompt %d chars > %d cap (prefill guard)",
                prompt_chars,
                self._LLM_MAX_PROMPT_CHARS,
            )
            return None
        schema = llm_prompt.narrow_schema(self._out_schema, observation)
        self._llm_consults += 1

        # ARC_LLM_DEBUG verbose-coordination capture (default OFF). `_dbg` carries
        # the raw proposal + parse outcome out of consult for the log.
        debug_on = self._llm_debug_on()
        _dbg: Dict[str, Any] = {
            "raw": None,
            "parsed": False,
            "declined": False,
            "parse_error": None,
        }

        def _parse(
            raw: Dict[str, Any]
        ) -> Optional[Tuple[int, Optional[Tuple[int, int]]]]:
            if debug_on:
                _dbg["raw"] = raw
            try:
                result = self._dispose_llm_move(raw, observation, legal)
            except Exception as exc:  # capture-then-reraise: consult still falls back
                if debug_on:
                    _dbg["parse_error"] = "%r" % (exc,)
                raise
            if debug_on:
                _dbg["declined"] = result is None
                _dbg["parsed"] = result is not None
            return result

        briefing = messages[-1]["content"]  # the user message (for the debug log)
        _t0 = time.time()
        proposal = consult(
            self._llm,
            messages,
            schema,
            parse=_parse,
            max_new_tokens=self._LLM_MAX_NEW_TOKENS,
        )
        # Accumulate REAL consult wall-clock into the per-game time guard (read by
        # _llm_active next turn). Counts whether the proposal was adopted or declined
        # -- the time was spent regardless.
        self._llm_time_spent += time.time() - _t0
        if proposal is None:
            self._llm_debug_emit(briefing, _dbg, decision="baseline", action_id=None)
            return None
        action_id, click_xy = proposal
        self._llm_click_xy = click_xy
        self._llm_debug_emit(briefing, _dbg, decision="llm", action_id=action_id)
        return action_id

    @staticmethod
    def _llm_debug_on() -> bool:
        """Whether ARC_LLM_DEBUG is truthy (verbose per-consult coordination log).
        Unset / a falsy token ("" / "0" / "off" / "false" / "no") -> OFF (default,
        byte-identical to today); any other value -> ON. Diagnosis-only; never
        changes the committed move."""
        raw = os.getenv("ARC_LLM_DEBUG")
        if raw is None:
            return False
        token = raw.strip().lower()
        if token in ("", "0", "off", "false", "no"):
            return False
        return True

    def _llm_debug_emit(
        self,
        briefing: str,
        dbg: Dict[str, Any],
        *,
        decision: str,
        action_id: Optional[int],
    ) -> None:
        """Emit ONE verbose record of this consult's agent<->backend exchange when
        ARC_LLM_DEBUG is on: the BRIEFING sent, the RAW proposal returned, the PARSE
        outcome (parsed action or the ACCURATE decline cause), and the FINAL decision
        (llm action used vs fallback to baseline). Pure logging (guarded so it can
        NEVER break the play loop); inert when OFF."""
        if not self._llm_debug_on():
            return
        try:
            raw = dbg.get("raw")
            if dbg.get("parsed"):
                outcome = "OK"
            elif dbg.get("declined"):
                # A clean DECLINE from disposition: report the ACTUAL cause (it is
                # NOT NO_ACTION anymore -- it is validation / no-mapping / low-conf).
                reason = self._llm_decline_reason or "unspecified"
                outcome = "DECLINE: %s -> classical fallback" % reason
            elif dbg.get("parse_error") is not None:
                outcome = "PARSE_FAIL %s" % dbg["parse_error"]
            elif raw is None:
                outcome = "NO_RAW (backend declined/raised; see 'LLM consult fell back')"
            else:
                outcome = "DECLINED"
            logger.info(
                "LLM_DEBUG consult #%d game=%s\n"
                "  briefing:\n%s\n"
                "  raw_proposal: %r\n"
                "  parse_outcome: %s\n"
                "  decision: %s action_id=%s",
                self._llm_consults,
                getattr(self, "game_id", "?"),
                "    " + briefing.replace("\n", "\n    "),
                raw,
                outcome,
                decision,
                action_id,
            )
        except Exception as exc:  # noqa: BLE001 - debug logging must never break the loop
            logger.debug("LLM_DEBUG emit skipped: %r", exc)

    def _llm_should_consult(
        self,
        situation: Any,
        latest_frame: Any,
        baseline_id: Optional[int],
        cur_mhash: Optional[bytes],
    ) -> bool:
        """Whether THIS turn warrants a consult: STUCK or STALLED only.

        True iff EITHER the masked board has not changed for ``>= _AIM_STUCK_TURNS``
        turns (directional stuck -- the same gate as navigate/click), OR no LEVEL has
        been completed for ``>= _LLM_STALL_TURNS`` turns (click-STALL: a click game
        whose board churns every click yet never wins, where blind click-coverage has
        run out of road). The LLM is a LAST-RESORT proposer: a mere ``target`` existing
        is NOT sufficient (firing on that let it displace a still-progressing classical
        win -- the cd82 regression / 0.01 lesson); only genuine stuckness/stall does,
        and classical click-navigate keeps move precedence so a game the classical side
        is winning is never overridden. Deterministic; no RNG, no hash.
        (``situation``/``latest_frame``/``baseline_id``/``cur_mhash`` kept for signature
        stability; the streaks live on ``self``.)"""
        return (
            self._no_progress_streak >= self._AIM_STUCK_TURNS
            or self._no_level_progress_streak >= self._LLM_STALL_TURNS
        )

    # -- LLM move disposition (validate + adopt/decline, the contract) ------- #

    def _dispose_llm_move(
        self, raw: Dict[str, Any], obs: Dict[str, Any], legal: frozenset
    ) -> Optional[Tuple[int, Optional[Tuple[int, int]]]]:
        """Dispose the LLM's raw ``{goal_prediction, move, ...}`` into
        ``(action_id, click_xy)`` or ``None`` (a clean DECLINE -> classical fallback).

        Order (IMPLEMENT-HANDOFF steps 7-8, + the user-confirmed low->explore gate):
          1. POST-PARSE VALIDATE the move/goal half against the observation
             (:func:`llm_prompt.validate`). ANY error -> DECLINE (reason 'validation').
          2. STORE the rolling ``note`` for next turn (passthrough, even on a
             declined move so the anti-thrash scratchpad survives).
          3. The advisory ``proposals`` / ``confidence_nudges`` channels are parsed,
             logged, and DEFERRED. They NEVER affect the move.
          4. MAP the move to a candidate ``(action_id, click_xy)`` -- ``{button:name}``
             -> the engine action id (legal NON-RESET); ``{click:{row,col}}`` ->
             ACTION6 with ``click_xy = (x=col, y=row)``. Any invalid / out-of-set /
             unmappable move -> DECLINE (reason 'no legal mapping').
          5. CONFIDENCE GATE (master disposition): the goal is NEVER adopted into
             classical state (that seam is deferred). For the MOVE:
               * medium / high -> ADOPT the candidate.
               * low -> ADOPT ONLY as GENUINE EXPLORATION: a button whose ``effect``
                 is null (ungrounded) this turn, OR a click (clicking an arbitrary
                 cell is inherently exploratory) -- AND it must not be a no-op REPEAT
                 (the last ``recent_actions`` entry being the SAME action with
                 ``changed == false``). Otherwise DECLINE (reason
                 'low-confidence non-exploratory'). This drives early exploration when
                 effects are ungrounded without blindly committing to a guessed goal.

        A returned tuple is an ACCEPTED move; ``None`` is a graceful decline (no
        warning noise). The decline cause is recorded on ``self._llm_decline_reason``
        for the debug log. No RNG/hash (DP-10)."""
        self._llm_decline_reason = None
        if not isinstance(raw, dict):
            self._llm_decline_reason = "validation"
            return None
        errs = llm_prompt.validate(raw, obs)
        # Store the note BEFORE any decline so the scratchpad carries across turns.
        self._store_llm_note(raw)
        self._apply_llm_advisory(raw)
        if errs:
            self._llm_decline_reason = "validation"
            logger.debug("LLM proposal failed validation -> decline: %s", errs)
            return None

        # Map the move to a candidate (unchanged validity rules). is_click flags a
        # click move; button_name is the chosen button (for the exploration check).
        move = raw.get("move") or {}
        candidate: Optional[Tuple[int, Optional[Tuple[int, int]]]] = None
        is_click = False
        button_name: Optional[str] = None
        if "click" in move:
            is_click = True
            click = move.get("click") or {}
            row = _coerce_int(click.get("row"))
            col = _coerce_int(click.get("col"))
            action_id = self._click_action_id(legal)
            # Engine ACTION6 takes x=col, y=row (the adapter's coordinate convention).
            if row is not None and col is not None and action_id is not None:
                x, y = col, row
                if 0 <= x <= 63 and 0 <= y <= 63:
                    candidate = (action_id, (x, y))
        elif "button" in move:
            button_name = move.get("button")
            action_id = game_io.action_for_button_name(button_name)
            if action_id is not None and action_id in legal:
                candidate = (action_id, None)
        if candidate is None:
            self._llm_decline_reason = "no legal mapping"
            return None

        gp = raw.get("goal_prediction") or {}
        if gp.get("confidence") != "low":
            return candidate  # medium / high -> adopt

        # low confidence: adopt ONLY as genuine exploration (the prompt's low->explore
        # directive), and never as a no-op repeat (anti-thrash).
        if not self._is_genuine_exploration(obs, is_click, button_name):
            self._llm_decline_reason = "low-confidence non-exploratory"
            return None
        if self._is_noop_repeat(obs, is_click, button_name, candidate):
            self._llm_decline_reason = "low-confidence no-op repeat"
            return None
        return candidate

    @staticmethod
    def _is_genuine_exploration(
        obs: Dict[str, Any], is_click: bool, button_name: Optional[str]
    ) -> bool:
        """Whether a low-confidence move is GENUINE EXPLORATION worth adopting:
        a CLICK (clicking an arbitrary cell is inherently exploratory), OR a BUTTON
        whose ``effect`` is null (ungrounded/unknown) in this turn's
        ``obs.inputs.buttons`` (the prompt tells the model to pick an unknown-effect
        button when low). A button with a KNOWN (non-null) effect is NOT exploration.
        Deterministic."""
        if is_click:
            return True
        if button_name is None:
            return False
        for b in obs.get("inputs", {}).get("buttons", []):
            if b.get("name") == button_name:
                return b.get("effect") is None
        return False

    @staticmethod
    def _is_noop_repeat(
        obs: Dict[str, Any],
        is_click: bool,
        button_name: Optional[str],
        candidate: Tuple[int, Optional[Tuple[int, int]]],
    ) -> bool:
        """Whether this move REPEATS the last recorded action that DID NOTHING: the
        last ``obs.recent_actions`` entry has ``changed == false`` AND is the same
        action (same button name, or same click cell). We just learned it is a no-op,
        so repeating it is futile thrash -> decline even though it is 'exploratory'.
        Deterministic."""
        recent = obs.get("recent_actions") or []
        if not recent:
            return False
        last = recent[-1]
        if last.get("changed") is not False:
            return False
        if is_click:
            click_xy = candidate[1]
            if click_xy is None:
                return False
            # recent_actions stores a click as {action:'click', x, y} (x=col, y=row).
            return (
                last.get("action") == "click"
                and last.get("x") == click_xy[0]
                and last.get("y") == click_xy[1]
            )
        return last.get("action") == button_name

    @staticmethod
    def _click_action_id(legal: frozenset) -> Optional[int]:
        """The legal COMPLEX (ACTION6 / click) action id this turn, or ``None``.
        Deterministic (smallest id among legal complex actions)."""
        complex_ids = sorted(
            a for a in legal if GameAction.from_id(a).is_complex()
        )
        return complex_ids[0] if complex_ids else None

    def _store_llm_note(self, raw: Dict[str, Any]) -> None:
        """Store the proposer's one-line ``note`` (the rolling scratchpad), to echo
        back verbatim as ``observation.last_note`` next turn (pure passthrough, not
        interpreted). A non-string / over-long / absent note clears it. Bounded to
        the contract's 140 chars. Deterministic; never raises."""
        note = raw.get("note")
        if isinstance(note, str) and note.strip():
            self._last_note = note.strip()[:140]
        else:
            self._last_note = None

    def _apply_llm_advisory(self, raw: Dict[str, Any]) -> None:
        """Handle the advisory ``proposals`` + ``confidence_nudges`` channels.

        DEFERRED SEAMS (logged, never fabricated): the runtime registry-admit path
        (``origin='proposed'`` via a ``Lexicon.add_word``-style seam) and the
        confidence-nudge belief store do NOT exist yet. Per the handoff, we parse +
        validate + LOG these (so the data flow is wired and observable) but DO NOT
        mutate any registry / belief state -- that is a precise TODO, not an
        invented mutation. Both are advisory and NEVER affect the move.

        TODO(API-04 advisory admit): when the runtime-admit seam lands, run each
        ``proposals.goal_patterns`` / ``proposals.roles`` predicate through the
        AssetLoader gate (``parse_predicate`` + ``_validate_predicate_operators`` +
        the ``_validate_solver_fks`` FK closures) and admit survivors at runtime
        (origin='proposed'), NEVER into the shipped TSVs; and apply each
        ``confidence_nudges`` ``{on,direction}`` as a small fixed clamped step
        weighted below the master's own evidence. Until then they are inert."""
        proposals = raw.get("proposals")
        nudges = raw.get("confidence_nudges")
        if not proposals and not nudges:
            return
        try:
            logger.debug(
                "LLM advisory (DEFERRED -- no registry/nudge admit seam yet): "
                "proposals=%r nudges=%r",
                proposals,
                nudges,
            )
        except Exception as exc:  # noqa: BLE001 - advisory logging must never break
            logger.debug("LLM advisory log skipped: %r", exc)

    # -- LLM observation builder (input-observation-schema.json instance) ---- #

    # Canonical role labels read off the projected situation for the observation.
    _TARGET_ROLE = "target"
    _FIELD_ROLES = frozenset({"field", "is_field", "background"})
    # Roles whose objects carry an absolute position (targets/clickables) so a
    # click can copy it directly.
    _POSITION_ROLES = frozenset({"target", "reference", "ref"})

    def build_observation(
        self, situation: Any, parse: Any, latest_frame: Any
    ) -> Dict[str, Any]:
        """Build a DETERMINISTIC instance of ``input-observation-schema.json`` for
        the move proposer (replaces the old ``_verbalize_for_llm`` briefing string).

        Honest-roles caveat: this ONLY translates what the agent already recognizes
        (the projected ``situation.objects`` role buckets + grounded affordance
        effects) into the model's semantic surface -- it invents nothing. ALL spatial
        facts are (row, col) geometry; NO direction words appear in the data (only
        button NAMES, which are game-agnostic ``aN`` handles, are 'names'). The same
        situation always renders the same dict (sorted; no rng/hash, DP-10).

        Fields (per the frozen contract):
          * ``inputs.buttons`` = usable simple controls this turn, each
            ``{name, effect}`` -- ``effect`` is the grounded (row,col) displacement of
            the controllable (null until grounded); ``inputs.click`` = is ACTION6 usable.
          * ``objects`` = semantic ref/role/color/shape/relative/position/flags,
            salient/role-bearing only, cap ~6, absolute position on targets/clickables.
          * ``goal_hint`` = the classical guess as a non-asserted CANDIDATE (null if none).
          * ``goal_templates`` = active goal_patterns; ``vocabulary`` = the loaded
            primitive catalogs; ``lexicon`` = synthesized words; ``last_note`` = the
            note the model wrote last turn (echoed verbatim)."""
        objects_map = getattr(situation, "objects", None) or {}
        legal = [a for a in (latest_frame.available_actions or []) if a != 0]

        # inputs.buttons (simple controls) + inputs.click (ACTION6). `effect` is the
        # robust grounded direction / "no-op" / "inconsistent" / null (_action_effect).
        buttons: List[Dict[str, Any]] = []
        click_ok = False
        for a in sorted(legal):
            if GameAction.from_id(a).is_complex():
                click_ok = True
                continue
            buttons.append(
                {
                    "name": game_io.button_name_for_action(a),
                    "effect": self._action_effect(a),
                }
            )

        # objects: semantic refs, capped, geometric. The controllable anchors
        # `relative`. Targets/clickables carry an absolute position.
        controllable_ref = self._controllable_centroid(objects_map)
        obs_objects = self._observation_objects(objects_map, controllable_ref)

        # world summary scalars (honest; from the read components).
        n_roles = len(objects_map)
        n_objects = sum(len(refs) for refs in objects_map.values())
        induced = (
            int(self._world_summary.get("rule_count", 0))
            if self._world_summary is not None
            else 0
        )
        viewport = (
            "scrolling"
            if (self._world_summary is not None
                and self._world_summary.get("is_scrolling"))
            else "static"
        )

        observation: Dict[str, Any] = {
            "meta": {
                "turn": int(self.action_counter),
                "level": int(getattr(latest_frame, "levels_completed", 0) or 0),
                # No standalone score field on FrameData; levels_completed is the
                # only progress signal the engine exposes, so it doubles as score
                # (the schema defines score as 'levels gained / game-defined').
                "score": int(getattr(latest_frame, "levels_completed", 0) or 0),
                "llm_budget_left": max(0, self._llm_budget - self._llm_consults),
            },
            "inputs": {"buttons": buttons, "click": click_ok},
            "world": {
                "n_objects": n_objects,
                "n_roles": n_roles,
                "induced_move_rules": induced,
                "viewport": viewport,
            },
            "objects": obs_objects,
        }

        # goal_hint: a NON-authoritative candidate (never asserted).
        if self._goal_id is not None:
            p = self._goal_id.pattern
            observation["goal_hint"] = {
                "kind": p.goal_kind,
                "description": "%s (%s)" % (p.id, p.goal_kind),
                "predicate": p.predicate,
                "confidence": "low",  # classical hint is never asserted as high
                "satisfied": bool(self._goal_id.satisfied),
            }
        else:
            observation["goal_hint"] = None

        # goal_templates (active patterns) + vocabulary (loaded catalogs).
        templates = self._goal_templates()
        observation["goal_templates"] = templates
        # Compact per-kind meanings, emitted ONCE (vs the old per-template summary
        # repeated ~20x): {kind: one-line win condition} for the kinds actually
        # present this turn. Lets the model read a template's `kind` without the bloat.
        legend = self._goal_kinds_legend(templates)
        if legend:
            observation["goal_kinds_legend"] = legend
        observation["vocabulary"] = self._vocabulary()
        # NOTE: the synthesized `lexicon` dump (~70 word ids, ~680 chars/turn) is
        # intentionally NOT emitted. It existed for the optional `proposals.lexicon`
        # channel (advisory, never changes the move); for a small model it was pure
        # noise drowning the objects/effects. Re-add behind a budget gate if the
        # proposer ever starts composing useful lexicon proposals.

        # recent_actions (anti-no-op / anti-loop): the last few played moves +
        # whether the board changed. Omitted when empty (optional schema field).
        recent = self._recent_actions_for_obs()
        if recent:
            observation["recent_actions"] = recent

        # last_note (rolling scratchpad echoed back verbatim).
        observation["last_note"] = self._last_note
        return observation

    def _controllable_centroid(
        self, objects_map: Mapping[str, Any]
    ) -> Optional[Tuple[int, int]]:
        """The grounded controllable's (row, col) centroid (the anchor for object
        `relative` offsets), or ``None`` when there is no controllable. Deterministic."""
        refs = objects_map.get(self._CONTROLLABLE_ROLE)
        if not refs:
            return None
        # Smallest-handle controllable (deterministic; there is usually one).
        ref = sorted(refs, key=lambda r: getattr(r, "handle", ""))[0]
        return self._centroid_rc(ref)

    def _observation_objects(
        self,
        objects_map: Mapping[str, Any],
        controllable_rc: Optional[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        """The bounded, semantic ``objects`` list (cap ~6): controllable + target(s)
        first, then other role-bearing objects. Each carries ref/role/color/shape/
        position/relative/flags; ALL spatial facts are (row, col). Deterministic
        (roles in a stable priority then alpha order; handles sorted)."""
        out: List[Dict[str, Any]] = []
        seen: set = set()

        def emit(role: str, ref: Any) -> None:
            rc = self._centroid_rc(ref)
            entry: Dict[str, Any] = {
                "ref": getattr(ref, "handle", "?"),
                "role": role if role in _SCHEMA_ROLES else "other",
                "position": {"row": rc[0], "col": rc[1]} if rc else {"row": 0, "col": 0},
            }
            color = self._ref_color(ref)
            if color is not None:
                entry["color"] = color
            shape = self._ref_shape(ref)
            if shape:
                entry["shape"] = shape
            if controllable_rc is not None and rc is not None:
                entry["relative"] = "from controllable: row %+d, col %+d" % (
                    rc[0] - controllable_rc[0],
                    rc[1] - controllable_rc[1],
                )
            flags = self._ref_flags(ref)
            if flags:
                entry["flags"] = flags
            out.append(entry)

        # Priority roles first (controllable, target), then the rest alpha.
        priority = [self._CONTROLLABLE_ROLE, self._TARGET_ROLE]
        ordered_roles = priority + [
            r for r in sorted(objects_map) if r not in priority
        ]
        for role in ordered_roles:
            refs = objects_map.get(role)
            if not refs:
                continue
            for ref in sorted(refs, key=lambda r: getattr(r, "handle", "")):
                handle = getattr(ref, "handle", "?")
                if handle in seen:
                    continue
                seen.add(handle)
                emit(role, ref)
                if len(out) >= self._LLM_OBSERVATION_MAX_OBJECTS:
                    return out
        return out

    @staticmethod
    def _centroid_rc(ref: Any) -> Optional[Tuple[int, int]]:
        """The integer (row, col) centroid of an ObjectRef footprint, or ``None``."""
        geom = getattr(ref, "geometry", None)
        cells = list(getattr(geom, "cells", None) or [])
        if not cells:
            return None
        rows = sum(int(r) for (r, _c) in cells)
        cols = sum(int(c) for (_r, c) in cells)
        n = len(cells)
        return rows // n, cols // n

    def _words_in_category(self, category: str) -> frozenset:
        """The set of lexicon word ids in ``category`` (e.g. 'color' / 'shape' /
        'behavior'), cached. Game-agnostic: derived from the loaded catalog, never
        hardcoded. Deterministic."""
        cache = self._word_category_cache
        if category not in cache:
            cache[category] = frozenset(
                w.id
                for w in getattr(self.assets.lexicon, "words", [])
                if getattr(w, "category", "") == category
            )
        return cache[category]

    def _ref_color(self, ref: Any) -> Optional[Any]:
        """The object's COLOR for the observation: a ``color``-category
        Characteristic word id if present, else ``None`` (never invented)."""
        colors = self._words_in_category("color")
        prof = getattr(ref, "profile", None)
        chars = list(getattr(prof, "characteristics", None) or [])
        for c in sorted(chars, key=lambda c: c.word_id):
            if getattr(c, "word_id", "") in colors:
                return c.word_id
        return None

    def _ref_shape(self, ref: Any) -> Optional[str]:
        """A coarse shape descriptor from a ``shape``-category Characteristic, or
        ``None`` (never invented)."""
        shapes = self._words_in_category("shape")
        prof = getattr(ref, "profile", None)
        chars = list(getattr(prof, "characteristics", None) or [])
        for c in sorted(chars, key=lambda c: c.word_id):
            if getattr(c, "word_id", "") in shapes:
                return c.word_id
        return None

    def _ref_flags(self, ref: Any) -> List[str]:
        """Salient behavior flags grounded on the object (e.g. ``marked`` /
        ``movable`` / ``clickable``), from its Profile; empty if none. Excludes the
        role axes (controllable / is_field) which are carried by ``role`` instead.
        Deterministic (sorted, deduped)."""
        behaviors = self._words_in_category("behavior")
        skip = {self._CONTROLLABLE_ROLE, "is_field"}
        prof = getattr(ref, "profile", None)
        chars = list(getattr(prof, "characteristics", None) or [])
        return sorted(
            {
                c.word_id
                for c in chars
                if getattr(c, "word_id", "") in behaviors
                and c.word_id not in skip
            }
        )

    # Max length of a goal_template summary (the handoff: "One-line: what winning
    # looks like"). A short abstract win condition, NOT the verbose goal_pattern
    # remark (which can run ~793 chars + carries move-direction words + file/line
    # citations, violating the no-direction-words-in-DATA invariant).
    _GOAL_SUMMARY_MAX = 80

    def _goal_templates(self) -> List[Dict[str, Any]]:
        """The ACTIVE goal_patterns executable this turn as ``{id, kind, summary}``
        (parsed predicate_tree present), id-ascending. Deterministic.

        ``summary`` is a SHORT one-line win condition sourced from the goal_kind's
        abstract description (goal_kinds.tsv), NOT the goal_pattern ``remark`` -- the
        remark is long (~793 chars) and carries move-direction words / file:line
        citations, which the frozen contract bars from the DATA. ALL ids are kept
        (that IS the real catalog); only the summary text shrinks."""
        out: List[Dict[str, Any]] = []
        for p in sorted(self.assets.goal_patterns, key=lambda p: p.id):
            if getattr(p, "predicate_tree", None) is None:
                continue
            # {id, kind} only -- the per-template `summary` was the goal_kind's
            # description, which is IDENTICAL across the many templates sharing a
            # kind (e.g. 11x "Bring the controllable"), so emitting it per row was
            # ~1 KB of repeated text. The `kind` id is self-descriptive and the full
            # kind set is already in `vocabulary.goal_kinds`; the per-kind meanings
            # ride once in `goal_kinds_legend` (see build_observation), not per row.
            out.append({"id": p.id, "kind": p.goal_kind})
        return out

    def _goal_kinds_legend(
        self, templates: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """``{kind: short one-line win condition}`` for the DISTINCT goal_kinds
        present in ``templates``, sourced from goal_kinds.tsv descriptions (the same
        text the old per-template ``summary`` carried, but emitted ONCE per kind).
        Deterministic (sorted keys); empty when no kinds resolve."""
        kind_desc = {k.id: k.description for k in self.assets.goal_kinds}
        kinds = sorted({t["kind"] for t in templates if t.get("kind")})
        legend: Dict[str, str] = {}
        for k in kinds:
            summary = self._short_goal_summary(kind_desc.get(k, ""))
            if summary:
                legend[k] = summary
        return legend

    @classmethod
    def _short_goal_summary(cls, description: str) -> str:
        """A short, one-line win-condition summary from a goal_kind description:
        the first clause (cut at the first sentence/paren/dash boundary), stripped
        of trailing punctuation, capped at :data:`_GOAL_SUMMARY_MAX` on a word
        boundary. Deterministic; introduces no move-direction words (the goal_kind
        phrasing is an abstract win condition)."""
        text = (description or "").strip()
        if not text:
            return ""
        for sep in (". ", " (", " -- ", " - "):
            idx = text.find(sep)
            if idx > 0:
                text = text[:idx]
                break
        text = text.rstrip(" .")
        if len(text) > cls._GOAL_SUMMARY_MAX:
            text = text[: cls._GOAL_SUMMARY_MAX].rsplit(" ", 1)[0].rstrip(" .")
        return text

    def _vocabulary(self) -> Dict[str, List[str]]:
        """The known primitive catalogs a proposal may compose from:
        ``{operators, roles, goal_kinds}`` (sorted ids). Deterministic.

        ``solver_kinds`` (50 ids) is DELIBERATELY omitted: solver SELECTION is the
        classical layer's job, not the move proposer's -- emitting the full solver
        catalog every turn was ~750 chars of noise the small model had to wade past
        to reach the 6 objects + grounded effects that actually drive the move."""
        lex = self.assets.lexicon
        operators = sorted(
            w.id
            for w in getattr(lex, "words", [])
            if (getattr(w, "impl_key", "") or "").startswith(("op_", "rel_"))
        )
        roles = sorted(self.assets.roles)
        goal_kinds = sorted(k.id for k in self.assets.goal_kinds)
        return {
            "operators": operators,
            "roles": roles,
            "goal_kinds": goal_kinds,
        }

    def _lexicon_words(self) -> List[str]:
        """The synthesized lexicon word ids (sorted) the model should reuse rather
        than invent jargon. Deterministic."""
        lex = self.assets.lexicon
        return sorted(w.id for w in getattr(lex, "words", []))

    # Max recent_actions entries surfaced to the model (bounded; the most recent
    # few are enough for the anti-no-op / anti-loop rule).
    _RECENT_ACTIONS_MAX = 3

    def _update_recent_actions(self, grid: Any) -> None:
        """Finalize the PREVIOUS turn's played move into the rolling
        ``recent_actions`` log now that its result board is visible.

        ``changed`` = did the full (unmasked) board signature differ from before the
        move (a no-op / futile move sets ``changed == false``, which the prompt's
        anti-no-op rule consumes). Bounded to :data:`_RECENT_ACTIONS_MAX`.
        Independent of the futility toggle. Deterministic (a plain board byte
        signature; no rng, no builtin ``hash``)."""
        try:
            sig = np.asarray(grid, dtype=np.int16).tobytes()
        except Exception as exc:  # noqa: BLE001 - recent-actions is a non-fatal overlay
            logger.debug("recent_actions signature skipped: %r", exc)
            return
        pending = self._pending_recent
        if pending is not None and self._prev_board_sig is not None:
            entry = dict(pending)
            entry["changed"] = sig != self._prev_board_sig
            self._recent_actions.append(entry)
            if len(self._recent_actions) > self._RECENT_ACTIONS_MAX:
                del self._recent_actions[
                    : len(self._recent_actions) - self._RECENT_ACTIONS_MAX
                ]
            self._pending_recent = None
        self._prev_board_sig = sig

    def _arm_recent_action(
        self, chosen_id: int, click_xy: Optional[Tuple[int, int]]
    ) -> None:
        """Record the move JUST committed this turn (its ``changed`` is resolved at
        the top of next turn by :meth:`_update_recent_actions`). The action name uses
        the same vocabulary as ``inputs`` (a button ``aN`` name, or ``"click"`` for a
        complex move, carrying its x/y). Deterministic; never raises."""
        try:
            if GameAction.from_id(chosen_id).is_complex():
                rec: Dict[str, Any] = {"action": "click"}
                if click_xy is not None:
                    rec["x"], rec["y"] = int(click_xy[0]), int(click_xy[1])
            else:
                rec = {"action": game_io.button_name_for_action(chosen_id)}
            self._pending_recent = rec
        except Exception as exc:  # noqa: BLE001 - non-fatal overlay
            logger.debug("arm recent_action skipped: %r", exc)
            self._pending_recent = None

    def _recent_actions_for_obs(self) -> List[Dict[str, Any]]:
        """The bounded recent_actions list for the observation (most recent last),
        each ``{action, changed, [x, y]}``. Empty until at least one move has been
        played and its result observed. Deterministic."""
        return list(self._recent_actions[-self._RECENT_ACTIONS_MAX:])

    def _action_displacements(self, legal: List[int]) -> Dict[int, set]:
        """Per legal action, the SET of distinct non-zero ``(dr, dc)`` footprint
        displacements it produced on the grounded controllable (read off the track
        history). Empty when there is no grounded controllable or no observed
        translation yet (never invents semantics). Deterministic (DP-10): reads the
        recorded displacements; no rng/hash. The shared displacement source for both
        the legacy briefing hint and the geometric ``effect`` in build_observation."""
        cid = self._controllable_id
        if cid is None:
            return {}
        history = self._track_history.get(cid, [])
        if len(history) < 2:
            return {}
        per_action: Dict[int, set] = {}
        for i in range(1, len(history)):
            action_id, obs = history[i]
            if action_id not in legal:
                continue
            shift = footprint_shift(history[i - 1][1].cells, obs.cells)
            if shift is not None and shift != (0, 0):
                per_action.setdefault(action_id, set()).add(shift)
        return per_action

    # Effect-grounding thresholds (DP-10 deterministic; no rng/hash):
    #   MIN_TRIES   -- transitions an action needs before a non-moving result is
    #                  reported as "no-op" (vs the early "unknown" null).
    #   PLURALITY   -- fraction the modal displacement must hold to set the
    #                  direction by itself (else fall back to per-axis sign consensus).
    _EFFECT_MIN_TRIES = 2
    _EFFECT_PLURALITY = 0.6

    def _action_effect(self, action_id: int) -> Optional[str]:
        """The ROBUST LLM-facing effect of ``action_id`` on the controllable, as a
        status string -- NEVER a direction word (the frozen contract: a direction
        word couples the model's choice to the button name, which a proxy test showed
        even a strong model gets wrong):

          * ``"row +d, col +d"`` -- a grounded dominant DIRECTION (unit signs). The
            controllable's translations under this action agree on a clear direction
            (modal displacement holds a plurality, else every nonzero axis-sign
            agrees). MAGNITUDE is intentionally dropped: the prompt compares the
            SIGN to position offsets, and the real step varies with walls / lane gaps
            (e.g. 10 vs 13), which the old strict single-exact-vector rule nulled.
          * ``"no-op"``        -- tried >= MIN_TRIES and the controllable NEVER
            translated (do not re-explore it to make positional progress; it may do
            something non-positional -- cross-reference recent_actions[].changed).
          * ``"inconsistent"`` -- it DID translate, but the directions contradict
            (no plurality AND mixed axis-signs) -> the control is state-dependent
            (hidden state); not a fixed vector.
          * ``None``           -- not enough evidence yet (explore to learn it).

        Reads the grounded controllable's own track history (DP-10: sorted, integer
        cell shifts; no rng/hash). ``None`` when there is no grounded controllable."""
        from collections import Counter

        cid = self._controllable_id
        if cid is None:
            return None
        history = self._track_history.get(cid, [])
        moves: "Counter[Tuple[int, int]]" = Counter()
        n_zero = 0
        for i in range(1, len(history)):
            if history[i][0] != action_id:
                continue
            shift = footprint_shift(history[i - 1][1].cells, history[i][1].cells)
            if shift is None:  # footprint deformed -> ambiguous, not counted
                continue
            if shift == (0, 0):
                n_zero += 1
            else:
                moves[shift] += 1
        if not moves:
            return "no-op" if n_zero >= self._EFFECT_MIN_TRIES else None
        direction = self._dominant_direction(moves)
        if direction is None:
            return "inconsistent"
        return "row %+d, col %+d" % direction

    @staticmethod
    def _sgn(v: int) -> int:
        return 1 if v > 0 else -1 if v < 0 else 0

    def _dominant_direction(self, moves: Any) -> Optional[Tuple[int, int]]:
        """Reduce a ``Counter`` of nonzero ``(dr, dc)`` displacements to a single
        UNIT direction ``(sgn dr, sgn dc)``, or ``None`` when the action's effect is
        self-contradictory. Two robust paths (DP-10 deterministic):

          1. PLURALITY: if the modal displacement holds >= ``_EFFECT_PLURALITY`` of
             the weight, take its signs (robust to a rare mis-track outlier).
          2. SIGN CONSENSUS: else, an axis is decided iff all its nonzero signs
             agree; if EITHER axis carries contradictory signs, the effect is not a
             fixed direction -> ``None``. (Handles "same magnitude family, no single
             mode" like {(2,0),(5,0)} -> row +.)"""
        total = sum(moves.values())
        # Path 1: modal plurality (deterministic tie-break: higher count, then
        # smaller shift tuple).
        top_shift, top_ct = sorted(moves.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        if total and top_ct / total >= self._EFFECT_PLURALITY:
            dr, dc = top_shift
            return (self._sgn(dr), self._sgn(dc))
        # Path 2: per-axis sign consensus.
        row_signs = {self._sgn(dr) for (dr, _dc) in moves if dr != 0}
        col_signs = {self._sgn(dc) for (_dr, dc) in moves if dc != 0}
        if len(row_signs) > 1 or len(col_signs) > 1:
            return None
        rr = next(iter(row_signs)) if row_signs else 0
        cc = next(iter(col_signs)) if col_signs else 0
        return None if (rr == 0 and cc == 0) else (rr, cc)

    def _action_displacement_vector(
        self, action_id: int
    ) -> Optional[Tuple[int, int]]:
        """A single robust ``(dr, dc)`` displacement WITH MAGNITUDE for navigate to
        predict the controllable's landing cell, or ``None`` when the action's effect
        is unreliable. Unlike the LLM-facing :meth:`_action_effect` (which drops
        magnitude to a unit direction), navigate needs the real step, so this returns
        the MODAL displacement -- robust to a rare mis-track outlier (plurality) and to
        a no-single-mode-but-consistent-sign family (e.g. {(2,0),(5,0)}). Returns
        ``None`` for a self-contradictory action (mixed axis-signs, no plurality) so
        navigate declines on it (same honesty as the strict rule it replaces, but far
        less trigger-happy). Reads only the grounded controllable's track history
        (DP-10: integer shifts, deterministic tie-break; no rng/hash)."""
        from collections import Counter

        cid = self._controllable_id
        if cid is None:
            return None
        history = self._track_history.get(cid, [])
        moves: "Counter[Tuple[int, int]]" = Counter()
        for i in range(1, len(history)):
            if history[i][0] != action_id:
                continue
            shift = footprint_shift(history[i - 1][1].cells, history[i][1].cells)
            if shift is not None and shift != (0, 0):
                moves[shift] += 1
        if not moves:
            return None
        total = sum(moves.values())
        top_shift, top_ct = sorted(moves.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        if total and top_ct / total >= self._EFFECT_PLURALITY:
            return top_shift
        # No clear mode: accept only if every nonzero axis-sign agrees, then use the
        # modal shift as the representative magnitude for that direction.
        row_signs = {self._sgn(dr) for (dr, _dc) in moves if dr != 0}
        col_signs = {self._sgn(dc) for (_dr, dc) in moves if dc != 0}
        if len(row_signs) > 1 or len(col_signs) > 1:
            return None
        return top_shift

    # -- navigate (GAP-1: the classical "actually solve" move) ---------------- #

    def _navigate_move(
        self,
        situation: Any,
        parse: Any,
        latest_frame: Any,
        legal: List[int],
        cur_mhash: Optional[bytes],
    ) -> Optional[int]:
        """NAVIGATE (GAP-1): pick the legal action that routes the controllable toward
        the target, or ``None`` (-> the caller falls back to the round-robin baseline).

        Two policies, strictly ordered so BFS dominates greedy and greedy dominates
        the baseline:

          1. WALL-AWARE BFS (:meth:`_navigate_bfs`): a breadth-first grid-distance field
             from the TARGET cells over WALKABLE cells (the field/background plane UNION
             the controllable's + target's own cells; everything else = walls). The
             action whose grounded effect lands the controllable on a WALKABLE cell with
             the smallest BFS-distance-to-target (strictly less than the current cell's)
             WINS. This ROUTES AROUND walls -- a corridor/maze where the target is
             straight ahead behind a wall is solved by the around path, not the
             into-the-wall greedy pick. Returns ``None`` when the walkable plane is
             unknown (no usable ``parse``), the target is unreachable, or no grounded
             action improves the BFS distance.
          2. GREEDY MANHATTAN (:meth:`_navigate_greedy`): the legacy fallback -- the
             grounded action that most reduces the straight-line Manhattan distance
             controllable->target. Used when BFS declines (e.g. field detection failed).

        ``None`` (defer to the baseline) whenever BOTH decline -- there is no
        recognized controllable/target, no robust grounded displacement (effects
        ungrounded early -> the baseline still explores to ground them), or no action
        improves either metric. A KNOWN-FUTILE action this turn (the futility
        WorkingMemory, when the toggle is on) is skipped by both policies so navigate
        never walks into a known wall / no-op.

        Determinism (DP-10): centroids are integer cell means; the BFS uses a
        deterministic FIFO queue with sorted neighbour expansion; displacements are
        read in sorted action order; every tie breaks by smallest action id. No RNG /
        builtin hash."""
        objects_map = getattr(situation, "objects", None) or {}
        controllable_rc = self._controllable_centroid(objects_map)
        target_rc = self._target_centroid(objects_map)
        if controllable_rc is None or target_rc is None:
            return None
        if not legal:
            return None
        # Early-out gate: if NO action has produced any grounded displacement yet,
        # there is nothing to navigate with (both policies would decline).
        if not self._action_displacements(legal):
            return None

        # 1. Wall-aware BFS first (strictly dominates greedy when the walkable plane
        #    is known). Guarded: a malformed parse must not break navigate.
        try:
            bfs_id = self._navigate_bfs(
                objects_map, parse, controllable_rc, target_rc, legal, cur_mhash
            )
        except Exception as exc:  # noqa: BLE001 - BFS is non-load-bearing; fall back
            logger.debug("navigate BFS skipped: %r", exc)
            bfs_id = None
        if bfs_id is not None:
            return bfs_id

        # 2. Greedy Manhattan fallback (the legacy policy; open fields + no-parse).
        return self._navigate_greedy(
            controllable_rc, target_rc, legal, cur_mhash
        )

    def _click_navigate_move(
        self,
        situation: Any,
        latest_frame: Any,
        cur_mhash: Optional[bytes],
    ) -> Optional[Tuple[int, Tuple[int, int]]]:
        """CLICK-NAVIGATE (GAP-1 for click-driven games): when a click action
        (ACTION6) is legal this turn AND one or more ``target``-role objects exist,
        return ``(action6_id, (x=col, y=row))`` clicking a TARGET CENTROID, else
        ``None`` (-> the caller falls back to the LLM proposer / baseline).

        This is the classical counterpart to directional :meth:`_navigate_move` for
        games whose ONLY usable control is a screen click. The classical side knows
        the target centroid (the ``marked`` detector surfaces it); we click THAT rather
        than letting the 1.5B LLM emit a garbage click coordinate.

        MULTIPLE targets / round-robin exploration (deterministic, no RNG): the
        distinct target centroids are sorted by ``(area asc, row, col)`` so the
        rarest/smallest candidate (the most goal-like marker) is tried first. A
        per-game index (:attr:`_click_candidate_idx`) selects the current candidate.
        ON NO-PROGRESS -- the masked board hash did NOT change since the turn we last
        clicked (``cur_mhash == self._click_last_mhash``) -- the index ADVANCES
        (round-robin) so the next distinct target is tried, until one wins. A
        candidate known-futile this turn (the futility WorkingMemory, when the toggle
        is on) is SKIPPED. Returns ``None`` when no click is legal, no target exists,
        or every candidate is known-futile.

        Determinism (DP-10): the candidate order is content-derived (sorted by area /
        row / col); the round-robin index lives on ``self`` and advances by +1 on
        no-progress; no RNG / builtin hash. Respects futility."""
        objects_map = getattr(situation, "objects", None) or {}
        targets = objects_map.get(self._TARGET_ROLE)
        if not targets:
            return None
        legal = frozenset(
            a for a in (latest_frame.available_actions or []) if a != 0
        )
        action_id = self._click_action_id(legal)
        if action_id is None:
            return None

        # Distinct target centroids, deterministically ordered (area asc, row, col):
        # the rarest/smallest marker first. Dedupe identical centroids (several refs
        # can share one cell footprint) keeping the smallest area seen.
        by_centroid: Dict[Tuple[int, int], int] = {}
        for ref in targets:
            geom = getattr(ref, "geometry", None)
            cells = list(getattr(geom, "cells", None) or [])
            if not cells:
                continue
            rc = self._centroid_rc(ref)
            if rc is None:
                continue
            area = len(cells)
            if rc not in by_centroid or area < by_centroid[rc]:
                by_centroid[rc] = area
        if not by_centroid:
            return None
        candidates = sorted(
            by_centroid, key=lambda rc: (by_centroid[rc], rc[0], rc[1])
        )

        # Round-robin: advance to the NEXT distinct candidate when the previous click
        # made no progress (the masked board hash is unchanged since we last clicked).
        # cur_mhash is None when futility is OFF -> no advance (stable single pick;
        # OFF-path determinism preserved). A reset/level change zeroes the hash chain
        # via the cold-start RESET, so a fresh board reads as progress (no advance).
        no_progress = (
            cur_mhash is not None
            and self._click_last_mhash is not None
            and cur_mhash == self._click_last_mhash
        )
        if no_progress:
            self._click_candidate_idx += 1
            # The candidate we just clicked from THIS state did nothing -> remember it
            # as known-futile-this-state so we never re-pick it while stuck here (a
            # coordinate-aware analogue of the action-id futility memory, which cannot
            # tell two clicks apart). Cleared the moment the board changes (below).
            if self._click_last_centroid is not None:
                self._click_futile_here.add(self._click_last_centroid)
        elif cur_mhash != self._click_last_mhash:
            # The board changed (progress, reset, or first click) -> the stuck set is
            # stale; clear it so each candidate is fresh on the new board.
            self._click_futile_here.clear()

        # Pick the current candidate, skipping any known-futile-here this turn. Scan at
        # most len(candidates) positions from the round-robin index so a fully-futile
        # set declines (rather than looping). Smallest-area-first order is preserved.
        n = len(candidates)
        for offset in range(n):
            idx = (self._click_candidate_idx + offset) % n
            rc = candidates[idx]
            x, y = rc[1], rc[0]  # ACTION6 takes x=col, y=row
            if not (0 <= x <= 63 and 0 <= y <= 63):
                continue
            if rc in self._click_futile_here:
                continue
            self._click_candidate_idx = idx
            self._click_last_mhash = cur_mhash
            self._click_last_centroid = rc
            return action_id, (x, y)
        return None

    def _navigate_greedy(
        self,
        controllable_rc: Tuple[int, int],
        target_rc: Tuple[int, int],
        legal: List[int],
        cur_mhash: Optional[bytes],
    ) -> Optional[int]:
        """GREEDY MANHATTAN navigate (the legacy GAP-1 policy): among legal NON-RESET
        actions with a ROBUST grounded ``(dr, dc)`` controllable displacement
        (:meth:`_action_displacement_vector` -- modal vector, declines only on a
        contradictory action), the one whose resulting centroid has the SMALLEST
        post-move Manhattan distance
        to the target, but ONLY if strictly less than the current distance; a tie
        breaks by smallest action id (DP-10). Skips known-futile actions when the
        futility toggle is on. ``None`` when no action strictly reduces the distance."""
        cur_dist = self._manhattan(controllable_rc, target_rc)
        best_id: Optional[int] = None
        best_dist: Optional[int] = None
        for action_id in sorted(legal):
            # Robust grounded displacement (modal vector; declines only on a truly
            # contradictory action -- far less trigger-happy than the old strict
            # exactly-one-vector rule that nulled varying-step / outlier actions).
            vec = self._action_displacement_vector(action_id)
            if vec is None:
                continue
            # Respect futility: never pick a known-futile (state, move) this turn.
            if (
                self._futility_on
                and cur_mhash is not None
                and self._working_memory.is_futile(cur_mhash, action_id)
            ):
                continue
            dr, dc = vec
            moved = (controllable_rc[0] + int(dr), controllable_rc[1] + int(dc))
            new_dist = self._manhattan(moved, target_rc)
            # Keep the strictly-distance-reducing survivor with the smallest new
            # distance; smallest action id breaks a tie (sorted iteration + strict
            # improve => the smallest id at the min is retained, DP-10).
            if new_dist < cur_dist and (best_dist is None or new_dist < best_dist):
                best_dist = new_dist
                best_id = action_id
        return best_id

    def _navigate_bfs(
        self,
        objects_map: Mapping[str, Any],
        parse: Any,
        controllable_rc: Tuple[int, int],
        target_rc: Tuple[int, int],
        legal: List[int],
        cur_mhash: Optional[bytes],
    ) -> Optional[int]:
        """WALL-AWARE BFS navigate: route the controllable toward the target THROUGH
        the walkable plane, around walls (the local-minima fix for maze/corridor games).

        Walkable cells = the FIELD/background leaf cells (perceive's field detection --
        the plane the avatar sits on) UNION the controllable's current cells UNION the
        target's cells; everything else (solid non-field objects) is a wall. A BFS
        distance field is computed FROM the target cells over the walkable set with
        4-connectivity, so ``dist[cell]`` = grid steps to the target (a cell behind a
        wall is unreachable = effectively +inf). For each legal NON-RESET action with a
        ROBUST grounded displacement (:meth:`_action_displacement_vector`), the
        controllable centroid is displaced; the
        resulting cell must be WALKABLE and have a FINITE BFS distance STRICTLY LESS than
        the controllable's current cell distance. The smallest such distance wins;
        smallest action id breaks a tie (DP-10).

        Returns ``None`` (-> greedy fallback) when the walkable plane cannot be derived
        (no usable ``parse`` / no field cells), the controllable's current cell is not on
        the walkable plane, the target is unreachable, or no grounded action improves the
        BFS distance. Skips known-futile actions when the futility toggle is on.

        Determinism (DP-10): the walkable/target cell sets are content-derived; the BFS
        queue is FIFO with neighbours expanded in a fixed (sorted) order; the action scan
        is sorted; the tie-break is a strict total order. No RNG / builtin hash. The BFS
        is bounded by the walkable-cell count (<= board area, e.g. 64*64 = 4096), each
        cell dequeued once -> O(cells) -- cheap even on a full 64x64 board."""
        walkable = self._walkable_cells(objects_map, parse)
        if not walkable:
            return None
        target_cells = self._role_cells(objects_map, self._TARGET_ROLE)
        if not target_cells:
            return None
        # The controllable must sit on the walkable plane for a step-wise route to make
        # sense; its centroid is the BFS source-side anchor.
        if controllable_rc not in walkable:
            return None

        dist = self._bfs_distance_field(walkable, target_cells)
        cur_d = dist.get(controllable_rc)
        if cur_d is None:  # controllable can't reach the target over walkable cells
            return None

        best_id: Optional[int] = None
        best_d: Optional[int] = None
        for action_id in sorted(legal):
            vec = self._action_displacement_vector(action_id)
            if vec is None:
                continue
            if (
                self._futility_on
                and cur_mhash is not None
                and self._working_memory.is_futile(cur_mhash, action_id)
            ):
                continue
            dr, dc = vec
            moved = (controllable_rc[0] + int(dr), controllable_rc[1] + int(dc))
            d = dist.get(moved)
            if d is None:  # lands on a wall / off-plane / unreachable cell -> skip
                continue
            if d < cur_d and (best_d is None or d < best_d):
                best_d = d
                best_id = action_id
        return best_id

    def _walkable_cells(
        self, objects_map: Mapping[str, Any], parse: Any
    ) -> FrozenSet[Tuple[int, int]]:
        """The WALKABLE cell set for this turn (the plane the avatar can occupy):
        the FIELD/background leaf cells (``parse.field_ids`` via perceive's detection)
        UNION the controllable's current cells UNION the target's cells. Everything
        else (solid non-field objects) is a wall. Returns ``frozenset()`` when no
        usable ``parse`` / no field cells are available (-> the caller falls back to
        greedy). Deterministic (set union, no RNG/hash)."""
        objects = list(getattr(parse, "objects", None) or [])
        field_ids = getattr(parse, "field_ids", None)
        if not objects or not field_ids:
            return frozenset()
        field: set = set()
        for obj in objects:
            if self._is_field(obj, parse):
                field |= set(getattr(obj, "cells", None) or ())
        if not field:
            return frozenset()
        walkable = field
        walkable |= self._role_cells(objects_map, self._CONTROLLABLE_ROLE)
        walkable |= self._role_cells(objects_map, self._TARGET_ROLE)
        return frozenset(walkable)

    def _role_cells(
        self, objects_map: Mapping[str, Any], role: str
    ) -> FrozenSet[Tuple[int, int]]:
        """The union of all footprint cells of the objects bucketed under ``role``
        in the projected situation (each ObjectRef exposes ``.geometry.cells``).
        Returns ``frozenset()`` when the role is absent. Deterministic."""
        cells: set = set()
        for ref in objects_map.get(role) or ():
            geom = getattr(ref, "geometry", None)
            cells |= set(getattr(geom, "cells", None) or ())
        return frozenset(cells)

    @staticmethod
    def _bfs_distance_field(
        walkable: FrozenSet[Tuple[int, int]],
        sources: FrozenSet[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], int]:
        """A 4-connected BFS distance field over ``walkable``, seeded at the
        ``sources`` cells (distance 0). Returns ``{cell: steps}`` for every walkable
        cell REACHABLE from a source; unreachable walkable cells (behind a wall) are
        ABSENT (a ``dict.get`` miss = +inf). Only ``sources`` that are themselves
        walkable seed the search.

        Determinism (DP-10): a FIFO queue (collections.deque) with neighbours
        expanded in a FIXED order (down, up, right, left); each cell is enqueued at
        most once, so the order is fully determined by the cell sets -- no RNG/hash.
        Bounded by ``len(walkable)`` (<= board area)."""
        from collections import deque

        dist: Dict[Tuple[int, int], int] = {}
        queue: deque = deque()
        for cell in sources:
            if cell in walkable and cell not in dist:
                dist[cell] = 0
                queue.append(cell)
        neighbours = ((1, 0), (-1, 0), (0, 1), (0, -1))
        while queue:
            r, c = queue.popleft()
            d = dist[(r, c)] + 1
            for dr, dc in neighbours:
                nxt = (r + dr, c + dc)
                if nxt in walkable and nxt not in dist:
                    dist[nxt] = d
                    queue.append(nxt)
        return dist

    def _target_centroid(
        self, objects_map: Mapping[str, Any]
    ) -> Optional[Tuple[int, int]]:
        """The recognized target's (row, col) centroid (smallest-handle target when
        several), or ``None`` when there is no target role. Deterministic."""
        refs = objects_map.get(self._TARGET_ROLE)
        if not refs:
            return None
        ref = sorted(refs, key=lambda r: getattr(r, "handle", ""))[0]
        return self._centroid_rc(ref)

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan (row, col) distance between two integer centroids."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # -- baseline action policy (DP-10: depends only on counter + available) --- #

    def _baseline_action_id(self, latest_frame: Any) -> Optional[int]:
        """The deterministic OBSERVABLE BASELINE move id, or ``None`` if the only
        legal action is RESET.

        Policy: cycle the legal non-RESET actions by ``action_counter`` (a
        deterministic round-robin = curiosity / coverage). Same
        ``(available_actions, action_counter)`` -> same id; no RNG, no hash."""
        legal = [a for a in (latest_frame.available_actions or []) if a != 0]
        if not legal:
            return None
        return legal[self.action_counter % len(legal)]

    def _baseline_click_xy(self, grid: Any) -> tuple:
        """A deterministic (x, y) for an ACTION6 click that CYCLES coverage over a
        coarse grid of board cells (DP-10: derived from ``action_counter``, no RNG).

        Scheme: walk a fixed COARSE x COARSE lattice of board cells in row-major
        order, one lattice node per turn, wrapping every COARSE*COARSE turns. The
        node ``k = action_counter`` maps to lattice cell
        ``(col=k % COARSE, row=(k // COARSE) % COARSE)`` and is scaled to board
        coordinates by ``step = board_size // COARSE`` and offset to the cell
        centre. GameIO clamps the result into ``0..63`` regardless."""
        height = int(grid.shape[0]) if hasattr(grid, "shape") else 64
        width = int(grid.shape[1]) if hasattr(grid, "shape") else 64
        coarse = 8  # 8x8 lattice over the board (64 distinct click targets)
        node = self.action_counter
        col_index = node % coarse
        row_index = (node // coarse) % coarse
        step_x = max(1, width // coarse)
        step_y = max(1, height // coarse)
        x = col_index * step_x + step_x // 2
        y = row_index * step_y + step_y // 2
        return x, y

    def _safe_fallback(self, latest_frame: Any) -> GameAction:
        """A safe legal action for the outer backstop: the first legal non-RESET
        action if one exists (built defensively without touching the core), else
        RESET. Never raises."""
        try:
            legal = [a for a in (latest_frame.available_actions or []) if a != 0]
            if legal:
                return game_io.move_to_action(
                    legal[0], x=0, y=0, game_id=getattr(latest_frame, "game_id", "")
                )
        except Exception:  # noqa: BLE001 - the fallback itself must never raise
            pass
        return GameAction.RESET

    # -- connected read components (goal-id + world summary) ---------------- #

    def _df(self, stage: str, callee: str, input_summary: Any, output_summary: Any) -> None:
        """Record one inter-component dataflow event (inert when observability is
        OFF). Guarded so logging can NEVER break the play loop."""
        if not self._observe_on:
            return
        try:
            self._dataflow.record(stage, callee, input_summary, output_summary)
        except Exception as exc:  # noqa: BLE001 - logging must never break the loop
            logger.debug("dataflow record skipped: %r", exc)

    def _compute_read_components(self, situation: Any) -> None:
        """Identify the current Goal (goal 同定) + summarize the WorldModel state
        for THIS turn, stashing them on ``self._goal_id`` / ``self._world_summary``
        for the briefing + observability overlay. Pure read; NEVER changes the
        committed action. Guarded (a non-fatal overlay must not break the loop)."""
        try:
            self._goal_id = identify_goal(
                situation, self.assets.goal_patterns, self.assets.lexicon
            )
            self._df(
                "goal-id",
                "identify_goal",
                "roles %s" % sorted(getattr(situation, "objects", {}) or {}),
                (
                    "%s d=%s"
                    % (self._goal_id.pattern.id, round(self._goal_id.distance, 3))
                    if self._goal_id is not None
                    else "none"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - goal-id is a non-fatal read overlay
            logger.debug("goal identification skipped: %r", exc)
            self._goal_id = None
        try:
            self._world_summary = summarize_world(
                self.world,
                affordance_evidence=self._affordance_evidence,
                controllable_id=self._controllable_id,
            )
            self._df(
                "world",
                "summarize_world",
                "situation -> rules/affordances",
                "%d rule(s), %d affordance(s)"
                % (
                    self._world_summary.get("rule_count", 0),
                    len(self._world_summary.get("affordances", {})),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - world summary is a non-fatal overlay
            logger.debug("world summary skipped: %r", exc)
            self._world_summary = None

    # -- capture-only solve leg --------------------------------------------- #

    def _solve_for_capture(self, situation: Any, latest_frame: Any):
        """Run the goal/solver selection purely to enrich the capture (descriptor
        only). Picks the first active GoalPattern with a parsed predicate tree,
        wraps it in a Goal, and asks SelectSolver for a plan. Returns ``None`` if
        no active pattern is available."""
        pattern = self._first_active_pattern()
        if pattern is None:
            return None
        goal_obj = Goal(predicate=pattern.predicate_tree)
        ctx = SolverContext(
            situation=situation,
            world=self.world,
            goal=goal_obj,
            moves=tuple(latest_frame.available_actions or ()),
            consult=self.generator,
        )
        return self.select.solve(ctx)

    def _first_active_pattern(self):
        """The first GoalPattern (id-ascending) whose ``predicate_tree`` is parsed
        (an ``active`` row), or ``None``."""
        patterns = sorted(self.assets.goal_patterns, key=lambda p: p.id)
        for pattern in patterns:
            if pattern.predicate_tree is not None:
                return pattern
        return None

    # -- controllable-object identification --------------------------------- #

    # The canonical "self" role label (agent/assets/roles.tsv: self/controllable).
    _CONTROLLABLE_ROLE = "controllable"
    # Min translate_support an object must clear to be the controllable.
    _CONTROLLABLE_TRANSLATE_THRESHOLD = 0.5
    # Bounded per-handle history window (keeps a long game's memory finite).
    _HISTORY_WINDOW = 12

    # Confidence stamped on the grounded controllable Characteristic (a definite
    # motion-grounded pick reads as full presence; has() default floor is 0.0).
    _CONTROLLABLE_CONFIDENCE = 1.0

    def _mark_controllable(self, parse: Any) -> None:
        """Stamp the ``controllable`` Characteristic onto the grounded pick's
        Profile so the canonical ``recognized_by = has(controllable)`` predicate
        tests true (FR-168: the motion pick stays the real decision; this only
        surfaces it on the predicate path). Idempotent per turn — the Profile is
        freshly rebuilt by detect_features each turn, so we never double-stamp.
        Guarded: a non-fatal overlay must never break the loop."""
        if self._controllable_id is None:
            return
        try:
            for obj in parse.objects:
                if obj.id != self._controllable_id:
                    continue
                if obj.profile.of(self._CONTROLLABLE_ROLE) is None:
                    obj.profile.characteristics.append(
                        Characteristic(
                            self._CONTROLLABLE_ROLE, 1.0, self._CONTROLLABLE_CONFIDENCE
                        )
                    )
                break
        except Exception as exc:  # noqa: BLE001 - non-fatal overlay
            logger.debug("controllable mark skipped: %r", exc)

    def _classify_roles(self, salient: List[Any]) -> Dict[str, str]:
        """The AnalogizeRoles pre-pass injected into StateAbstraction (FR-R-1):
        classify the salient objects into functional roles by evaluating each
        roles.tsv ``recognized_by`` predicate over them (controllable via
        has(controllable); field via has(static)/has(is_field); target via
        inside(self, box)). Any object no predicate claims falls through to
        ``default_role_of`` (the situation default), so bucketing is identical to
        a default StateAbstraction for un-classified objects (no regression).

        Reconcile the controllable stamp FIRST (no-regression of the pick): the
        salient set can include STALE tracker objects (a handle absent this frame
        keeps its last GameObject, still ``visible``, and that GameObject may carry a
        ``controllable`` Characteristic stamped on a PAST turn when it was the pick).
        The grounded pick is the single source of truth (FR-168), so strip the stamp
        from every salient object whose id is not the current pick before
        classifying -- this makes ``has(controllable)`` true for EXACTLY the pick,
        matching the retired id-keyed role_of override."""
        for obj in salient:
            if obj.id != self._controllable_id:
                self._strip_controllable(obj)
        return classify_roles(
            salient,
            self.assets.roles,
            self.assets.lexicon,
            default_role_of=default_role_of,
        )

    def _strip_controllable(self, obj: Any) -> None:
        """Remove a stale ``controllable`` Characteristic from ``obj``'s Profile
        (a past-pick stamp persisted by the ObjectTracker). Idempotent / guarded."""
        try:
            chars = obj.profile.characteristics
            kept = [c for c in chars if c.word_id != self._CONTROLLABLE_ROLE]
            if len(kept) != len(chars):
                obj.profile.characteristics = kept
        except Exception as exc:  # noqa: BLE001 - non-fatal overlay
            logger.debug("controllable strip skipped: %r", exc)

    def _identify_controllable(self, parse: Any, grid: Any) -> None:
        """Update :attr:`_controllable_id` from the per-object motion trajectory.

        Appends this turn's per-object :class:`Observation` (footprint +
        colour-counts) to :attr:`_track_history`, keyed on the (action_id) that
        PRECEDED this frame (``self.last_action_id``; a ``-1`` sentinel for the
        first / post-RESET frame keeps the tuple type ``(int, Observation)``). Once
        any track has >= 2 transitions (>= 3 entries), it runs
        ``world_model.affordance_evidence`` over the whole history and picks the
        non-field candidate maximizing ``(translate_support, distinct_action_count)``
        with ``translate_support >= 0.5`` — EXCLUDING an autonomous drifter that
        moved the SAME way under every action (it is a self-mover, not the
        controllable object).

        Determinism (DP-10): candidate ids are visited in sorted order and ties
        break by the SMALLEST id; no RNG, no builtin ``hash``."""
        action_id = self.last_action_id if self.last_action_id is not None else -1

        # Append this turn's observation per parsed object, then bound the window.
        this_turn_handles = set()
        for obj in parse.objects:
            obs = Observation(
                handle=obj.id,
                cells=frozenset(obj.cells),
                color_counts=self._color_counts(obj.cells, grid),
            )
            history = self._track_history.setdefault(obj.id, [])
            history.append((action_id, obs))
            if len(history) > self._HISTORY_WINDOW:
                del history[: len(history) - self._HISTORY_WINDOW]
            this_turn_handles.add(obj.id)

        # Drop a stale controllable absent for THIS turn and the previous one (re-id).
        if (
            self._controllable_id is not None
            and self._controllable_id not in this_turn_handles
            and self._controllable_id not in self._prev_turn_handles
        ):
            self._controllable_id = None

        # Need at least one track with >= 2 transitions (>= 3 entries) overall.
        have_transitions = any(
            len(h) >= 3 for h in self._track_history.values()
        )
        if have_transitions:
            evidence = affordance_evidence(self._track_history)
            # Expose this turn's evidence to detect_features (step 3b).
            self._affordance_evidence = dict(evidence)
            # Candidates: objects parsed THIS turn that are NOT the field.
            candidate_ids = sorted(
                obj.id
                for obj in parse.objects
                if not self._is_field(obj, parse)
            )
            # Footprints parsed THIS turn (for the single-connected-component gate).
            this_turn_cells = {
                obj.id: frozenset(obj.cells) for obj in parse.objects
            }
            best_id = None
            best_key: Optional[Tuple[int, float, int]] = None
            for cid in candidate_ids:
                aff = evidence.get(cid)
                if (
                    aff is None
                    or aff.translate_support < self._CONTROLLABLE_TRANSLATE_THRESHOLD
                ):
                    continue
                distinct = self._distinct_action_count(cid)
                # Autonomous-drifter guard: a self-mover that drifted the SAME way
                # under EVERY action it was seen with is action-independent motion,
                # not control. Skip it (it has one displacement vector across >= 2
                # distinct actions).
                if aff.autonomous and self._single_displacement(cid):
                    continue
                # CM-3 FLICKER GUARD: a real controllable (e.g. a cursor) is a
                # COHERENT, PERSISTENT RIGID UNIT -- it keeps a STABLE footprint size
                # and is a SINGLE connected component, translating cleanly. ACTION1/2
                # edits that toggle colour cells board-wide make many fragments LOOK
                # like they translate (noisy translate_support) but they vary in size
                # / split into pieces. `coherent` is 1 for a stable-size single-blob
                # mover, 0 otherwise; it is the PRIMARY ranking key so a clean unit
                # outranks a noisy fragment of equal translate_support. Deterministic
                # (reads only footprints; no rng/hash, DP-10).
                coherent = (
                    1
                    if (
                        self._stable_footprint_size(cid)
                        and self._single_component(this_turn_cells.get(cid, frozenset()))
                    )
                    else 0
                )
                # Rank by (coherent, translate_support, distinct_action_count); on a
                # tie the SMALLEST id wins (candidate_ids ascending + strict-improve
                # replace => the first/smallest id at the max is retained, DP-10).
                key = (coherent, aff.translate_support, distinct)
                if best_key is None or key > best_key:
                    best_key = key
                    best_id = cid
            if best_id is not None:
                self._controllable_id = best_id

        self._prev_turn_handles = frozenset(this_turn_handles)

    def _stable_footprint_size(self, handle: str) -> bool:
        """True iff ``handle``'s footprint CARDINALITY is stable across the frames
        where it was present (a coherent rigid unit keeps its cell count; a
        flickering colour-toggle fragment gains/loses cells). Tolerates frames
        where it was absent (occlusion). With < 2 present frames there is no
        instability evidence, so it is treated as stable (True). Deterministic; no
        rng/hash (DP-10)."""
        history = self._track_history.get(handle, [])
        sizes = [len(obs.cells) for (_a, obs) in history if obs.cells]
        if len(sizes) < 2:
            return True
        return min(sizes) == max(sizes)

    @staticmethod
    def _single_component(cells: frozenset) -> bool:
        """True iff ``cells`` form ONE 4-connected component (a coherent blob), or
        the set is empty/singleton (vacuously coherent). A controllable cursor is a
        single connected unit; scattered toggle cells are not. Deterministic
        flood-fill from the min cell; no rng/hash (DP-10)."""
        if len(cells) <= 1:
            return True
        cells = frozenset(cells)
        start = min(cells)
        seen = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r + dr, c + dc)
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return len(seen) == len(cells)

    def _distinct_action_count(self, handle: str) -> int:
        """Number of DISTINCT action ids under which ``handle`` actually translated
        (a non-zero rigid footprint shift). Deterministic; reads the same history
        ``affordance_evidence`` reads."""
        return len(self._translating_actions(handle))

    def _single_displacement(self, handle: str) -> bool:
        """True iff ``handle`` exhibited exactly ONE distinct non-zero displacement
        vector across its whole history (the autonomous-drifter signature: it moved
        the same way regardless of the action). False if it never moved or moved in
        >= 2 distinct directions."""
        displacements = set()
        history = self._track_history.get(handle, [])
        for i in range(1, len(history)):
            shift = footprint_shift(history[i - 1][1].cells, history[i][1].cells)
            if shift is not None and shift != (0, 0):
                displacements.add(shift)
        return len(displacements) == 1

    def _translating_actions(self, handle: str) -> set:
        """The set of action ids under which ``handle`` translated (non-zero rigid
        shift) at least once across its history."""
        actions = set()
        history = self._track_history.get(handle, [])
        for i in range(1, len(history)):
            action_id, obs = history[i]
            shift = footprint_shift(history[i - 1][1].cells, obs.cells)
            if shift is not None and shift != (0, 0):
                actions.add(action_id)
        return actions

    # -- perception helpers ------------------------------------------------- #

    @staticmethod
    def _color_counts(cells, grid) -> dict:
        """Per-colour cell counts of ``cells`` over ``grid`` (the FeatureContext
        colour histogram). Deterministic."""
        counts: dict = {}
        for (row, col) in cells:
            color = int(grid[row, col])
            counts[color] = counts.get(color, 0) + 1
        return counts

    @staticmethod
    def _is_field(obj, parse) -> bool:
        """Whether ``obj`` is the is_field leaf (its int leaf id is in
        ``parse.field_ids``). ``obj.id`` is ``obj_%04d`` so the leaf id is the
        integer suffix."""
        try:
            leaf_id = int(obj.id[4:])
        except (ValueError, IndexError):
            return False
        return leaf_id in parse.field_ids

    # GOAL-MARKER threshold: a colour is RARE if its total non-field cell count is
    # at most this FRACTION of all non-field cells. Board-DERIVED (a fraction, not a
    # game literal): conservative so a HUD/legend glyph that happens to be small but
    # whose colour tiles a large strip is NOT marked, while a tiny isolated goal dot
    # is. The component-count escape (<= this many connected components of the
    # colour) catches a rare colour used by only one or two distinct objects even if
    # those objects are not minuscule. Both are board-relative, no magic per game.
    _MARKED_RARE_AREA_FRACTION = 0.05
    _MARKED_RARE_MAX_COMPONENTS = 2

    def _marked_object_ids(self, parse: Any, grid: Any) -> FrozenSet[str]:
        """The ids of top-level objects that are GOAL-MARKERS by the generalizing
        rare-colour rule (no game literal): a NON-FIELD object whose DOMINANT colour
        is rare over the board -- its colour's total NON-FIELD cell area is at most
        ``_MARKED_RARE_AREA_FRACTION`` of all non-field cells, OR that colour appears
        in at most ``_MARKED_RARE_MAX_COMPONENTS`` top-level non-field objects --
        EXCEPT the single most-common non-field colour (the foreground bulk), which
        is never a marker (it guards the component escape from firing when the board
        has only a couple of non-field objects).

        Board context (the per-colour non-field census) is computed ONCE here, then
        each object is tested against it. Deterministic (DP-10): sorted/stable
        iteration, no RNG / builtin hash. Returns ``frozenset()`` if there are no
        non-field cells (total-function guard) -- nothing is marked.

        The controllable is NOT excluded here on purpose: it gets `controllable` in
        Wave A and `target` is a Wave-C predicate over the REMAINING objects, so a
        rare-coloured avatar still won't be claimed as target. Field objects ARE
        excluded (a marker is a foreground signal, not the background plane)."""
        objects = list(getattr(parse, "objects", []) or [])
        if not objects:
            return frozenset()

        # Per-object dominant colour + non-field flag (one pass, stable order).
        dom: Dict[str, Optional[int]] = {}
        non_field: Dict[str, bool] = {}
        for obj in objects:
            non_field[obj.id] = not self._is_field(obj, parse)
            dom[obj.id] = attributes.dominant_color(self._color_counts(obj.cells, grid))

        # Board census over NON-FIELD objects only: per-colour total cell area and
        # the count of distinct non-field objects (components) carrying that colour.
        color_area: Dict[int, int] = {}
        color_components: Dict[int, int] = {}
        for obj in objects:
            if not non_field[obj.id]:
                continue
            color = dom[obj.id]
            if color is None:
                continue
            color_area[color] = color_area.get(color, 0) + len(obj.cells)
            color_components[color] = color_components.get(color, 0) + 1

        total_non_field = sum(color_area.values())
        if total_non_field <= 0:
            return frozenset()
        area_threshold = self._MARKED_RARE_AREA_FRACTION * total_non_field

        # The single most-common non-field colour (largest area; ties -> lowest
        # index, DP-10) is the foreground "bulk" -- it is NEVER a marker regardless
        # of its component count. This guards the component-count escape from
        # firing on an area-DOMINANT colour when the board has only a couple of
        # non-field objects (then every colour trivially has <= 2 components): a
        # colour covering most of the foreground is not rare. The component escape
        # then only catches a genuinely small-footprint colour used by 1-2 objects.
        bulk_color = min(color_area, key=lambda c: (-color_area[c], c))

        def _is_rare(color: int) -> bool:
            if color == bulk_color:
                return False  # the foreground bulk is never a marker
            return (
                color_area[color] <= area_threshold
                or color_components[color] <= self._MARKED_RARE_MAX_COMPONENTS
            )

        marked = {
            obj.id
            for obj in objects
            if non_field[obj.id]
            and dom[obj.id] is not None
            and _is_rare(dom[obj.id])
        }
        return frozenset(marked)

    # -- observability record (ARC_DATAFLOW) -------------------------------- #

    def _observability_record(self, situation: Any) -> Dict[str, Any]:
        """Build the per-turn observability overlay (ARC_DATAFLOW ON): the three
        views the inspector renders. Pure read + deterministic; guarded fields so a
        malformed input never raises (the caller is itself inside _emit_capture's
        try/except, but keep this total too).

        1. ``dataflow``   — the ordered inter-component stage events.
        2. ``lexicon``    — the Lexicon size + Words + delta vs last turn.
        3. ``verbalize``  — World / per-object / Goal NL rendering (canonical)."""
        # Lexicon growth (delta vs the previous turn's snapshot), then advance.
        growth = lexicon_growth(self.assets.lexicon, self._prev_lexicon_words)
        self._prev_lexicon_words = tuple(growth.get("words", ()))

        world_line = (
            verbalize_world(self._world_summary, situation)
            if self._world_summary is not None
            else "world summary unavailable."
        )
        return {
            "dataflow": self._dataflow.events(),
            "lexicon": growth,
            "verbalize": {
                "world": world_line,
                "goal": verbalize_goal(self._goal_id),
                "objects": verbalize_objects(situation),
            },
            "goal": (
                {
                    "pattern": self._goal_id.pattern.id,
                    "goal_kind": self._goal_id.pattern.goal_kind,
                    "predicate": self._goal_id.pattern.predicate,
                    "distance": round(float(self._goal_id.distance), 3),
                    "satisfied": self._goal_id.satisfied,
                }
                if self._goal_id is not None
                else None
            ),
            "world_summary": self._world_summary,
        }

    # -- capture emission (tools/inspector sink) ---------------------------- #

    def _emit_capture(
        self,
        grid: Any,
        parse: Any,
        situation: Any,
        chosen_id: int,
        latest_frame: Any,
        solver_id: Optional[str],
    ) -> None:
        """Emit one JSON line per turn into the ARC_INTROSPECT sink (inert if no
        writer). Carries the capture-schema CORE fields (t / level / grid /
        action) plus v14 overlays under distinct keys (parse_objects /
        situation_hash / solver). A tool-input sink, never a report; any failure
        is swallowed so capture can never break the play loop."""
        # Invalidate the row FIRST: it is stamped only after a successful write, so a
        # skipped/failed capture (or capture disabled) leaves it None and the paired
        # TurnRecord emits capture_ref.row=null rather than silently pointing at the
        # PREVIOUS turn's frame (H1 lockstep fix).
        self._last_capture_row = None
        if self._capture_writer is None:
            return
        try:
            situation_hash = situation.hash() if hasattr(situation, "hash") else None
            record = {
                # capture-schema CORE (held-out contract: t / level / grid / action).
                "t": self.action_counter,
                "level": latest_frame.levels_completed,
                "grid": grid.tolist(),
                "action": chosen_id,
                # v14 overlays (distinct keys; optional per the schema).
                "parse_objects": [
                    {"id": o.id, "cells": sorted(o.cells)} for o in parse.objects
                ],
                "situation_hash": situation_hash,
                "solver": solver_id,
                # GAP-2/GAP-3: the identified controllable object id (or null).
                # capture-schema permits an optional overlay keyed off the Core four.
                "controllable_id": self._controllable_id,
                # API-04: which leg chose this move ("llm" | "baseline"). A replay
                # then shows exactly when the LLM acted. Always "baseline" when the
                # proposer is OFF (default), so this overlay is inert-stable.
                "move_source": self._last_move_source,
            }
            # OBSERVABILITY overlay (ARC_DATAFLOW ON only). Three views: the inter-
            # component dataflow log, the Lexicon-growth snapshot, and the canonical
            # World/object/Goal verbalization. Folded under a single "observe" key so
            # the existing capture-schema fields are untouched (OFF -> absent).
            if self._observe_on:
                record["observe"] = self._observability_record(situation)
            self._capture_writer.write(json.dumps(record) + "\n")
            self._capture_writer.flush()
            # Stamp the row this line landed on (the 0-based capture LINE index) so
            # the paired TurnRecord's capture_ref.row points at the right line, then
            # advance. RESET turns write NO capture line, so this counter (NOT the
            # action_counter) is the correct join key.
            self._last_capture_row = self._capture_row
            self._capture_row += 1
        except Exception as exc:  # noqa: BLE001 - capture must never break the loop
            logger.debug("capture emit skipped: %r", exc)

    # -- trace emission (CMP-37 / FR-C-12 -- the PAIRED decision trace) ------ #

    def _emit_trace(
        self,
        grid: Any,
        parse: Any,
        situation: Any,
        plan: Any,
        chosen_id: int,
        latest_frame: Any,
    ) -> None:
        """Emit one TurnRecord (trace-schema.md v1.0) into the ARC_TRACE sink, in
        LOCKSTEP with ``_emit_capture`` (inert if no writer). game_io owns the
        serialization (CMP-37 adapter); THIS method owns the data + the file IO and
        hands plain VALUES to ``game_io.turn_record`` (game_io never reaches into a
        situation/world). Fully guarded -- like ``_emit_capture``, the trace can
        NEVER break the play loop (load-bearing backstop).

        ``capture_ref.row`` is ``self._last_capture_row`` (the row the just-written
        capture line landed on -- NOT the action_counter, which diverges on a
        RESET-skip). ``move_effect`` / ``goal_predicate`` are null by GAP (F2 /
        GAP-3); ``game_plan.moves`` is [] (GAP-1 descriptor-only)."""
        if self._trace_writer is None:
            return
        try:
            # Lazy header on the first turn: game_id is per-frame, so we know the
            # game tag + capture basename only now (recommended simplest-correct).
            if not self._trace_header_written:
                header = game_io.trace_header(
                    game=getattr(latest_frame, "game_id", "") or "",
                    capture_file=self._capture_basename,
                )
                self._trace_writer.write(json.dumps(header) + "\n")
                self._trace_writer.flush()
                self._trace_header_written = True

            # Pre-compute per-object colour ids + shape bases here (needs the grid +
            # agent.core.attributes, which game_io -- the adapter -- must not touch).
            color_ids: dict = {}
            shape_bases: dict = {}
            for obj in parse.objects:
                counts = self._color_counts(obj.cells, grid)
                color_ids[obj.id] = attributes.dominant_color(counts)
                shape_bases[obj.id] = attributes.shape_base(obj.cells)

            situation_hash = situation.hash() if hasattr(situation, "hash") else None

            # game_plan (GAP-1: descriptor-only, moves=[]). Read the chosen Solver's
            # id + verification_horizon straight off the plan; null when no plan.
            game_plan = None
            chosen_solver = plan.chosen if plan is not None else None
            if chosen_solver is not None:
                game_plan = {
                    "solver": chosen_solver.id,
                    "moves": [],
                    "horizon": chosen_solver.verification_horizon,
                }

            height = int(grid.shape[0]) if hasattr(grid, "shape") else 64
            width = int(grid.shape[1]) if hasattr(grid, "shape") else 64

            # Per-object role map (handle -> role) from the projected situation:
            # situation.objects is role -> frozenset[ObjectRef], so invert it to
            # ref.handle -> role. This carries the controllable assignment (and any
            # default-bucketed role) into each GameObject's trace `role` field --
            # the trace-schema contract has NO top-level controllable/avatar field;
            # the per-object role is the single source of truth.
            object_roles: Dict[str, str] = {}
            for role, refs in (getattr(situation, "objects", None) or {}).items():
                for ref in refs:
                    object_roles[ref.handle] = role

            record = game_io.turn_record(
                turn=self.action_counter,
                level=latest_frame.levels_completed,
                capture_file=self._capture_basename,
                capture_row=self._last_capture_row,
                game_objects=list(parse.objects),
                object_color_ids=color_ids,
                object_shape_bases=shape_bases,
                object_roles=object_roles,
                situation=situation,
                situation_hash=situation_hash,
                rule_count=len(self.world.rules),
                goal_predicate=None,  # GAP-3: no real goal SELECTION (honest null).
                game_plan=game_plan,
                move_id=chosen_id,
                move_name=GameAction.from_id(chosen_id).name,
                grid_height=height,
                grid_width=width,
            )
            # API-04: stamp the move's source ("llm" | "baseline") onto the
            # game_move map so a replayed TurnRecord shows when the LLM acted.
            # Additive only (game_io owns the base schema; we annotate post-build),
            # so the trace-schema contract stays intact and "baseline" when OFF.
            if isinstance(record.get("game_move"), dict):
                record["game_move"]["source"] = self._last_move_source
            self._trace_writer.write(json.dumps(record) + "\n")
            self._trace_writer.flush()
        except Exception as exc:  # noqa: BLE001 - trace must never break the loop
            logger.debug("trace emit skipped: %r", exc)
