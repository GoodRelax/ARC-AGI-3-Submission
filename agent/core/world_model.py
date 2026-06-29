"""v14 World-View dynamics model — entities and values (agent.core.world_model).

A clean-room port of the canonical domain model's *world* facet
(``docs/StrictDoc-specs/_assets/gr-arc-3-domain-model.json`` v031):

    WorldModel{rules, abstraction, representation, confidence}   (entity)
        .predict(situation, move) -> (AbstractSituation, Outcome?)
        .classify(before, after, goal_distance) -> MoveEffect
    InteractionRule{trigger, effect: EffectSignature[*], confidence} (entity)
    EffectSignature{driver, target_type, feature, operator, parameter_kind} (value)
    Outcome{state, score, is_full_reset, level}                  (value)
    Observation{handle, cells, color_counts}                     (value)

Controlled vocabularies (MoveEffect / OutcomeState / Driver / TransformOperator /
ParameterKind / TrackingState / Representation) are module-level ``frozenset`` of
plain strings —
NOT salted Enums — so equality / logging / memo keys are deterministic across
processes (DP-10: no RNG, no builtin ``hash()`` of mutable state for identity).

This module is game-literal-free (NFR-6): every typed handle is a Role label, a
``Word.id`` axis, or a TransformOperator kind — never a colour number, a
coordinate, or a glyph. The EffectSignature 5-tuple is the surface-free
recognition/transfer key (verbalization §5.2, TS-25).

Mirrors the style of ``agent.core.model`` (frozen ``@dataclass`` values, plain
``@dataclass`` entities, controlled vocabularies as frozensets). Collaborators
that belong to step 6-3.6 (AbstractSituation / GameObject / ObjectTracker /
StateAbstraction) are declared here only as ``typing.Protocol`` stubs — the real
classes live elsewhere; tests pass fakes that satisfy the protocols.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from agent.core.model import WorldMap

# --------------------------------------------------------------------------- #
# Controlled vocabularies (frozensets of stable string labels — DP-10).
# Each value carries the bare string; the frozenset is the membership oracle.
# --------------------------------------------------------------------------- #

# MoveEffect — one move's effect on the salient state (3 values).
# futile == not progress (invariant=frame unchanged / no_progress=changed but
# the milestone did not advance).
MOVE_EFFECTS = frozenset({"invariant", "no_progress", "progress"})

# OutcomeState — one turn's observed/predicted terminal axis (env signal).
OUTCOME_STATES = frozenset({"ongoing", "win", "over"})

# Driver — attribution of *what* changed an axis.
#   direct      = the GameMove itself
#   indirect    = a contact / Relation (e.g. overlap with another object)
#   independent = a clock / phase (action-agnostic)
DRIVERS = frozenset({"direct", "indirect", "independent"})

# TransformOperator — the typed verb an EffectSignature applies to one axis.
TRANSFORM_OPERATORS = frozenset(
    {"translate", "rotate", "scale", "recolor", "cycle", "set"}
)

# ParameterKind — the KIND of an effect's parameter (never the concrete value).
# Provisional: pending §5d fit-signature canonicalization (the fit loop may refine
# or rename these kinds). induce_signature only ever produces a kind in this set.
PARAMETER_KINDS = frozenset({"vector", "angle", "factor", "index", "value"})

# TrackingState — a tracked object's visibility (fog verbalization).
TRACKING_STATES = frozenset({"visible", "remembered", "unknown"})

# Representation — the WorldModel's internal representation switch.
REPRESENTATIONS = frozenset({"Symbolic", "Learned", "Graph"})


def futile(effect: str) -> bool:
    """True iff ``effect`` is NOT progress (i.e. ``invariant`` or ``no_progress``).

    Both non-progress MoveEffect labels are futile (planner hard-prunes
    ``invariant``; ``no_progress`` gets one try). This is the TS-12 futility
    predicate.
    """
    return effect != "progress"


# --------------------------------------------------------------------------- #
# Depend-on-contracts (Protocol stubs — the REAL classes belong to step 6-3.6).
# Declared structurally so this module never imports its collaborators; tests
# supply fakes that satisfy these shapes.
# --------------------------------------------------------------------------- #

@runtime_checkable
class AbstractSituation(Protocol):
    """The abstract board used for search/prediction (domain «value»).

    A «value»: it is compared by value-equality (``==`` / ``__eq__``), not by a
    ``canonical()`` method. Only the surface this module needs is declared:
      * ``scalar(name)`` -> the VALUE component of a Markov scalar gauge (e.g.
        move_budget, a health gauge) as a plain number, or ``None`` if absent —
        the loss-trigger substrate. It returns the VALUE, NOT the ``(value, cap)``
        pair: the real step-6-3.6 AbstractSituation stores a ``(value, cap)``
        gauge and MUST unwrap it here so ``scalar`` yields the number alone.
    """

    def scalar(self, name: str) -> Optional[float]:
        ...


@runtime_checkable
class GameObject(Protocol):
    """A tracked board object (domain «entity»). Identity is ``id``."""

    id: str
    tracking_state: str


@runtime_checkable
class ObjectTracker(Protocol):
    """Frame-to-frame identity repository (domain «entity»)."""

    def associate(self, frame: object) -> object:
        ...


@runtime_checkable
class StateAbstraction(Protocol):
    """The projector raw-frame -> AbstractSituation (domain «function»)."""

    def project(self, frame: object, prev: object) -> AbstractSituation:
        ...


# --------------------------------------------------------------------------- #
# Outcome (value)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Outcome:
    """One turn's observed (or predicted) result — a «value» (domain v031).

    Field is ``is_full_reset`` (NOT ``full_reset``). Frozen + primitive fields =>
    deterministic equality (no RNG, no builtin-hash of mutable state). Win/loss
    symmetry: a WIN is decided by the Goal predicate (another cluster); an OVER is
    decided by an :class:`InteractionRule` loss trigger (here).
    """

    state: str = "ongoing"
    score: float = 0.0
    is_full_reset: bool = False
    level: int = 0

    def __post_init__(self) -> None:
        if self.state not in OUTCOME_STATES:
            raise ValueError(
                f"Outcome.state {self.state!r} not in {sorted(OUTCOME_STATES)}"
            )

    def is_over(self) -> bool:
        return self.state == "over"

    def is_win(self) -> bool:
        return self.state == "win"

    def is_ongoing(self) -> bool:
        return self.state == "ongoing"

    @staticmethod
    def ongoing(score: float = 0.0, level: int = 0) -> "Outcome":
        return Outcome(state="ongoing", score=score, level=level)

    @staticmethod
    def win(score: float = 0.0, level: int = 0) -> "Outcome":
        return Outcome(state="win", score=score, level=level)

    @staticmethod
    def over(score: float = 0.0, is_full_reset: bool = False, level: int = 0) -> "Outcome":
        return Outcome(
            state="over", score=score, is_full_reset=is_full_reset, level=level
        )


# --------------------------------------------------------------------------- #
# EffectSignature (value) — the per-axis fingerprint / recognition key (TERM-17).
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EffectSignature:
    """The per-axis effect fingerprint (verbalization §5.2 canonical form).

    Five typed handles ONLY — no colour / coordinate / glyph literal (TS-25(c)):
      * ``driver``         — a :data:`DRIVERS` label (direct/indirect/independent);
      * ``target_type``    — the Role label the effect lands on (e.g. 'self'/'ref');
      * ``feature``        — a ``Word.id`` axis (position/colour/orientation/...);
      * ``operator``       — a :data:`TRANSFORM_OPERATORS` verb;
      * ``parameter_kind`` — the KIND of parameter (a :data:`PARAMETER_KINDS`
        label, e.g. 'vector'/'angle'/'factor'/'index'/'value'), NOT the value.

    Two objects are *analogous* iff their signatures match, *contrasted* iff the
    surface is similar but the signature differs. :meth:`canonical` is the
    surface-free 5-tuple used as the role-matching / transfer key.
    """

    driver: str
    target_type: str
    feature: str
    operator: str
    parameter_kind: str

    def __post_init__(self) -> None:
        if self.driver not in DRIVERS:
            raise ValueError(
                f"EffectSignature.driver {self.driver!r} not in {sorted(DRIVERS)}"
            )
        if self.operator not in TRANSFORM_OPERATORS:
            raise ValueError(
                f"EffectSignature.operator {self.operator!r} not in "
                f"{sorted(TRANSFORM_OPERATORS)}"
            )

    def canonical(self) -> Tuple[str, str, str, str, str]:
        """The surface-free 5-tuple key (driver, target_type, feature, operator,
        parameter_kind). Contains only typed handles (TS-25(c))."""
        return (
            self.driver,
            self.target_type,
            self.feature,
            self.operator,
            self.parameter_kind,
        )


# --------------------------------------------------------------------------- #
# InteractionRule (entity) — a local transition rule.
# --------------------------------------------------------------------------- #

# A trigger predicate is a situation-pattern test ``AbstractSituation -> bool``.
TriggerPredicate = Callable[[AbstractSituation], bool]


@dataclass
class InteractionRule:
    """A learned local transition rule (domain «entity»).

    ``trigger`` = a situation-pattern predicate (``AbstractSituation -> bool``)
    plus an optional ``move`` id the rule fires for (``None`` = move-agnostic).
    ``effect`` = a per-axis list of :class:`EffectSignature` (each axis change +
    its driver attribution). A LOSS trigger is expressed AS a typed
    EffectSignature on a presence / scalar axis — there is no separate
    ``outcome_effect`` field (the terminal claim is read out of the effect list by
    :class:`WorldModel`, keyed on the gauge VALUE reaching 0, not on the
    operator). ``confidence`` in [0, 1] is refined from
    history; it does not reorder ``predict`` beyond registration order (DP-10).
    """

    trigger: TriggerPredicate = field(default=lambda _s: True)
    move: Optional[int] = None
    effect: List[EffectSignature] = field(default_factory=list)
    confidence: float = 1.0

    def applies(self, situation: AbstractSituation, move: Optional[int]) -> bool:
        """True iff this rule fires for ``(situation, move)``: the move matches
        (or the rule is move-agnostic) AND the trigger predicate holds."""
        if self.move is not None and move is not None and self.move != move:
            return False
        return bool(self.trigger(situation))


# --------------------------------------------------------------------------- #
# Observation (value) — the per-frame object view feeding affordance evidence.
# world_model OWNS this «value».
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Observation:
    """One associated track's per-frame view (a «value» owned by world_model).

    ``handle`` is the within-track stable id; ``cells`` is the footprint
    (frozenset of (row, col)); ``color_counts`` maps colour index -> cell count.
    The footprint is the truth for translate/vanish/spawn; ``color_counts`` (a
    NON-position axis) backs the ``response`` channel. Frozen + a frozenset
    footprint => deterministic equality.
    """

    handle: str
    cells: frozenset = field(default_factory=frozenset)
    color_counts: Mapping[int, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Affordance evidence (verbalization §4) — single deterministic forward pass.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Affordance:
    """The per-handle affordance evidence (verbalization §4 derivation source).

    Four ratios in [0, 1] over the ordered (action, Observation) pairs of a
    track, plus an ``autonomous`` flag. This is exactly what ``feat_afford`` will
    later read (NOT registered now — R1):
      * ``translate_support`` — footprint shifted (rigid displacement) under an
        action (movable);
      * ``vanish_support``    — visible -> absent AND the handle never reappears
        through the end of the track (destroy, NOT occlude: a present->absent
        transition whose handle reappears LATER is an occlusion and is not
        counted — object-schema §4 occlude != destroy, TS-04);
      * ``spawn_support``     — absent -> visible (spawning);
      * ``response_support``  — a non-position axis (colour) changed while the
        footprint did NOT translate (recolorable);
      * ``autonomous``        — the SAME displacement under >= 2 DISTINCT action
        ids (moves itself, action-independent).
    """

    translate_support: float = 0.0
    vanish_support: float = 0.0
    spawn_support: float = 0.0
    response_support: float = 0.0
    autonomous: bool = False


def footprint_shift(before: frozenset, after: frozenset) -> Optional[Tuple[int, int]]:
    """The rigid (dr, dc) that maps ``before`` exactly onto ``after`` (a pure
    translation), or ``None`` if no single shift does (deform / no move / empty).

    Deterministic: anchors on the min cell of each set (lexicographic) and checks
    that applying that delta to every ``before`` cell reproduces ``after`` exactly
    (same cardinality, same shape). The zero shift is treated as "no translation"
    by the caller."""
    if not before or not after or len(before) != len(after):
        return None
    br, bc = min(before)
    ar, ac = min(after)
    dr, dc = ar - br, ac - bc
    shifted = frozenset((r + dr, c + dc) for (r, c) in before)
    if shifted == after:
        return (dr, dc)
    return None


def affordance_evidence(
    track_history: Mapping[str, List[Tuple[int, "Observation"]]],
) -> Dict[str, Affordance]:
    """Derive per-handle :class:`Affordance` from associated track histories.

    Parameters
    ----------
    track_history:
        ``handle -> ordered list of (action_id, Observation)`` — the per-frame
        view of ONE associated track. Association is given (the tracker upstream
        kept identity); a track's absence is trusted as destroy ONLY when the
        handle never reappears later in the track — a present->absent transition
        followed by a later reappearance is an occlusion, not a destroy
        (object-schema §4, TS-04). Each Observation's ``handle`` is expected to
        equal the map key.

    Returns
    -------
    dict
        ``handle -> Affordance``. Ratios are exact integer fractions
        (supporting transitions / counted transitions) so the result is
        reproducible (TS-29). Handles are visited in ``sorted`` order; the single
        forward pass over each track is order-preserving.

    Channels (per consecutive (action, Observation) pair within a track):
      * present := the footprint is non-empty;
      * translate := both present AND a non-zero rigid shift maps before->after;
      * vanish := present -> absent AND the footprint never becomes present again
        later in the track (a true destroy; a present->absent transition whose
        handle reappears is an occlusion and is NOT a vanish hit);
      * spawn := absent -> present;
      * response := both present AND footprint did NOT translate AND a non-position
        axis (color_counts) changed.
    Each channel's support is (#supporting pairs) / (#applicable pairs); a channel
    with no applicable pair scores 0.0 (no evidence). ``autonomous`` is True iff
    the same non-zero displacement was observed under >= 2 distinct action ids.
    """
    result: Dict[str, Affordance] = {}
    for handle in sorted(track_history):
        history = track_history[handle]
        translate_hits = translate_pairs = 0
        vanish_hits = vanish_pairs = 0
        spawn_hits = spawn_pairs = 0
        response_hits = response_pairs = 0
        # displacement -> set of distinct action ids that produced it (non-zero).
        disp_actions: Dict[Tuple[int, int], set] = {}

        for i in range(1, len(history)):
            action_id, obs = history[i]
            prev_obs = history[i - 1][1]
            before, after = prev_obs.cells, obs.cells
            present_before = bool(before)
            present_after = bool(after)

            if present_before and present_after:
                shift = footprint_shift(before, after)
                translated = shift is not None and shift != (0, 0)
                # translate channel (movable)
                translate_pairs += 1
                if translated:
                    translate_hits += 1
                    disp_actions.setdefault(shift, set()).add(action_id)
                # response channel: non-position axis changed without translating
                response_pairs += 1
                if not translated and prev_obs.color_counts != obs.color_counts:
                    response_hits += 1
            elif present_before and not present_after:
                # vanish channel: a present->absent transition. It is a DESTROY
                # only if the footprint never becomes present again later in the
                # track; if it reappears it is an OCCLUSION, not a vanish hit
                # (object-schema §4 occlude != destroy, TS-04).
                vanish_pairs += 1
                reappears = any(bool(history[j][1].cells) for j in range(i + 1, len(history)))
                if not reappears:
                    vanish_hits += 1
            elif not present_before and present_after:
                # spawn channel
                spawn_pairs += 1
                spawn_hits += 1
            # absent -> absent: no channel applies.

        autonomous = any(len(actions) >= 2 for actions in disp_actions.values())
        result[handle] = Affordance(
            translate_support=_ratio(translate_hits, translate_pairs),
            vanish_support=_ratio(vanish_hits, vanish_pairs),
            spawn_support=_ratio(spawn_hits, spawn_pairs),
            response_support=_ratio(response_hits, response_pairs),
            autonomous=autonomous,
        )
    return result


def _ratio(hits: int, total: int) -> float:
    """Exact integer fraction in [0, 1] (0.0 when ``total`` is 0 — no evidence)."""
    if total <= 0:
        return 0.0
    return hits / total


# --------------------------------------------------------------------------- #
# Signature induction (object-naming §5; active probe/fit loop DEFERRED).
# --------------------------------------------------------------------------- #

# Default parameter-kind labels per operator (the KIND, never the value).
_OPERATOR_PARAMETER_KIND: Mapping[str, str] = {
    "translate": "vector",
    "rotate": "angle",
    "scale": "factor",
    "recolor": "index",
    "cycle": "index",
    "set": "value",
}


def induce_signature(
    before: "Observation",
    after: "Observation",
    *,
    role: str,
    driver: str,
) -> EffectSignature:
    """Classify the SINGLE changed axis between two Observations into an
    :class:`EffectSignature` (classification only — NO xf_* application).

    Parameters
    ----------
    before, after:
        The OBJ view before/after the interaction (one changed axis assumed).
    role:
        The Role label the effect lands on (``target_type``).
    driver:
        The attribution of WHAT caused the change, supplied by the caller's
        context — a :data:`DRIVERS` label (``direct`` / ``indirect`` /
        ``independent``). It is NOT inferred from "did it move": a mover may be
        ``independent`` (a clock/phase autonomous mover) and a non-position change
        may be ``direct``, so collapsing the attribution onto the diff would
        conflate the three drivers. A ``driver`` outside :data:`DRIVERS` raises
        ``ValueError``.

    Returns
    -------
    EffectSignature
        ``feature`` is the axis that changed (``position`` if the footprint moved,
        else ``colour`` if ``color_counts`` changed, else ``presence`` if the
        object appeared/vanished); ``operator`` is a best-guess TransformOperator
        from the single-axis diff (translate / recolor / set); ``parameter_kind``
        is the KIND for that operator (always in :data:`PARAMETER_KINDS`), never
        the concrete value. This classifies only the OBSERVABLE single-axis diff
        and takes ``driver`` / ``target_type`` from the caller's context:
        cycle-vs-recolor disambiguation and the active probe -> attribute -> fit
        -> re-type loop (object-naming §8-12) are DEFERRED.
    """
    if driver not in DRIVERS:
        raise ValueError(
            f"induce_signature.driver {driver!r} not in {sorted(DRIVERS)}"
        )

    shift = footprint_shift(before.cells, after.cells)
    moved = shift is not None and shift != (0, 0)
    presence_changed = bool(before.cells) != bool(after.cells)
    color_changed = before.color_counts != after.color_counts

    if moved:
        feature, operator = "position", "translate"
    elif presence_changed:
        feature, operator = "presence", "set"
    elif color_changed:
        feature, operator = "colour", "recolor"
    else:
        # No discernible change on the modelled axes: a no-op set on presence.
        feature, operator = "presence", "set"

    parameter_kind = _OPERATOR_PARAMETER_KIND[operator]
    return EffectSignature(
        driver=driver,
        target_type=role,
        feature=feature,
        operator=operator,
        parameter_kind=parameter_kind,
    )


# --------------------------------------------------------------------------- #
# WorldModel (entity)
# --------------------------------------------------------------------------- #

# Axes whose value reaching 0 asserts a terminal OVER (a monotone gauge reaching
# zero — TS-15 / TERM-32). Terminality keys on the gauge VALUE, not the operator:
# a presence/scalar axis driven to 0 is terminal whether the effect set/dec'd it.
# (``move_budget`` is NOT a feature axis — it is read via scalar() by name; the
# non-axis literal must not leak into the feature set.)
_TERMINAL_FEATURES = frozenset({"presence", "scalar"})


@dataclass
class WorldModel:
    """The learned world dynamics (domain «entity», v031).

    Holds an ordered list of :class:`InteractionRule` and depend-on-contract
    collaborators (``abstraction`` : StateAbstraction, ``tracking`` :
    ObjectTracker — both Protocol stubs here). ``representation`` is one of
    :data:`REPRESENTATIONS`. ``confidence`` = coverage x prediction-accuracy
    (stored/exposed; the full update is deferred). Mutable (it accumulates rules)
    but never uses RNG or builtin ``hash`` for identity.

    ``is_scrolling`` is an observation-updated bool (default ``False``; no
    "belief" wording — TERM-56) that records whether the frame is a window onto a
    larger world. It GATES :class:`agent.core.model.WorldMap` creation: while it
    is ``False`` there is no ``map`` and camera_offset is ``(0, 0)`` — a true
    no-op (a non-scrolling game creates no WorldMap at all). ``map`` is the lazily
    owned WorldMap (``0..1``); ``None`` until :func:`track_viewport` raises
    ``is_scrolling`` after observing cumulative window motion (DEFERRED).
    """

    rules: List[InteractionRule] = field(default_factory=list)
    abstraction: Optional["StateAbstraction"] = None
    representation: str = "Symbolic"
    tracking: Optional["ObjectTracker"] = None
    confidence: float = 0.0
    is_scrolling: bool = False
    map: Optional[WorldMap] = None

    def __post_init__(self) -> None:
        if self.representation not in REPRESENTATIONS:
            raise ValueError(
                f"WorldModel.representation {self.representation!r} not in "
                f"{sorted(REPRESENTATIONS)}"
            )

    def add_rule(self, rule: InteractionRule) -> "WorldModel":
        """Register an :class:`InteractionRule` (deterministic order). Returns
        ``self`` so registrations chain."""
        self.rules.append(rule)
        return self

    def predict(
        self, situation: "AbstractSituation", move: Optional[int]
    ) -> Tuple["AbstractSituation", Optional["Outcome"]]:
        """Predict ``(next AbstractSituation, Optional[Outcome])`` for
        ``(situation, move)``.

        Order (deterministic, DP-10): the FIRST registered :class:`InteractionRule`
        whose trigger fires for ``(situation, move)`` wins. Its EffectSignature
        list is interpreted as an axis-delta on the situation (full xf_*
        application deferred — the next AbstractSituation is the same object until
        a projector applies the deltas). If any effect asserts a terminal (a
        monotone scalar/presence axis whose VALUE in the situation is <= 0,
        regardless of the effect's operator), :meth:`Outcome.over` is returned;
        otherwise :meth:`Outcome.ongoing`. An unknown move (no rule fires) is
        predicted inert + ongoing.
        """
        for rule in self.rules:
            if rule.applies(situation, move):
                outcome = self._terminal_outcome(rule, situation)
                return situation, outcome
        # No rule fired: inert + ongoing (optimistic default).
        return situation, Outcome.ongoing()

    @staticmethod
    def _terminal_outcome(
        rule: "InteractionRule", situation: "AbstractSituation"
    ) -> "Outcome":
        """``Outcome.over`` iff a loss-trigger effect's gauge reads <= 0 in
        ``situation``; else ``Outcome.ongoing``.

        Terminality keys on the gauge VALUE, not the operator: a presence/scalar
        axis reaching 0 is terminal whether the effect set/dec'd it (TS-15). The
        value is the scalar VALUE component (a number); a stray tuple (e.g. a raw
        ``(value, cap)`` pair from a mis-wired situation) is guarded against via
        ``isinstance`` so it cannot raise.
        """
        for sig in rule.effect:
            if sig.feature in _TERMINAL_FEATURES:
                value = situation.scalar(sig.feature)
                if isinstance(value, (int, float)) and value <= 0:
                    return Outcome.over()
        return Outcome.ongoing()

    def classify(
        self,
        before: "AbstractSituation",
        after: "AbstractSituation",
        goal_distance: Callable[["AbstractSituation"], object],
    ) -> str:
        """Classify a move's observed effect into a :data:`MOVE_EFFECTS` label
        (TS-12).

        ``invariant`` iff ``before == after`` by value-equality (the
        AbstractSituation is a «value»; nothing salient changed); else
        ``progress`` iff ``goal_distance(after) < goal_distance(before)``; else
        ``no_progress``. ``goal_distance`` is an INJECTED callable (this module
        never imports goal.py); it must yield a TOTAL ORDER (only the order
        relation is used) and :meth:`classify` does NOT catch exceptions raised
        by it. ``futile`` == result != 'progress'.
        """
        if before == after:
            return "invariant"
        before_distance = goal_distance(before)
        after_distance = goal_distance(after)
        if after_distance < before_distance:
            return "progress"
        return "no_progress"


# --------------------------------------------------------------------------- #
# TrackViewport (UseCase) — per-move ego-motion estimation (TERM-58 / CMP-08).
# A UseCase (peer of Verbalize / NameObjects), NOT a plug family — there is no
# catalog of interchangeable members. The active consensus Δ-estimation body is
# DEFERRED until a real scrolling game is observed (handoff 2-stage); the stub
# returns the no-scroll safe default so the wiring is present but inert.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ViewportDelta:
    """The per-move ego-motion verdict — a «value».

    ``delta`` is the window's translation increment ``(row, col)`` to add to
    ``camera_offset`` this move; ``is_scrolling`` is the updated scroll verdict.
    The no-scroll safe default is ``delta=(0, 0)``, ``is_scrolling=False`` (a true
    no-op). Frozen + integer tuple => deterministic value-equality (DP-10).

    IMPL CARRIER, not a domain node: this is a thin two-field return for
    ``track_viewport`` whose components are already canon — ``delta`` = the
    ``Viewport.origin`` / camera_offset increment (TERM-53), ``is_scrolling`` =
    the scroll verdict (TERM-56). It is intentionally NOT in the domain model
    (like ``Affordance``), so it carries no independent semantics to canonicalize."""

    delta: Tuple[int, int] = (0, 0)
    is_scrolling: bool = False


def track_viewport(
    prev_static: Mapping[str, "Observation"],
    curr_static: Mapping[str, "Observation"],
    *,
    action_correlated: FrozenSet[str] = frozenset(),
) -> ViewportDelta:
    """Estimate this move's window ego-motion Δ (= camera_offset increment) and
    update the scroll verdict. The TrackViewport UseCase (TERM-58 / CMP-08).

    Parameters
    ----------
    prev_static, curr_static:
        The STATIC tracks' footprints (``handle -> Observation``) one frame apart
        (one-frame-lagged input contract). Only static cells contribute to the
        ego-motion population — dynamic objects are excluded.
    action_correlated:
        Handles whose displacement correlates with the issued action; they are
        EXCLUDED from the population (ego-motion separation, consistent with L1
        §7.1) so a controllable object's own move is not mistaken for a pan.

    Returns
    -------
    ViewportDelta
        ``delta`` = the dominant window translation ``(row, col)``;
        ``is_scrolling`` = whether scrolling holds this move.

    Contract (the DEFERRED active body must satisfy):
      1. BOOTSTRAP — an association-free GLOBAL best-shift over the static
         population (the single translation that maps the most ``prev`` cells onto
         ``curr``).
      2. REFINE — the consensus translation = the MAX-INLIER set of static tracks
         sharing one exact integer translation (epsilon = 0, exact match — NOT a
         median, which a majority of distractors could lock).
      3. SCROLL holds IFF the dominant translation is a STRICT MAJORITY of the
         static population; otherwise no-scroll (safe degenerate).
      4. EGO-MOTION SEPARATION — any track in ``action_correlated`` is excluded
         from the population before steps 1-3.
      5. DETERMINISTIC — tie-break by translation vector ``(row, col)`` ascending;
         no RNG, no builtin ``hash()`` in the returned value (DP-10).

    # DEFERRED body: returns the no-scroll safe default (delta=(0, 0),
    #   is_scrolling=False) so TrackViewport is INERT until the real consensus
    #   Δ-estimation algorithm lands after observing a real scrolling game
    #   (handoff 2-stage — do NOT invent an unvalidated algorithm here).
    """
    return ViewportDelta(delta=(0, 0), is_scrolling=False)
