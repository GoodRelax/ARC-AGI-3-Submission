"""[agent/core] WorldModel -- the dynamics the agent LEARNS from observation (no game literals).

It learns three things from watching its own moves:
  * move_delta[action]   -- how the controllable translates for each action id;
  * walls                -- colours the controllable was BLOCKED by (everything else is
                            treated as passable: optimistic, corrected by replanning);
  * triggers + pose_succ -- objects whose fresh footprint-overlap advances the carried
                            state's POSE, and the observed pose transition.

These feed the search (solver.bfs). Nothing here is ls20-specific; it is the generic
"learn move effect + interaction rule by prediction-vs-observation" loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from .situation import AbstractSituation


class WorldModel:
    def __init__(self) -> None:
        self.move_delta: dict[int, tuple] = {}     # action id -> (dr, dc)
        self.walls: set[int] = set()               # colours known to block the controllable
        self.passable: set[int] = set()            # colours the controllable was seen to occupy
        self.triggers: list[frozenset] = []        # cell-sets that advance carried pose on fresh overlap
        self.pose_succ: dict[frozenset, frozenset] = {}  # carried pose -> next carried pose

    # ----------------------------------------------------------------- learning
    def learn_move(self, action: int, disp: tuple) -> None:
        if disp != (0, 0):
            self.move_delta[action] = disp

    def learn_wall(self, color: int) -> None:
        self.walls.add(int(color))

    def learn_passable(self, colors) -> None:
        self.passable |= set(int(c) for c in colors)

    def learn_trigger(self, cells, pose_before, pose_after) -> None:
        cells = frozenset(cells)
        if cells and cells not in self.triggers:
            self.triggers.append(cells)
        if pose_before != pose_after:
            self.pose_succ[pose_before] = pose_after

    def moves_known(self) -> bool:
        return len(self.move_delta) >= 4

    # ----------------------------------------------------------------- prediction (for search)
    def cell_passable(self, grid, r, c, allow: frozenset = frozenset()) -> bool:
        h, w = grid.shape
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in allow:
            return True
        return int(grid[r, c]) not in self.walls   # optimistic: unknown colours are passable

    def footprint_passable(self, grid, offsets, pos, allow: frozenset = frozenset()) -> bool:
        r0, c0 = pos
        return all(self.cell_passable(grid, r0 + dr, c0 + dc, allow) for dr, dc in offsets)


# =====================================================================================
# General dynamics layer (CMP-07 InteractionRule / CMP-08 WorldModel / CMP-12 MoveEffect /
# CMP-17 Outcome / CMP-26 ModelWorld). Built ON TOP of the L1 ``WorldModel`` above, which
# stays byte-for-byte the concrete instance the solver/play L1 slice imports. This layer
# learns InteractionRules over the *abstract* AbstractSituation (agent/core/situation.py), so it is
# game-literal-free (NFR-6) and deterministic (DP-10: canonical tuples / no RNG / no builtin
# ``hash()`` for stable identity -- equality uses AbstractSituation.canonical()).
#
# Canon (cite, never duplicate):
#   - _assets/gr-arc-3-terms.md
#       TERM-11 MODEL_WORLD       -- model the dynamics; compress to a minimal-sufficient
#                                    abstract AbstractSituation; Markov is the GOAL -- when the visible
#                                    features are non-Markov, ADD a HIDDEN variable. (SC-07)
#       TERM-31 AbstractSituation / TERM-24 StateAbstraction -- the abstract state + its projector.
#       TERM-32 Move Budget       -- loss trigger = InteractionRule -> Outcome.over; learned
#                                    from a monotone-decreasing Dimension reaching 0 (e.g.
#                                    move_budget==0 -> over, OR any health gauge ==0 -> over);
#                                    win/loss symmetry: win = GoalPredicate / loss = rule.
#       TERM-33 Outcome           -- one turn's observed result {state, score, full_reset,
#                                    level}; predicted by WorldModel.predict; a «value».
#   - _assets/gr-arc-3-domain-model.md
#       WorldModel := composition of InteractionRule(s) + StateAbstraction;
#       InteractionRule : (AbstractSituation, GameMove) -> next  (trigger=target+relations,
#       effect=target placement, confidence); WorldModel.predict -> next AbstractSituation AND
#       -> Outcome (the two are symmetric); InteractionRule refined from TurnRecord.
#   - 04-specification SC-07 / SC-17 ; 05-test-strategy TS-07 / TS-12 / TS-15 ;
#     sequence choose-action (refine = per-move feedback; predict xN in look-ahead).
#
# LIM-1: the in-house sim is APPROXIMATE -- predictions are compared on the *salient*
# AbstractSituation (AbstractSituation.canonical()), never on full frames.
# =====================================================================================


# --------------------------------------------------------------------------- MoveEffect
# The three MoveEffect «values» (TERM: a move's observed result axis, recorded by
# TurnRecord). These are stable string labels (not a salted Enum) so equality / logging /
# serialization are deterministic across processes (DP-10). This is the EXACT vocabulary
# TS-12 (futility) consumes: ``invariant`` and ``no_progress`` are *futile*; only
# ``progress`` advances the goal.
class MoveEffect:
    """Namespace of the 3 MoveEffect labels + the futility predicate (CMP-12).

    Classification oracle (05-test-strategy TS-12 / memos/futility-detection.md):
      * ``invariant``   -- the next AbstractSituation is IDENTICAL (the move changed nothing salient);
      * ``no_progress`` -- the AbstractSituation changed but the goal-distance is UNCHANGED;
      * ``progress``    -- the goal-distance DECREASED.
    ``futile(value) == (value != PROGRESS)`` -- both non-progress labels are futile (G1: the
    planner hard-prunes ``invariant``; ``no_progress`` gets one try). The classifier takes the
    goal-distance as an INJECTED callable (the SearchHeuristic, API-05), so this module never
    imports goal.py and stays decoupled.
    """

    INVARIANT: str = "invariant"
    NO_PROGRESS: str = "no_progress"
    PROGRESS: str = "progress"

    ALL: tuple = (INVARIANT, NO_PROGRESS, PROGRESS)

    @staticmethod
    def futile(value: str) -> bool:
        """True iff ``value`` is NOT progress (i.e. ``invariant`` or ``no_progress``)."""
        return value != MoveEffect.PROGRESS


def classify_move_effect(
    before: AbstractSituation,
    after: AbstractSituation,
    goal_distance: Callable[[AbstractSituation], object],
) -> str:
    """Classify a move's observed result into the 3 MoveEffect labels (CMP-12; TS-12 API).

    Parameters
    ----------
    before, after:
        The salient :class:`AbstractSituation` before and after the move (StateAbstraction output).
        Identity is compared via :meth:`AbstractSituation.canonical` (DP-10), never builtin ``hash``.
    goal_distance:
        The injected SearchHeuristic ``AbstractSituation -> distance`` (API-05). Only the ORDER
        relation ``after < before`` is used, so the distance may be any comparable value
        (int / float / tuple). It is read for ``before`` and ``after`` only -- no goal.py
        dependency, no game literal.

    Returns
    -------
    str
        ``MoveEffect.INVARIANT`` if ``after`` equals ``before`` (no salient change);
        else ``MoveEffect.PROGRESS`` if ``goal_distance(after) < goal_distance(before)``;
        else ``MoveEffect.NO_PROGRESS``. ``futile == value != PROGRESS``.
    """
    if before.canonical() == after.canonical():
        return MoveEffect.INVARIANT
    if goal_distance(after) < goal_distance(before):
        return MoveEffect.PROGRESS
    return MoveEffect.NO_PROGRESS


# ----------------------------------------------------------------------------- Outcome
# The Outcome states (TERM-33). String labels, not a salted Enum -> deterministic equality.
class OutcomeState:
    """The three terminal/non-terminal Outcome states (TERM-33)."""

    ONGOING: str = "ongoing"
    WIN: str = "win"
    OVER: str = "over"

    ALL: tuple = (ONGOING, WIN, OVER)


@dataclass(frozen=True)
class Outcome:
    """One turn's observed (or predicted) result -- a «value» (TERM-33).

    ``state`` is one of :class:`OutcomeState`. ``score`` / ``full_reset`` / ``level`` mirror
    the env signal (FrameData). Win/loss symmetry (TERM-32): a WIN is decided by a
    GoalPredicate (another cluster); an OVER is decided by an :class:`InteractionRule` (a loss
    trigger). Frozen + primitive fields => deterministic equality (no RNG, no builtin-hash of
    mutable state)."""

    state: str = OutcomeState.ONGOING
    score: int = 0
    full_reset: bool = False
    level: int = 0

    def is_over(self) -> bool:
        return self.state == OutcomeState.OVER

    def is_win(self) -> bool:
        return self.state == OutcomeState.WIN

    def is_ongoing(self) -> bool:
        return self.state == OutcomeState.ONGOING

    @staticmethod
    def ongoing(score: int = 0, level: int = 0) -> "Outcome":
        return Outcome(state=OutcomeState.ONGOING, score=score, level=level)

    @staticmethod
    def win(score: int = 0, level: int = 0) -> "Outcome":
        return Outcome(state=OutcomeState.WIN, score=score, level=level)

    @staticmethod
    def over(score: int = 0, full_reset: bool = False, level: int = 0) -> "Outcome":
        return Outcome(state=OutcomeState.OVER, score=score, full_reset=full_reset, level=level)


# --------------------------------------------------------------------- InteractionRule
# A learned local transition rule (CMP-07): {applicability(precondition) -> effect}. It maps
# (AbstractSituation, move) -> next AbstractSituation (and may also assert an Outcome). The L1 move_delta /
# walls / triggers above are CONCRETE instances of this idea; this is the GENERAL form over
# the abstract AbstractSituation, with NO game literal (NFR-6) and a confidence for refinement.
@dataclass(frozen=True)
class InteractionRule:
    """A learned rule ``{applicability(observed precondition) -> effect}`` (CMP-07).

    Fields
    ------
    move:
        The GameMove id this rule fires for (``None`` = any move). Matching keys off the move
        AND the applicability predicate, mirroring the L1 ``move_delta[action]``.
    applicability:
        A predicate ``AbstractSituation -> bool`` -- the observed PRECONDITION (trigger = target +
        relations). Default: applies to every AbstractSituation.
    effect:
        A transform ``AbstractSituation -> AbstractSituation`` -- the next salient AbstractSituation (effect = target
        placement). Default: identity (the move changes nothing salient).
    outcome_effect:
        Optional ``AbstractSituation -> Optional[Outcome]`` -- asserts a predicted Outcome (e.g. a loss
        trigger returns :meth:`Outcome.over` when a gauge hits 0; ``None`` = no terminal claim).
    confidence:
        Refinement confidence in [0, 1] (TERM-25-style); raised by confirmation, lowered by
        contradiction. Carried for the solver/explore wave; not used by ``predict`` ordering
        beyond the deterministic registration order.
    name:
        A stable, game-literal-free label for logs/handles (e.g. ``"move"`` / ``"loss:<dim>"``).

    The rule holds no mutable state and is hashable via its frozen identity tuple (the
    callables are by-identity); equality of the RESULT it produces is decided on
    :meth:`AbstractSituation.canonical` by callers, not on the callable objects (DP-10)."""

    move: Optional[int] = None
    applicability: Callable[[AbstractSituation], bool] = field(default=lambda _s: True)
    effect: Callable[[AbstractSituation], AbstractSituation] = field(default=lambda s: s)
    outcome_effect: Optional[Callable[[AbstractSituation], Optional[Outcome]]] = None
    confidence: float = 1.0
    name: str = ""

    def applies(self, situation: AbstractSituation, move: Optional[int]) -> bool:
        """True iff this rule fires for ``(situation, move)``: the move matches (or the rule is
        move-agnostic) AND the applicability precondition holds."""
        if self.move is not None and move is not None and self.move != move:
            return False
        return bool(self.applicability(situation))

    def apply_effect(self, situation: AbstractSituation) -> AbstractSituation:
        """The next salient AbstractSituation this rule predicts from ``situation``."""
        return self.effect(situation)

    def apply_outcome(self, situation: AbstractSituation) -> Optional[Outcome]:
        """The Outcome this rule predicts for ``situation`` (``None`` = no terminal claim)."""
        if self.outcome_effect is None:
            return None
        return self.outcome_effect(situation)


def _scalar_value(situation: AbstractSituation, dim_id: str):
    """Read a scalar gauge ``dim_id`` from a AbstractSituation (``None`` if absent). The Markov scalars
    (e.g. move_budget, a health Dimension) live in ``AbstractSituation.scalars`` (TERM-32)."""
    return situation.scalars.get(dim_id)


def loss_trigger_rule(dim_id: str, move: Optional[int] = None) -> InteractionRule:
    """Build the GENERAL loss-trigger InteractionRule: *any* monotone-decreasing scalar
    Dimension reaching 0 predicts ``Outcome.over`` (TERM-32; SC-17).

    This is game-literal-free: ``dim_id`` is a Dimension NAME (e.g. the observed move_budget
    gauge, or a health gauge), never a colour/coordinate. The SAME factory serves move_budget
    and any other gauge -- generalising beyond move_budget as SC-17(c) requires. The rule only
    READS the scalar already projected into the AbstractSituation; the projector decides the name.
    """

    def _is_zero(situation: AbstractSituation) -> bool:
        v = _scalar_value(situation, dim_id)
        return v is not None and v <= 0

    def _over(situation: AbstractSituation) -> Optional[Outcome]:
        return Outcome.over() if _is_zero(situation) else None

    return InteractionRule(
        move=move,
        applicability=_is_zero,
        effect=lambda s: s,            # a terminal state's salient config is unchanged here
        outcome_effect=_over,
        name="loss:%s" % dim_id,
    )


# ------------------------------------------------------------------------- hidden state
# The hidden-state key: the canonical AbstractSituation MINUS the hidden scalars, so two occasions
# that look identical on the visible features but carry different hidden values are
# distinguished. Built from primitives (DP-10).
def _visible_key(situation: AbstractSituation, hidden: Sequence[str]) -> tuple:
    """Canonical key of ``situation`` with the ``hidden`` scalar names REMOVED -- i.e. the
    VISIBLE-feature signature. When two transitions share this key but differ in their next
    state, the dynamics are non-Markov on the visible features (SC-07)."""
    hidden_set = frozenset(hidden)
    roles, relations, _scalars = situation.canonical()
    visible_scalars = _sorted_visible(situation.scalars, hidden_set)
    return (roles, relations, visible_scalars)


def _sorted_visible(scalars: Mapping[str, object], hidden_set: frozenset) -> tuple:
    """Deterministic sorted tuple of the non-hidden scalar (name, value) pairs."""
    items = [(name, val) for name, val in scalars.items() if name not in hidden_set]
    return tuple(sorted(items, key=lambda kv: (kv[0], repr(kv[1]))))


def _with_scalar(situation: AbstractSituation, name: str, value) -> AbstractSituation:
    """Return a copy of ``situation`` with scalar ``name`` set to ``value`` (immutable copy)."""
    scalars = dict(situation.scalars)
    scalars[name] = value
    return AbstractSituation(roles=situation.roles, relations=situation.relations, scalars=scalars)


# Default name of the synthetic hidden variable ``refine`` adds. It is a *scalar* Dimension
# (a learned phase index), NOT a game literal -- the value is observation-induced.
HIDDEN_PHASE: str = "phase"


@dataclass
class ModelWorld:
    """The learned world dynamics over abstract AbstractSituations (CMP-08 / CMP-26).

    Holds an ordered list of :class:`InteractionRule`. ``predict`` applies the first matching
    rule (deterministic registration order, DP-10); ``refine`` learns from observed
    (AbstractSituation, move, next) transitions and, when the visible features are non-Markov, ADDS a
    HIDDEN scalar variable (a phase) so prediction matches observation on the salient
    components (SC-07). Mutable (it accumulates rules + a learned phase schedule), but it
    never uses RNG or builtin ``hash`` for identity.

    This is the GENERAL counterpart of the L1 :class:`WorldModel`; the two coexist (L1 stays
    the concrete grid sim the solver imports). ``ModelWorld`` operates purely on AbstractSituations.
    """

    rules: list = field(default_factory=list)
    hidden: list = field(default_factory=list)          # names of added hidden scalars
    # Learned transition table over (visible_key, hidden-tuple, move) -> next AbstractSituation, used
    # to drive predict once a hidden variable has been added. Deterministic dict (insertion
    # order is irrelevant -- lookups are by key).
    _table: dict = field(default_factory=dict)
    # Learned hidden-state schedule: visible_key -> ordered list of observed hidden values,
    # plus a successor map so a hidden value advances deterministically (the phase cycle).
    _phase_succ: dict = field(default_factory=dict)     # (visible_key, hidden_val) -> hidden_val
    _phase_seq: dict = field(default_factory=dict)      # visible_key -> [hidden_val, ...] (FIFO)
    _outcome_table: dict = field(default_factory=dict)  # transition-key -> Outcome

    # ----------------------------------------------------------------- rule registration
    def add_rule(self, rule: InteractionRule) -> "ModelWorld":
        """Register an :class:`InteractionRule` (deterministic order). Returns ``self`` so
        registrations chain."""
        self.rules.append(rule)
        return self

    def learn_loss_trigger(self, dim_id: str, move: Optional[int] = None) -> "ModelWorld":
        """Convenience: register the general loss trigger for scalar ``dim_id`` (SC-17)."""
        return self.add_rule(loss_trigger_rule(dim_id, move))

    # --------------------------------------------------------------------------- predict
    def predict(self, situation: AbstractSituation, move: Optional[int]) -> AbstractSituation:
        """Predict the next SALIENT :class:`AbstractSituation` for ``(situation, move)``.

        Order (deterministic):
          1. if a learned transition is known for this (visible-key, hidden-value, move),
             return it -- this covers BOTH the Markov case (hidden-value ``None``, learned
             directly) AND the non-Markov case (a phase value, which is what makes the
             augmented model match observation on a non-Markov system -- SC-07);
          2. else apply the first registered :class:`InteractionRule` whose precondition holds
             (the L1 move_delta / trigger analogue, lifted to AbstractSituations);
          3. else return ``situation`` unchanged (an unknown move is predicted inert -- the
             optimistic default, corrected by ``refine``).

        Per LIM-1 the result is the salient AbstractSituation only; callers compare via
        :meth:`AbstractSituation.canonical`, never full frames.
        """
        if self._table:
            hit = self._predict_from_table(situation, move)
            if hit is not None:
                return hit
        for rule in self.rules:
            if rule.applies(situation, move):
                return rule.apply_effect(situation)
        return situation

    def predict_outcome(self, situation: AbstractSituation, move: Optional[int]) -> Outcome:
        """Predict the :class:`Outcome` for ``(situation, move)`` (symmetric to ``predict``).

        The first registered rule that asserts an Outcome wins (e.g. a loss trigger ->
        :meth:`Outcome.over` when a monotone gauge is 0); otherwise a learned terminal table
        entry is used; otherwise the play is ``ongoing`` (win is decided by the GoalPredicate
        cluster, not here -- win/loss symmetry, TERM-32).
        """
        for rule in self.rules:
            if rule.applies(situation, move):
                oc = rule.apply_outcome(situation)
                if oc is not None:
                    return oc
        key = self._transition_key(situation, move)
        if key in self._outcome_table:
            return self._outcome_table[key]
        return Outcome.ongoing()

    def _predict_from_table(self, situation: AbstractSituation, move: Optional[int]):
        vkey = _visible_key(situation, self.hidden)
        hidden_val = self._current_hidden(situation, vkey)
        key = (vkey, hidden_val, move)
        return self._table.get(key)

    def _current_hidden(self, situation: AbstractSituation, vkey: tuple):
        """The hidden value to use for ``situation``: its own hidden scalar if already set,
        else the next value in the learned phase schedule for ``vkey`` (deterministic)."""
        if self.hidden:
            own = _scalar_value(situation, self.hidden[-1])
            if own is not None:
                return own
        seq = self._phase_seq.get(vkey)
        if seq:
            return seq[0]
        return None

    # ---------------------------------------------------------------------------- refine
    def refine(
        self,
        transitions: Sequence[tuple],
        hidden_name: str = HIDDEN_PHASE,
    ) -> bool:
        """Learn InteractionRules from observed ``(situation, move, observed_next)`` triples;
        add a HIDDEN state variable when the visible features are non-Markov (SC-07).

        A system is non-Markov on the visible features iff two transitions share a
        visible-key (same AbstractSituation+move ignoring hidden scalars) but lead to DIFFERENT next
        states. When detected, ``refine``:
          * records the distinct observed next-states per visible-key in observation order;
          * assigns each occurrence a hidden value (a phase index 0, 1, 2, ... cycling) and a
            successor map (phase_i -> phase_{i+1 mod period}) -- a deterministic schedule;
          * stores ``(visible_key, phase, move) -> next`` so that, after augmentation, the
            predicted next from the hidden-augmented state matches the observed next on the
            salient components.

        Returns ``True`` iff a hidden variable was added (the dynamics were non-Markov);
        ``False`` iff the visible features were already Markov (every visible-key mapped to one
        next-state, which is recorded directly).

        Determinism (DP-10): occurrences are processed in the given order; phase indices are
        assigned by first-seen order; no RNG. The same transition log yields the same model.
        """
        # Group observed next-states by visible-key, preserving observation order.
        by_vkey: dict = {}
        order: list = []
        for situation, move, observed_next in transitions:
            vkey = _visible_key(situation, [])         # hidden not yet added -> visible = all
            bucket = by_vkey.setdefault((vkey, move), [])
            if not bucket:
                order.append((vkey, move))
            bucket.append((situation, observed_next))

        non_markov = any(
            len({nxt.canonical() for _s, nxt in bucket}) > 1
            for bucket in by_vkey.values()
        )

        if not non_markov:
            # Markov on visible features: record each (visible_key, move) -> the single next.
            for (vkey, move), bucket in by_vkey.items():
                _situation, observed_next = bucket[0]
                self._table[(vkey, None, move)] = observed_next
            return False

        # Non-Markov: add the hidden phase variable and learn the phase schedule.
        if hidden_name not in self.hidden:
            self.hidden.append(hidden_name)

        for (vkey, move) in order:
            bucket = by_vkey[(vkey, move)]
            # Distinct next-states in first-seen order define the phase period.
            distinct: list = []
            seen: set = set()
            for _situation, nxt in bucket:
                canon = nxt.canonical()
                if canon not in seen:
                    seen.add(canon)
                    distinct.append(nxt)
            period = len(distinct)
            phases = list(range(period))
            # Phase schedule + successor (cyclic), recorded against the visible key.
            self._phase_seq[vkey] = phases[:]
            for i in phases:
                self._phase_succ[(vkey, i)] = phases[(i + 1) % period]
                # Augmented transition: (visible_key, phase_i, move) -> the i-th observed next.
                self._table[(vkey, i, move)] = distinct[i]
        return True

    def augment(self, situation: AbstractSituation, visible_key: Optional[tuple] = None) -> AbstractSituation:
        """Return ``situation`` with the learned hidden phase scalar attached (the NEXT phase
        from the schedule), so a caller can carry the hidden state forward across predictions.

        Used by tests / the planner to turn a raw (visible-only) AbstractSituation into the
        hidden-augmented AbstractSituation whose ``predict`` matches observation (SC-07). No-op when no
        hidden variable has been added."""
        if not self.hidden:
            return situation
        name = self.hidden[-1]
        if _scalar_value(situation, name) is not None:
            return situation
        vkey = visible_key if visible_key is not None else _visible_key(situation, self.hidden)
        phase = self._current_hidden(situation, vkey)
        if phase is None:
            return situation
        return _with_scalar(situation, name, phase)

    def advance_hidden(self, situation: AbstractSituation) -> AbstractSituation:
        """Advance the hidden phase of ``situation`` to its successor (the period-cycle step).
        Returns ``situation`` unchanged when no hidden variable is present or no successor is
        known. Lets a caller roll the phase forward deterministically between moves."""
        if not self.hidden:
            return situation
        name = self.hidden[-1]
        cur = _scalar_value(situation, name)
        if cur is None:
            return situation
        vkey = _visible_key(situation, self.hidden)
        nxt = self._phase_succ.get((vkey, cur))
        if nxt is None:
            return situation
        return _with_scalar(situation, name, nxt)

    # ------------------------------------------------------------------- terminal learning
    def learn_outcome(self, situation: AbstractSituation, move: Optional[int], outcome: Outcome) -> None:
        """Record an observed terminal Outcome for a transition (used to confirm a loss trigger
        was both PREDICTED and OBSERVED as ``over`` -- SC-17)."""
        self._outcome_table[self._transition_key(situation, move)] = outcome

    def _transition_key(self, situation: AbstractSituation, move: Optional[int]) -> tuple:
        vkey = _visible_key(situation, self.hidden)
        return (vkey, self._current_hidden(situation, vkey), move)

    # --------------------------------------------------------------------- MoveEffect API
    def classify(
        self,
        before: AbstractSituation,
        after: AbstractSituation,
        goal_distance: Callable[[AbstractSituation], object],
    ) -> str:
        """Instance shortcut for :func:`classify_move_effect` (the TS-12 entry point)."""
        return classify_move_effect(before, after, goal_distance)
