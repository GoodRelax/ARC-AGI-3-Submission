"""[Use Case] ModelBasedPolicy — the Phase B turn loop, belief owner, explorer wirer.

A new `DecisionPolicy` (FR-100) substitutable for `GraphExplorerPolicy` with NO change
to `OurSearchAgent` (ADR-009). It OWNS all Phase-B belief as instance fields (model,
goal, novelty counts, held plan, counters) and resets THEM itself on a level boundary
(FR-116) — it adds NO field to `Memory` and never calls `Memory.reset()` (that is the
owned explorer's job, F-03). It OWNS ONE `GraphExplorerPolicy` that is the SOLE owner
of segmentation + the state-graph + level-reset bookkeeping, used both to keep state
coherent and as the fallback action source (FR-140).

Each turn (spec §3.5, §4.3):
  1. reset OWN belief fields if `levels_completed` changed (FR-116);
  2. let the owned explorer process the frame (segments once, resets Memory on a
     boundary, picks a legal explorer-class action) — F-02/F-03;
  3. read the resulting `ObjectSet`/`observed_hash`; validate vs the predicted hash,
     learning from the last transition and updating mismatch/replan counters;
  4. on a win, synthesize/refine the `GoalPredicate` (FR-130/132);
  5. EITHER emit a planned action (model trusted AND goal/novelty plan in budget) OR
     return the explorer's fallback action (FR-140). Plan-in-simulation only (ADR-007).

Recoverable confidence trust (ADR-010, FR-164..FR-166, v0.6): trust is PURELY the
continuous `self._confidence in [0,1]` — it DECAYS `*CONFIDENCE_DECAY` on a
predict-vs-observe mismatch and RECOVERS `+CONFIDENCE_RECOVER*(1-conf)` on a correct
prediction, resetting to `CONFIDENCE_INIT` on a level boundary. The model is "trusted
to plan" iff `confidence >= CONFIDENCE_TRUST_THRESHOLD` (plus the structural §1.8
predicates). There is NO per-level permanent untrust latch and NO `M_MAX_REPLANS`
rest-of-level guard: a level that dipped below the threshold RECOVERS in-level on the
next correct prediction. Abduction is DECOUPLED from model trust (FR-153): directed
exploration (FR-153b) runs regardless of confidence; only abduce-PLAN (FR-153a)
requires `confidence >= threshold`. RESET is billed (C-10): prefer fallback, cap at
`MAX_RESETS_PER_LEVEL` (FR-145).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np

from arcengine import FrameData, GameAction

from agent.goal import (
    Affordance,
    GoalPredicate,
    Identity,
    MIN_CONTROLLABILITY_SUPPORT,
    SelectionContext,
    _controllable,
)
from agent.goal_inference import GoalHypothesis, GoalInference, GoalProposer
from agent.learn import learn_transition
from agent.planner import Plan, Planner
from agent.policy import (
    GraphExplorerPolicy,
    _action_keys,
    _cap_reasoning,
    _to_game_action,
)
from agent.segment import ObjectSet, detect_hud, latest_grid, node_hash, segment
from agent.state_graph import RESET_ID, ActionKey, Memory
from agent.world_model import WorldModel, affordance_map

logger = logging.getLogger(__name__)

__all__ = [
    "ModelBasedPolicy",
    "MAX_CONTRADICTION_RATE",
    "CONFIDENCE_INIT",
    "CONFIDENCE_DECAY",
    "CONFIDENCE_RECOVER",
    "CONFIDENCE_TRUST_THRESHOLD",
    "update_confidence",
    "MAX_RESETS_PER_LEVEL",
    "ABDUCTION_ACTION_BUDGET",
    "RECONSIDER_WINDOW",
    "AVATAR_MARGIN",
    "MAX_AVATAR_COAST",
]

# Trust predicate threshold (§1.8): fall back if the worst contradiction rate exceeds
# this. Module-level, no concrete game value (NFR-103).
MAX_CONTRADICTION_RATE: float = 0.34

# --- Recoverable model confidence (ADR-010, FR-164..FR-166) -------------------------
# Trust is PURELY this continuous signal, fully recoverable in-level. The v0.4 binary
# `_model_untrusted_for_level` latch AND the v0.5 `M_MAX_REPLANS`/`_replan_count`-latch
# are RETIRED entirely (the planner is bounded per turn by node/depth caps — NFR-101 —
# so unbounded within-turn replanning is not a risk).
CONFIDENCE_INIT: float = 1.0  # confidence at level start (FR-164); reset on a boundary
CONFIDENCE_DECAY: float = 0.7  # multiplicative drop on a MISMATCH (FR-164):
#                                1.0 -> 0.7 -> 0.49 -> 0.343, so 3 straight mismatches drop trust
CONFIDENCE_RECOVER: float = 0.34  # asymptotic recovery toward 1.0 on a MATCH (FR-164):
#                                   from 0.343 one match -> 0.343+0.34*0.657 = 0.566 >= 0.4
CONFIDENCE_TRUST_THRESHOLD: float = 0.4  # trusted-enough-to-plan iff confidence >= this (FR-165)

MAX_RESETS_PER_LEVEL: int = 2  # billed RESET cap per level (FR-145)
# Max real actions spent cycling abductive hypotheses per level (FR-156, N-04). On
# reaching it the policy stops abducing and falls back to novelty, then the explorer.
ABDUCTION_ACTION_BUDGET: int = 40
# REAL-ACTION turns a hypothesis is pursued without an evidence gain before it is
# down-weighted (FR-154, N-13). Counted from when the hypothesis becomes active; reset
# on a hypothesis switch or a level boundary.
RECONSIDER_WINDOW: int = 3

# --- avatar state observer (spec 87) -------------------------------------------------
# The static-fallback regime TRACKS the avatar as a state estimate instead of binding a
# never-moving static fragment. `AVATAR_MARGIN`: gate slack added to |velocity| — a present
# in-colour object matches the prediction only within `|velocity| + AVATAR_MARGIN` cells.
# `MAX_AVATAR_COAST`: turns the prediction is carried through a miss before the track is
# dropped. General constants, no game value (NFR-103/NFR-130).
AVATAR_MARGIN: float = 4.0
MAX_AVATAR_COAST: int = 5
# Sentinel pin used WHILE COASTING: colour -1 can never exist, so `goal._controllable`
# resolves it to None (the controllable is genuinely UNBOUND this turn — for the policy AND
# the planner) instead of falling through to the static fragment (spec 87 FR-195/M-D).
_AVATAR_COAST_PIN: Identity = (-1, -1)


def update_confidence(confidence: float, matched: bool) -> float:
    """Recoverable model-confidence update (FR-164), deterministic + bounded (NFR-114).

    A pure function of `(prior, event, named constants)` — no randomness, no I/O. Both
    branches map `[0, 1] -> [0, 1]`: decay multiplies by `CONFIDENCE_DECAY in (0,1)`;
    recovery moves a `CONFIDENCE_RECOVER` fraction of the remaining gap toward 1.0, so
    the result never exceeds 1.0. A correctly predicted no-op (FR-162) is a MATCH and
    feeds recovery, not decay.
    """
    if matched:  # correct prediction (including a correctly predicted no-op)
        return confidence + CONFIDENCE_RECOVER * (1.0 - confidence)  # asymptotic -> 1.0
    return confidence * CONFIDENCE_DECAY  # multiplicative drop


class ModelBasedPolicy:
    """Phase B DecisionPolicy: predict -> plan -> act -> validate -> update (FR-100)."""

    def __init__(
        self,
        explorer: Optional[GraphExplorerPolicy] = None,
        planner: Optional[Planner] = None,
        goal_inference: Optional[GoalInference] = None,
    ) -> None:
        # OWNED explorer: sole segmentation owner + fallback source (FR-140). Injectable
        # for tests; defaults to a fresh instance.
        self._explorer: GraphExplorerPolicy = explorer or GraphExplorerPolicy()
        self._planner: Planner = planner or Planner()
        self._goal_inference: GoalInference = goal_inference or GoalInference()
        # Proactive abductive goal proposer over the fixed v1 library (ADR-0002).
        self._proposer: GoalProposer = GoalProposer()

        # --- OWN Phase-B belief state (NOT in Memory; reset on level boundary) ---
        self._model: WorldModel = WorldModel()
        self._goal: Optional[GoalPredicate] = None
        self._novelty_counts: dict[int, int] = {}
        self._held_plan: tuple[ActionKey, ...] = ()
        self._next_step_predicted_hash: Optional[int] = None
        self._prev_objset: Optional[ObjectSet] = None
        self._prev_action: Optional[ActionKey] = None
        self._prev_win_pre: Optional[ObjectSet] = None  # pre-win ObjectSet for goal diff
        # Recoverable model confidence in [0,1] (FR-164). Trust is PURELY this signal,
        # fully recoverable in-level. RETIRED (v0.6): `_mismatch_streak`,
        # `_replan_count`-as-latch, `M_MAX_REPLANS`, `_model_untrusted_for_level`.
        self._confidence: float = CONFIDENCE_INIT
        self._resets_emitted: int = 0
        self._belief_level: Optional[int] = None

        # --- abduction belief (ADR-0002, FR-150..FR-156) -------------------------
        self._ranked_hyps: tuple[GoalHypothesis, ...] = ()  # ranked abduced goals (FR-152)
        self._active_hyp_idx: int = 0           # which ranked hypothesis is pursued (FR-153)
        self._abduction_actions: int = 0        # real actions spent on abduction (FR-156)
        self._hyp_active_turns: int = 0          # real-action turns the active hyp ran (FR-154)
        self._last_goal_distance: Optional[int] = None  # for the no-progress check (FR-154)
        # Observed distance delta credited to each action under directed exploration
        # (FR-153b): the shaping heuristic ordering the frontier toward smaller goal-
        # distance over OBSERVED states (no model/prediction needed).
        self._action_distance_delta: dict[ActionKey, int] = {}
        # Pursuit hardening: the goal's `controllable` anchor identity last turn + whether it
        # FLIPPED this turn. Goal-distance is only comparable across turns when the anchor bound
        # to the SAME object; a flip (e.g. the mover transiently HUD-masked) makes the distance
        # jump, poisoning the FR-153b shaping and falsely tripping the no-progress down-weight —
        # so on a flip we skip BOTH the delta credit and the no-progress count (extends FR-171
        # from the grounded pin to the actual per-turn goal binding).
        self._last_ctrl_ident: Optional[Identity] = None
        self._binding_unstable: bool = False

        # --- grounded controllability (ADR-012, FR-167..FR-171) -------------------
        # The per-turn SelectionContext (affordance map + sticky pinned identity) threaded
        # into ALL role binding; the sticky controllable identity (FR-171 stickiness) and
        # whether it changed this turn (FR-171 baseline reset).
        self._selection_context: Optional[SelectionContext] = None
        self._controllable_identity: Optional[Identity] = None
        self._controllable_binding_changed: bool = False

        # --- avatar state observer (spec 87, FR-192) ------------------------------
        # A tracked estimate (colour, centroid, velocity) seeded ONLY from grounded movers;
        # in the static-fallback regime it predicts the avatar (own velocity) + matches the
        # nearest in-colour object, else coasts UNBOUND — never the static fragment. `_bind_source`
        # records how the controllable was bound this turn (grounded/observer/coast/none) for FR-199.
        self._avatar_track: Optional[
            tuple[int, tuple[float, float], tuple[float, float]]
        ] = None
        self._avatar_coast: int = 0
        self._bind_source: str = "none"

        # --- introspection capture (DEBUG; env-gated, no behavior change when off) ------
        # When ARC_INTROSPECT names a file, append one JSON line per turn (grid + masks +
        # affordances + controllable + goal + action) for the offline viewer (tools/introspect.py).
        self._capture_path: Optional[str] = os.environ.get("ARC_INTROSPECT") or None
        self._cap: Optional[dict] = None
        self._cap_t: int = 0
        self._cap_grid = None
        self._cap_hud = None

        # --- scripted action replay (DEBUG; env-gated, no behavior change when off) ------
        # ARC_REPLAY="1,1,4,..." plays a fixed action sequence (one id per turn) THROUGH the
        # full observe -> learn -> from_win -> capture loop, so a KNOWN solution can drive a
        # first WIN that bootstraps goal inference (FR-130). The sequence is supplied at RUNTIME
        # (no game literal in the source). Only simple integer (movement) actions are supported.
        _replay_env = (os.environ.get("ARC_REPLAY") or "").replace(" ", "")
        self._replay: Optional[list[ActionKey]] = (
            [int(x) for x in _replay_env.split(",") if x] or None
        )
        self._replay_idx: int = 0

    # ------------------------------------------------------------------ decide

    def decide(self, observation: FrameData, memory: Memory) -> GameAction:
        """One Phase-B turn (FR-100). Never crashes: degrades to the explorer (FR-142)."""
        # 1. Reset OWN belief on any level-boundary change (FR-116). The explorer resets
        #    Memory ITSELF when it processes the frame (F-03) — we never call it here.
        win_this_turn = False
        if self._belief_level is not None and observation.levels_completed != self._belief_level:
            win_this_turn = observation.levels_completed > self._belief_level
            # capture pre-win ObjectSet BEFORE wiping beliefs (for goal diff, FR-130).
            pre_win = self._prev_objset
            self._reset_belief()
            self._prev_win_pre = pre_win
        self._belief_level = observation.levels_completed

        # 2. Owned explorer processes the frame: single segmentation + Memory reset +
        #    a legal explorer-class action (the fallback). F-02/F-03.
        try:
            explorer_action = self._explorer.decide(observation, memory)
        except Exception:  # pragma: no cover - explorer is robust, defensive only
            logger.exception("owned explorer failed; emitting RESET")
            return GameAction.RESET

        # 3. Read the resulting ObjectSet / observed hash. We re-derive the ObjectSet
        #    from the SAME grid + the explorer's OWN HUD history so it matches the
        #    explorer's segmentation exactly (single logical segmentation, F-02). On any
        #    failure, just return the explorer's action (FR-142).
        try:
            obj_set, observed_hash = self._segment_like_explorer(observation, memory)
        except Exception:
            logger.exception("phase-B segmentation read failed; using explorer action")
            return explorer_action

        # 4. On a win, synthesize/refine the goal from the pre-win transition (FR-130/132).
        if win_this_turn and self._prev_win_pre is not None:
            self._update_goal(self._prev_win_pre, obj_set)
            self._prev_win_pre = None

        # 5. Validate prediction vs observation; learn from the last transition (FR-121).
        try:
            self._validate_and_learn(obj_set, observed_hash)
        except Exception:
            logger.exception("phase-B learn/validate failed; using explorer action")
            self._prev_objset, self._prev_action = obj_set, None
            return explorer_action

        # v0.7 (ADR-012): build this turn's SelectionContext from the just-updated model —
        # the affordance map + the sticky/FR-171 controllable binding — so all downstream
        # role binding (distance shaping, abduction, planning) grounds `controllable` in
        # OBSERVED dynamics rather than a static color/size guess.
        self._update_selection_context(obj_set)
        if self._capture_path:
            self._cap = self._build_capture(observation, obj_set)

        # bump the novelty visit count for the observed state (FR-131).
        self._novelty_counts[observed_hash] = self._novelty_counts.get(observed_hash, 0) + 1

        # DIAGNOSTIC (option-1, 2026-05-31): does the STATIC _controllable guess agree
        # with the object the world model has actually OBSERVED to move? Directed
        # exploration (FR-153b) binds operand A to the static guess; if that guess is
        # not the truly-controllable object, every goal-distance measurement steers the
        # wrong blob. Emitted AFTER learning so the model reflects this turn's transition.
        self._log_controllable_diag(obj_set)

        legal = _action_keys(observation.available_actions, obj_set)

        # Track per-action distance deltas for directed exploration BEFORE choosing
        # this turn's action (credit the LAST abductive action with the observed change
        # in goal-distance over OBSERVED states — the FR-153b shaping signal, N-08).
        active_goal = self._active_goal()
        if active_goal is not None:
            self._credit_distance_progress(active_goal, obj_set)

        # DEBUG (env-gated): scripted action replay. Emit the next action of ARC_REPLAY AFTER
        # this turn's learning / win-handling / capture have run, so a known solution drives the
        # real loop (and a first WIN reaches from_win). No-op when ARC_REPLAY is unset.
        if self._replay is not None and self._replay_idx < len(self._replay):
            scripted = self._replay[self._replay_idx]
            self._replay_idx += 1
            if self._legal(scripted, legal):
                return self._emit_planned(scripted, obj_set, legal, mode="replay")
            logger.warning(
                "ARC_REPLAY action %s illegal at idx %d; legal=%s",
                scripted, self._replay_idx - 1, legal,
            )

        # 6. If we still hold a valid plan and prediction matched, continue it (FR-122).
        held_action = self._next_held_action(legal)
        if held_action is not None:
            return self._emit_planned(held_action, obj_set, legal, mode="plan")

        # 7. With a CONFIRMED goal: plan toward it when trusted (FR-125). The pre-win
        #    (no-confirmed-goal) regime is handled by ABDUCTION first (step 7b), which
        #    only degrades to plain novelty as its DEEPEST fallback (FR-153, §4.3c) —
        #    so we do NOT run the generic novelty planner ahead of abduction.
        try:
            if self._goal is not None and self._is_trusted(obj_set, legal):
                # No replan latch (v0.6): the planner is bounded per call by node/depth
                # caps (NFR-101), and confidence (decayed on each mismatch) bounds
                # replanning across turns. There is NO rest-of-level fallback (FR-146).
                plan = self._make_plan(obj_set, legal)
                if plan is not None and plan.actions:
                    self._held_plan = plan.actions
                    first = plan.actions[0]
                    if self._legal(first, legal):
                        return self._emit_planned(first, obj_set, legal, mode="plan")
        except Exception:
            logger.exception("phase-B planning failed; using explorer action")

        # 7b. Proactive ABDUCTION (ADR-0002): no CONFIRMED goal yet, within budget.
        #     Pursue the top-ranked hypothesis DUAL-MODE (N-08): a model-based plan when
        #     a usable WorldModel exists (FR-153a), else DIRECTED EXPLORATION ordering the
        #     explorer's frontier by goal.distance over observed states (FR-153b).
        try:
            abductive = self._pursue_abduction(obj_set, legal, observation)
            if abductive is not None:
                return abductive
        except Exception:
            logger.exception("phase-B abduction failed; using explorer action")

        # 7c. DEEPEST fallback before the explorer: plain count-based novelty (FR-131),
        #     used only when no hypothesis is confidently rankable / the budget is spent.
        try:
            if self._goal is None and self._is_trusted(obj_set, legal):
                plan = self._make_plan(obj_set, legal)
                if plan is not None and plan.actions and self._legal(plan.actions[0], legal):
                    self._held_plan = plan.actions
                    return self._emit_planned(plan.actions[0], obj_set, legal, mode="novelty")
        except Exception:
            logger.exception("phase-B novelty fallback failed; using explorer action")

        # 8. Fallback to the owned explorer's action (FR-140). Prefer it over RESET
        #    unless RESET is the only legal action and within the per-level cap (FR-145).
        self._held_plan = ()
        self._next_step_predicted_hash = None
        self._prev_objset, self._prev_action = obj_set, self._key_of(explorer_action, obj_set)
        logger.info(
            "PB mode=fallback rules=%d hyps=%d goal_known=%s conf=%.2f abduct=%d",
            len(self._model.rules), len(self._ranked_hyps),
            self._goal is not None, self._confidence,
            self._abduction_actions,
        )
        self._flush_capture(self._prev_action, "fallback")
        return self._guard_reset(explorer_action, observation)

    # ------------------------------------------------------------- sub-steps

    def _segment_like_explorer(
        self, observation: FrameData, memory: Memory
    ) -> tuple[ObjectSet, int]:
        """Re-derive the ObjectSet/hash the explorer just computed (F-02).

        Uses the explorer's OWN `_grid_history` (already updated by its `decide`) minus
        the current grid, so the HUD mask matches what the explorer used this turn.
        `memory.current_hash` is the explorer's authoritative observed hash.
        """
        grid = latest_grid(observation.frame)
        history = list(self._explorer._grid_history[:-1])  # exclude current (just pushed)
        hud_mask = detect_hud(history, grid)
        obj_set = segment(grid, hud_mask=hud_mask)
        observed_hash = (
            memory.current_hash if memory.current_hash is not None else node_hash(obj_set)
        )
        self._cap_grid, self._cap_hud = grid, hud_mask  # for the introspection capture
        return obj_set, observed_hash

    def _validate_and_learn(self, obj_set: ObjectSet, observed_hash: int) -> None:
        """Compare predicted vs observed; learn the last transition (FR-121, FR-102)."""
        if self._prev_objset is not None and self._prev_action is not None:
            # Learn the (before, action, after) transition regardless (FR-102).
            self._model = learn_transition(
                self._model, self._prev_objset, self._prev_action, obj_set
            )
            # Validate against the prediction we recorded last turn (FR-121, FR-164).
            # Trust is purely recoverable confidence: a mismatch DECAYS it (and discards
            # the stale plan), a match RECOVERS it toward 1.0. No latch, no streak counter
            # — a level that dipped below the threshold climbs back on the next match.
            if self._next_step_predicted_hash is not None:
                matched = observed_hash == self._next_step_predicted_hash
                self._confidence = update_confidence(self._confidence, matched)
                if not matched:
                    self._held_plan = ()  # discard stale plan -> replan (FR-121)
        self._next_step_predicted_hash = None

    def _update_goal(self, pre: ObjectSet, post: ObjectSet) -> None:
        """Synthesize or refine the GoalPredicate from a win transition (FR-130/132)."""
        if self._goal is None:
            self._goal = self._goal_inference.from_win(pre, post)
        else:
            self._goal = self._goal_inference.refine(self._goal, pre, post)

    def _next_held_action(self, legal: list[ActionKey]) -> Optional[ActionKey]:
        """Pop the held plan's next action if still valid+legal (FR-122), else None."""
        if not self._held_plan:
            return None
        first, rest = self._held_plan[0], self._held_plan[1:]
        if not self._legal(first, legal):
            self._held_plan = ()  # illegal step -> replan (FR-141)
            return None
        self._held_plan = rest
        return first

    def _is_trusted(self, obj_set: ObjectSet, legal: list[ActionKey]) -> bool:
        """Boolean trust predicate (§1.8, FR-165). Any failing clause => fall back THIS
        TURN ONLY (FR-140); recovers automatically once confidence climbs back (FR-164).

        Trust is PURELY the recoverable confidence gate plus the structural predicates
        (contradiction rate, start-state rule match, no non-deterministic rule on path).
        There is NO permanent latch and NO rest-of-level fallback (ADR-010).
        """
        if self._confidence < CONFIDENCE_TRUST_THRESHOLD:  # recoverable gate (FR-165)
            return False
        if self._model.max_contradiction_rate() > MAX_CONTRADICTION_RATE:
            return False
        # A rule must match the start state for SOME candidate action; OR we have no
        # goal yet (novelty mode), in which case unknown effects are themselves useful.
        if self._goal is not None:
            if not any(self._model.matches(obj_set, a) for a in legal):
                return False
            # No planned action may touch a non-deterministic rule.
            if any(self._model.has_nondeterministic_match(obj_set, a) for a in legal):
                return False
        return True

    def _make_plan(self, obj_set: ObjectSet, legal: list[ActionKey]) -> Optional[Plan]:
        """Plan in simulation toward the goal, else toward novelty (FR-123/131)."""
        return self._planner.plan(
            start=obj_set,
            model=self._model,
            goal=self._goal,
            legal=legal,
            novelty=self._goal_inference.novelty,
            counts=self._novelty_counts,
            context=self._selection_context,  # grounded controllable (FR-170)
        )

    # ------------------------------------------------------------- emission

    def _emit_planned(
        self,
        action_key: ActionKey,
        obj_set: ObjectSet,
        legal: list[ActionKey],
        mode: str,
    ) -> GameAction:
        """Emit one planned action and record the predicted successor hash (FR-120/144)."""
        succ = None
        try:
            succ = self._model.predict(obj_set, action_key)
        except Exception:
            succ = None
        self._next_step_predicted_hash = node_hash(succ) if succ is not None else None
        self._prev_objset, self._prev_action = obj_set, action_key
        reasoning = self._reasoning(mode, action_key)
        # Instrumentation (PRINCIPLES: instrumentation over compactness): make the
        # learning observable per turn — mode, #learned rules, #abduced hypotheses,
        # whether a confirmed goal exists, the recoverable CONFIDENCE (so in-level
        # recovery is observable, FR-164), abduction budget spent.
        logger.info(
            "PB mode=%s rules=%d hyps=%d goal_known=%s conf=%.2f abduct=%d",
            mode, len(self._model.rules), len(self._ranked_hyps),
            self._goal is not None, self._confidence,
            self._abduction_actions,
        )
        self._flush_capture(action_key, mode)
        return _to_game_action(action_key, reasoning)

    def _guard_reset(
        self, explorer_action: GameAction, observation: FrameData
    ) -> GameAction:
        """Prefer the explorer's non-RESET action; bound billed RESETs (FR-145, C-10)."""
        if explorer_action is GameAction.RESET:
            # Try to substitute a legal non-RESET explorer-class action first.
            for aid in observation.available_actions:
                if aid != RESET_ID and aid != 6:
                    if self._resets_emitted >= MAX_RESETS_PER_LEVEL:
                        action = GameAction.from_id(int(aid))
                        action.reasoning = _cap_reasoning(
                            {"policy": "ModelBasedPolicy", "mode": "fallback-no-reset"}
                        )
                        return action
            if self._resets_emitted < MAX_RESETS_PER_LEVEL:
                self._resets_emitted += 1
                reset = GameAction.RESET
                reset.reasoning = _cap_reasoning(
                    {"policy": "ModelBasedPolicy", "mode": "fallback-reset"}
                )
                return reset
            # Over the cap and no non-RESET legal: still emit RESET (recovery, C-5).
            return GameAction.RESET
        return explorer_action

    # ------------------------------------------------------- abduction (ADR-0002)

    def _active_goal(self) -> Optional[GoalPredicate]:
        """The goal pursued this turn (§4.3c). CONFIRMED goal wins; else the top
        non-down-weighted ranked hypothesis (FR-153); else None (-> novelty, FR-131).

        Honors the ABDUCTION_ACTION_BUDGET ceiling (FR-156, N-04): once spent, abduction
        stops and we report no abductive goal so the deeper novelty fallback takes over.
        """
        if self._goal is not None:  # CONFIRMED (FR-155)
            return self._goal
        if self._abduction_actions >= ABDUCTION_ACTION_BUDGET:  # bound (FR-156)
            return None
        # v0.6 (H-1): abduction gates on the budget + per-hypothesis down-weighting ONLY,
        # NEVER on model trust. The retired `_model_untrusted_for_level` check is removed,
        # so directed-exploration abduction (FR-153b) survives a low-confidence stretch.
        for h in self._ranked_hyps[self._active_hyp_idx:]:
            if not h.down_weighted:
                return h.predicate
        return None

    def _model_is_usable(self, obj_set: ObjectSet, legal: list[ActionKey]) -> bool:
        """Dual-mode switch (N-08): a WorldModel is usable once a rule matches the
        current state and is not flagged non-deterministic. Before that (pre-first-win
        regime) abduction runs in DIRECTED-EXPLORATION mode (FR-153b)."""
        if not self._model.rules:
            return False
        return any(
            self._model.matches(obj_set, a)
            and not self._model.has_nondeterministic_match(obj_set, a)
            for a in legal
        )

    def _abduce_plan_enabled(self, obj_set: ObjectSet, legal: list[ActionKey]) -> bool:
        """FR-153a gate: model-based abduce-PLAN requires a usable model AND confidence
        at/above the trust threshold. Otherwise abduction runs in directed-exploration
        mode (FR-153b), which does NOT use the model and runs REGARDLESS of confidence
        (H-1). This is the ONLY place confidence touches abduction — low confidence
        flips abduce-plan to abduce-explore, it never abandons abduction.
        """
        return (
            self._model_is_usable(obj_set, legal)
            and self._confidence >= CONFIDENCE_TRUST_THRESHOLD
        )

    def _credit_distance_progress(
        self, goal: GoalPredicate, obj_set: ObjectSet
    ) -> None:
        """FR-153b shaping over OBSERVED states: credit the last abductive action with
        the observed change in goal-distance, and run the FR-154 no-progress check.

        Records the per-action distance delta (smaller goal-distance == progress) so the
        directed-exploration ordering prefers actions that reduce the distance. Also
        bumps the active-hypothesis evidence on a decrease and counts no-progress turns.
        """
        # Pursuit hardening: skip progress accounting across a `controllable` anchor FLIP — the
        # distance is not comparable when the goal's controllable bound to a DIFFERENT object
        # this turn (extends FR-171 from the grounded pin to the actual goal binding).
        self._binding_unstable = False
        if "controllable" in goal.selectors:
            ctrl = _controllable(obj_set, self._selection_context)
            ident = (ctrl.shape_hash, ctrl.color) if ctrl is not None else None
            # FR-198 (spec 87, code-review M-1): an anchor FLIP that breaks distance comparability is a
            # change of the bound OBJECT — keyed on its COLOUR (the observer's carried token) plus
            # to/from unbound. A same-COLOUR `shape_hash` drift (the observer tracking the avatar AS its
            # shape changes) is the SAME anchor and must NOT be flagged, else the FR-153b shaping is
            # suppressed on exactly the tracked-drift regime the observer exists to serve.
            colour = ident[1] if ident is not None else None
            prev_colour = self._last_ctrl_ident[1] if self._last_ctrl_ident is not None else None
            if colour != prev_colour:
                self._binding_unstable = True
            self._last_ctrl_ident = ident
            if self._binding_unstable:
                self._last_goal_distance = None  # baseline invalid across a flip
                return
        try:
            dist = goal.distance(obj_set, self._selection_context)
        except Exception:
            return
        prev = self._last_goal_distance
        if prev is not None and self._prev_action is not None:
            delta = dist - prev  # negative == got closer (progress)
            key = self._prev_action
            # Keep the BEST (most negative) observed delta per action (shaping, N-08).
            if key not in self._action_distance_delta or delta < self._action_distance_delta[key]:
                self._action_distance_delta[key] = delta
            if delta < 0:  # progress: evidence gain, reset the no-progress window (N-06)
                self._bump_active_evidence(1)
                self._hyp_active_turns = 0
        self._last_goal_distance = dist

    def _bump_active_evidence(self, delta: int) -> None:
        """Immutable evidence update on the active hypothesis (FR-152, N-06)."""
        if self._goal is not None or not self._ranked_hyps:
            return
        i = self._active_hyp_idx
        if 0 <= i < len(self._ranked_hyps):
            self._ranked_hyps = (
                self._ranked_hyps[:i]
                + (self._ranked_hyps[i].with_evidence(delta),)
                + self._ranked_hyps[i + 1:]
            )

    def _pursue_abduction(
        self, obj_set: ObjectSet, legal: list[ActionKey], observation: FrameData
    ) -> Optional[GameAction]:
        """Pursue the top-ranked goal hypothesis DUAL-MODE (FR-153). None => no
        abductive action (caller falls to novelty/explorer)."""
        # A CONFIRMED goal supersedes abduction (handled by the step-7 plan path).
        if self._goal is not None:
            return None
        # v0.6 (H-1): the ONLY gates are the abduction budget (FR-156) and per-hypothesis
        # down-weighting (FR-154). Model trust NEVER suspends abduction — a low-confidence
        # stretch merely flips abduce-plan to abduce-explore below.
        if self._abduction_actions >= ABDUCTION_ACTION_BUDGET:  # bound (FR-156, N-04)
            return None

        # (Re)propose if we have no live hypothesis set yet (ranks then truncates, N-05).
        if not self._ranked_hyps:
            self._ranked_hyps = tuple(
                self._proposer.hypotheses(obj_set, self._selection_context)
            )
            self._active_hyp_idx = 0
            self._hyp_active_turns = 0
            self._last_goal_distance = None

        goal = self._active_goal()
        if goal is None:
            return None  # nothing confidently rankable -> novelty (FR-153 last clause)

        # FR-154 no-progress reconsideration: a hypothesis pursued RECONSIDER_WINDOW
        # real-action turns without an evidence gain is down-weighted; advance to the
        # next-ranked one (the per-hypothesis down-weight fires BEFORE the FR-146 latch).
        if self._hyp_active_turns >= RECONSIDER_WINDOW:
            self._down_weight_active()
            goal = self._active_goal()
            if goal is None:
                return None

        action: Optional[ActionKey] = None
        mode = "abduce-explore"
        if self._abduce_plan_enabled(obj_set, legal):  # FR-153a: usable model AND trusted
            action = self._plan_toward(goal, obj_set, legal)
            mode = "abduce-plan"
        if action is None:
            # FR-153b: directed exploration ordering the frontier by goal.distance over
            # OBSERVED states. Also the graceful path when no model-plan was found — a
            # missing plan is NOT a surprise, so it must not down-weight the hypothesis.
            action = self._directed_exploration_action(goal, obj_set, legal)
            mode = "abduce-explore"
        if action is None:
            return None

        self._abduction_actions += 1   # real action spent on abduction (FR-156)
        # Pursuit hardening: a turn where the `controllable` anchor FLIPPED is not a fair
        # no-progress turn (distance was unmeasurable) — don't advance the reconsider window.
        if not self._binding_unstable:
            self._hyp_active_turns += 1
        return self._emit_planned(action, obj_set, legal, mode=mode)

    def _plan_toward(
        self, goal: GoalPredicate, obj_set: ObjectSet, legal: list[ActionKey]
    ) -> Optional[ActionKey]:
        """FR-153a: plan toward the hypothesis via the planner; emit actions[0]."""
        plan = self._planner.plan(
            start=obj_set,
            model=self._model,
            goal=goal,
            legal=legal,
            novelty=self._goal_inference.novelty,
            counts=self._novelty_counts,
            context=self._selection_context,  # grounded controllable (FR-170)
        )
        # No replan latch (v0.6): the planner is bounded per call (NFR-101); confidence
        # bounds replanning across turns. No `M_MAX_REPLANS`, no rest-of-level fallback.
        if plan is not None and plan.actions and self._legal(plan.actions[0], legal):
            self._held_plan = plan.actions
            return plan.actions[0]
        # No plan found is NOT a surprise — the caller falls to directed exploration
        # (FR-153b). Down-weighting is driven only by the FR-154 no-progress window.
        return None

    def _directed_exploration_action(
        self, goal: GoalPredicate, obj_set: ObjectSet, legal: list[ActionKey]
    ) -> Optional[ActionKey]:
        """FR-153b: order the explorer's frontier toward smaller goal-distance.

        Pre-model directed exploration over OBSERVED states (no prediction): among the
        legal SIMPLE actions, prefer the one whose past emission most reduced the
        observed goal-distance (the shaping heuristic). Untried actions are explored
        before known-useless ones so the agent discovers which action moves the
        controllable object toward the target, then exploits it.
        """
        simple = sorted(k for k in legal if not isinstance(k, tuple))
        if not simple:
            return None
        deltas = self._action_distance_delta

        def key(a: ActionKey) -> tuple[int, int, int]:
            # untried (delta unknown) sorts AFTER a known-progress action but BEFORE a
            # known-useless one, so we keep exploiting a good action yet still probe.
            known = a in deltas
            d = deltas.get(a, 0)
            # 1st: known-progress (d<0) first; 2nd: smaller delta; 3rd: action id.
            tier = 0 if (known and d < 0) else (1 if not known else 2)
            return (tier, d, int(a))

        return min(simple, key=key)

    def _down_weight_active(self) -> None:
        """Down-weight the active hypothesis and advance to the next-ranked (FR-154).

        Immutable: rebuild the frozen hypothesis. A down-weighted hypothesis is not
        re-pursued until a level-boundary reset (FR-154). Reset the per-hypothesis
        window/distance so the next hypothesis starts clean (N-13).
        """
        if not self._ranked_hyps:
            return
        i = self._active_hyp_idx
        if 0 <= i < len(self._ranked_hyps):
            self._ranked_hyps = (
                self._ranked_hyps[:i]
                + (self._ranked_hyps[i].down_weight(),)
                + self._ranked_hyps[i + 1:]
            )
        self._active_hyp_idx += 1
        self._hyp_active_turns = 0
        self._last_goal_distance = None

    # ------------------------------------------------ grounded controllability (ADR-012)

    def _update_selection_context(self, obj_set: ObjectSet) -> None:
        """Build this turn's SelectionContext from the current model (FR-167/FR-170).

        Derives the affordance map (pure query — emits no real actions, NFR-117), updates
        the sticky controllable binding (FR-168/FR-171), and stores a context pinned to that
        identity so every consumer (distance shaping, abduction, planning) binds the SAME
        observed avatar this turn.
        """
        amap = affordance_map(self._model)
        self._update_controllable_binding(amap, obj_set)
        self._selection_context = SelectionContext(
            affordances=amap, pinned_controllable=self._controllable_identity
        )

    def _update_controllable_binding(
        self, amap: dict[Identity, Affordance], obj_set: ObjectSet
    ) -> None:
        """Pick the sticky grounded controllable identity (FR-168/FR-171, M4).

        The candidate is the selector's own grounded pick (reusing its shape_hash-invariant
        tie-break for consistency). Stickiness: keep the current binding while it still
        qualifies and is PRESENT, switching only when the candidate STRICTLY exceeds its
        translate support — so the binding does not churn near the threshold. On a CHANGE,
        reset the no-progress baseline so a referent switch is not mistaken for progress
        (FR-171); `_credit_distance_progress` then skips the cross-switch delta.
        """
        probe = _controllable(obj_set, SelectionContext(affordances=amap))
        cand: Optional[Identity] = None
        if probe is not None:
            ident = (probe.shape_hash, probe.color)
            aff = amap.get(ident)
            if aff is not None and aff.translate_support >= MIN_CONTROLLABILITY_SUPPORT:
                cand = ident

        prev = self._controllable_identity
        new_ident = cand
        if prev is not None and prev != cand:
            prev_aff = amap.get(prev)
            prev_present = any((o.shape_hash, o.color) == prev for o in obj_set.objects)
            if (
                prev_aff is not None
                and prev_present
                and prev_aff.translate_support >= MIN_CONTROLLABILITY_SUPPORT
                and (
                    cand is None
                    or amap[cand].translate_support <= prev_aff.translate_support
                )
            ):
                new_ident = prev  # sticky: keep prev unless cand STRICTLY exceeds it

        # --- avatar observer (spec 87): seed the track from a GROUNDED bind; otherwise (the old
        # static-fallback regime) TRACK the avatar by predicted position + colour rather than bind
        # the never-moving static fragment. The grounded branch above is untouched (NFR-131).
        if new_ident is not None:
            self._seed_avatar_track(new_ident, obj_set)
            self._bind_source = "grounded"
        else:
            new_ident = self._observer_bind(obj_set)

        # FR-171 + code-review M-1/L-3: reset the no-progress baseline only on a real referent switch
        # (a COLOUR change, or to/from the unbound coast sentinel), NOT on a same-colour `shape_hash`
        # refresh (the observer tracking the avatar through drift) — keyed identically to the FR-198
        # guard so the two baseline-reset sites cannot diverge.
        new_colour = new_ident[1] if new_ident is not None else None
        prev_colour = prev[1] if prev is not None else None
        self._controllable_binding_changed = new_colour != prev_colour
        if self._controllable_binding_changed:
            self._last_goal_distance = None  # skip the cross-switch progress delta
        self._controllable_identity = new_ident

    def _seed_avatar_track(self, ident: Identity, obj_set: ObjectSet) -> None:
        """Seed/update the avatar track from a GROUNDED bind (spec 87 FR-193). Velocity is the
        avatar's own grounded displacement since last turn (a constant-velocity estimate)."""
        bound = next(
            (o for o in obj_set.objects if (o.shape_hash, o.color) == ident), None
        )
        if bound is None:
            return
        prev = self._avatar_track
        vel = (
            (bound.centroid[0] - prev[1][0], bound.centroid[1] - prev[1][1])
            if prev is not None and prev[0] == bound.color
            else (0.0, 0.0)
        )
        self._avatar_track = (bound.color, bound.centroid, vel)
        self._avatar_coast = 0

    def _observer_bind(self, obj_set: ObjectSet) -> Optional[Identity]:
        """Static-fallback regime (no grounded mover): predict the avatar by its own velocity,
        take the diff against the in-colour objects, correct on a gated match, else coast —
        NEVER the static fragment (spec 87 FR-194..FR-196).

        Returns the matched avatar's CURRENT identity (pin refreshed this frame so it survives
        shape drift, C-B), the absent coast sentinel (so `_controllable` resolves to None while
        coasting, M-D), or None when there is no track to predict from.
        """
        track = self._avatar_track
        if track is None:
            # FR-196: no track yet -> fall through to the CURRENT cold-start behaviour (None pin ->
            # the FR-169 static pick). This is kept deliberately: the static fallback is the only
            # controllable in worlds whose mover is immovable / not-yet-moved, which the abduction
            # BOOTSTRAP relies on. The observer takes over the moment a grounded mover seeds a track,
            # so the white cross survives only on the first few cold-start turns (live obs run: 7/250).
            self._bind_source = "none"
            return None
        colour, (cr, cc), (vr, vc) = track
        pr, pc = cr + vr, cc + vc  # predict (the virtual level)
        cands = [o for o in obj_set.objects if o.color == colour]
        if cands:
            best = min(
                cands,
                key=lambda o: (
                    (o.centroid[0] - pr) ** 2 + (o.centroid[1] - pc) ** 2,
                    -o.size, o.bbox[0], o.bbox[1],   # deterministic tie-break (M-C)
                ),
            )
            d2 = (best.centroid[0] - pr) ** 2 + (best.centroid[1] - pc) ** 2
            gate = (vr * vr + vc * vc) ** 0.5 + AVATAR_MARGIN  # |velocity| + margin (relative)
            if d2 <= gate * gate:  # take diff -> within gate -> correct
                self._avatar_track = (
                    best.color, best.centroid,
                    (best.centroid[0] - cr, best.centroid[1] - cc),
                )
                self._avatar_coast = 0
                self._bind_source = "observer"
                return (best.shape_hash, best.color)  # pin to the CURRENT identity (C-B)
        # coast (FR-195): carry the prediction; controllable UNBOUND via the absent sentinel pin
        self._avatar_coast += 1
        self._avatar_track = (colour, (pr, pc), (vr, vc))
        if self._avatar_coast > MAX_AVATAR_COAST:
            self._avatar_track = None
        self._bind_source = "coast"
        return _AVATAR_COAST_PIN

    # ----------------------------------------------- introspection capture (DEBUG, env-gated)

    def _build_capture(self, observation: FrameData, obj_set: ObjectSet) -> dict:
        """The per-turn observation half of the capture (action half added on flush).

        Pure read of state already computed this turn — emits NO action (RHAE). The viewer
        re-segments `grid`+`masked` (segment() is pure) and overlays these captured beliefs.
        """
        ctx = self._selection_context
        amap = ctx.affordances if ctx is not None else {}
        ctrl = _controllable(obj_set, ctx)
        ctrl_ident = (ctrl.shape_hash, ctrl.color) if ctrl is not None else None
        grounded = False
        if ctrl_ident is not None:
            aff = amap.get(ctrl_ident)
            grounded = aff is not None and aff.translate_support >= MIN_CONTROLLABILITY_SUPPORT
        goal = self._active_goal()
        masked = (
            [[int(r), int(c)] for r, c in np.argwhere(self._cap_hud)]
            if self._cap_hud is not None
            else []
        )
        return {
            "t": self._cap_t,
            "level": observation.levels_completed,
            # Policy (2026-05-31): record whether an ANIMATION occurred this action (frame is the list
            # of every grid rendered until the action completed — arcengine base_game). n_frames>1 => an
            # animation (e.g. a spring slide) played; keep all grids then so its meaning can be analysed
            # offline. DEBUG-only (env-gated by ARC_INTROSPECT); the agent itself still uses frame[-1].
            "n_frames": len(observation.frame),
            "frames": (observation.frame if len(observation.frame) > 1 else None),
            "grid": [[int(v) for v in row] for row in self._cap_grid.tolist()],
            "masked": masked,
            "affordances": {
                f"{k[0]},{k[1]}": [a.translate_support, a.response_support,
                                   a.vanish_support, a.spawn_support, int(a.autonomous)]
                for k, a in amap.items()
            },
            "controllable": list(ctrl_ident) if ctrl_ident is not None else None,
            "grounded": bool(grounded),
            "source": self._bind_source,  # spec 87 FR-199: grounded/observer/coast/none
            "goal": None
            if goal is None
            else {"selectors": list(goal.selectors), "relations": [list(r) for r in goal.relations]},
        }

    def _flush_capture(self, action_key: Optional[ActionKey], mode: str) -> None:
        """Append the completed capture record (obs + action) as one JSON line."""
        if self._cap is None or not self._capture_path:
            return
        self._cap["action"] = (
            list(action_key) if isinstance(action_key, tuple) else action_key
        )
        self._cap["mode"] = mode
        try:
            with open(self._capture_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(self._cap, separators=(",", ":")) + "\n")
        except Exception:  # pragma: no cover - capture must never break the turn
            logger.exception("introspection capture write failed")
        self._cap = None
        self._cap_t += 1

    # ------------------------------------------------------------- diagnostic

    def _log_controllable_diag(self, obj_set: ObjectSet) -> None:
        """Compare the STATIC controllable guess vs the world model's OBSERVED mover.

        Diagnostic only (option-1): never throws, never changes behaviour. Logs one
        `PB-DIAG` line per turn so a live run reveals whether `_controllable` (color/size
        heuristic) names the same object the model has learned a TRANSLATE effect for.
        `agree=False` while a mover exists is the signature of the wall: directed
        exploration is steering the wrong object.
        """
        try:
            static = _controllable(obj_set)
            static_ident = (
                (static.shape_hash, frozenset({("color", static.color)}))
                if static is not None
                else None
            )
            ident, support = _observed_mover_identity(self._model)
            if ident is None:
                agree = "n/a-no-translate-learned"
                mover_desc = "none(support=0)"
            else:
                agree = str(static_ident == ident)
                role, attrs = ident
                cands = [
                    o
                    for o in obj_set.objects
                    if o.shape_hash == role
                    and frozenset({("color", o.color)}) == attrs
                ]
                mv = cands[0] if cands else None
                mover_desc = (
                    f"color={mv.color},size={mv.size},"
                    f"centroid=({mv.centroid[0]:.1f},{mv.centroid[1]:.1f}),support={support}"
                    if mv is not None
                    else f"absent-in-frame,support={support}"
                )
            static_desc = (
                f"color={static.color},size={static.size},"
                f"centroid=({static.centroid[0]:.1f},{static.centroid[1]:.1f})"
                if static is not None
                else "none"
            )
            # v0.7: the GROUNDED pick actually used in planning/shaping (with the context).
            # Post-fix this should AGREE with the observed mover where the static one did not.
            grounded = _controllable(obj_set, self._selection_context)
            grounded_ident = (
                (grounded.shape_hash, frozenset({("color", grounded.color)}))
                if grounded is not None
                else None
            )
            grounded_desc = (
                f"color={grounded.color},size={grounded.size}"
                if grounded is not None
                else "none"
            )
            grounded_agree = str(grounded_ident == ident) if ident is not None else "n/a"
            logger.info(
                "PB-DIAG controllable static[%s] observed_mover[%s] static_agree=%s "
                "grounded[%s] grounded_agree=%s pin=%s",
                static_desc, mover_desc, agree, grounded_desc, grounded_agree,
                self._controllable_identity,
            )
        except Exception:  # pragma: no cover - diagnostic must never break the turn
            logger.exception("controllable diagnostic failed")

    # ------------------------------------------------------------- helpers

    def _reset_belief(self) -> None:
        """Reset ALL OWN Phase-B belief fields (FR-116). Never touches Memory (F-03)."""
        self._model = WorldModel()
        self._goal = None
        self._novelty_counts = {}
        self._held_plan = ()
        self._next_step_predicted_hash = None
        self._prev_objset = None
        self._prev_action = None
        self._prev_win_pre = None
        # Confidence starts fresh each level (FR-116, FR-164): a new level/attempt is not
        # penalised for the previous level's mismatches.
        self._confidence = CONFIDENCE_INIT
        self._resets_emitted = 0
        # abduction belief (down-weighted hyps forgotten across the boundary, FR-154).
        self._ranked_hyps = ()
        self._active_hyp_idx = 0
        self._abduction_actions = 0
        self._hyp_active_turns = 0
        self._last_goal_distance = None
        self._action_distance_delta = {}
        self._last_ctrl_ident = None
        self._binding_unstable = False
        # grounded controllability (ADR-012): the binding does not cross a level boundary.
        self._selection_context = None
        self._controllable_identity = None
        self._controllable_binding_changed = False
        # avatar observer (spec 87, FR-192/M-B): the track does not cross a level boundary.
        self._avatar_track = None
        self._avatar_coast = 0
        self._bind_source = "none"

    @staticmethod
    def _legal(action_key: ActionKey, legal: list[ActionKey]) -> bool:
        """A planned key is legal iff it is in the per-frame legal key set (FR-141)."""
        return action_key in legal

    @staticmethod
    def _key_of(action: GameAction, obj_set: ObjectSet) -> Optional[ActionKey]:
        """Recover the ActionKey the explorer emitted, for next-turn learning."""
        if action is GameAction.RESET:
            return None
        if action.value == 6:
            data = getattr(action, "action_data", None)
            if data is not None:
                return (6, int(data.x), int(data.y))
            return None
        return int(action.value)

    def _reasoning(self, mode: str, action_key: ActionKey) -> dict[str, object]:
        """Structured reasoning record, capped to 16 KB (FR-143)."""
        return _cap_reasoning(
            {
                "policy": "ModelBasedPolicy",
                "mode": mode,
                "plan_len": len(self._held_plan) + 1,
                "conf": round(self._confidence, 3),
                "goal_known": self._goal is not None,
                "chosen": list(action_key)
                if isinstance(action_key, tuple)
                else action_key,
            }
        )


# --- module-level diagnostic helper (option-1) --------------------------------------


def _observed_mover_identity(
    model: WorldModel,
) -> tuple[Optional[tuple[int, frozenset[tuple[str, int]]]], int]:
    """The object identity the model has OBSERVED to move the most (diagnostic only).

    Sums `translate` hypothesis support per `(role, attrs)` identity across SIMPLE
    (non-tuple = movement-class) actions, ignoring non-deterministic hypotheses. Returns
    `(identity, total_translate_support)`, or `(None, 0)` if the model has learned no
    deterministic translate effect yet. This is the world model's empirical answer to
    "which object responds to movement?" — the ground truth `_controllable` only guesses.
    """
    support_by_ident: dict[tuple[int, frozenset[tuple[str, int]]], int] = {}
    for rule in model.rules:
        if isinstance(rule.action, tuple):  # skip clicks (6, x, y); movement is a plain int
            continue
        for h in rule.hypotheses:
            if h.effect.kind == "translate" and not h.nondeterministic:
                ident = (h.precondition.role, h.precondition.attrs)
                support_by_ident[ident] = support_by_ident.get(ident, 0) + h.support
    if not support_by_ident:
        return None, 0
    # Deterministic, shape_hash-INVARIANT tie-break (NFR-116, H1): highest support, then by
    # COLOR — NEVER by role (= shape_hash), which would break relabel-invariance.
    best = max(
        support_by_ident.items(),
        key=lambda kv: (kv[1], dict(kv[0][1]).get("color", 0)),
    )
    return best[0], best[1]
