"""[Adapter] OurSearchAgent — the framework Agent contract, owning the guards.

This is the ONLY code we own inside the play loop (10-io-ports.md). It subclasses
the vendored `Agent` directly (ADR-004 — so the registry auto-discovers it) and
implements exactly two methods:

    choose_action(frames, latest_frame) -> GameAction   # decide
    is_done(frames, latest_frame)       -> bool          # stop?

Clean Architecture role (3.1): the Adapter satisfies the framework contract and
delegates the actual decision to a `DecisionPolicy` (DIP — depends on the
abstraction, ADR-002). It OWNS the safety guards so the policy only ever sees a
playable frame:

  * FR-002: state NOT_PLAYED / GAME_OVER  -> RESET before any policy logic.
  * FR-004: no legal non-RESET action     -> RESET rather than stall.
  * FR-005: is_done True iff state is WIN.

Everything else (fetching frames, submitting/recording actions, scoring) is the
vendored Input/Output ports and is untouched.
"""

from __future__ import annotations

import logging
import os

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

from agent.policy import DecisionPolicy
from agent.model_policy import ModelBasedPolicy
from agent.state_graph import RESET_ID, Memory

logger = logging.getLogger(__name__)


class OurSearchAgent(Agent):
    """Phase A graph-explorer adapter onto the framework `Agent` contract.

    Holds one `DecisionPolicy` (the swap seam) and one per-level `Memory`. The
    policy is injectable so Phase B/C policies plug in with no adapter change
    (ADR-002, LSP). Run as: ``main.py --agent oursearchagent --game <id>``.
    """

    # Framework loop-safety cap on turns (L-4), configurable. Not a game rule; keeps
    # a stuck search visible rather than looping forever. Raised from 80 to 250 to
    # give the explorer room for real exploration (live-run finding, spec v0.3).
    # Env-gated override (ARC_MAX_ACTIONS) for long debug replays that exceed 250
    # (e.g. the L6 verification = 200-move prefix + 84-move plan = 284); default
    # 250 leaves the scored submission unchanged, matching the ARC_REPLAY pattern.
    MAX_ACTIONS = int(os.environ.get("ARC_MAX_ACTIONS", "250"))

    def __init__(self, *args: object, **kwargs: object) -> None:
        # The framework constructs agents positionally (see agents/agent.py); we
        # forward everything and only ADD our policy + memory. A custom policy may
        # be passed via kwarg for tests / future phases without touching the loop.
        policy = kwargs.pop("policy", None)
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        # Env-gated opt-in (ARC_POLICY=belief) to the partial-observability belief-state explorer
        # (R12, spec 89) for fog/torch levels like ls20-L7. Unset -> default unchanged, so scored
        # behaviour + the test suite are untouched (mirrors the ARC_REPLAY/ARC_MAX_ACTIONS pattern).
        if policy is None and os.environ.get("ARC_POLICY") == "belief":
            from agent.belief_policy import BeliefExplorerPolicy
            policy = BeliefExplorerPolicy()
        # Default to the Phase B ModelBasedPolicy (ADR-009): it OWNS a
        # GraphExplorerPolicy as its segmentation owner + fallback, so no Phase A
        # capability is lost. The `policy` kwarg stays injectable for tests.
        self._policy: DecisionPolicy = (
            policy if policy is not None else ModelBasedPolicy()  # type: ignore[assignment]
        )
        self._memory = Memory()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Stop only when we win (FR-005).

        GAME_OVER is intentionally NOT a stop condition: choose_action issues a
        RESET on GAME_OVER so the agent keeps trying within its action budget.
        """
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Return exactly one GameAction (FR-001). Guard first, then delegate.

        The Adapter owns BOTH guards (§4.3 precondition F-01) so `decide()` never
        runs on a terminal frame and never has to invent legality:
          * FR-002: NOT_PLAYED / GAME_OVER -> RESET, do NOT call decide().
          * FR-004: no legal non-RESET action offered -> RESET, do NOT call decide().
        Only a playable (NOT_FINISHED) frame with a legal action reaches the policy.
        """
        # FR-002: terminal states get RESET before any segmentation runs.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        # FR-004: if the env offers no legal non-RESET action, RESET rather than
        # stall. Checked HERE (not in the policy) so the policy precondition holds.
        if not any(a != RESET_ID for a in latest_frame.available_actions):
            return GameAction.RESET

        # Playable frame -> delegate to the policy (ADR-002, SC-013).
        action = self._policy.decide(latest_frame, self._memory)

        # Defensive NFR-002 backstop: a non-RESET action MUST be in this frame's
        # available_actions. By construction the policy only picks legal actions;
        # if that invariant is ever violated, log and fall back to RESET rather
        # than emit an illegal action (error table §4.5).
        if action is not GameAction.RESET and action.value not in latest_frame.available_actions:
            logger.error(
                "policy emitted illegal action %s not in %s; forcing RESET",
                action.name,
                latest_frame.available_actions,
            )
            return GameAction.RESET
        return action
