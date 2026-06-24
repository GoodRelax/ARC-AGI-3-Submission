"""[Adapter] BeliefExplorerPolicy — wires the partial-observability belief-state agent (R12, spec 89)
into the framework's DecisionPolicy seam. Thin shell: extract the settled grid + legal actions from the
FrameData, locate the controllable (general rigid-mover finder, no game literals), step the
BeliefExplorer brain, and emit a GameAction. Selectable at runtime via ARC_POLICY=belief (default agent
behaviour is unchanged when unset). NOT yet live-verified — its closed-loop validation is a live run.
"""
from __future__ import annotations

import os

import numpy as np

from arcengine import FrameData, GameAction

from agent.policy import DecisionPolicy
from agent.state_graph import Memory, RESET_ID
from agent.segment import latest_grid
from agent.belief_explorer import BeliefExplorer
from agent.controllable import find_controllable


class BeliefExplorerPolicy(DecisionPolicy):
    def __init__(self) -> None:
        self._brain = BeliefExplorer()
        self._ctrl_state = None
        self._last_action_id = None
        # Optional scripted PREFIX (ARC_REPLAY) to reach a target level before belief-exploring — e.g.
        # play the L1-L6 winning sequence (284 moves) to arrive at the L7 fog level, then take over.
        env = (os.environ.get("ARC_REPLAY") or "").replace(" ", "")
        self._prefix = [int(x) for x in env.split(",") if x]
        self._pidx = 0
        # Own debug capture (ARC_INTROSPECT): ModelBasedPolicy's capture is bypassed under this policy,
        # so write our own per-turn {grid, level, action, coverage} for offline stitching/analysis.
        self._cap_path = os.environ.get("ARC_INTROSPECT") or None
        self._cap_t = 0

    def _capture(self, observation, grid, aid):
        if not self._cap_path:
            return
        import json
        rec = {"t": self._cap_t, "level": int(getattr(observation, "levels_completed", -1)),
               "grid": grid.tolist(), "action": int(aid),
               "coverage": self._brain.belief.coverage(), "fog": self._brain.fog}
        with open(self._cap_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self._cap_t += 1

    def decide(self, observation: FrameData, memory: Memory) -> GameAction:
        grid = np.asarray(latest_grid(observation.frame), dtype=int)
        avail = [int(a) for a in observation.available_actions if int(a) != RESET_ID]
        if not avail:
            return GameAction.RESET
        if self._pidx < len(self._prefix):                  # replay the prefix to reach the target level
            aid = self._prefix[self._pidx]
            self._pidx += 1
            self._capture(observation, grid, aid)
            return GameAction.from_id(aid)
        # NB: pass only `fog` as background, NOT `passable` — `passable` includes the avatar's OWN colours,
        # which would mask the avatar out of the segmentation and break controllable-finding.
        cells, self._ctrl_state = find_controllable(grid, self._ctrl_state, fog=self._brain.fog)
        if cells is None:
            aid = avail[0]                       # no controllable yet -> act to induce detectable motion
        else:
            aid = self._brain.act(grid, cells, avail, last_action=self._last_action_id)
        self._last_action_id = aid
        self._capture(observation, grid, aid)
        return GameAction.from_id(aid)
