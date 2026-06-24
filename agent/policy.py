"""[Use Case] DecisionPolicy seam + the Phase A GraphExplorerPolicy (§4.3).

This layer is the imperative shell that orchestrates the pure Entity core. It MAY
import `arcengine` (`GameAction`, `FrameData`) — only `search_agent.py` imports the
`Agent` base class. It MUST NOT import `agents.agent`.

The `DecisionPolicy` Protocol is the swap seam (ADR-002): "graph explorer now,
planner later, LLM maybe" becomes a substitution, not a rewrite. `OurSearchAgent`
depends on this abstraction (DIP), never on the concrete policy.

`GraphExplorerPolicy.decide` implements the §4.3 procedure verbatim. PRECONDITION
(F-01 / §4.3): `decide()` is called ONLY on a playable (`NOT_FINISHED`) frame; the
Adapter owns the FR-002 / FR-004 guards, so we never segment a terminal frame here.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol, runtime_checkable

import numpy as np
from arcengine import FrameData, GameAction

from agent.segment import (
    GridObject,
    ObjectSet,
    detect_hud,
    latest_grid,
    node_hash,
    segment,
)
from agent.state_graph import (
    RESET_ID,
    ActionKey,
    ClickKey,
    Memory,
    StateGraph,
    StateNode,
)

logger = logging.getLogger(__name__)

# Serialized `reasoning` must stay <= 16 KB (FR-017; 20-api-and-data.md §1). We cap
# below the wire limit so the JSON envelope + key names never push us over.
_REASONING_BYTE_CAP = 16 * 1024

# Fixed change-likelihood ordering of the SIMPLE action ids (FR-012, NFR-005).
# This is a generic prior, NOT game-specific (NFR-004): it merely says "try the
# lower-numbered movement-style actions before UNDO". UNDO (7) is last because it
# is a backtrack that costs budget for no exploration value (L-2). The ordering is
# documented and frozen; there is no randomness anywhere in the policy.
_SIMPLE_ACTION_PRIOR: tuple[int, ...] = (1, 2, 3, 4, 5, 7)


@runtime_checkable
class DecisionPolicy(Protocol):
    """The seam (ADR-002): algorithm varies independently of the framework adapter."""

    def decide(self, observation: FrameData, memory: Memory) -> GameAction: ...


# Coarse full-board click sweep stride (FR-014). Stride 8 -> an 8x8 = 64-point grid
# of cell centres, so click targets that are NOT object centroids/corners are still
# reachable. This is a generic uniform-board probe, NOT game-specific (NFR-004).
_GRID_SWEEP_STRIDE = 8


def _click_keys_from_objects(obj_set: ObjectSet) -> list[ClickKey]:
    """Object-derived click targets, salience-ordered (FR-014).

    Each object contributes its CENTROID and its four bounding-box CORNERS, so a
    level that needs a click on an object's edge/corner (not just its middle) is
    reachable — the earlier centroid-only heuristic was too narrow (it reached only
    ~3 states on vc33). Larger / rarer-color objects first (more likely interactive).
    Coordinates clamped to 0..63 (FR-018). `x` = column, `y` = row (API click is
    (x=col, y=row); centroid is (row_mean, col_mean)).
    """
    # Rarity = inverse color frequency: count objects per color, rarer ranks first.
    color_counts: dict[int, int] = {}
    for o in obj_set.objects:
        color_counts[o.color] = color_counts.get(o.color, 0) + 1

    def salience_key(o: GridObject) -> tuple[int, int, int, int]:
        # Sort DESC by size, then ASC by color frequency (rarer first), then a
        # deterministic geometric tie-break so the order is fully reproducible.
        return (-o.size, color_counts[o.color], o.bbox[0], o.bbox[1])

    def clamp(col: float, row: float) -> ClickKey:
        return (6, max(0, min(63, int(round(col)))), max(0, min(63, int(round(row)))))

    keys: list[ClickKey] = []
    seen: set[tuple[int, int]] = set()
    for o in sorted(obj_set.objects, key=salience_key):
        r0, c0, r1, c1 = o.bbox
        cr, cc = o.centroid
        # centroid first, then the four bbox corners (col, row) pairs.
        for col, row in ((cc, cr), (c0, r0), (c1, r0), (c0, r1), (c1, r1)):
            k = clamp(col, row)
            if (k[1], k[2]) not in seen:
                seen.add((k[1], k[2]))
                keys.append(k)
    return keys


def _grid_sweep_clicks() -> list[ClickKey]:
    """Coarse, deterministic full-board click sweep (FR-014 fallback)."""
    half = _GRID_SWEEP_STRIDE // 2
    return [
        (6, x, y)
        for y in range(half, 64, _GRID_SWEEP_STRIDE)
        for x in range(half, 64, _GRID_SWEEP_STRIDE)
    ]


def _click_candidates(obj_set: ObjectSet) -> list[ClickKey]:
    """All click targets to consider this frame (FR-014): salient object points
    (centroid + corners) FIRST, then a coarse board sweep for non-object cells.

    Deduped + deterministic. The explorer's no-op detection (FR-021) and
    least-recently-tried ordering then prune the dead targets cheaply, so the broad
    candidate set costs little while letting the agent reach progress cells that are
    not object centroids (needed for pure-click games like vc33).
    """
    out = _click_keys_from_objects(obj_set)
    seen = {(k[1], k[2]) for k in out}
    for k in _grid_sweep_clicks():
        if (k[1], k[2]) not in seen:
            seen.add((k[1], k[2]))
            out.append(k)
    return out


def order_actions(
    untested: frozenset[ActionKey] | set[ActionKey],
    obj_set: ObjectSet,
    graph: StateGraph,
    src_hash: int,
    action_usage: dict[ActionKey, int] | None = None,
) -> list[ActionKey]:
    """Order untested actions by the change-likelihood / click-salience prior (§4.3).

    Rules (all deterministic, NFR-005):
      1. Simple actions first, ordered LEAST-RECENTLY-TRIED-first: the sort key is
         `(is_noop, usage_count, fixed_prior_index)` (FR-012). `usage_count` is how
         many times the policy has emitted that action this level (`action_usage`);
         the fixed `_SIMPLE_ACTION_PRIOR` index is only a tie-break among equally
         used actions. This makes the explorer round-robin ACTION1..4 instead of
         fixating on the lowest-numbered action while every fresh node still shows
         every action as untested.
      2. Then click actions, ALSO least-recently-tried-first within their salience
         order: key `(is_noop, usage_count, salience_tier)` (FR-012/FR-014). Without
         the `usage_count` term the explorer re-picks the SAME top-salience click on
         every fresh-looking node (the "clicks the same point again and again" bug on
         pure-click games); the round-robin spreads clicks across distinct targets.
      3. Known no-op actions at this node sink to the BOTTOM of their group (FR-021),
         so a self-looping action is never re-selected ahead of an untried one.
    """
    usage = action_usage or {}
    noops = graph.noop_actions.get(src_hash, set())
    simple = [k for k in untested if not isinstance(k, tuple)]
    clicks = [k for k in untested if isinstance(k, tuple)]

    # Salience order is computed over ALL object-derived clicks; we then keep only
    # those that are actually in this node's untested set, preserving tier order.
    salient = [k for k in _click_candidates(obj_set) if k in set(clicks)]
    # Any untested click not derived from an object (shouldn't normally happen) is
    # appended deterministically so nothing is silently dropped.
    salient += sorted(k for k in clicks if k not in set(salient))

    simple_sorted = sorted(
        simple,
        key=lambda k: (
            k in noops,  # no-ops last (False < True)
            usage.get(k, 0),  # least-recently-tried first (round-robin)
            _SIMPLE_ACTION_PRIOR.index(k) if k in _SIMPLE_ACTION_PRIOR else 99,
        ),
    )
    clicks_sorted = sorted(
        range(len(salient)),
        # no-ops last, then least-recently-tried first (round-robin), then tier order
        key=lambda i: (salient[i] in noops, usage.get(salient[i], 0), i),
    )
    ordered_clicks = [salient[i] for i in clicks_sorted]
    return simple_sorted + ordered_clicks


def _action_keys(available_actions: list[int], obj_set: ObjectSet) -> list[ActionKey]:
    """Map this frame's legal ids to ActionKeys, excluding RESET (FR-010).

    Simple ids map 1:1. ACTION6 (click) expands into one ClickKey per salience tier
    derived from the segmented objects (FR-014). If clicking is legal but no object
    yielded a target, fall back to the grid centre so the frontier is never empty
    for a no-reason. RESET is never a frontier action.
    """
    keys: list[ActionKey] = []
    for action_id in available_actions:
        if action_id == RESET_ID:
            continue
        if action_id == 6:
            click_keys = _click_candidates(obj_set)
            if not click_keys:
                click_keys = [(6, 32, 32)]  # centre fallback, in-range (FR-018)
            keys.extend(click_keys)
        else:
            keys.append(int(action_id))
    return keys


def _to_game_action(key: ActionKey, reasoning: dict[str, object]) -> GameAction:
    """Translate an ActionKey into a filled GameAction with capped reasoning.

    Clicks call `set_data` with x,y in 0..63 (FR-018). `reasoning` is JSON and is
    truncated to stay <= 16 KB (FR-017). Reasoning is attached as an attribute on
    the enum member, which is how the framework reads it back (`getattr(action,
    "reasoning", None)` in agents/agent.py).
    """
    if isinstance(key, tuple):  # ClickKey -> ACTION6
        _, x, y = key
        action = GameAction.ACTION6
        action.set_data({"x": int(x), "y": int(y)})
    else:
        action = GameAction.from_id(int(key))
    action.reasoning = _cap_reasoning(reasoning)
    return action


def _cap_reasoning(reasoning: dict[str, object]) -> dict[str, object]:
    """Ensure the serialized reasoning stays <= 16 KB (FR-017).

    If the JSON blob would exceed the cap, drop the heavy free-text `rationale`
    field and replace it with a truncation marker, keeping the structured keys
    (policy name, chosen action) that matter for offline debugging.
    """
    try:
        raw = json.dumps(reasoning, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        return {"policy": "GraphExplorerPolicy", "note": "unserializable-reasoning"}
    if len(raw) <= _REASONING_BYTE_CAP:
        return reasoning
    capped = dict(reasoning)
    capped["rationale"] = "<truncated to fit 16KB cap>"
    return capped


class GraphExplorerPolicy:
    """Phase A DecisionPolicy: systematic, non-redundant frontier exploration (§4.3).

    Holds NO per-game state of its own (NFR-004 / Concurrency Safety): all mutable
    state lives in the `Memory` passed in, so one policy instance is safe to reuse
    and there is no cross-game shared state. Frame history (for HUD detection) is
    threaded through `Memory.graph`? No — it is local to `decide` via the framework
    `frames` list, which the adapter does not forward; instead we keep a bounded
    rolling history on the policy keyed only for HUD heuristics, which is per-agent
    (one thread per game) and therefore race-free.
    """

    # Bounded rolling window of recent grids for the change-rate HUD heuristic
    # (FR-008a). Small + per-agent; reset on every level boundary (NFR-007).
    _HUD_HISTORY = 8

    def __init__(self) -> None:
        self._grid_history: list[np.ndarray] = []
        self._history_level: int | None = None

    def decide(self, observation: FrameData, memory: Memory) -> GameAction:
        # 1. Per-level / new-attempt reset on ANY level-boundary change (FR-016).
        #    Covers both an increase (level cleared) and a decrease/equal-but-
        #    different value (new attempt after a full RESET) — identical handling.
        if observation.levels_completed != memory.levels_completed:
            memory.reset()
            memory.levels_completed = observation.levels_completed
            self._grid_history = []  # HUD history must not cross a boundary
            self._history_level = observation.levels_completed

        # 2. Reduce multi-grid frame (FR-020) then segment + HUD-mask + hash.
        #    detect_hud is the one history-dependent step (FR-008a); on any failure
        #    in segmentation/hashing we degrade gracefully (FR-019).
        try:
            grid = latest_grid(observation.frame)
            hud_mask = detect_hud(self._grid_history, grid)
            obj_set = segment(grid, hud_mask=hud_mask)
            h = node_hash(obj_set)
        except Exception:  # FR-019: never crash on a bad frame
            logger.exception("segmentation/hashing failed; falling back")
            return self._fallback_action(observation, reason="segment-error")

        self._remember_grid(grid)

        # 3. Register node + record the edge we just traversed (FR-010, FR-011, FR-021).
        legal = _action_keys(observation.available_actions, obj_set)
        if h not in memory.graph.nodes:
            memory.graph.add_node(StateNode(h, untested_actions=frozenset(legal)))
        if memory.last_node_action is not None:
            src_h, src_a = memory.last_node_action
            memory.graph.add_edge((src_h, src_a), h)
            if src_h == h:  # no-op self-loop (FR-021): nothing changed
                memory.graph.flag_noop(src_h, src_a)

        graph = memory.graph
        node = graph.nodes[h]
        untested = graph.untested_at(h)

        # 4. Frontier choice (FR-012, FR-013, FR-015, FR-022, FR-023).
        if untested:
            ordered = order_actions(
                frozenset(untested), obj_set, graph, h, memory.action_usage
            )
            action_key = ordered[0]
            rationale = "local-untested"
        elif graph.frontier_nodes:
            target = self._nearest_frontier(graph, h)
            path = graph.shortest_path(h, target) if target is not None else []
            if path:
                action_key = path[0]  # first step toward nearest frontier (FR-013)
                rationale = "navigate-to-frontier"
            else:
                # Frontier exists but is unreachable via known edges (FR-023):
                # prefer a local untested action (none here), else dead-end + RESET.
                key = self._local_untested_or_reset(graph, h)
                action_key = key
                rationale = "unreachable-frontier-reset"
        else:
            # Level mapped: global frontier empty (FR-022) -> exploit best known.
            graph.mark_dead(h)
            action_key = self._exploit_best_known(graph, h, observation)
            rationale = "frontier-empty-exploit"

        # 5. Bookkeep + emit (FR-017, FR-018). Tally the emitted action so the
        #    least-recently-tried ordering round-robins across nodes (FR-012).
        memory.action_usage[action_key] = memory.action_usage.get(action_key, 0) + 1

        # RESET key short-circuits to RESET.
        if action_key == RESET_ID:
            memory.current_hash = h
            memory.last_node_action = None  # RESET breaks the edge chain
            reset = GameAction.RESET
            reset.reasoning = _cap_reasoning(
                {"policy": "GraphExplorerPolicy", "rationale": rationale}
            )
            return reset

        memory.current_hash = h
        memory.last_node_action = (h, action_key)
        reasoning = self._trace(node, action_key, rationale, untested)
        return _to_game_action(action_key, reasoning)

    # --- helpers (the §4.3 named sub-procedures) ------------------------------

    def _remember_grid(self, grid: np.ndarray) -> None:
        """Append to the bounded HUD-history window (FR-008a)."""
        self._grid_history.append(grid)
        if len(self._grid_history) > self._HUD_HISTORY:
            self._grid_history.pop(0)

    @staticmethod
    def _nearest_frontier(graph: StateGraph, src_hash: int) -> int | None:
        """Pick the nearest frontier node by BFS hop distance (FR-013).

        One bounded BFS over the graph (NFR-010). Deterministic tie-break: among
        equal-distance frontier nodes, the smaller node_hash wins (NFR-005).
        """
        if not graph.frontier_nodes:
            return None
        from collections import deque

        queue: deque[tuple[int, int]] = deque([(src_hash, 0)])
        seen = {src_hash}
        best: tuple[int, int] | None = None  # (distance, node_hash)
        while queue:
            cur, dist = queue.popleft()
            if cur in graph.frontier_nodes and cur != src_hash:
                cand = (dist, cur)
                if best is None or cand < best:
                    best = cand
            for (e_src, _), e_dst in graph.edges.items():
                if e_src == cur and e_dst not in seen:
                    seen.add(e_dst)
                    queue.append((e_dst, dist + 1))
        if best is not None:
            return best[1]
        # Frontier node(s) exist but none reachable; return one deterministically so
        # shortest_path then returns [] and FR-023 fallback fires.
        return min(graph.frontier_nodes)

    def _local_untested_or_reset(self, graph: StateGraph, src_hash: int) -> ActionKey:
        """FR-023 fallback: any local untested action, else mark dead + RESET."""
        local = graph.untested_at(src_hash)
        if local:
            # Should be empty in this branch, but honor the spec literally.
            return sorted(
                local, key=lambda k: (isinstance(k, tuple), str(k))
            )[0]
        graph.mark_dead(src_hash)  # local dead-end (FR-023)
        return RESET_ID

    def _exploit_best_known(
        self, graph: StateGraph, src_hash: int, observation: FrameData
    ) -> ActionKey:
        """FR-022: emit the LEAST-RECENTLY-TRIED legal non-RESET action, else RESET.

        Deterministic (NFR-005): tried_order preserves first-try order, so the
        front of that list is the least-recently-tried. We only emit actions that
        are still legal this frame (FR-003). If none, RESET.
        """
        # exploit only re-issues simple legal actions deterministically (clicks need
        # object context that may have shifted; simple actions are always safe).
        legal_simple = [
            int(a)
            for a in observation.available_actions
            if a != RESET_ID and a != 6
        ]
        order = graph.tried_order.get(src_hash, [])
        for key in order:
            if not isinstance(key, tuple) and key in legal_simple:
                return key
        # Nothing tried yet that is legal: fall back to first legal simple action.
        if legal_simple:
            return min(legal_simple)
        return RESET_ID

    @staticmethod
    def _fallback_action(observation: FrameData, reason: str) -> GameAction:
        """FR-019: first legal non-RESET action, else RESET, never crash."""
        for action_id in observation.available_actions:
            if action_id == RESET_ID:
                continue
            action = GameAction.from_id(int(action_id))
            if action.is_complex():
                action.set_data({"x": 32, "y": 32})  # in-range default (FR-018)
            action.reasoning = {"policy": "GraphExplorerPolicy", "fallback": reason}
            return action
        reset = GameAction.RESET
        reset.reasoning = {"policy": "GraphExplorerPolicy", "fallback": reason}
        return reset

    @staticmethod
    def _trace(
        node: StateNode,
        action_key: ActionKey,
        rationale: str,
        untested: set[ActionKey],
    ) -> dict[str, object]:
        """Structured decision record for offline debugging (FR-017)."""
        return {
            "policy": "GraphExplorerPolicy",
            "node_hash": node.node_hash,
            "rationale": rationale,
            "chosen": list(action_key) if isinstance(action_key, tuple) else action_key,
            "untested_count": len(untested),
        }
