"""[Use Case] Planner: BFS / IDA* search over the WorldModel, in SIMULATION only.

Use-Case layer. The search expands action sequences ENTIRELY in memory over the
immutable `WorldModel` (FR-123) and returns a `Plan` (a list of `ActionKey`s) without
emitting any real action — so internal backtracking costs zero budget (ADR-007). Pure
function of its inputs (NFR-105): no I/O, no global mutation.

Strategy (FR-124):
  * **Goal mode** (`goal` known): find the SHORTEST action sequence whose simulated
    terminal `ObjectSet` satisfies the predicate. IDA* with the admissible
    distance-to-goal heuristic `goal.distance` (never overestimates), iterative
    deepening so very-short horizons are effectively BFS.
  * **Novelty mode** (`goal is None`): maximize the count-based novelty surrogate over
    least-visited masked object-states (FR-131); an unknown-effect successor is treated
    as MAXIMALLY novel (FR-127), biasing the search toward learning.

Per-step branching is capped (FR-124a): simple legal actions + top-`PLAN_CLICK_K`
salient `ClickKey`s, so `b ≈ len(simple) + PLAN_CLICK_K`. Bounded by
`PLAN_NODE_BUDGET` / `PLAN_DEPTH_BUDGET`; on exhaustion returns `None` (FR-126) —
never a partial/invalid plan.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

from agent.goal import GoalPredicate, SelectionContext, select_role
from agent.policy import _click_keys_from_objects
from agent.segment import ObjectSet, node_hash
from agent.state_graph import ActionKey, ClickKey
from agent.world_model import WorldModel

__all__ = ["Plan", "Planner", "PLAN_CLICK_K", "PLAN_NODE_BUDGET", "PLAN_DEPTH_BUDGET"]

# Budgets (NFR-101, "to be calibrated"). PLAN_CLICK_K caps click branching (FR-124a).
PLAN_CLICK_K: int = 6
PLAN_NODE_BUDGET: int = 5000
PLAN_DEPTH_BUDGET: int = 12

# Novelty function signature: (node_hash, visit_counts) -> novelty score (FR-131).
NoveltyFn = Callable[[int, dict[int, int]], float]

# Successor score for an UNKNOWN-effect action: maximally novel (FR-127). Above any
# count-based novelty (which is <= 1.0) so the search prefers learning the unknown.
_MAX_NOVELTY = 1e9


@dataclass(frozen=True, slots=True)
class Plan:
    """An ordered list of real actions reaching the goal/novelty target (Glossary)."""

    actions: tuple[ActionKey, ...]
    plan_terminal_hash: int  # node_hash of the simulated terminal state (F-11, F-16)


@dataclass(frozen=True, slots=True)
class _Budget:
    nodes: int
    depth: int


def _simple_legal(legal: list[ActionKey]) -> list[ActionKey]:
    """Legal non-click actions, in fixed ascending order (deterministic)."""
    return sorted(k for k in legal if not isinstance(k, tuple))


def _top_k_clicks(
    objset: ObjectSet, legal: list[ActionKey], k: int
) -> list[ClickKey]:
    """Top-k salient ClickKeys for this state (FR-124a).

    Uses the SAME salience order as the owned explorer (`_click_keys_from_objects`),
    restricted to clicks that are legal this frame, capped to k. A ClickKey is legal
    iff `ACTION6` (id 6) is in `legal` (the policy passes a `(6, x, y)` sentinel when
    clicking is available).
    """
    if not any(isinstance(x, tuple) and x[0] == 6 for x in legal):
        return []
    return _click_keys_from_objects(objset)[:k]


def _successor_actions(
    objset: ObjectSet, legal: list[ActionKey]
) -> list[ActionKey]:
    """Capped per-step action candidates (FR-124a): simple + top-K clicks."""
    return _simple_legal(legal) + list(_top_k_clicks(objset, legal, PLAN_CLICK_K))


@dataclass
class _DfsResult:
    plan: Optional[list[ActionKey]]
    state: Optional[ObjectSet]
    expanded: int


class Planner:
    """Pure simulation searcher (FR-123). One method: `plan`."""

    def plan(
        self,
        start: ObjectSet,
        model: WorldModel,
        goal: Optional[GoalPredicate],
        legal: list[ActionKey],
        novelty: Optional[NoveltyFn] = None,
        counts: Optional[dict[int, int]] = None,
        node_budget: int = PLAN_NODE_BUDGET,
        depth_budget: int = PLAN_DEPTH_BUDGET,
        context: Optional[SelectionContext] = None,
    ) -> Optional[Plan]:
        """Search for a plan in simulation; return None if none found in budget (FR-126).

        Goal mode (goal given): IDA* on f = g + goal.distance toward `holds`. Novelty
        mode (goal None): best-novelty bounded DFS reaching a least-visited state.

        `context` (FR-170) grounds the `controllable` role; it is ROOT-FIXED here (H2) and
        forwarded to every per-node `goal.holds`/`goal.distance` (C1).
        """
        budget = _Budget(nodes=node_budget, depth=depth_budget)
        counts = counts or {}

        if goal is not None:
            sim_context = self._root_fix(start, goal, context)
            if goal.holds(start, sim_context):
                return Plan(actions=(), plan_terminal_hash=node_hash(start))
            return self._ida_star(start, model, goal, legal, budget, sim_context)
        return self._novelty_search(start, model, legal, novelty, counts, budget)

    @staticmethod
    def _root_fix(
        start: ObjectSet,
        goal: GoalPredicate,
        context: Optional[SelectionContext],
    ) -> Optional[SelectionContext]:
        """Pin the `controllable` role to a concrete identity at the plan ROOT (FR-170/H2).

        Keeps the grounded binding STABLE across every simulated node so the heuristic
        stays admissible (a per-node re-resolve could fall back to the static blob and
        OVERESTIMATE). Reuses an existing pin (the policy's sticky binding); pins nothing
        when there is no context or the goal does not use the `controllable` selector.
        """
        if context is None or context.pinned_controllable is not None:
            return context
        if not goal.selectors or "controllable" not in goal.selectors:
            return context
        ctrl = select_role("controllable", start, context)
        if ctrl is None:
            return context
        return replace(context, pinned_controllable=(ctrl.shape_hash, ctrl.color))

    # --- goal-directed IDA* ---------------------------------------------------

    def _ida_star(
        self,
        start: ObjectSet,
        model: WorldModel,
        goal: GoalPredicate,
        legal: list[ActionKey],
        budget: _Budget,
        context: Optional[SelectionContext] = None,
    ) -> Optional[Plan]:
        """Iterative-deepening A* on f = g + h (FR-124). h = goal.distance (admissible).

        Iterative deepening makes very-short horizons behave like BFS (shortest plan
        first), while bounding memory. Total expanded nodes are capped at budget.nodes.
        `context` (root-fixed by the caller) is forwarded to every node's goal eval (C1).
        """
        expanded_total = 0
        for depth_limit in range(1, budget.depth + 1):
            res = self._dfs_goal(
                start, [], 0, depth_limit, model, goal, legal, context
            )
            expanded_total += res.expanded
            if res.plan is not None and res.state is not None:
                return Plan(
                    actions=tuple(res.plan),
                    plan_terminal_hash=node_hash(res.state),
                )
            if expanded_total > budget.nodes:
                return None  # budget exhausted (FR-126)
        return None  # depth budget exhausted, goal unreached (FR-126)

    def _dfs_goal(
        self,
        state: ObjectSet,
        path: list[ActionKey],
        g: int,
        depth_limit: int,
        model: WorldModel,
        goal: GoalPredicate,
        legal: list[ActionKey],
        context: Optional[SelectionContext] = None,
    ) -> _DfsResult:
        """Depth-bounded DFS for the goal; prunes by f = g + h > depth_limit.

        `context` (root-fixed) grounds the `controllable` binding at every node (C1/H2).
        """
        if goal.holds(state, context):
            return _DfsResult(plan=list(path), state=state, expanded=0)
        f = g + goal.distance(state, context)
        if f > depth_limit:
            return _DfsResult(plan=None, state=None, expanded=0)

        expanded = 1
        for action in _successor_actions(state, legal):
            succ = model.predict(state, action)
            if succ is None:
                continue  # unknown effect: not useful for a deterministic goal plan
            if succ.signature == state.signature:
                continue  # no-op self-loop: never progresses a goal search
            res = self._dfs_goal(
                succ, path + [action], g + 1, depth_limit, model, goal, legal, context
            )
            expanded += res.expanded
            if res.plan is not None:
                res.expanded = expanded
                return res
        return _DfsResult(plan=None, state=None, expanded=expanded)

    # --- novelty-directed bounded search --------------------------------------

    def _novelty_search(
        self,
        start: ObjectSet,
        model: WorldModel,
        legal: list[ActionKey],
        novelty: Optional[NoveltyFn],
        counts: dict[int, int],
        budget: _Budget,
    ) -> Optional[Plan]:
        """Bounded DFS returning the path to the most-novel reachable state (FR-131).

        An UNKNOWN-effect action (model.predict -> None) is treated as MAXIMALLY novel
        (FR-127): the search returns a one-step plan emitting it so the policy learns
        the unknown. Otherwise it returns the best-scoring reachable terminal path
        within the depth/node budget; None only if nothing better than the start was
        found (caller then falls back).
        """
        if novelty is None:
            return None
        best_score = -1.0
        best_path: list[ActionKey] = []
        best_state: Optional[ObjectSet] = start
        expanded = 0

        # Greedy depth-bounded DFS, deterministic action order (NFR-104).
        stack: list[tuple[ObjectSet, list[ActionKey]]] = [(start, [])]
        seen: set[int] = {start.signature}
        while stack and expanded <= budget.nodes:
            state, path = stack.pop()
            if len(path) >= budget.depth:
                continue
            for action in _successor_actions(state, legal):
                expanded += 1
                succ = model.predict(state, action)
                if succ is None:
                    # Maximally novel: emit this learning probe immediately (FR-127).
                    return Plan(
                        actions=tuple(path + [action]),
                        plan_terminal_hash=state.signature,
                    )
                if succ.signature in seen:
                    continue
                seen.add(succ.signature)
                score = novelty(succ.signature, counts)
                if score > best_score:
                    best_score = score
                    best_path = path + [action]
                    best_state = succ
                stack.append((succ, path + [action]))

        if best_path:
            return Plan(
                actions=tuple(best_path),
                plan_terminal_hash=(best_state or start).signature,
            )
        return None
