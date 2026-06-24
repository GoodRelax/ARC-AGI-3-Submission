"""[Entity] The per-level state graph: StateNode, StateGraph, Memory.

These are the mutable Entity containers (schema §4.2). The imperative shell (the
policy) owns their lifecycle; methods stay simple (CQS: commands mutate + return
None, queries are pure). `StateNode` is a frozen value object; `StateGraph` and
`Memory` are mutable because a graph that is explored is inherently stateful.

Hard rule (task / spec 3.1): this Entity module MUST NOT import the framework
(`agents` / `arcengine`). `ActionKey` is therefore defined here as plain ints /
tuples so the policy can map them to `GameAction` without this layer ever knowing
about `GameAction`. It depends only on stdlib + `segment` (also Entity).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Union

from agent.segment import GridObject  # re-export so callers have one import site

__all__ = [
    "GridObject",
    "ActionKey",
    "SimpleKey",
    "ClickKey",
    "RESET_ID",
    "StateNode",
    "StateGraph",
    "Memory",
]

# --- Action keys (schema §4.2) -------------------------------------------------
# A SimpleKey is a GameAction id for a non-click action (1,2,3,4,5,7). A ClickKey
# carries the click target so two clicks at different coordinates are DISTINCT
# frontier entries (FR-014). We deliberately do NOT import GameAction here (keeps
# the Entity layer framework-free); the policy translates keys <-> GameAction.
SimpleKey = int  # GameAction id in {1,2,3,4,5,7}
ClickKey = tuple[Literal[6], int, int]  # (6, x, y), x,y in 0..63
ActionKey = Union[SimpleKey, ClickKey]

RESET_ID = 0  # GameAction.RESET.value — never a frontier action (recovery only)


@dataclass(frozen=True, slots=True)
class StateNode:
    """A node in the state graph (schema §4.2).

    Frozen: a node's *identity* (`node_hash`) and its initial frontier are fixed
    once observed. The MUTABLE per-node bookkeeping (which actions were tried, in
    what order, which are no-ops) lives on `StateGraph` keyed by `node_hash`, not
    on this frozen value object — so we never need to rebuild the node to update it.
    """

    node_hash: int  # == ObjectSet.signature (FR-009)
    untested_actions: frozenset[ActionKey]  # per-node frontier (initial set)
    dead: bool = False


@dataclass
class StateGraph:
    """The per-level directed graph (schema §4.2, §4.4).

    nodes:          node_hash -> StateNode
    edges:          (src_hash, action) -> dst_hash
    frontier_nodes: nodes that still have at least one untested action

    Extra bookkeeping (not on the frozen StateNode):
    untested:       node_hash -> remaining untested action set (mutable mirror)
    tried_order:    node_hash -> list of actions in the order first tried
                    (used by exploit_best_known for least-recently-tried, FR-022)
    noop_actions:   node_hash -> actions known to self-loop (FR-021), deprioritized
    """

    nodes: dict[int, StateNode] = field(default_factory=dict)
    edges: dict[tuple[int, ActionKey], int] = field(default_factory=dict)
    frontier_nodes: set[int] = field(default_factory=set)
    untested: dict[int, set[ActionKey]] = field(default_factory=dict)
    tried_order: dict[int, list[ActionKey]] = field(default_factory=dict)
    noop_actions: dict[int, set[ActionKey]] = field(default_factory=dict)

    # --- commands (mutate, return None) ---------------------------------------

    def add_node(self, node: StateNode) -> None:
        """Insert a freshly observed node and initialize its frontier (FR-010).

        Idempotent: re-observing a known node must not wipe its exploration
        progress (its `untested` set may already be partly drained), so we no-op
        if the hash is already present.
        """
        if node.node_hash in self.nodes:
            return
        self.nodes[node.node_hash] = node
        self.untested[node.node_hash] = set(node.untested_actions)
        self.tried_order.setdefault(node.node_hash, [])
        self.noop_actions.setdefault(node.node_hash, set())
        if node.untested_actions:
            self.frontier_nodes.add(node.node_hash)

    def add_edge(self, src: tuple[int, ActionKey], dst_hash: int) -> None:
        """Record a traversed edge and drain the action from src's frontier (FR-011).

        Also appends to `tried_order` (first time only) so exploit_best_known can
        pick the least-recently-tried action deterministically (FR-022). When src's
        untested set empties, src leaves the global frontier (§4.4 lifecycle).
        """
        src_hash, action = src
        self.edges[(src_hash, action)] = dst_hash

        remaining = self.untested.get(src_hash)
        if remaining is not None:
            remaining.discard(action)
            if not remaining:
                self.frontier_nodes.discard(src_hash)

        order = self.tried_order.setdefault(src_hash, [])
        if action not in order:
            order.append(action)

    def flag_noop(self, src_hash: int, action: ActionKey) -> None:
        """Flag a no-op self-loop edge so it is deprioritized at src (FR-021).

        Recorded separately from `untested` because the action is already drained
        once tried; this set lets `order_actions` sink known no-ops to the bottom
        and feeds the change-likelihood prior.
        """
        self.noop_actions.setdefault(src_hash, set()).add(action)

    def mark_dead(self, node_hash: int) -> None:
        """Mark a node dead and remove it from the global frontier (FR-015).

        Rebuilds the frozen StateNode with `dead=True` (value objects are replaced,
        not mutated) and clears its untested set so navigation never targets it.
        """
        node = self.nodes.get(node_hash)
        if node is not None and not node.dead:
            self.nodes[node_hash] = StateNode(
                node_hash=node.node_hash,
                untested_actions=frozenset(),
                dead=True,
            )
        self.untested[node_hash] = set()
        self.frontier_nodes.discard(node_hash)

    # --- queries (pure, no mutation) ------------------------------------------

    def untested_at(self, node_hash: int) -> set[ActionKey]:
        """Remaining untested actions at a node (empty set if unknown/dead)."""
        return self.untested.get(node_hash, set())

    def shortest_path(self, src_hash: int, dst_hash: int) -> list[ActionKey]:
        """BFS shortest path of ACTIONS from src to dst (FR-013).

        Returns the list of action keys to traverse, or `[]` if dst is unreachable
        via known directed edges (FR-023 fallback trigger) or src == dst already.
        Exactly one bounded BFS, O(nodes + edges) (NFR-010) — no unbounded
        re-search. Edges are directed; we only follow recorded transitions.
        """
        if src_hash == dst_hash:
            return []

        # Adjacency built from recorded directed edges only (no assumptions).
        came_from: dict[int, tuple[int, ActionKey]] = {}
        queue: deque[int] = deque([src_hash])
        seen = {src_hash}
        while queue:
            cur = queue.popleft()
            for (e_src, action), e_dst in self.edges.items():
                if e_src != cur or e_dst in seen:
                    continue
                seen.add(e_dst)
                came_from[e_dst] = (cur, action)
                if e_dst == dst_hash:
                    return self._reconstruct(came_from, src_hash, dst_hash)
                queue.append(e_dst)
        return []  # unreachable (FR-023)

    @staticmethod
    def _reconstruct(
        came_from: dict[int, tuple[int, ActionKey]],
        src_hash: int,
        dst_hash: int,
    ) -> list[ActionKey]:
        """Walk parent links back from dst to src and return forward actions."""
        path: list[ActionKey] = []
        cur = dst_hash
        while cur != src_hash:
            prev, action = came_from[cur]
            path.append(action)
            cur = prev
        path.reverse()
        return path


@dataclass
class Memory:
    """Per-level mutable store (Glossary 1.8, schema §4.2, §4.4).

    Holds the StateGraph, the current node hash, the last (node, action) edge taken
    (to record the next edge), and the last recorded `levels_completed` (to detect a
    level-boundary change, FR-016). Reset on EVERY level-boundary change.
    """

    graph: StateGraph = field(default_factory=StateGraph)
    current_hash: int | None = None
    last_node_action: tuple[int, ActionKey] | None = None
    levels_completed: int = 0
    # Per-level tally of how many times each action key was EMITTED (FR-012). Used by
    # `order_actions` to round-robin untested simple actions least-recently-tried-
    # first, so the explorer does not fixate on the lowest-numbered action. Reset
    # with the rest of the per-level beliefs (FR-016).
    action_usage: dict[ActionKey, int] = field(default_factory=dict)

    def reset(self) -> None:
        """Drop all beliefs: fresh empty graph, cleared traversal threading (FR-016).

        Allocates a brand-new StateGraph so the previous level's nodes are released
        (NFR-007 memory bound). `levels_completed` is intentionally NOT touched here;
        the caller records the new value right after, so a half-applied reset can't
        leave a stale boundary marker.
        """
        self.graph = StateGraph()
        self.current_hash = None
        self.last_node_action = None
        self.action_usage = {}
