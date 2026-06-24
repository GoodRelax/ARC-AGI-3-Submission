"""[agent/core] solver -- the ONE Solver of the slice: bounded BFS over the AbstractSituation.

State = (controllable_pos, carried_pose, consumed_triggers). Transitions come from the
learned WorldModel (move_delta + footprint passability + trigger->pose_succ). Two searches:
  * ``bfs_solve`` -- shortest actions to WIN (footprint inside container AND carried pose ==
    target pose); the orientation axis is part of the state, so a delivery in the wrong
    orientation is not mistaken for a win.
  * ``bfs_to`` -- shortest actions so the footprint OVERLAPS some cells (curiosity navigation
    to provoke an unknown interaction).
"""

from __future__ import annotations

from collections import deque


def _foot(offsets, pos):
    r0, c0 = pos
    return set((r0 + dr, c0 + dc) for dr, dc in offsets)


def _won(pos, pose, target_pose, offsets, container):
    return pose == target_pose and _foot(offsets, pos) <= container


def bfs_solve(model, grid, offsets, start_pos, start_pose, target_pose, container,
              max_depth=120, max_expand=40000):
    if _won(start_pos, start_pose, target_pose, offsets, container):
        return []
    start = (start_pos, start_pose, frozenset())
    seen = {start}
    q = deque([(start, [])])
    expand = 0
    while q and expand < max_expand:
        (pos, pose, cons), path = q.popleft()
        expand += 1
        if len(path) >= max_depth:
            continue
        for a, (dr, dc) in model.move_delta.items():
            npos = (pos[0] + dr, pos[1] + dc)
            if not model.footprint_passable(grid, offsets, npos, container):
                continue
            foot = _foot(offsets, npos)
            npose, ncons = pose, cons
            for i, trig in enumerate(model.triggers):
                if i not in cons and (foot & trig):
                    ncons = ncons | {i}
                    if pose in model.pose_succ:
                        npose = model.pose_succ[pose]
            key = (npos, npose, ncons)
            if key in seen:
                continue
            seen.add(key)
            if _won(npos, npose, target_pose, offsets, container):
                return path + [a]
            q.append(((npos, npose, ncons), path + [a]))
    return None


def bfs_to(model, grid, offsets, start_pos, goal_cells, allow=frozenset(),
           max_depth=120, max_expand=40000):
    """Shortest actions so the controllable footprint overlaps ``goal_cells``."""
    if _foot(offsets, start_pos) & goal_cells:
        return []
    seen = {start_pos}
    q = deque([(start_pos, [])])
    expand = 0
    while q and expand < max_expand:
        pos, path = q.popleft()
        expand += 1
        if len(path) >= max_depth:
            continue
        for a, (dr, dc) in model.move_delta.items():
            npos = (pos[0] + dr, pos[1] + dc)
            if npos in seen:
                continue
            if not model.footprint_passable(grid, offsets, npos, allow):
                continue
            seen.add(npos)
            if _foot(offsets, npos) & goal_cells:
                return path + [a]
            q.append((npos, path + [a]))
    return None


# =====================================================================================
# Solver / planning (core) cluster -- PlanMoves simulated look-ahead (CMP-28),
# CheckFutility prune (CMP-31), Milestone-roadmap ordering. Built ON TOP of the L1 BFS
# above (which stays the byte-for-byte concrete grid search ``play.py`` imports). This layer
# plans over the *abstract* AbstractSituation (agent/core/situation.py) using the GENERAL built-in
# simulator (world_model.ModelWorld.predict) ranked by the GoalPredicate distance
# (SearchHeuristic, API-05), so it is game-literal-free (NFR-6) and deterministic (DP-10:
# canonical tie-breaks / no RNG / no builtin ``hash()`` for stable identity).
#
# Canon (cite, never duplicate):
#   - _assets/gr-arc-3-terms.md
#       TERM-13 BuiltInSimulator  -- the in-house approximate sim; look-ahead + verify at ZERO
#                                    scored moves (RHAE-0); ``action`` is the SOLE scored unit.
#       TERM-31 AbstractSituation         -- the «value» search node; canonical() is a stable memo key.
#   - _assets/gr-arc-3-domain-model.md
#       GamePlan := ordered moves 1..* (by-value GameMove list); aims at the CURRENT Milestone;
#         the predicted trajectory is NOT stored -- re-derived from WorldModel.predict each turn.
#       Solver -> produces -> GamePlan; Conception owns the plan + the ordered Milestone roadmap.
#       Milestone := one ordered step; roadmap order is a HARD track a fixed implementation
#         enforces (NOT delegated to an LLM -- the belief-agent thrash lesson).
#       Probe ... DetectFutility -- futility's OBSERVE side (rides with play, not here).
#   - 04-specification SC-10 (RHAE-0 look-ahead) / SC-12 (prune side) / SC-20 (roadmap order) ;
#     05-test-strategy TS-10 / TS-12 / TS-19 ; sequence v005 (choose-action (4) PLAN look-ahead
#     xN RHAE-0 ; #9 CheckFutility) ; Ch3 API-03 (Simulator = 0 scored moves) /
#     API-05 (SearchHeuristic = goal distance).
#
# RHAE-0 (NFR-3): nothing here touches the real environment. The ONLY dynamics call is
# ``world.predict`` (the offline sim); a planning run sends EXACTLY 0 actions to the env
# (TS-10 asserts this with an env spy). The «GameMove» is an opaque action-id token (int);
# no colour / coordinate / game literal is read (NFR-6).
# =====================================================================================

from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional, Protocol, Sequence, Tuple

from .goal import And, Atom, GoalPredicate, Roadmap
from .situation import AbstractSituation
from .world_model import MoveEffect, classify_move_effect


# A GameMove is the single «1手»: an opaque action-id token. The planner never interprets it
# (no button/colour/coordinate semantics) -- it only feeds it to the sim and orders ties by it.
GameMove = int


class _Simulator(Protocol):
    """The structural contract the planner needs from the built-in simulator (ModelWorld):
    a pure ``predict(situation, move) -> next AbstractSituation`` that scores NOTHING (API-03). Any
    object exposing this method works -- the TS-10 spy wraps a real ModelWorld to count that
    the planner calls ONLY ``predict`` and sends 0 actions to the environment."""

    def predict(self, situation: AbstractSituation, move: Optional[int]) -> AbstractSituation: ...


# A goal-distance callable: the SearchHeuristic (API-05). ``GoalPredicate.distance`` satisfies
# it (total, non-negative, 0-iff-test, monotone). Injected so the planner never hard-binds a
# concrete GoalPredicate subclass and the futility classifier stays decoupled (world_model).
Heuristic = Callable[[AbstractSituation], object]


# ------------------------------------------------------------------------------- GamePlan
@dataclass(frozen=True)
class GamePlan:
    """An ordered list of GameMove ids that a Solver produced to reach a goal (CMP-28 output).

    A «value» (frozen): two plans with the same ``moves`` and ``goal`` handle compare equal.
    The predicted trajectory is deliberately NOT stored (domain model: re-derived from
    ``predict`` each turn); only the ordered ``moves``, the ``goal_distance`` the look-ahead
    reached (0 iff the plan satisfies its goal), and the optional ``aimed_at`` Milestone-goal
    handle (the goal this plan was planned toward -- the current roadmap step) are kept.

    Ergonomics: iterating / ``len`` / indexing a GamePlan yields its moves, so callers can do
    ``plan[0]`` (the MPC "commit one move" idiom) or ``for m in plan`` without reaching into
    ``.moves``. ``is_empty()`` is True for the zero-move plan (goal already satisfied, or no
    progress possible)."""

    moves: Tuple[GameMove, ...] = ()
    goal_distance: object = None
    aimed_at: Optional[GoalPredicate] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "moves", tuple(self.moves))

    def is_empty(self) -> bool:
        """True iff the plan commits no move (goal already met, or nothing makes progress)."""
        return len(self.moves) == 0

    def first(self) -> Optional[GameMove]:
        """The first GameMove to commit this turn (MPC), or ``None`` for an empty plan."""
        return self.moves[0] if self.moves else None

    def __iter__(self):
        return iter(self.moves)

    def __len__(self) -> int:
        return len(self.moves)

    def __getitem__(self, index):
        return self.moves[index]


# Default search bounds. Deterministic and game-literal-free; chosen so the abstract look-ahead
# terminates cheaply. ``horizon`` caps plan length (look-ahead depth); ``max_expand`` caps node
# expansions (a safety valve mirroring the L1 BFS bounds).
DEFAULT_HORIZON: int = 32
DEFAULT_MAX_EXPAND: int = 20000


def _sorted_moves(moves: Sequence[GameMove]) -> List[GameMove]:
    """Deterministic candidate order: ascending GameMove id (the stable tie-break, DP-10).
    Duplicates are dropped (a move set), preserving the ascending order. No builtin ``hash``
    is used for identity -- ints sort by value."""
    seen: set = set()
    out: List[GameMove] = []
    for m in sorted(moves):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


# ---------------------------------------------------------------------------- CheckFutility
def check_futility(
    world: _Simulator,
    situation: AbstractSituation,
    goal: GoalPredicate,
    moves: Sequence[GameMove],
) -> List[GameMove]:
    """CheckFutility (CMP-31): keep only the candidate moves whose PREDICTED effect is PROGRESS.

    For each candidate it predicts the next AbstractSituation on the built-in sim and classifies the
    move effect via ``classify_move_effect(situation, predict(situation, move), goal.distance)``
    (the exact TS-12 oracle): an ``invariant`` (next AbstractSituation identical) or ``no_progress``
    (changed but goal-distance unchanged) candidate is *futile* and is PRUNED; only a
    ``progress`` (goal-distance decreased) candidate survives (NFR-3). Returns the surviving
    moves in the deterministic ascending-id order.

    RHAE-0: only ``world.predict`` is called -- zero actions reach the environment (the sim does
    not score, API-03). This is the PREDICT side of futility; the OBSERVE side (DetectFutility
    recording the *actually-played* move's MoveEffect every turn) rides with the play cluster,
    not here (the predict/verify symmetry of SC-12)."""
    survivors: List[GameMove] = []
    for move in _sorted_moves(moves):
        nxt = world.predict(situation, move)
        effect = classify_move_effect(situation, nxt, goal.distance)
        if not MoveEffect.futile(effect):        # effect == PROGRESS
            survivors.append(move)
    return survivors


# -------------------------------------------------------------------------- PlanMoves / plan
def plan(
    world: _Simulator,
    situation: AbstractSituation,
    goal: GoalPredicate,
    moves: Sequence[GameMove],
    horizon: int = DEFAULT_HORIZON,
    max_expand: int = DEFAULT_MAX_EXPAND,
    aimed_at: Optional[GoalPredicate] = None,
) -> GamePlan:
    """PlanMoves (CMP-28): bounded look-ahead over the built-in sim, ranked by goal distance.

    A best-first (greedy / uniform-cost-on-the-heuristic) search over ``world.predict``: from
    ``situation`` it expands candidate ``moves`` (ascending-id), scoring each successor by
    ``goal.distance`` (the SearchHeuristic, API-05), and returns the ordered ``GamePlan`` of
    GameMove ids that first reaches ``goal.test`` within ``horizon`` steps. When the goal is
    unreachable within the bounds, it returns the BEST partial plan found (the prefix that
    minimised the goal distance) so the agent still makes progress (MPC re-plans next turn);
    when nothing makes progress, the plan is empty.

    RHAE-0 / API-03 (NFR-3): the ONLY dynamics call is ``world.predict`` -- the search is
    entirely virtual and sends EXACTLY 0 actions to the real environment (TS-10's env-spy
    asserts the count is 0). Determinism (DP-10): candidate moves are tried in ascending id
    order; ties on the heuristic are broken by (shorter path, then lexicographically-smaller
    move sequence); the visited set keys on ``AbstractSituation.canonical()`` -- never builtin ``hash``
    of mutable state -- and no RNG is used, so the same inputs yield the same plan.

    Parameters
    ----------
    world:
        The built-in simulator (a :class:`world_model.ModelWorld` or any ``.predict`` provider).
    situation:
        The current abstract :class:`AbstractSituation` (the StateAbstraction output).
    goal:
        The :class:`GoalPredicate` to satisfy; ``goal.distance`` ranks moves, ``goal.test`` ends
        the search.
    moves:
        The available GameMove ids (e.g. the env's available_actions, lifted to opaque tokens).
    horizon:
        Max plan length / look-ahead depth (a single roadmap step's reach).
    max_expand:
        Safety cap on node expansions (a deterministic bound, never RNG-driven).
    aimed_at:
        The Milestone goal this plan aims at (recorded on the returned plan for TS-19's
        planned-toward-goal trace). Defaults to ``goal``.

    Returns
    -------
    GamePlan
        An ordered GameMove sequence; empty iff the goal already holds or nothing progresses.
    """
    aim = aimed_at if aimed_at is not None else goal
    start_d = goal.distance(situation)
    if goal.test(situation):
        return GamePlan(moves=(), goal_distance=start_d, aimed_at=aim)

    candidates = _sorted_moves(moves)
    start_key = situation.canonical()
    # best-so-far partial: the path that minimised the goal distance (for the unreachable case).
    best_path: Tuple[GameMove, ...] = ()
    best_d = start_d
    # Visited set keyed on canonical AbstractSituation identity (DP-10): collapse equal salient configs.
    seen = {start_key}
    # Frontier ordered by (distance, depth, path) so expansion is best-first AND fully
    # deterministic -- a smaller heuristic wins; ties prefer the shorter, then lexicographically
    # smaller, move sequence. A plain sorted list is used (no heapq) to keep the order total and
    # obvious; the frontier stays small under the horizon/expand bounds.
    frontier: List[Tuple[object, int, Tuple[GameMove, ...], AbstractSituation]] = [
        (start_d, 0, (), situation)
    ]
    expand = 0
    while frontier and expand < max_expand:
        frontier.sort(key=lambda item: (_dist_key(item[0]), item[1], item[2]))
        _d, depth, path, state = frontier.pop(0)
        expand += 1
        if depth >= horizon:
            continue
        for move in candidates:
            nxt = world.predict(state, move)
            key = nxt.canonical()
            if key in seen:
                continue
            seen.add(key)
            npath = path + (move,)
            if goal.test(nxt):
                return GamePlan(moves=npath, goal_distance=goal.distance(nxt), aimed_at=aim)
            nd = goal.distance(nxt)
            if _dist_key(nd) < _dist_key(best_d):
                best_d, best_path = nd, npath
            frontier.append((nd, depth + 1, npath, nxt))
    # Goal not reached within the bounds: return the best progress-making prefix (possibly
    # empty if nothing lowered the distance). MPC re-plans next turn from the new observation.
    return GamePlan(moves=best_path, goal_distance=best_d, aimed_at=aim)


def _dist_key(distance: object) -> tuple:
    """A TOTAL, deterministic ordering key for a goal distance (API-05 allows int/float/tuple).
    Wraps the value with its type name so heterogeneous-but-comparable distances never raise on
    ``<`` and the order is stable across processes (DP-10) -- e.g. an ``int`` 3 and a ``tuple``
    never collide. For the homogeneous int distances ``GoalPredicate`` returns, this is just
    the integer order."""
    return (type(distance).__name__, distance)


@dataclass
class PlanMoves:
    """The PlanMoves use-case object (CMP-28): plans a :class:`GamePlan` over the built-in sim.

    A thin, stateless handle (mirrors the domain-model use-case and the legacy ``Planner``
    call shape ``PlanMoves().plan(...)``) delegating to the module-level :func:`plan` /
    :func:`check_futility` / :func:`plan_roadmap`. Holds default search bounds so a caller can
    fix a horizon once and re-use it each turn. No per-board state, no RNG -- deterministic."""

    horizon: int = DEFAULT_HORIZON
    max_expand: int = DEFAULT_MAX_EXPAND

    def plan(
        self,
        world: _Simulator,
        situation: AbstractSituation,
        goal: GoalPredicate,
        moves: Sequence[GameMove],
        aimed_at: Optional[GoalPredicate] = None,
    ) -> GamePlan:
        """Bounded look-ahead toward ``goal`` (see :func:`plan`)."""
        return plan(
            world, situation, goal, moves,
            horizon=self.horizon, max_expand=self.max_expand, aimed_at=aimed_at,
        )

    def check_futility(
        self,
        world: _Simulator,
        situation: AbstractSituation,
        goal: GoalPredicate,
        moves: Sequence[GameMove],
    ) -> List[GameMove]:
        """Prune non-progress candidates (see :func:`check_futility`)."""
        return check_futility(world, situation, goal, moves)

    def plan_roadmap(
        self,
        world: _Simulator,
        situation: AbstractSituation,
        roadmap: Roadmap,
        moves: Sequence[GameMove],
    ) -> GamePlan:
        """Plan toward the current (first-unmet) Milestone of ``roadmap`` (see
        :func:`plan_roadmap`)."""
        return plan_roadmap(
            world, situation, roadmap, moves,
            horizon=self.horizon, max_expand=self.max_expand,
        )


# -------------------------------------------------------------- Milestone-roadmap ordering
def plan_roadmap(
    world: _Simulator,
    situation: AbstractSituation,
    roadmap: Roadmap,
    moves: Sequence[GameMove],
    horizon: int = DEFAULT_HORIZON,
    max_expand: int = DEFAULT_MAX_EXPAND,
) -> GamePlan:
    """Roadmap-ordered planning (TS-19): plan toward the CURRENT Milestone's goal only.

    The current milestone is ``roadmap.current(situation)`` -- the FIRST (in roadmap order)
    whose goal does not yet hold. The planner aims at THAT step's goal, never a later one,
    until its predecessor's goal holds; the overall target is ``roadmap.final_goal()``. This is
    the固定実装 that enforces order (NOT an LLM -- the belief-agent thrash lesson): a later
    milestone can only become the planned-toward goal once every predecessor is met, because
    ``current`` advances strictly in order.

    Returns the :class:`GamePlan` toward the current step (its ``aimed_at`` records WHICH goal
    was targeted, for the TS-19 planned-toward-goal trace). When the whole roadmap is already
    satisfied, returns the empty plan aimed at the final goal. An empty roadmap yields an empty
    plan (no goal to pursue)."""
    current = roadmap.current(situation)
    if current is None:
        # Every milestone is met -> roadmap complete; nothing left to plan.
        return GamePlan(moves=(), goal_distance=0, aimed_at=roadmap.final_goal())
    return plan(
        world, situation, current.goal, moves,
        horizon=horizon, max_expand=max_expand, aimed_at=current.goal,
    )


# =====================================================================================
# Solver escalation / SolverLibrary cluster (CMP-14 Solver / CMP-15 SolverLibrary /
# CMP-35 SelectSolver / CMP-36 ScoreCandidates / CMP-16 Conception). Built ON TOP of the
# solver-core ``plan`` above (the base "navigate" Solver) -- it stays the byte-for-byte
# look-ahead the play cluster imports; this layer DISPATCHES among Solvers, COMPOSES them by
# axis (Law E), and -- when nothing baked fits -- has the built-in LLM GENERATE a small
# program mid-game, registering it to the overlay. Every stage's candidate is scored by
# OBSERVATION (ScoreCandidates) BEFORE adoption.
#
# Canon (cite, never duplicate):
#   - _assets/gr-arc-3-terms.md
#       TERM-13 BuiltInSimulator -- look-ahead + verify at ZERO scored moves (RHAE-0).
#       TERM-17 EffectSignature  -- the per-axis effect key {event,target,attribute,operator,
#                                   params_class}; ``attribute`` IS the independent axis
#                                   (position / orientation / form / colour / count ...) Law E
#                                   factors the goal along (attributes.effect_signature).
#       TERM-22 proposer / TERM-23 Qwen -- the in-core LOCAL/OFFLINE LLM; proposes only, the
#                                   master (ScoreCandidates) disposes (NFR-1).
#   - _assets/gr-arc-3-domain-model.plan.md
#       Solver := a typed GoalPredicate->GamePlan; {kind, applicability(structure->confidence,
#         observation-updated), verification_horizon(=futility progress window, adaptive),
#         parts(composite = Law E axis-factoring), impl(port)}. A noun «entity», SINGULAR.
#       SolverLibrary := the catalog; base (LTM prior, read-only) + overlay (runtime,
#         synthesized); SelectSolver ranks by applicability + synthesizes-when-stuck.
#       Conception := the answer bundle (WorldModel + GoalPredicate + GamePlan + roadmap +
#         provenance Solver), confidence-ranked.
#   - _assets/gr-arc-3-sequence-solve-core.md (sheet 2): competing solvers scored by
#     ScoreCandidates by OBSERVATION (no training answer exists in ARC-3); the Local LLM is
#     "just another proposer" -- its typed output is scored EXACTLY like any candidate.
#   - docs/memos/solver-management.md (design SSOT): structure -> ranked candidates ->
#     cheap-first parallel -> observe-score -> sharpen; ``verification_horizon`` = the
#     futility progress window; ``parts`` = axis-factoring (Law E).
#   - 04-specification SC-14 / 02-requirements FR-C-11 (staged escalation; Law E axis
#     decomposition; observation post-scoring) ; 05-test-strategy TS-14 ; NFR-1 (LLM offline).
#
# Hard rules honoured here:
#   * RHAE-0 (NFR-3): the only dynamics call is ``world.predict`` -- 0 actions reach the env.
#   * No game literals (NFR-6): a Solver keys on an axis NAME (a Dimension/attribute label) and
#     opaque GameMove ids only -- never a colour number, coordinate, or glyph.
#   * Determinism (DP-10): no RNG; no builtin ``hash()`` for identity (Solvers carry a stable
#     ``kind``/axis string id; ranking ties break on (-, ascending id)); the stub generator the
#     test injects is deterministic. The LLM is a PLUGGABLE Protocol -- the real one is the T3
#     offline-risk and is never imported here.
# =====================================================================================

from typing import Dict, FrozenSet, Iterable  # (supplements the module-level typing import)


# ----------------------------------------------------------------- structure signature
# A Solver's ``applicability`` is a function of the STRUCTURE SIGNATURE (the partial Goal
# logical-form x World transition-structure -- memos/solver-management.md). We model it as a
# frozenset of axis NAMES the goal touches (e.g. {"position"}, {"colour", "position"}) plus an
# optional ``kind`` tag, because the staged escalation keys on which independent axes the goal
# decomposes into (Law E). It is game-literal-free: an axis name is a Dimension/attribute label
# (the EffectSignature.attribute vocabulary), never a colour/coordinate.
@dataclass(frozen=True)
class StructureSignature:
    """The dispatch key SelectSolver ranks Solvers against (CMP-35 input).

    ``axes`` -- the independent attribute axes the (partial) goal touches, as a frozenset of
    axis-name strings drawn from the EffectSignature.attribute vocabulary
    (``position`` / ``orientation`` / ``form`` / ``colour`` / ``count`` / ``presence`` /
    ``scalar``); these are the axes Law E factors the goal along. ``kind`` -- an optional
    coarse transition-structure tag (e.g. ``"navigate"``). Frozen «value»: deterministic
    equality from primitives only (DP-10)."""

    axes: FrozenSet[str] = field(default_factory=frozenset)
    kind: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", frozenset(str(a) for a in self.axes))

    def with_axes(self, axes: Iterable[str]) -> "StructureSignature":
        """A copy carrying ``axes`` (same kind) -- used when restricting to one axis (Law E)."""
        return StructureSignature(axes=frozenset(str(a) for a in axes), kind=self.kind)


# An applicability function: STRUCTURE SIGNATURE -> confidence in [0, 1] (a *function*, not a
# static scalar -- memos/solver-management.md; observation-updated like InteractionRule.conf).
Applicability = Callable[[StructureSignature], float]

# The implementation of a Solver: it turns (world, situation, goal, moves) into a GamePlan.
# The base navigate Solver's impl is the module-level ``plan``; a composite's impl composes its
# parts; an LLM-generated Solver's impl runs the generated program. Kept as a port (memos:
# ``impl`` = which port: SearchHeuristic / Simulator / ConstrainedGenerator).
SolverImpl = Callable[["_Simulator", AbstractSituation, GoalPredicate, Sequence[GameMove]], GamePlan]


# ------------------------------------------------------------------------------- Solver
@dataclass(frozen=True)
class Solver:
    """A typed ``GoalPredicate -> GamePlan`` connector -- the morphism (CMP-14).

    Attributes (domain-model.plan + memos/solver-management.md):
      * ``kind``                 -- the solver family / a stable game-literal-free id
                                    (search | order | modular | ... ; or "navigate" for the
                                    base). Used as the deterministic identity + tie-break.
      * ``applicability``        -- a FUNCTION ``StructureSignature -> confidence`` (not a static
                                    scalar); the dispatch-ranking signal, observation-updatable
                                    (see :meth:`reweight`).
      * ``verification_horizon`` -- how many moves to invest to verify the hypothesis = the
                                    futility progress window (G1/P1); adaptive. An RHAE
                                    investment, so it doubles as the look-ahead bound here.
      * ``parts``                -- sub-Solvers composed by axis-factoring (Law E); empty for a
                                    leaf. A composite's :meth:`solve` chains its parts so each
                                    independent axis is advanced (and observed) separately.
      * ``impl``                 -- the port that produces the GamePlan (the base = module
                                    ``plan``; a composite = :func:`_compose_parts`; an
                                    LLM-generated one = the generated program). ``None`` for a
                                    composite (its parts carry the impl).
      * ``axis``                 -- for a per-axis leaf in a Law-E composite: the single axis
                                    name this Solver is responsible for (so the test can assert
                                    each axis got a DISTINCT Solver and its effect is observed
                                    separately). Empty for a whole-goal Solver.
      * ``origin``               -- provenance tag: ``"base"`` (LTM prior) / ``"composite"`` /
                                    ``"llm"`` (overlay, just-generated). Lets ScoreCandidates /
                                    the test trace where a candidate came from.

    Frozen «value»-ish: identity is the (kind, axis, origin) triple via :meth:`solver_id`
    (a stable string -- NEVER builtin ``hash()`` of the callables), so ranking is deterministic
    (DP-10). The callables compare by identity, which is fine -- ordering never depends on them.
    """

    kind: str
    applicability: Applicability = field(default=lambda _sig: 0.0)
    verification_horizon: int = DEFAULT_HORIZON
    parts: Tuple["Solver", ...] = ()
    impl: Optional[SolverImpl] = None
    axis: str = ""
    origin: str = "base"

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", tuple(self.parts))

    def is_composite(self) -> bool:
        """True iff this Solver composes sub-Solvers (a Law-E axis-factored solver)."""
        return len(self.parts) > 0

    def solver_id(self) -> str:
        """A stable, process-independent identity string (DP-10): ``origin:kind[:axis]``.
        Used as the ranking tie-break and the overlay de-dup key -- never builtin ``hash``."""
        base = "%s:%s" % (self.origin, self.kind)
        return ("%s:%s" % (base, self.axis)) if self.axis else base

    def confidence(self, signature: StructureSignature) -> float:
        """The applicability confidence for ``signature`` (the dispatch signal), clamped to
        [0, 1]. A composite reports the MIN confidence across its parts (it is only as
        applicable as its least-applicable axis), so a composite never out-ranks a leaf that
        genuinely covers the whole goal unless every axis is covered."""
        if self.is_composite():
            return min((p.confidence(signature) for p in self.parts), default=0.0)
        try:
            c = float(self.applicability(signature))
        except Exception:
            return 0.0
        return 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)

    def reweight(self, signature: StructureSignature, confidence: float) -> "Solver":
        """Return a copy whose applicability is PINNED to ``confidence`` on ``signature``
        (observation-update: ScoreCandidates raises/lowers a Solver's applicability after
        seeing its effect). Deterministic; other signatures fall back to the old function."""
        old = self.applicability
        pinned = max(0.0, min(1.0, float(confidence)))
        target = signature

        def _updated(sig: StructureSignature) -> float:
            if sig == target:
                return pinned
            return old(sig)

        return Solver(
            kind=self.kind, applicability=_updated,
            verification_horizon=self.verification_horizon, parts=self.parts,
            impl=self.impl, axis=self.axis, origin=self.origin,
        )

    def solve(
        self,
        world: "_Simulator",
        situation: AbstractSituation,
        goal: GoalPredicate,
        moves: Sequence[GameMove],
    ) -> GamePlan:
        """Produce a :class:`GamePlan` for ``goal`` from ``situation`` (RHAE-0: only
        ``world.predict`` is called). A composite chains its parts (each axis advanced in turn
        = Law E); a leaf runs its ``impl`` (defaulting to the base navigate ``plan`` bounded by
        ``verification_horizon``). Never sends an action to the environment."""
        if self.is_composite():
            return _compose_parts(self.parts, world, situation, goal, moves)
        if self.impl is not None:
            return self.impl(world, situation, goal, moves)
        # Leaf with no explicit impl == the base navigate Solver: bounded look-ahead toward goal.
        return plan(world, situation, goal, moves, horizon=self.verification_horizon)


def restrict_goal_to_axis(goal: GoalPredicate, axis: str) -> GoalPredicate:
    """Restrict an AND-of-axis-relations ``goal`` to the conjunction of ITS atoms that constrain
    ``axis`` (the Law-E projection of the goal onto one independent axis). A relation NAME maps to
    the axis it constrains via :data:`_RELATION_AXIS`. Returns the (sub-)GoalPredicate over only
    that axis's atoms; falls back to the WHOLE ``goal`` when no atom matches (so a leaf never
    plans toward a vacuous predicate). Instance-invariant (the kept atoms are role-keyed)."""
    matching = []
    for atom in goal.atoms():
        rel_name = getattr(atom, "name", None)
        axes = _RELATION_AXIS.get(rel_name, ("position",)) if rel_name is not None else ()
        if axis in axes and isinstance(atom, Atom):
            matching.append(atom)
    if not matching:
        return goal                               # nothing on this axis: plan the whole goal
    if len(matching) == 1:
        return matching[0]
    return And(operands=tuple(matching))


def _axis_leaf_impl(axis: str, horizon: int) -> SolverImpl:
    """The impl of a per-axis leaf Solver: plan toward only ITS axis's projection of the goal
    (:func:`restrict_goal_to_axis`), using only the moves that act on ITS axis (the registered
    per-axis probe moves -- the interactions this Solver OWNS), bounded by ``horizon``. RHAE-0
    (only ``world.predict``). A leaf therefore addresses exactly one independent axis through the
    interactions it owns (the Law-E building block); a single such leaf cannot satisfy a
    multi-axis goal alone -- it lacks the OTHER axes' moves -- which is the escalation cue (SC-14).

    When the axis has NO registered probe moves, the leaf falls back to all available ``moves``
    (a Solver with an unknown move repertoire plans over everything -- still game-literal-free)."""

    def _impl(world: "_Simulator", situation: AbstractSituation, goal: GoalPredicate,
              moves: Sequence[GameMove]) -> GamePlan:
        axis_goal = restrict_goal_to_axis(goal, axis)
        own = _AXIS_PROBE_MOVES.get(axis)
        axis_moves = list(own) if own else list(moves)
        return plan(world, situation, axis_goal, axis_moves, horizon=horizon, aimed_at=goal)

    return _impl


def navigate_solver(horizon: int = DEFAULT_HORIZON) -> Solver:
    """The base "navigate" Solver (the existing ``plan``): addresses the single POSITION axis (the
    reach/deliver family) -- the LTM prior every library ships with. Its applicability is high when
    the goal touches the ``position`` axis and falls off as more axes appear, and its impl plans
    toward ONLY the position projection of the goal: a single navigate cannot satisfy a multi-axis
    goal alone -- that is what forces escalation (SC-14)."""

    def _applies(sig: StructureSignature) -> float:
        if "position" not in sig.axes:
            return 0.0
        # Full confidence for a pure-position goal; decays as other axes join (escalation cue).
        return 1.0 / float(len(sig.axes))

    return Solver(
        kind="navigate",
        applicability=_applies,
        verification_horizon=horizon,
        impl=_axis_leaf_impl("position", horizon),
        axis="position",
        origin="base",
    )


def axis_solver(axis: str, horizon: int = DEFAULT_HORIZON) -> Solver:
    """A single-axis Solver responsible for ONE independent attribute ``axis`` (colour / shape /
    position / orientation ...). It plans toward only the part of the goal on its axis
    (:func:`restrict_goal_to_axis`) -- the building block a Law-E composite factors the goal into.
    Applicability is full iff the goal is a SINGLE-axis goal on ITS axis, else 0: an axis Solver
    is not a standalone candidate for a multi-axis goal (those route to a Law-E composite, which
    pulls axis Solvers in by axis name). This keeps the stage-1 whole-goal ranking to whole-goal
    Solvers (navigate); the axis leaves are the composite's building blocks."""

    def _applies(sig: StructureSignature) -> float:
        return 1.0 if sig.axes == frozenset({axis}) else 0.0

    return Solver(
        kind="axis-%s" % axis,
        applicability=_applies,
        verification_horizon=horizon,
        impl=_axis_leaf_impl(axis, horizon),
        axis=axis,
        origin="base",
    )


def _compose_parts(
    parts: Tuple[Solver, ...],
    world: "_Simulator",
    situation: AbstractSituation,
    goal: GoalPredicate,
    moves: Sequence[GameMove],
) -> GamePlan:
    """Run a Law-E composite: each part advances its OWN axis from the running situation, and the
    per-axis move sub-sequences are concatenated in a deterministic (part) order. Each part plans
    against the WHOLE goal but is bounded by its own horizon, so its contribution is the moves
    that advance ITS axis; the running situation is rolled forward on the sim between parts (so
    the axes are advanced -- and, by construction, OBSERVED -- separately). RHAE-0: only
    ``world.predict`` is used. Returns the concatenated :class:`GamePlan` aimed at ``goal``."""
    state = situation
    out_moves: List[GameMove] = []
    for part in parts:
        sub = part.solve(world, state, goal, moves)
        for mv in sub.moves:
            out_moves.append(mv)
            state = world.predict(state, mv)
    return GamePlan(moves=tuple(out_moves), goal_distance=goal.distance(state), aimed_at=goal)


# ------------------------------------------------------------------- per-axis observation
@dataclass(frozen=True)
class AxisEffect:
    """The separately-observed effect of advancing ONE axis (Law E evidence, TS-14 (a)).

    Records, for a single axis Solver, the move it committed and the MoveEffect that resulted
    when classified against an axis-restricted goal-distance -- so the test can assert each
    independent axis's effect is observed SEPARATELY (not lumped into one whole-goal delta).
    ``axis`` is the attribute name; ``effect`` is a :class:`world_model.MoveEffect` label;
    ``move`` is the opaque id committed (``None`` iff the axis-plan was empty)."""

    axis: str
    effect: str
    move: Optional[GameMove] = None


def observe_axis_effects(
    composite: Solver,
    world: "_Simulator",
    situation: AbstractSituation,
    axis_distance: Mapping[str, Heuristic],
) -> Tuple[AxisEffect, ...]:
    """Advance each axis-part of a Law-E ``composite`` ONE move and classify ITS effect with
    THAT axis's own goal-distance (``axis_distance[axis]``) -- the per-axis separate observation
    of SC-14 / TS-14. Returns one :class:`AxisEffect` per part, in part order.

    For each part: plan its sub-plan from the running state, commit its first move (MPC), predict
    the next state, and classify the move via ``classify_move_effect`` against the part's axis
    distance (so the colour move is scored on the colour axis, the position move on the position
    axis, etc. -- the effects are SEPARATE). RHAE-0 (only ``world.predict``); deterministic."""
    state = situation
    effects: List[AxisEffect] = []
    for part in composite.parts:
        axis = part.axis
        dist = axis_distance.get(axis)
        # Plan the sub-move against a throwaway goal whose distance IS the injected per-axis
        # distance, so the classifier reads the SAME axis metric the caller asserts on. The
        # candidate moves are this axis's registered probe moves (opaque ids, no game literal).
        if dist is not None:
            axis_goal: GoalPredicate = _AxisGoal(axis=axis, dist=dist)
            sub = part.solve(world, state, axis_goal, _AXIS_PROBE_MOVES.get(axis, ()))
        else:
            sub = part.solve(world, state, _TRIVIAL_GOAL, ())
        move = sub.first()
        if move is None:
            effects.append(AxisEffect(axis=axis, effect=MoveEffect.INVARIANT, move=None))
            continue
        nxt = world.predict(state, move)
        effect = (classify_move_effect(state, nxt, dist) if dist is not None
                  else MoveEffect.INVARIANT)
        effects.append(AxisEffect(axis=axis, effect=effect, move=move))
        state = nxt
    return tuple(effects)


# A per-axis probe-move table the composite-observation helper consults: which opaque GameMove
# advances each axis. It is supplied by the caller through ``register_axis_probe`` (the test
# wires its synthetic axis->move map); empty by default (game-literal-free -- ids are opaque).
_AXIS_PROBE_MOVES: Dict[str, Tuple[GameMove, ...]] = {}


def register_axis_probe(axis: str, moves: Iterable[GameMove]) -> None:
    """Register the opaque GameMove ids that advance ``axis`` (used only by
    :func:`observe_axis_effects` to pick a probing move per axis). Deterministic; ids are
    opaque tokens (no game literal)."""
    _AXIS_PROBE_MOVES[axis] = tuple(moves)


class _AxisGoal(GoalPredicate):
    """A minimal GoalPredicate wrapping an injected per-axis distance (used internally by
    :func:`observe_axis_effects` so the planner/classifier reads exactly the caller's axis
    metric). ``test`` holds iff that distance is 0; instance-invariant (no literal terms)."""

    def __init__(self, axis: str, dist: Heuristic) -> None:
        self._axis = str(axis)
        self._dist = dist

    def test(self, situation: AbstractSituation) -> bool:
        return _as_int(self._dist(situation)) <= 0

    def distance(self, situation: AbstractSituation):
        return self._dist(situation)

    def atoms(self) -> tuple:
        return ()

    def canonical(self) -> tuple:
        return ("axisgoal", self._axis)

    def describe(self) -> str:
        return "axis:%s" % self._axis


class _TrivialGoal(GoalPredicate):
    """A constant-true GoalPredicate (distance 0): a safe fallback when no axis distance is
    supplied, so :func:`observe_axis_effects` never crashes on a missing metric."""

    def test(self, situation: AbstractSituation) -> bool:
        return True

    def distance(self, situation: AbstractSituation) -> int:
        return 0

    def atoms(self) -> tuple:
        return ()

    def canonical(self) -> tuple:
        return ("trivial",)

    def describe(self) -> str:
        return "true"


_TRIVIAL_GOAL = _TrivialGoal()


def _as_int(value) -> int:
    """Coerce a distance to int for a >0 / <=0 comparison (distances are int-like here)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


# ---------------------------------------------------------------------- ConstrainedGenerator
class ConstrainedGenerator(Protocol):
    """The pluggable, LOCAL/OFFLINE program generator port (the built-in LLM, NFR-1; CMP-35
    crux). The agent's core consults it ONLY to PROPOSE a small program when nothing baked fits
    (sequence sheet 2 "propose a NEW structure when stuck"); the proposal is then scored by
    ScoreCandidates EXACTLY like any other candidate (the master disposes). It is a PROTOCOL so
    the real Qwen+vLLM generator is injected at runtime and a DETERMINISTIC STUB is injected in
    the test -- the real LLM is the T3 offline-risk and is NEVER imported here.

    ``generate(signature, situation, goal, moves) -> SolverImpl`` returns the IMPL of a fresh
    Solver: a callable ``(world, situation, goal, moves) -> GamePlan``. Implementations MUST be
    offline and deterministic (NFR-1 / DP-10); they may read only the opaque ``moves`` and the
    abstract situation/goal -- never the real environment, never a game literal."""

    def generate(
        self,
        signature: StructureSignature,
        situation: AbstractSituation,
        goal: GoalPredicate,
        moves: Sequence[GameMove],
    ) -> SolverImpl: ...


def llm_generated_solver(
    generator: ConstrainedGenerator,
    signature: StructureSignature,
    situation: AbstractSituation,
    goal: GoalPredicate,
    moves: Sequence[GameMove],
    horizon: int = DEFAULT_HORIZON,
) -> Solver:
    """Wrap a program from ``generator`` as a fresh overlay Solver (origin ``"llm"``; CMP-35
    stage 3). The generator returns a :class:`SolverImpl`; we box it in a Solver with full
    nominal applicability on ``signature`` (its REAL worth is decided by ScoreCandidates, not by
    this prior). NFR-1: ``generator`` is local/offline by contract; nothing here goes online."""
    impl = generator.generate(signature, situation, goal, moves)

    def _applies(sig: StructureSignature) -> float:
        return 1.0 if sig == signature else 0.5

    return Solver(
        kind="llm-generated",
        applicability=_applies,
        verification_horizon=horizon,
        impl=impl,
        axis="",
        origin="llm",
    )


# ------------------------------------------------------------------------- SolverLibrary
class SolverLibrary:
    """The two-layer Solver catalog (CMP-15): ``base`` (LTM prior, read-only) + ``overlay``
    (runtime, incl. just-generated). Parallels :class:`attributes.Lexicon` and
    :class:`goal.GoalPatterns`. ``select`` ranks the applicable Solvers for a goal+situation by
    applicability confidence (deterministic tie-break).

    Determinism (DP-10): both layers are ordered lists kept in registration order; ``select``
    sorts by (-confidence, ``solver_id``) -- a total, process-independent order, never builtin
    ``hash``. ``add`` appends to ``overlay`` (de-duped on ``solver_id``) -- the in-ops synthesis
    target. No game literal lives in a Solver key (NFR-6)."""

    def __init__(self, base: Optional[Sequence[Solver]] = None) -> None:
        self._base: List[Solver] = list(base) if base is not None else _default_base_solvers()
        self._overlay: List[Solver] = []

    # ---- layers ----
    def base(self) -> Tuple[Solver, ...]:
        """The read-only baked prior Solvers (LTM), in registration order."""
        return tuple(self._base)

    def overlay(self) -> Tuple[Solver, ...]:
        """The runtime-synthesized Solvers (incl. just-generated), in registration order."""
        return tuple(self._overlay)

    def all_solvers(self) -> Tuple[Solver, ...]:
        """Base then overlay (the full catalog), in a deterministic order."""
        return tuple(self._base) + tuple(self._overlay)

    def add(self, solver: Solver) -> Solver:
        """Register a synthesized ``solver`` into the OVERLAY (e.g. the LLM-generated program --
        sequence sheet 2 S2). De-duped on ``solver_id`` (a re-add updates in place). Base is
        never mutated (read-only LTM). Returns the registered Solver."""
        sid = solver.solver_id()
        for i, existing in enumerate(self._overlay):
            if existing.solver_id() == sid:
                self._overlay[i] = solver
                return solver
        self._overlay.append(solver)
        return solver

    def reweight(self, solver: Solver, signature: StructureSignature, confidence: float) -> Solver:
        """Observation-update a Solver's applicability (overlay only; base stays read-only). The
        reweighted twin is registered to the overlay, shadowing a base Solver of the same id
        without mutating the read-only base. ``add`` de-dupes on ``solver_id``. Returns it."""
        return self.add(solver.reweight(signature, confidence))

    # ---- selection ----
    def select(
        self,
        goal: GoalPredicate,
        situation: AbstractSituation,
        signature: Optional[StructureSignature] = None,
    ) -> Tuple[Solver, ...]:
        """Return the applicable Solvers RANKED by applicability confidence for this goal +
        situation (CMP-35 step 1). ``signature`` defaults to one derived from the goal via
        :func:`signature_of` (the axes the goal touches). Only Solvers with positive confidence
        are returned; ties break on ``solver_id`` (ascending) for a total, deterministic order
        (DP-10). The highest-ranked single Solver is the stage-1 candidate."""
        sig = signature if signature is not None else signature_of(goal, situation)
        scored = [(s.confidence(sig), s) for s in self.all_solvers()]
        applicable = [(c, s) for c, s in scored if c > 0.0]
        applicable.sort(key=lambda cs: (-cs[0], cs[1].solver_id()))
        return tuple(s for _c, s in applicable)


def _default_base_solvers() -> List[Solver]:
    """The shipped base catalog (LTM prior, read-only): the navigate Solver plus one per
    foundational independent axis (colour / shape / position / orientation), so the library can
    factor a multi-axis goal by Law E. A generalisation prior, NOT learned; no game literal."""
    return [
        navigate_solver(),
        axis_solver("colour"),
        axis_solver("shape"),
        axis_solver("position"),
        axis_solver("orientation"),
    ]


# ---------------------------------------------------------------- structure signature derive
# The foundational independent axes Law E factors a goal along (the EffectSignature.attribute
# vocabulary, TERM-17). These are axis NAMES (Dimension labels), not game literals.
FOUNDATIONAL_AXES: Tuple[str, ...] = ("colour", "shape", "position", "orientation")


def signature_of(goal: GoalPredicate, situation: AbstractSituation) -> StructureSignature:
    """Derive the :class:`StructureSignature` of a (partial) goal: the set of independent axes it
    touches. The axes come from the goal's relation atoms -- a relation NAME maps to the axis it
    constrains via :data:`_RELATION_AXIS` (e.g. ``inside``/``overlaps`` -> position;
    ``matches`` -> the match axes). Unknown relation names fall back to ``position`` (the default
    navigation axis). Game-literal-free: only relation NAMES (words) and axis labels are read."""
    axes: set = set()
    for atom in goal.atoms():
        name = atom.terms[0] if atom.terms and isinstance(atom.terms[0], str) else None
        rel_name = getattr(atom, "name", None) or name
        for ax in _RELATION_AXIS.get(rel_name, ("position",)):
            axes.add(ax)
    if not axes:
        axes.add("position")
    return StructureSignature(axes=frozenset(axes), kind="navigate")


# Which independent axes a relation NAME constrains (memos: Law E factors by axis). A "matches"
# goal constrains the surface axes (colour + shape + orientation); a spatial relation constrains
# position. These are structural axis labels, never game literals.
_RELATION_AXIS: Dict[str, Tuple[str, ...]] = {
    "inside": ("position",),
    "overlaps": ("position",),
    "adjacent": ("position",),
    "matches": ("colour", "shape", "orientation"),
    "same-colour": ("colour",),
    "same-shape": ("shape",),
    "same-orientation": ("orientation",),
}


# ------------------------------------------------------------------------- ScoreCandidates
@dataclass(frozen=True)
class CandidateScore:
    """The observation post-score of one candidate Solver (CMP-36 output).

    ``solver`` -- the scored Solver. ``plan`` -- the :class:`GamePlan` it produced (carried so
    the adopted Conception holds the exact scored plan -- no re-derivation needed). ``effect`` --
    the :class:`world_model.MoveEffect` of that plan's first move, classified by OBSERVATION
    against the goal-distance (the same signal DetectFutility reads). ``progressed`` -- True iff
    that effect is PROGRESS (goal-distance decreased): only a progressed candidate is adopted.
    ``delta`` -- the goal-distance BEFORE minus AFTER (>0 iff it advanced); the ranking key among
    positive candidates. ``adopted`` -- the post-scoring adoption verdict (== ``progressed``).
    Frozen «value» (DP-10)."""

    solver: Solver
    plan: GamePlan
    effect: str
    progressed: bool
    delta: int = 0
    adopted: bool = False


class ScoreCandidates:
    """Observation post-scoring of candidate Solvers (CMP-36; sequence sheet 2 ScoreCandidates).

    ``score`` runs each candidate's plan on the built-in sim, commits its first move (MPC), and
    classifies the OBSERVED effect via ``classify_move_effect(before, after, goal.distance)`` --
    the SAME observation machinery that drives DetectFutility (sheet 2 C3 connection). A
    candidate is ADOPTED only iff its effect is PROGRESS (it advanced the goal). The LLM-generated
    candidate goes through the IDENTICAL scoring -- it is NOT adopted unscored (TS-14 (b)).

    RHAE-0 (NFR-3): only ``world.predict`` is called; 0 actions reach the env (the sim does not
    score, API-03). Determinism (DP-10): candidates are scored in the given order; the adopted
    set is filtered by the PROGRESS predicate; ranking ties break on ``solver_id``. No game
    literal is read (the score is a goal-distance delta over the abstract AbstractSituation)."""

    def score_one(
        self,
        solver: Solver,
        world: "_Simulator",
        situation: AbstractSituation,
        goal: GoalPredicate,
        moves: Sequence[GameMove],
    ) -> CandidateScore:
        """Score a SINGLE candidate by observation: plan it, commit one move, classify the
        effect, and decide adoption (PROGRESS => adopted). An empty plan (or a move the sim
        leaves inert) scores as non-progress and is NOT adopted."""
        game_plan = solver.solve(world, situation, goal, moves)
        move = game_plan.first()
        if move is None:
            return CandidateScore(solver=solver, plan=game_plan, effect=MoveEffect.INVARIANT,
                                  progressed=False, delta=0, adopted=False)
        after = world.predict(situation, move)
        effect = classify_move_effect(situation, after, goal.distance)
        before_d = _as_int(goal.distance(situation))
        after_d = _as_int(goal.distance(after))
        delta = before_d - after_d
        progressed = (effect == MoveEffect.PROGRESS)
        return CandidateScore(solver=solver, plan=game_plan, effect=effect,
                              progressed=progressed, delta=delta, adopted=progressed)

    def score(
        self,
        candidates: Sequence[Solver],
        world: "_Simulator",
        situation: AbstractSituation,
        goal: GoalPredicate,
        moves: Sequence[GameMove],
    ) -> Tuple[CandidateScore, ...]:
        """Score every candidate by observation (in order). Returns one :class:`CandidateScore`
        each; the caller adopts the scored-positive ones. The LLM candidate is just another
        entry here -- scored identically (TS-14 (b))."""
        return tuple(
            self.score_one(s, world, situation, goal, moves) for s in candidates
        )

    def best_adopted(
        self,
        candidates: Sequence[Solver],
        world: "_Simulator",
        situation: AbstractSituation,
        goal: GoalPredicate,
        moves: Sequence[GameMove],
    ) -> Optional[CandidateScore]:
        """The single best ADOPTED candidate (highest goal-distance ``delta``, ties on
        ``solver_id``), or ``None`` if NONE progressed (nothing scored-positive => adopt nothing,
        and the caller escalates to the next stage)."""
        adopted = [cs for cs in self.score(candidates, world, situation, goal, moves)
                   if cs.adopted]
        if not adopted:
            return None
        adopted.sort(key=lambda cs: (-cs.delta, cs.solver.solver_id()))
        return adopted[0]


# ------------------------------------------------------------------------- Conception
@dataclass(frozen=True)
class Conception:
    """The answer bundle (CMP-16): the selected/composed solution for the current turn.

    Binds the chosen :class:`Solver` (provenance -- which solver produced the plan), the
    :class:`GamePlan` it produced, the :class:`GoalPredicate` it aimed at, and the observation
    ``score`` that justified adoption, plus the escalation ``stage`` that yielded it
    (``"single"`` / ``"composite"`` / ``"llm"``). ``confidence`` ranks competing answers (the
    adopted score's goal-distance delta, clamped) -- distinct from the game score. Frozen «value»
    (DP-10): equality from the (stage, solver_id, plan) content, no builtin ``hash`` of mutable
    state."""

    solver: Solver
    plan: GamePlan
    goal: GoalPredicate
    stage: str = "single"
    score: Optional[CandidateScore] = None
    confidence: float = 0.0

    def solver_id(self) -> str:
        """The provenance Solver's stable id (DP-10)."""
        return self.solver.solver_id()

    def is_empty(self) -> bool:
        """True iff the bundled plan commits no move."""
        return self.plan.is_empty()


# ------------------------------------------------------------------------- SelectSolver
# The three escalation stages (CMP-35; SC-14). Stable string labels for deterministic logging.
STAGE_SINGLE: str = "single"
STAGE_COMPOSITE: str = "composite"
STAGE_LLM: str = "llm"

# A record of one attempted stage (for the test/trace): the stage label, the candidate Solvers it
# offered, the scores ScoreCandidates produced, and whether it was adopted. Lets TS-14 assert the
# ORDER of stages AND that each stage's candidate passed scoring before adoption.
@dataclass(frozen=True)
class StageAttempt:
    """One escalation stage's record (CMP-35 trace): ``stage`` label, the ``candidates`` offered,
    the ``scores`` from ScoreCandidates, the ``adopted`` CandidateScore (``None`` if the stage
    failed scoring and escalation continued), and -- for the composite stage -- the per-axis
    ``axis_effects`` (Law E: each independent axis's effect observed SEPARATELY). Frozen «value»
    (DP-10)."""

    stage: str
    candidates: Tuple[Solver, ...]
    scores: Tuple[CandidateScore, ...]
    adopted: Optional[CandidateScore] = None
    axis_effects: Tuple[AxisEffect, ...] = ()

    def passed_scoring(self) -> bool:
        """True iff this stage adopted a candidate via observation scoring (its effect was
        PROGRESS). A stage NEVER adopts unscored -- ``adopted`` is always one of ``scores``."""
        return self.adopted is not None and self.adopted.adopted


@dataclass(frozen=True)
class Escalation:
    """The full staged-escalation result (CMP-35 output; the TS-14 oracle bundle).

    ``conception`` -- the adopted answer (``None`` iff every stage failed scoring). ``attempts``
    -- the ordered :class:`StageAttempt`s, so the test can assert the stages ran in the order
    single -> composite -> llm AND that the adopted candidate at each reached-stage passed
    ScoreCandidates' observation scoring BEFORE adoption (an unscored candidate is never adopted).
    Frozen «value» (DP-10)."""

    conception: Optional[Conception]
    attempts: Tuple[StageAttempt, ...]

    def stages(self) -> Tuple[str, ...]:
        """The stage labels in the order they were attempted (single[, composite[, llm]])."""
        return tuple(a.stage for a in self.attempts)

    def adopted_stage(self) -> Optional[str]:
        """The stage whose candidate was adopted (``None`` iff none adopted)."""
        return self.conception.stage if self.conception is not None else None


class SelectSolver:
    """Staged solver escalation with observation post-scoring (CMP-35; SC-14 / FR-C-11).

    :meth:`escalate` runs, IN ORDER, until a stage's candidate is ADOPTED by ScoreCandidates:
      1. **single**    -- the highest-ranked single baked Solver
                          (``library.select(goal, situation)[0]``);
      2. **composite** -- a Law-E axis-factored composite: each independent axis of the goal
                          (``signature.axes``) routes to its OWN axis Solver, composed via
                          ``parts``; the per-axis effects are observed SEPARATELY
                          (:func:`observe_axis_effects`);
      3. **llm**       -- the built-in LOCAL/OFFLINE generator proposes a small program, which
                          is registered to ``library.overlay`` and then scored.

    CRITICAL (TS-14 (b)): EVERY stage's candidate passes through :class:`ScoreCandidates`'
    observation scoring BEFORE adoption -- including the LLM-generated one (scored identically;
    NOT adopted unscored). A stage that scores no progress is recorded and escalation continues.

    RHAE-0 (NFR-3): only ``world.predict`` is used. NFR-1: the generator is local/offline by the
    :class:`ConstrainedGenerator` contract. Determinism (DP-10): stages and candidates run in a
    fixed order; adoption is the PROGRESS predicate; no RNG; no builtin ``hash`` for identity.
    """

    def __init__(
        self,
        library: SolverLibrary,
        scorer: Optional[ScoreCandidates] = None,
        generator: Optional[ConstrainedGenerator] = None,
    ) -> None:
        self.library = library
        self.scorer = scorer if scorer is not None else ScoreCandidates()
        self.generator = generator

    def escalate(
        self,
        world: "_Simulator",
        situation: AbstractSituation,
        goal: GoalPredicate,
        moves: Sequence[GameMove],
        axis_distance: Optional[Mapping[str, Heuristic]] = None,
    ) -> Escalation:
        """Run the staged escalation and return the :class:`Escalation` (adopted Conception +
        the ordered stage attempts). ``axis_distance`` optionally injects a per-axis goal
        distance so the composite stage can observe each axis's effect SEPARATELY (Law E); when
        omitted, the composite is still built and scored on the whole-goal distance.

        Stops at the FIRST stage whose candidate ScoreCandidates adopts (PROGRESS). If no stage
        progresses, ``conception`` is ``None`` and every attempted stage is recorded."""
        sig = signature_of(goal, situation)
        attempts: List[StageAttempt] = []

        # -------- stage 1: single baked Solver --------
        ranked = self.library.select(goal, situation, sig)
        single_candidates: Tuple[Solver, ...] = (ranked[0],) if ranked else ()
        attempt1 = self._score_stage(STAGE_SINGLE, single_candidates, world, situation, goal, moves)
        attempts.append(attempt1)
        if attempt1.passed_scoring():
            return self._finish(attempt1, attempts, goal)

        # -------- stage 2: composite via axis decomposition (Law E) --------
        composite = self._build_composite(sig)
        composite_candidates: Tuple[Solver, ...] = (composite,) if composite is not None else ()
        # Per-axis SEPARATE observation (Law E evidence) for the caller/test to assert on.
        axis_effects: Tuple[AxisEffect, ...] = ()
        if composite is not None and axis_distance is not None:
            axis_effects = observe_axis_effects(composite, world, situation, axis_distance)
        attempt2 = self._score_stage(STAGE_COMPOSITE, composite_candidates, world, situation,
                                     goal, moves, axis_effects=axis_effects)
        attempts.append(attempt2)
        if attempt2.passed_scoring():
            return self._finish(attempt2, attempts, goal)

        # -------- stage 3: built-in LLM generates a program, registered to overlay --------
        if self.generator is not None:
            llm = llm_generated_solver(self.generator, sig, situation, goal, moves)
            self.library.add(llm)                       # register to overlay (just-generated)
            attempt3 = self._score_stage(STAGE_LLM, (llm,), world, situation, goal, moves)
            attempts.append(attempt3)
            if attempt3.passed_scoring():
                return self._finish(attempt3, attempts, goal)

        return Escalation(conception=None, attempts=tuple(attempts))

    # ---- stage scoring (the SAME observation scoring for every stage, incl. LLM) ----
    def _score_stage(
        self,
        stage: str,
        candidates: Tuple[Solver, ...],
        world: "_Simulator",
        situation: AbstractSituation,
        goal: GoalPredicate,
        moves: Sequence[GameMove],
        axis_effects: Tuple[AxisEffect, ...] = (),
    ) -> StageAttempt:
        """Score ``candidates`` by observation and build the :class:`StageAttempt`. Adoption is
        ONLY via the score (PROGRESS) -- a candidate is never adopted unscored (TS-14 (b))."""
        scores = self.scorer.score(candidates, world, situation, goal, moves)
        adopted = None
        positives = [cs for cs in scores if cs.adopted]
        if positives:
            positives.sort(key=lambda cs: (-cs.delta, cs.solver.solver_id()))
            adopted = positives[0]
        return StageAttempt(stage=stage, candidates=candidates, scores=scores,
                            adopted=adopted, axis_effects=axis_effects)

    def _finish(self, attempt: StageAttempt, attempts: List[StageAttempt],
                goal: GoalPredicate) -> Escalation:
        """Build the adopted :class:`Conception` from a passing ``attempt`` and close out. The
        bundled plan is the EXACT one ScoreCandidates already produced (carried on the score), so
        no re-derivation is needed; ``confidence`` is the adopted goal-distance delta (clamped)."""
        cs = attempt.adopted
        conception = Conception(
            solver=cs.solver,
            plan=cs.plan,
            goal=goal,
            stage=attempt.stage,
            score=cs,
            confidence=max(0.0, min(1.0, float(cs.delta))),
        )
        return Escalation(conception=conception, attempts=tuple(attempts))

    # ---- composite construction (Law E) ----
    def _build_composite(self, signature: StructureSignature) -> Optional[Solver]:
        """Build a Law-E composite Solver: ONE distinct axis Solver per independent axis in
        ``signature.axes`` (each axis gets its own Solver -- the SC-14 decomposition), composed
        via ``parts``. Returns ``None`` if the goal has fewer than 2 axes (nothing to compose --
        a single Solver already covers it). The axis Solvers are drawn from the library's base
        (each ``axis-<name>``); a missing one is built on the fly."""
        axes = sorted(signature.axes)              # deterministic axis order (DP-10)
        if len(axes) < 2:
            return None
        by_axis = {s.axis: s for s in self.library.all_solvers() if s.axis}
        parts: List[Solver] = []
        for ax in axes:
            parts.append(by_axis.get(ax) or axis_solver(ax))
        return Solver(
            kind="composite",
            applicability=lambda _sig: 1.0,
            parts=tuple(parts),
            axis="",
            origin="composite",
        )
