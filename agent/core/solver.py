"""Solver selection / ranking / staged escalation — the SelectSolver cluster.

The descriptor-only realization of the domain ``Solver`` dispatch (stage 6-3.7).
It ports the plug-architecture ``SolverContext`` (Port) / ``SolverLibrary`` /
``SelectSolver`` / ``ScoreCandidates`` cluster against the live data model
(``agent.core.model.Solver`` «value», seeded from ``agent/assets/solvers.tsv``).

SCOPE — descriptor-only (★★). :class:`SelectSolver` chooses WHICH typed solver
family applies and returns a :class:`SolverPlan` DESCRIPTOR (chosen solver(s) +
rank trace + why). It does NOT execute the 47 concrete backend algorithms
(z3 / ortools / networkx / pymunk / ...): those Adapters live behind the three
``backend`` ports (SearchHeuristic / ConstrainedGenerator / Simulator) and are a
later step. It also does NOT do navigate A* / ConcreteSituation grounding — the
selector runs on the abstract :class:`agent.core.situation.AbstractSituation`.
Identity note: the ``shortest_path`` Solver IS the ``navigate`` family (domain
``navigate`` Solver, position axis); when realized its concrete backend runs A*
on a ConcreteSituation while every other family stays abstract.

The selection model (plug-architecture / domain v031 Solver entity):
  * ``StructuralSignature`` — the recognition key: the Goal's logical form +
    derived ``goal_kind`` (FK), and the World's transition structure (a small
    deterministic label set). :func:`signature_of` builds it.
  * ``applicability`` — the signature-match score in [0, 1] = goal_kind FK
    exact-match x world_signature keyword-overlap (signature v1).
  * ``confidence`` — the observation-updated posterior :class:`ScoreCandidates`
    keeps (same shape as ``InteractionRule.confidence`` / ``affordance_evidence``
    running fractions — deterministic, no RNG, no builtin ``hash()``).

Determinism (DP-10): every ordering is TOTAL with an explicit id-ascending
tie-break; the escalation synthesize loop is bounded by a hard attempt cap; no
builtin ``hash()`` / RNG anywhere. NULL FALLBACK (DP-20 / NFR-1):
:meth:`SelectSolver.solve` ALWAYS returns a bounded :class:`SolverPlan` — with a
:class:`agent.core.llm.NullGenerator` it falls back to the highest-applicability
prior solver, never None and never blocking.

Naming matches the plug-architecture vocabulary exactly (``SolverContext`` Port,
``StructuralSignature`` + ``signature_of``, ``SolverPlan`` result descriptor,
``applicability`` / ``confidence``). ``SolverPlan`` does NOT collide with the
domain ``GamePlan`` (a different, retired name).

Mirrors the module style of ``agent.core.model`` / ``agent.core.world_model``
(frozen ``@dataclass`` for «value», plain ``@dataclass`` for runtime entities,
rich docstrings, controlled vocabularies as module-level maps).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Tuple,
    Union,
)

from agent.core.llm import ConstrainedGenerator, NullGenerator, consult
from agent.core.model import Goal, GoalPattern, Solver
from agent.core.world_model import _TERMINAL_FEATURES

# --------------------------------------------------------------------------- #
# Verification-horizon cost map (DP-10 ranking key). Free-text horizon ->
# numeric cost; cheaper-to-verify solvers rank first (RHAE: fewer wasted moves
# before the sim can disprove the plan). The map is the SSOT for the order.
# --------------------------------------------------------------------------- #

# The literal free-text horizon labels in solvers.tsv -> numeric cost.
_HORIZON_COST: Dict[str, int] = {
    "~0": 0,
    "low": 1,
    "low-med": 2,
    "med": 3,
    "med-high": 4,
    "high": 5,
}

# Unknown / unmappable horizon (defensive): treat as the most expensive (5) so an
# un-recognised row never sorts ahead of a known-cheap one.
_HORIZON_UNKNOWN = 5

# The composite row's horizon is the free text "max(parts)": its cost is derived
# as the max over the row's parts' costs at runtime (a composite is only as cheap
# as its most expensive sub-solver). This is the literal that triggers that path.
_HORIZON_MAX_PARTS = "max(parts)"


# --------------------------------------------------------------------------- #
# SolverContext (Port) — the read-only bundle a selection runs against.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SolverContext:
    """The selection Port: everything :class:`SelectSolver` reads — a «value».

    Frozen + by-value: a selection is a pure function of its context (DP-10).
    Fields:
      * ``situation`` — the abstract board (:class:`AbstractSituation`) the solver
        will run on (the selector is abstract-only; navigate/Concrete is deferred).
      * ``world``     — the learned :class:`agent.core.world_model.WorldModel`
        (its transition structure is the world half of the signature).
      * ``goal``      — the :class:`agent.core.model.Goal` (its logical form +
        derived goal_kind is the goal half of the signature).
      * ``moves``     — the available move ids (the action alphabet; carried for
        the concrete backend, unused by descriptor-only selection).
      * ``consult``   — the :class:`ConstrainedGenerator` propose-only seam for
        LLM synthesis (defaults to :class:`NullGenerator` = classical-only). The
        master is classical and disposes; the LLM only proposes (DP-20).
    """

    situation: Any = None
    world: Any = None
    goal: Optional[Goal] = None
    moves: Tuple[int, ...] = ()
    consult: ConstrainedGenerator = field(default_factory=NullGenerator)


# --------------------------------------------------------------------------- #
# StructuralSignature (value) — the recognition key (Goal logical form x World
# transition structure). signature v1: goal_kind FK + world keyword set.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StructuralSignature:
    """The structure key a Solver's ``applicability`` matches against — a «value».

    Two halves (plug-architecture / domain v031 Solver.applicability):
      * GOAL half — ``goal_kind`` (the FK into ``goal_kinds.tsv``, derived from the
        Goal via :class:`GoalPattern` hints; ``None`` if no kind can be derived)
        plus ``goal_form`` (the deterministic ``goal_canonical`` logical-form key,
        carried for the trace / future graded matching).
      * WORLD half — ``world_keywords`` = a small deterministic label set of the
        World's transition structure (e.g. ``reversible`` / ``bounded-branch`` /
        ``scrolling`` / ``stochastic`` / ``adversarial``), the overlap target for
        the ``world_signature`` half of ``applicability``.

    Frozen + a frozenset keyword field => deterministic value-equality (DP-10:
    no RNG, no builtin ``hash()`` of mutable state)."""

    goal_kind: Optional[str] = None
    goal_form: Union[Tuple, str, None] = None
    world_keywords: FrozenSet[str] = field(default_factory=frozenset)


# Stop-words dropped from a Solver.world_signature before keyword overlap — pure
# connective / filler tokens that carry no structural discrimination. Keeping the
# set tiny and explicit keeps the overlap honest (DP-10 deterministic).
_WORLD_STOP_WORDS: FrozenSet[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "be", "of", "on", "in", "to", "as",
        "and", "or", "with", "you", "your", "it", "its", "that", "this",
        "+", "/", "(", ")",
    }
)


def _world_tokens(text: str) -> FrozenSet[str]:
    """Tokenize a free-text ``world_signature`` into a deterministic keyword set.

    Lower-cased, split on non-alphanumeric runs (so ``bounded-branch`` ->
    {``bounded``, ``branch``}), with stop-words and empty tokens dropped. Pure /
    deterministic (no RNG): the same string always yields the same frozenset."""
    tokens: List[str] = []
    cur: List[str] = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                tokens.append("".join(cur))
                cur = []
    if cur:
        tokens.append("".join(cur))
    return frozenset(t for t in tokens if t and t not in _WORLD_STOP_WORDS)


def _world_keywords(world: Any) -> FrozenSet[str]:
    """Derive the World transition-structure label set from a WorldModel.

    A small, deterministic set of structural labels read off the model's CHEAPLY
    AVAILABLE attributes (no game literals — NFR-6):
      * ``scrolling``  — the frame is a window onto a larger world
        (``WorldModel.is_scrolling``);
      * ``bounded-branch`` — a finite move alphabet drives a bounded branching
        factor (always true for ARC-AGI-3's fixed action set; a safe default);
      * ``reversible`` — every learned :class:`InteractionRule` is non-terminal
        (no loss trigger fires) — a CHEAPLY-available reversibility proxy; absent
        when a rule can drive a gauge terminal (irreversible / hazardous). The
        terminal-feature set is :data:`agent.core.world_model._TERMINAL_FEATURES`
        (imported, not duplicated, so the proxy cannot drift from the SSOT).
    The labels overlap the vocabulary the ``world_signature`` column uses
    (``reversible`` / ``bounded-branch`` / ``scrolling``), so a signature built
    here scores against the prior solvers without a translation table. Defensive:
    a ``None`` / attribute-less world yields ``{bounded-branch}`` (the safe
    default), never raises."""
    labels = {"bounded-branch"}
    if getattr(world, "is_scrolling", False):
        labels.add("scrolling")
    rules = getattr(world, "rules", None)
    if rules is not None:
        # Reversible proxy: no learned rule carries a terminal (loss) effect.
        # SSOT: the terminal-feature set lives in world_model (a rule effect on a
        # terminal axis can drive a gauge to OVER) — import it so this cheap proxy
        # can never silently drift from the world model's own definition.
        reversible = all(
            not any(sig.feature in _TERMINAL_FEATURES for sig in rule.effect)
            for rule in rules
        )
        if reversible:
            labels.add("reversible")
    return frozenset(labels)


def _goal_kind_of(goal: Optional[Goal], assets: Any) -> Optional[str]:
    """Derive the Goal's ``goal_kind`` FK via :class:`GoalPattern` hints (runtime
    bridge). ``None`` if no kind can be derived (the caller then falls back to
    world-only matching — it does NOT crash).

    Bridge: a Goal carries a ``predicate`` Relation, not a goal_kind. The
    GoalPatternLibrary (``assets.goal_patterns``) maps a goal logical-form template
    -> ``goal_kind`` + ``solver_kinds``. v1 matches by the goal's canonical
    logical form against each active pattern's ``predicate_tree`` canonical form;
    the first deterministic (id-ascending) match donates its ``goal_kind``. If no
    pattern matches (or there is no goal / no library) the result is ``None``.
    Deterministic: patterns are visited in id-ascending order."""
    if goal is None:
        return None
    patterns: Tuple[GoalPattern, ...] = tuple(getattr(assets, "goal_patterns", ()))
    if not patterns:
        return None
    # Local import keeps the goal interpreter dependency lazy (avoids importing
    # the registry-backed evaluator at module import time).
    from agent.core.goal import canonical, goal_canonical

    try:
        goal_form = goal_canonical(goal)
    except Exception:  # noqa: BLE001 - a malformed predicate must not crash select
        return None
    for pattern in sorted(patterns, key=lambda p: p.id):
        tree = pattern.predicate_tree
        if tree is None:
            continue
        if canonical(tree) == goal_form:
            return pattern.goal_kind
    return None


def signature_of(
    goal: Optional[Goal], world: Any, assets: Any
) -> StructuralSignature:
    """Build the :class:`StructuralSignature` for a ``(goal, world)`` pair.

    The GOAL half: the goal's canonical logical form (``goal_canonical``) +
    ``goal_kind`` derived via :func:`_goal_kind_of` (GoalPattern bridge; ``None``
    if not derivable -> world-only matching downstream). The WORLD half:
    :func:`_world_keywords` over the WorldModel's cheaply-available transition
    structure. Pure + deterministic (DP-10)."""
    goal_form: Union[Tuple, str, None] = None
    if goal is not None:
        from agent.core.goal import goal_canonical

        try:
            goal_form = goal_canonical(goal)
        except Exception:  # noqa: BLE001 - tolerate a malformed predicate
            goal_form = None
    return StructuralSignature(
        goal_kind=_goal_kind_of(goal, assets),
        goal_form=goal_form,
        world_keywords=_world_keywords(world),
    )


# --------------------------------------------------------------------------- #
# applicability (signature-match score in [0, 1]).
# --------------------------------------------------------------------------- #

def applicability(solver: Solver, signature: StructuralSignature) -> float:
    """The signature-match score in [0, 1] (signature v1).

    Two factors, AVERAGED so the score stays in [0, 1]:
      * GOAL factor — goal_kind FK EXACT match. ``1.0`` iff the signature's
        ``goal_kind`` is one of the solver's ``goal_kinds`` (the FK list). If the
        signature has NO derivable goal_kind, or the solver is POLYMORPHIC (empty
        ``goal_kinds`` — ``composite`` / ``nrpa_adaptive_playout``), the goal
        factor is NEUTRAL (1.0): a goal-agnostic solver is not penalised on the
        goal axis (it applies to any goal_kind), and a kind-less signature falls
        back to world-only discrimination.
      * WORLD factor — keyword overlap = ``|sig.world_keywords ∩
        solver_world_tokens| / |sig.world_keywords|`` (the fraction of the
        signature's structural labels the solver's ``world_signature`` covers).
        Empty signature keywords => neutral 1.0 (nothing to discriminate on).

    The result = ``0.5 * goal_factor + 0.5 * world_factor`` in [0, 1].
    Deterministic / pure (DP-10). Solver-kind exact-match (not fuzzy) keeps the
    FK semantics; world overlap is graded so a partial structural match still
    ranks.

    Combine rule = WEIGHTED AVERAGE, not a product (signature v1, deliberate).
    The world signature is intentionally COARSE — just a few structural keywords
    (``reversible`` / ``bounded-branch`` / ``scrolling`` / ...). A product would
    multiply the two factors, so any solver with even partial world overlap
    (world_factor < 1) would be dragged toward 0 and, where the world half is 0,
    zeroed outright — forcing nearly everything to look inapplicable and escalate,
    which defeats the whole point of ranking. The average instead keeps a solver
    with a STRONG goal-kind match applicable even on partial world overlap (a
    perfect goal match alone floors the score at 0.5), so the ranking still
    discriminates. (A sharper, calibrated combine is a later signature version.)"""
    # GOAL factor.
    if not solver.goal_kinds or signature.goal_kind is None:
        goal_factor = 1.0  # polymorphic solver or kind-less signature => neutral
    else:
        goal_factor = 1.0 if signature.goal_kind in solver.goal_kinds else 0.0

    # WORLD factor.
    sig_kw = signature.world_keywords
    if not sig_kw:
        world_factor = 1.0
    else:
        solver_kw = _world_tokens(solver.world_signature)
        overlap = len(sig_kw & solver_kw)
        world_factor = overlap / len(sig_kw)

    return 0.5 * goal_factor + 0.5 * world_factor


# --------------------------------------------------------------------------- #
# Horizon cost + ranking key (DP-10 total order).
# --------------------------------------------------------------------------- #

def horizon_cost(
    solver: Solver,
    library: Optional["SolverLibrary"] = None,
    _visited: Optional[FrozenSet[str]] = None,
) -> int:
    """The numeric verification-horizon cost of ``solver`` (cheaper = sooner a
    bad plan is disproved; lower sorts first).

    Maps the free-text ``verification_horizon`` via :data:`_HORIZON_COST`. The
    composite literal ``"max(parts)"`` is DERIVED as the max over the row's
    parts' costs (a composite is only as cheap as its most expensive part); an
    empty-parts composite (the runtime-bound default) costs :data:`_HORIZON_UNKNOWN`.
    An unmapped horizon also costs :data:`_HORIZON_UNKNOWN` (defensive). Resolving
    parts needs the :class:`SolverLibrary` (to look the parts up by id); without
    one a ``max(parts)`` row falls back to :data:`_HORIZON_UNKNOWN`.

    Cycle-safe (DP-20: ``solve`` never raises / is always bounded). The recursion
    over ``parts`` threads a ``_visited`` set of solver ids; a self-referential or
    mutually-referential composite (an id re-encountered on its own resolution
    chain) is treated as the :data:`_HORIZON_UNKNOWN` ceiling rather than
    recursing forever. ``_visited`` is internal — callers pass only ``solver`` /
    ``library``."""
    horizon = solver.verification_horizon.strip()
    if horizon == _HORIZON_MAX_PARTS:
        if library is None or not solver.parts:
            return _HORIZON_UNKNOWN
        # Cycle guard: if this solver is already on the resolution chain we are in
        # a self-/mutual-reference loop -> the unknown ceiling (never recurse).
        if _visited is not None and solver.id in _visited:
            return _HORIZON_UNKNOWN
        chain = (_visited or frozenset()) | {solver.id}
        part_costs = [
            horizon_cost(part, library, chain)
            for part in (library.by_id(pid) for pid in solver.parts)
            if part is not None
        ]
        return max(part_costs) if part_costs else _HORIZON_UNKNOWN
    return _HORIZON_COST.get(horizon, _HORIZON_UNKNOWN)


def _rank_key(
    solver: Solver, signature: StructuralSignature, library: "SolverLibrary"
) -> Tuple[float, int, str]:
    """The total-order ranking key (DP-10): applicability DESC, horizon_cost ASC,
    id ASC. Negating applicability turns the DESC primary into an ASC sort so the
    whole key sorts ascending; ``id`` is the content-stable final tie-break (every
    solver id is unique, so the order is total)."""
    return (
        -applicability(solver, signature),
        horizon_cost(solver, library),
        solver.id,
    )


# --------------------------------------------------------------------------- #
# SolverLibrary (runtime entity) — prior catalog + synthesized additions.
# --------------------------------------------------------------------------- #

@dataclass
class SolverLibrary:
    """The two-layer solver dictionary (domain v031 ``SolverLibrary`` entity).

    ``prior`` = the read-only LTM catalog (``LoadedAssets.solvers`` = the
    ``solvers.tsv`` row «value»s, seeded once at load). ``synthesized`` = the
    per-run append log of solvers synthesized DURING this run (e.g. an LLM-proposed
    family added by :class:`SelectSolver`). Mutable on the synthesized layer only;
    the prior tuple is never mutated. Lookups visit prior THEN synthesized in
    id-ascending order so :meth:`all` / :meth:`by_id` are deterministic (DP-10).

    Mirrors the Lexicon's old two-layer shape, but unlike Lexicon (which folded
    base/overlay into ``Word.origin`` in v14) the two layers live HERE — the
    prior/synthesized split is the SolverLibrary's own structure."""

    prior: Tuple[Solver, ...] = ()
    synthesized: List[Solver] = field(default_factory=list)

    @classmethod
    def from_assets(cls, assets: Any) -> "SolverLibrary":
        """Build a library from a ``LoadedAssets`` (its ``.solvers`` become the
        prior layer; synthesized starts empty)."""
        return cls(prior=tuple(getattr(assets, "solvers", ())))

    def all(self) -> Tuple[Solver, ...]:
        """Every solver (prior + synthesized), deterministic id-ascending order."""
        return tuple(sorted((*self.prior, *self.synthesized), key=lambda s: s.id))

    def by_id(self, solver_id: str) -> Optional[Solver]:
        """The solver with this id (prior wins over a same-id synthesized; ``None``
        if absent). Deterministic."""
        for s in self.prior:
            if s.id == solver_id:
                return s
        for s in self.synthesized:
            if s.id == solver_id:
                return s
        return None

    def add_synthesized(self, solver: Solver) -> Solver:
        """Append a synthesized :class:`Solver` to the run-local layer (a no-op if
        a solver with the same id is already present in either layer). Returns the
        solver (the existing one if a duplicate id, else the added one)."""
        existing = self.by_id(solver.id)
        if existing is not None:
            return existing
        self.synthesized.append(solver)
        return solver


# --------------------------------------------------------------------------- #
# SolverPlan (result descriptor) — the chosen solver(s) + rank trace + why.
# --------------------------------------------------------------------------- #

# How a SolverPlan was reached (the escalation tier that produced it). Strings
# (not an Enum) for deterministic logging / equality (DP-10).
PLAN_SOURCES = frozenset(
    {"single", "composite", "synthesized", "null-fallback"}
)


@dataclass(frozen=True)
class SolverPlan:
    """The DESCRIPTOR :meth:`SelectSolver.solve` returns — a «value».

    NOT an execution engine: it names WHICH typed solver family (or composed
    parts) was chosen and WHY, leaving execution to the (deferred) backend
    Adapters. Fields:
      * ``solvers``    — the chosen :class:`Solver`(s): one for a single pick, the
        ordered parts for a composite, the synthesized/fallback row otherwise.
      * ``signature``  — the :class:`StructuralSignature` selection ran against.
      * ``source``     — the escalation tier that produced the plan (a
        :data:`PLAN_SOURCES` label: single / composite / synthesized /
        null-fallback).
      * ``rank_trace`` — the ranked ``(solver_id, applicability, horizon_cost)``
        tuples that were considered (the audit trail of the order).
      * ``why``        — a short human-readable rationale (for structured logs).

    Frozen + tuple fields => deterministic value-equality (DP-10). ``SolverPlan``
    is a distinct name from the domain ``GamePlan`` (no collision)."""

    solvers: Tuple[Solver, ...] = ()
    signature: Optional[StructuralSignature] = None
    source: str = "single"
    rank_trace: Tuple[Tuple[str, float, int], ...] = ()
    why: str = ""

    def __post_init__(self) -> None:
        if self.source not in PLAN_SOURCES:
            raise ValueError(
                f"SolverPlan.source {self.source!r} not in {sorted(PLAN_SOURCES)}"
            )

    @property
    def chosen(self) -> Optional[Solver]:
        """The primary chosen solver (the first of :attr:`solvers`), or ``None``
        for the degenerate empty plan."""
        return self.solvers[0] if self.solvers else None


# --------------------------------------------------------------------------- #
# SelectSolver (use case) — rank + staged escalation -> SolverPlan.
# --------------------------------------------------------------------------- #

# Hard attempt cap for the synthesize-escalation loop (DP-10: the loop must be
# bounded — never spin on a generator that keeps proposing unusable rows).
_MAX_SYNTHESIZE_ATTEMPTS = 3

# The JSON schema the LLM synthesize proposal is constrained to (propose-only).
# Mirrors the solvers.tsv columns the descriptor needs; the master disposes.
_SYNTHESIZE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "category": {"type": "string"},
        "world_signature": {"type": "string"},
        "verification_horizon": {"type": "string"},
        "backend": {"type": "string"},
        "algorithm": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["id", "backend", "algorithm"],
}

_SYNTHESIZE_BACKENDS = frozenset(
    {"SearchHeuristic", "ConstrainedGenerator", "Simulator"}
)


def _parse_synthesized(raw: Dict[str, Any]) -> Solver:
    """Parse an LLM synthesize proposal into a :class:`Solver` «value».

    Raises ``KeyError`` / ``ValueError`` / ``TypeError`` on a malformed proposal
    (``consult`` swallows those into a None fallback). The backend is validated
    against the 3 ports; goal_kinds / parts default empty (a synthesized family is
    runtime-bound). The id is prefixed ``syn-`` so a synthesized row never shadows
    a prior catalog id."""
    sid = str(raw["id"]).strip()
    if not sid:
        raise ValueError("synthesized solver: empty id")
    backend = str(raw["backend"]).strip()
    if backend not in _SYNTHESIZE_BACKENDS:
        raise ValueError(f"synthesized solver: bad backend {backend!r}")
    algorithm = str(raw["algorithm"]).strip()
    if not algorithm:
        raise ValueError("synthesized solver: empty algorithm")
    return Solver(
        category=str(raw.get("category", "synthesized")).strip() or "synthesized",
        id=sid if sid.startswith("syn-") else f"syn-{sid}",
        goal_kinds=(),
        world_signature=str(raw.get("world_signature", "")).strip(),
        verification_horizon=str(raw.get("verification_horizon", "high")).strip()
        or "high",
        backend=backend,
        parts=(),
        algorithm=algorithm,
        description=str(raw.get("description", "")).strip(),
        remark="synthesized at runtime (SelectSolver + ConstrainedGenerator)",
    )


@dataclass
class SelectSolver:
    """Solve = pick the applicable typed solver family for a context, descriptor-
    only (domain v031 ``SelectSolver`` use case).

    :meth:`rank` orders the library by the ranking key (applicability DESC,
    horizon_cost ASC, id ASC). :meth:`solve` runs the staged escalation:
      1. SINGLE      — the top-ranked applicable solver.
      2. COMPOSITE   — if the top pick is a composite (Law E axis-factoring) with
         bound parts, expand it into its ordered parts.
      3. SYNTHESIZE  — if nothing applies (all applicability == 0), consult the
         ConstrainedGenerator (bounded by :data:`_MAX_SYNTHESIZE_ATTEMPTS`) for a
         proposed family, added to the library's synthesized layer.
      4. NULL FALLBACK — if synthesis declines (NullGenerator / exhausted), return
         the HIGHEST-APPLICABILITY prior solver as a bounded plan (DP-20 / NFR-1:
         always a plan, never None, never blocking).

    Holds the :class:`SolverLibrary` and the ``assets`` (for the goal_kind
    bridge). Deterministic: ranking is a total order; the synthesize loop is
    bounded."""

    library: SolverLibrary
    assets: Any = None

    def rank(self, ctx: SolverContext) -> Tuple[Solver, ...]:
        """The library ranked best-first for ``ctx`` (total order, DP-10)."""
        signature = self._signature(ctx)
        return tuple(
            sorted(
                self.library.all(),
                key=lambda s: _rank_key(s, signature, self.library),
            )
        )

    def solve(self, ctx: SolverContext) -> SolverPlan:
        """Run the staged escalation and return a bounded :class:`SolverPlan`.

        ALWAYS returns a plan (DP-20): single -> composite -> synthesize ->
        null-fallback. Never None, never blocking."""
        signature = self._signature(ctx)
        ranked = self.rank(ctx)
        trace = self._rank_trace(ranked, signature)

        top = ranked[0] if ranked else None
        top_applicability = (
            applicability(top, signature) if top is not None else 0.0
        )

        # 1/2. Something applies (> 0) — single, or composite expansion.
        if top is not None and top_applicability > 0.0:
            if self._is_composite(top) and top.parts:
                parts = self._expand_parts(top)
                return SolverPlan(
                    solvers=parts,
                    signature=signature,
                    source="composite",
                    rank_trace=trace,
                    why=(
                        f"composite {top.id!r} expanded into "
                        f"{[p.id for p in parts]} (Law E axis-factoring); "
                        f"applicability={top_applicability:.3f}"
                    ),
                )
            return SolverPlan(
                solvers=(top,),
                signature=signature,
                source="single",
                rank_trace=trace,
                why=(
                    f"single best {top.id!r}; applicability="
                    f"{top_applicability:.3f}, horizon_cost="
                    f"{horizon_cost(top, self.library)}"
                ),
            )

        # 3. Nothing applies — try bounded LLM synthesis.
        synthesized = self._synthesize(ctx, signature)
        if synthesized is not None:
            return SolverPlan(
                solvers=(synthesized,),
                signature=signature,
                source="synthesized",
                rank_trace=trace,
                why=f"synthesized {synthesized.id!r} (no prior applied)",
            )

        # 4. NULL FALLBACK — highest-applicability prior solver (bounded; DP-20).
        return self._null_fallback(ranked, signature, trace)

    # -- internals --------------------------------------------------------- #

    def _signature(self, ctx: SolverContext) -> StructuralSignature:
        return signature_of(ctx.goal, ctx.world, self.assets)

    def _rank_trace(
        self, ranked: Tuple[Solver, ...], signature: StructuralSignature
    ) -> Tuple[Tuple[str, float, int], ...]:
        """The audit trail: ``(id, applicability, horizon_cost)`` per ranked solver.

        The reported cost is the SAME RESOLVED :func:`horizon_cost` the ranking key
        sorts by (passing ``self.library`` so a composite's ``max(parts)`` horizon
        resolves), so the trace can never disagree with the actual order — a raw
        map lookup would report the literal ``max(parts)`` as the unknown ceiling
        while the sort used the derived part cost."""
        return tuple(
            (
                s.id,
                applicability(s, signature),
                horizon_cost(s, self.library),
            )
            for s in ranked
        )

    @staticmethod
    def _is_composite(solver: Solver) -> bool:
        """A composite / Law E axis-factoring solver (category ``meta`` whose
        horizon is ``max(parts)``)."""
        return solver.verification_horizon.strip() == _HORIZON_MAX_PARTS

    def _expand_parts(self, composite: Solver) -> Tuple[Solver, ...]:
        """The composite's bound parts as ordered :class:`Solver`s (id-ascending,
        unresolved ids dropped). Empty when no part resolves."""
        parts = [self.library.by_id(pid) for pid in composite.parts]
        resolved = [p for p in parts if p is not None]
        return tuple(sorted(resolved, key=lambda s: s.id))

    def _synthesize(
        self, ctx: SolverContext, signature: StructuralSignature
    ) -> Optional[Solver]:
        """Bounded LLM synthesis: consult the generator up to
        :data:`_MAX_SYNTHESIZE_ATTEMPTS` times for a usable proposed family; add
        the first one to the library's synthesized layer and return it. ``None``
        if every attempt declines / fails to parse (the classical fallback path).
        Deterministic + bounded (DP-10): a fixed attempt cap, no RNG."""
        briefing = self._briefing(signature)
        for _ in range(_MAX_SYNTHESIZE_ATTEMPTS):
            proposed = consult(
                ctx.consult,
                briefing,
                _SYNTHESIZE_SCHEMA,
                _parse_synthesized,
            )
            if proposed is not None:
                return self.library.add_synthesized(proposed)
        return None

    @staticmethod
    def _briefing(signature: StructuralSignature) -> str:
        """The propose-only briefing for the synthesize seam (re-grounded each
        call; no memory). Names the signature so the generator proposes a family
        keyed to it."""
        return (
            "Propose ONE typed solver family for a goal/world with "
            f"goal_kind={signature.goal_kind!r} and world structure "
            f"{sorted(signature.world_keywords)}. Return id, backend "
            "(SearchHeuristic|ConstrainedGenerator|Simulator), and algorithm."
        )

    def _null_fallback(
        self,
        ranked: Tuple[Solver, ...],
        signature: StructuralSignature,
        trace: Tuple[Tuple[str, float, int], ...],
    ) -> SolverPlan:
        """The bounded DP-20 fallback: the highest-applicability prior solver
        (the rank head, which is total-ordered even at applicability 0). Returns a
        plan with an EMPTY ``solvers`` only if the library itself is empty (the
        truly degenerate case) — otherwise always names a concrete solver."""
        if not ranked:
            return SolverPlan(
                solvers=(),
                signature=signature,
                source="null-fallback",
                rank_trace=trace,
                why="empty library: no solver available (degenerate)",
            )
        head = ranked[0]
        return SolverPlan(
            solvers=(head,),
            signature=signature,
            source="null-fallback",
            rank_trace=trace,
            why=(
                f"null-fallback to highest-applicability prior {head.id!r} "
                f"(applicability={applicability(head, signature):.3f}); "
                "no LLM backend / synthesis declined (DP-20 bounded plan)"
            ),
        )


# --------------------------------------------------------------------------- #
# ScoreCandidates (use case) — posterior accept/reject + confidence update.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CandidateScore:
    """One candidate solver's running posterior — a «value».

    The same shape as ``InteractionRule.confidence`` / ``affordance_evidence``
    running fractions: ``confidence`` = ``accepts / trials`` (an exact integer
    fraction; 0.0 with no trials = no evidence). ``accepts`` / ``trials`` are the
    deterministic running counts (NO RNG, NO builtin ``hash()``). Frozen — an
    update returns a NEW score (immutability keeps the history honest)."""

    solver_id: str
    accepts: int = 0
    trials: int = 0

    @property
    def confidence(self) -> float:
        """The posterior in [0, 1]: ``accepts / trials`` (0.0 if no trials)."""
        return self.accepts / self.trials if self.trials > 0 else 0.0

    def updated(self, accepted: bool) -> "CandidateScore":
        """A NEW score with this trial folded in (``accepts`` += accepted,
        ``trials`` += 1). Deterministic running-fraction update."""
        return replace(
            self,
            accepts=self.accepts + (1 if accepted else 0),
            trials=self.trials + 1,
        )


@dataclass
class ScoreCandidates:
    """Anytime parallel posterior scoring of candidate solvers (domain v031
    ``ScoreCandidates`` use case; FR-C-11 / FR-S-9).

    Runs the SelectSolver candidates in parallel and posterior-scores each against
    observation: a candidate whose predicted progress materialised is ACCEPTED
    (confidence up), one that diverged is REJECTED (confidence down). The update
    is the deterministic running fraction (``accepts / trials``) — the SAME shape
    as ``InteractionRule.confidence`` / ``affordance_evidence`` (NO RNG, NO
    builtin ``hash()``; DP-10). Pairs with :class:`SelectSolver` to form the
    solver-operation loop (select -> score -> reselect on divergence).

    Holds ``scores : solver_id -> CandidateScore`` (deterministic id-keyed).
    :meth:`best` returns the highest-confidence candidate, id-ascending tie-break."""

    scores: Dict[str, CandidateScore] = field(default_factory=dict)

    def observe(self, solver_id: str, accepted: bool) -> CandidateScore:
        """Fold one observation for ``solver_id`` (accept = predicted progress
        materialised; reject = diverged) and return its updated score. Creates the
        score on first observation. Deterministic running-fraction update."""
        current = self.scores.get(solver_id, CandidateScore(solver_id=solver_id))
        updated = current.updated(accepted)
        self.scores[solver_id] = updated
        return updated

    def confidence(self, solver_id: str) -> float:
        """The posterior confidence of ``solver_id`` (0.0 if never observed)."""
        score = self.scores.get(solver_id)
        return score.confidence if score is not None else 0.0

    def accept(self, solver_id: str, *, threshold: float = 0.5) -> bool:
        """The accept/reject verdict: confidence >= ``threshold`` AND at least one
        trial (an unobserved candidate is not accepted — no evidence)."""
        score = self.scores.get(solver_id)
        return score is not None and score.trials > 0 and score.confidence >= threshold

    def best(self) -> Optional[CandidateScore]:
        """The highest-confidence observed candidate (id-ascending tie-break), or
        ``None`` if nothing has been observed. Deterministic total order."""
        if not self.scores:
            return None
        return sorted(
            self.scores.values(),
            key=lambda s: (-s.confidence, s.solver_id),
        )[0]
