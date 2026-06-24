"""[agent/core] goal -- read a deliver+match goal from the frame (abduction, no game literals).

The L1 goal is a two-object conjunction discovered structurally:
  * a TARGET = a salient mark enclosed in a box (the goal);
  * its mutable TWIN = the object with the SAME colour and rotation/scale-invariant shape
    but a (currently) different pose -- the carried state the agent can change;
  * WIN = bring the controllable's footprint inside the target's container AND make the twin's
    pose MATCH the target (over colour + shape + orientation, at any scale).

The target/twin split is settled by OBSERVATION (the twin is the one whose pose changes when
the agent acts), so nothing here keys on a specific colour, coordinate, or scale.

This module ALSO hosts the general Goal / abduction cluster (CMP-09 GoalPredicate /
CMP-10 GoalPatterns / CMP-21 Milestone / CMP-27 ModelGoal) built ON TOP of the abstract
AbstractSituation (agent/core/situation.py). The two layers coexist: ``twin_of`` / ``container_of``
above stay the concrete L1-slice helpers ``play.py`` imports; everything below operates purely
on AbstractSituations and is game-literal-free (NFR-6) and deterministic (DP-10). See the cluster
banner further down for the canon citations.
"""

from __future__ import annotations

import numpy as np

from agent.core import attributes as A


def twin_of(objs, obj):
    """An object with the same colour AND rotation/scale-invariant shape as ``obj`` (its twin
    in another pose/scale), excluding ``obj`` itself by position."""
    for o in objs:
        if o.pos != obj.pos and o.dom_color == obj.dom_color and A.shape(o) == A.shape(obj):
            return o
    return None


def _component_around(grid, col, tbox):
    """Bbox of the connected ``col`` component whose bbox encloses ``tbox`` (the box border)."""
    grid = np.asarray(grid, dtype=int)
    tr0, tc0, tr1, tc1 = tbox
    mask = grid == col
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                st = [(r, c)]
                seen[r, c] = True
                cells = []
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                rs = [y for y, _ in cells]
                cs = [x for _, x in cells]
                br0, bc0, br1, bc1 = min(rs), min(cs), max(rs), max(cs)
                if br0 <= tr0 and bc0 <= tc0 and br1 >= tr1 and bc1 >= tc1:
                    return (br0, bc0, br1, bc1)
    return None


def container_of(grid, target, max_box=14):
    """Delivery region = outer bbox of the smallest single-colour box enclosing ``target``.
    Generic 'a mark inside a frame'; the border colour is whatever rings the target."""
    grid = np.asarray(grid, dtype=int)
    tr0, tc0, tr1, tc1 = target.bbox
    ring = set()
    for c in range(tc0 - 1, tc1 + 2):
        for r in (tr0 - 1, tr1 + 1):
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                ring.add(int(grid[r, c]))
    for r in range(tr0 - 1, tr1 + 2):
        for c in (tc0 - 1, tc1 + 1):
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                ring.add(int(grid[r, c]))
    best = None
    for col in ring:
        box = _component_around(grid, col, (tr0, tc0, tr1, tc1))
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if (r1 - r0) <= max_box and (c1 - c0) <= max_box:
            area = (r1 - r0) * (c1 - c0)
            if best is None or area < best[0]:
                best = (area, box)
    return best[1] if best else None


# =====================================================================================
# Goal / abduction cluster (CMP-09 GoalPredicate / CMP-10 GoalPatterns / CMP-21 Milestone /
# CMP-27 ModelGoal). Operates purely on the abstract AbstractSituation (agent/core/situation.py),
# so it is game-literal-free (NFR-6) and deterministic (DP-10: comparable distances from
# canonical content, no RNG, no builtin ``hash()`` for stable identity).
#
# Canon (cite, never duplicate):
#   - _assets/gr-arc-3-terms.md
#       TERM-12 MODEL_GOAL        -- model the win predicate; verify-WEAK (unknowable until a
#                                    win); the HARDEST facet.
#       TERM-19 goal back-inference -- abduce "what was the goal" from the FIRST win-diff, then
#                                    switch to goal-driven search (the bootstrap / 呼び水).
#   - _assets/gr-arc-3-domain-model.md
#       GoalPredicate := a win / sub-goal truth condition over a AbstractSituation; AND/OR/NOT
#         Composite tree, Atom leaves; carries no time order (Milestone roadmap does);
#         describe(lexicon) renders one sentence; conditions over objects use quantifiers
#         (count / forall / exists); confidence = abduction support.
#       GoalPatterns := baked PRIOR win-pattern templates (roles + relations); instantiate
#         (objects) -> concrete GoalPredicate; a generalisation prior, not learned.
#       Milestone := one ordered攻略 step; Milestone.goal = a GoalPredicate; the固定実装
#         enforces order. Conception.roadmap = ordered 1..* Milestones.
#   - 04-specification SC-08 / SC-09 ; 05-test-strategy TS-08 / TS-09 / TS-19 ;
#     sequence v005 goal-bootstrap (no-win -> provisional-from-prior; first-win ->
#     induce-from-win-diff + raise the matched prior's confidence).
#
# INSTANCE-INVARIANCE (NFR-6), enforced everywhere below: a predicate Atom keys ONLY on a
# AbstractSituation RELATION descriptor whose terms are ROLE LABELS (the keys of AbstractSituation.roles) or
# Profile dimension axes -- never an absolute coordinate, a colour number, or a fixed object
# id. The same predicate therefore transfers to any board with the same roles. The AbstractSituation
# relation descriptors are themselves role-keyed tuples (see ``relation``), so a GoalPredicate
# built from them is invariant by construction.
# =====================================================================================

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Tuple

from agent.core.situation import AbstractSituation


# --------------------------------------------------------------------- relation descriptors
# A AbstractSituation's ``relations`` is a frozenset of hashable descriptors. The cluster standardises
# on a ROLE-KEYED tuple form so predicates stay instance-invariant (NFR-6): a binary relation
# is ``(name, role_a, role_b)`` and a unary one ``(name, role)`` -- the terms are role LABELS
# (AbstractSituation.roles keys), never a coordinate / colour / object id. ``name`` is a relation word
# (e.g. "overlaps", "inside", "matches"), a structural axis, not a game literal.
RelDesc = Tuple  # ("name", role_a[, role_b]) -- role labels only


def relation(name: str, *roles: str) -> RelDesc:
    """Build a role-keyed relation descriptor ``(name, *roles)`` for use in a AbstractSituation's
    ``relations`` set and in a :class:`GoalPredicate` atom. The terms are role LABELS only, so
    any predicate built from it is instance-invariant (NFR-6). ``name`` is interned as ``str``
    and roles are kept in given order (a directed relation keeps its direction)."""
    return (str(name),) + tuple(str(r) for r in roles)


def relation_terms(desc: RelDesc) -> tuple:
    """The role-label terms of a relation descriptor (everything after the leading name)."""
    return tuple(desc[1:])


# ----------------------------------------------------------------------------- the literal ban
# Anything that would make a predicate instance-SPECIFIC. A term is "literal" iff it is NOT a
# role label string -- i.e. it is a number (colour / coordinate / id) or a coordinate tuple.
# Used by ``GoalPredicate.is_instance_invariant`` (the NFR-6 self-check the TS-09 test asserts).
def _is_literal_term(term: object) -> bool:
    """True iff ``term`` is a game LITERAL (a colour/coord/id number or a coordinate tuple),
    rather than an instance-invariant role label. Role labels are plain ``str``."""
    if isinstance(term, bool):           # bool is an int subclass; treat as non-literal flag
        return False
    if isinstance(term, (int, float, complex)):
        return True
    if isinstance(term, (tuple, list)):  # e.g. an (r, c) coordinate or a bbox
        return True
    return False


# ============================================================================== GoalPredicate
class GoalPredicate:
    """A win / sub-goal truth condition over a :class:`AbstractSituation` -- an AND/OR/NOT Composite
    tree with relation Atoms at the leaves (CMP-09). Holds NO state; ``test`` evaluates it and
    ``distance`` scores how far a AbstractSituation is from satisfying it (the SearchHeuristic, API-05).

    Instance-invariance (NFR-6): a leaf Atom keys ONLY on a role-keyed relation descriptor
    (see :func:`relation`); :meth:`is_instance_invariant` verifies the WHOLE tree carries no
    coordinate / colour / object-id literal. ``source_pattern`` records the GoalPattern id this
    predicate was instantiated from (``None`` for a directly-induced one), so a provisional goal
    stays traceable to its prior (TS-08 (b)).

    This base class is abstract-ish: callers build trees with :class:`Atom`, :class:`And`,
    :class:`Or`, :class:`Not` (and the quantifier wrappers :class:`Exists` / :class:`Forall`).
    Determinism (DP-10): ``distance`` is derived from canonical content with deterministic
    tie-breaks and no RNG / no builtin ``hash()``; equality is by the canonical tree key.
    """

    source_pattern: Optional[str] = None

    # ---- evaluation (subclasses implement) ----
    def test(self, situation: AbstractSituation) -> bool:
        """True iff ``situation`` satisfies this predicate (the win? test)."""
        raise NotImplementedError

    def distance(self, situation: AbstractSituation) -> int:
        """A comparable, MONOTONE-toward-satisfaction goal-distance for ``situation`` (the
        SearchHeuristic, API-05). ``0`` iff :meth:`test` holds; strictly positive otherwise, and
        it does NOT increase as more constituent atoms become satisfied. Used by the planner to
        rank candidate moves and by MoveEffect/futility (world_model.classify_move_effect)."""
        raise NotImplementedError

    # ---- structure (subclasses implement) ----
    def atoms(self) -> tuple:
        """All leaf :class:`Atom`s in this tree (deterministic order), for invariance checks
        and describe(); a flat view of the conditions the predicate tests."""
        raise NotImplementedError

    def canonical(self) -> tuple:
        """A fully-ordered canonical key the predicate's equality/hash derive from (DP-10)."""
        raise NotImplementedError

    def describe(self) -> str:
        """A deterministic one-line, game-literal-free rendering of the predicate (logs / LLM /
        viewer). Mirrors Profile.render: derived, never saved. ASCII operators."""
        raise NotImplementedError

    # ---- shared services ----
    def is_instance_invariant(self) -> bool:
        """True iff NO leaf carries a coordinate / colour / object-id literal -- every term is a
        role label (NFR-6). The TS-09 oracle asserts this on an induced goal."""
        for atom in self.atoms():
            if any(_is_literal_term(t) for t in atom.terms):
                return False
        return True

    def literal_terms(self) -> tuple:
        """Every literal (non-role) term across all atoms -- empty iff instance-invariant. A
        debugging companion to :meth:`is_instance_invariant`."""
        out: list = []
        for atom in self.atoms():
            out.extend(t for t in atom.terms if _is_literal_term(t))
        return tuple(out)

    def with_source(self, pattern_id: Optional[str]) -> "GoalPredicate":
        """Tag this predicate with the GoalPattern id it came from (traceability) and return
        ``self`` so construction chains. ``None`` clears the tag (a directly-induced goal)."""
        self.source_pattern = pattern_id
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GoalPredicate):
            return NotImplemented
        return self.canonical() == other.canonical()

    def __hash__(self) -> int:
        return hash(self.canonical())

    def __repr__(self) -> str:
        return "GoalPredicate(%s)" % self.describe()


def _holding(situation: AbstractSituation) -> frozenset:
    """The set of relation descriptors that HOLD in ``situation`` (its ``relations`` frozenset),
    normalised to tuples so membership tests are order-independent and hashable."""
    return frozenset(tuple(rel) for rel in situation.relations)


@dataclass(eq=False)
class Atom(GoalPredicate):
    """A leaf condition: the relation descriptor ``desc`` must HOLD in the AbstractSituation (CMP-09
    Atom). ``desc`` is role-keyed (:func:`relation`), so the atom is instance-invariant. The
    optional ``negated`` flag lets an Atom express "this relation must NOT hold" cheaply without
    a wrapping :class:`Not` (still one literal-free leaf)."""

    desc: RelDesc
    negated: bool = False
    source_pattern: Optional[str] = None

    def __post_init__(self) -> None:
        self.desc = tuple(self.desc)

    @property
    def name(self) -> str:
        """The relation word (the descriptor's leading element)."""
        return self.desc[0]

    @property
    def terms(self) -> tuple:
        """The role-label terms (everything after the name) -- what invariance is checked on."""
        return tuple(self.desc[1:])

    def test(self, situation: AbstractSituation) -> bool:
        present = tuple(self.desc) in _holding(situation)
        return (not present) if self.negated else present

    def distance(self, situation: AbstractSituation) -> int:
        return 0 if self.test(situation) else 1

    def atoms(self) -> tuple:
        return (self,)

    def canonical(self) -> tuple:
        return ("atom", bool(self.negated), tuple(self.desc))

    def describe(self) -> str:
        body = "%s(%s)" % (self.desc[0], ", ".join(self.desc[1:]))
        return ("NOT " + body) if self.negated else body


@dataclass(eq=False)
class And(GoalPredicate):
    """Conjunction: holds iff EVERY operand holds; distance = SUM of operand distances (so it is
    0 iff all hold and decreases monotonically as operands are satisfied -- an admissible-style
    SearchHeuristic for AND-decomposed goals)."""

    operands: Tuple[GoalPredicate, ...]
    source_pattern: Optional[str] = None

    def __post_init__(self) -> None:
        self.operands = tuple(self.operands)

    def test(self, situation: AbstractSituation) -> bool:
        return all(op.test(situation) for op in self.operands)

    def distance(self, situation: AbstractSituation) -> int:
        return sum(op.distance(situation) for op in self.operands)

    def atoms(self) -> tuple:
        out: list = []
        for op in self.operands:
            out.extend(op.atoms())
        return tuple(out)

    def canonical(self) -> tuple:
        # Order-independent: AND is commutative, so sort operand keys deterministically.
        return ("and",) + tuple(sorted((op.canonical() for op in self.operands), key=repr))

    def describe(self) -> str:
        return "(" + " AND ".join(op.describe() for op in self.operands) + ")"


@dataclass(eq=False)
class Or(GoalPredicate):
    """Disjunction: holds iff ANY operand holds; distance = MIN of operand distances (0 iff some
    operand holds; the closest branch drives the heuristic). An EMPTY ``Or`` is the degenerate
    disjunction -- logically FALSE (no disjunct can satisfy it), so ``test`` is ``False`` and
    ``distance`` is the positive unit ``1`` (NOT a ``min([])`` crash): the heuristic contract
    (API-05) requires ``distance`` to be total and ``0`` iff ``test`` -- both are preserved."""

    operands: Tuple[GoalPredicate, ...]
    source_pattern: Optional[str] = None

    def __post_init__(self) -> None:
        self.operands = tuple(self.operands)

    def test(self, situation: AbstractSituation) -> bool:
        return any(op.test(situation) for op in self.operands)

    def distance(self, situation: AbstractSituation) -> int:
        # Empty disjunction is unsatisfiable: a positive, finite distance (never min([]) -> raise).
        if not self.operands:
            return 1
        return min(op.distance(situation) for op in self.operands)

    def atoms(self) -> tuple:
        out: list = []
        for op in self.operands:
            out.extend(op.atoms())
        return tuple(out)

    def canonical(self) -> tuple:
        return ("or",) + tuple(sorted((op.canonical() for op in self.operands), key=repr))

    def describe(self) -> str:
        return "(" + " OR ".join(op.describe() for op in self.operands) + ")"


@dataclass(eq=False)
class Not(GoalPredicate):
    """Negation: holds iff the inner predicate does NOT hold. Distance is 0 when satisfied else
    1 (a structural flip cannot borrow the inner gradient meaningfully)."""

    operand: GoalPredicate
    source_pattern: Optional[str] = None

    def test(self, situation: AbstractSituation) -> bool:
        return not self.operand.test(situation)

    def distance(self, situation: AbstractSituation) -> int:
        return 0 if self.test(situation) else 1

    def atoms(self) -> tuple:
        return self.operand.atoms()

    def canonical(self) -> tuple:
        return ("not", self.operand.canonical())

    def describe(self) -> str:
        return "NOT " + self.operand.describe()


# --------------------------------------------------------------------------- quantifiers
# GoalPredicate conditions over objects use quantifiers (count / forall / exists -- domain
# model). They are expressed by quantifying a relation NAME over the participating roles of a
# AbstractSituation: e.g. EXISTS a pair holding "matches", or FORALL roles "inside". The bound terms are
# still role labels (NFR-6); the quantifier just lifts the role choice.
@dataclass(eq=False)
class Exists(GoalPredicate):
    """There EXISTS a tuple of (distinct) roles in the AbstractSituation for which relation ``name``
    holds. ``arity`` is the number of roles the relation takes (1 or 2). Instance-invariant: it
    quantifies over role LABELS, binding none to a literal. Distance is 0 iff some binding holds,
    else 1 (presence/absence)."""

    name: str
    arity: int = 2
    source_pattern: Optional[str] = None

    def _bindings_hold(self, situation: AbstractSituation) -> bool:
        holding = _holding(situation)
        for desc in holding:
            if desc and desc[0] == self.name and len(desc) - 1 == self.arity:
                return True
        return False

    def test(self, situation: AbstractSituation) -> bool:
        return self._bindings_hold(situation)

    def distance(self, situation: AbstractSituation) -> int:
        return 0 if self.test(situation) else 1

    def atoms(self) -> tuple:
        # A quantified leaf: the term is the relation NAME plus a structural arity marker, no
        # role/literal bound. Surfaced as an Atom-like leaf for the invariance scan (no literal).
        return (_QuantLeaf(name=self.name, arity=self.arity, quant="exists"),)

    def canonical(self) -> tuple:
        return ("exists", self.name, int(self.arity))

    def describe(self) -> str:
        return "EXISTS roles: %s/%d" % (self.name, self.arity)


@dataclass(eq=False)
class Forall(GoalPredicate):
    """For ALL participating roles, the unary relation ``name`` holds on that role. (Defined for
    arity-1; a board with no roles is vacuously true.) Instance-invariant. Distance = COUNT of
    roles for which it fails (monotone: 0 iff all satisfied)."""

    name: str
    roles_of: Optional[Callable[[AbstractSituation], Iterable[str]]] = None
    source_pattern: Optional[str] = None

    def _roles(self, situation: AbstractSituation) -> tuple:
        if self.roles_of is not None:
            return tuple(self.roles_of(situation))
        return tuple(sorted(situation.roles.keys()))

    def distance(self, situation: AbstractSituation) -> int:
        holding = _holding(situation)
        missing = 0
        for role in self._roles(situation):
            if (self.name, role) not in holding:
                missing += 1
        return missing

    def test(self, situation: AbstractSituation) -> bool:
        return self.distance(situation) == 0

    def atoms(self) -> tuple:
        return (_QuantLeaf(name=self.name, arity=1, quant="forall"),)

    def canonical(self) -> tuple:
        return ("forall", self.name)

    def describe(self) -> str:
        return "FORALL roles: %s" % self.name


@dataclass(eq=False)
class _QuantLeaf(GoalPredicate):
    """An invariance-scan stand-in for a quantified leaf: it exposes a ``terms`` of role-free
    structural markers (the relation name + an arity tag), so :meth:`GoalPredicate.atoms` can
    include quantified predicates in the NFR-6 scan WITHOUT introducing a literal. Never tested
    directly -- it has no ``test`` semantics of its own beyond the structural marker."""

    name: str = ""
    arity: int = 0
    quant: str = ""
    source_pattern: Optional[str] = None

    @property
    def terms(self) -> tuple:
        # Role-free: only the relation name (a word) and an arity marker string -- both
        # non-literal, so the invariance scan passes.
        return (self.name, "arity:%d" % self.arity)

    def test(self, situation: AbstractSituation) -> bool:  # pragma: no cover - structural marker
        return True

    def distance(self, situation: AbstractSituation) -> int:  # pragma: no cover - structural marker
        return 0

    def atoms(self) -> tuple:
        return (self,)

    def canonical(self) -> tuple:
        return ("quantleaf", self.quant, self.name, int(self.arity))

    def describe(self) -> str:  # pragma: no cover - structural marker
        return "%s:%s/%d" % (self.quant, self.name, self.arity)


# ============================================================================== GoalPatterns
@dataclass(frozen=True)
class GoalPattern:
    """One baked PRIOR win-pattern template (CMP-10 element): a relation schema over role SLOTS
    plus a default ``confidence``. ``relations`` is a tuple of ``(name, slot_a[, slot_b])`` where
    the slots are ABSTRACT role names the template expects (e.g. "carried-state", "target"); a
    binding maps each slot to an observed object's role label. ``pattern_id`` is a stable,
    game-literal-free handle (e.g. "deliver-and-match"). No coordinate / colour / id appears."""

    pattern_id: str
    relations: Tuple[RelDesc, ...]
    confidence: float = 0.5
    slots: Tuple[str, ...] = ()

    def required_slots(self) -> frozenset:
        """All role slots the template's relations mention (the roles a board must offer)."""
        out: set = set()
        for rel in self.relations:
            out.update(rel[1:])
        if self.slots:
            out.update(self.slots)
        return frozenset(out)


def _default_patterns() -> dict:
    """The shipped PRIOR library (a generalisation prior, NOT learned -- domain model). Each
    pattern is a relation schema over abstract role SLOTS; none names a colour / coordinate / id.
    Confidences are PRIORS (deliver+match is the strongest baked guess -- it is the ls20 family
    shape -- but every value is a role-only template, never a board fact)."""
    return {
        # Deliver the carried state to the target AND make it match the goal mark.
        "deliver-and-match": GoalPattern(
            pattern_id="deliver-and-match",
            relations=(
                ("inside", "carried-state", "target"),
                ("matches", "carried-state", "target"),
            ),
            confidence=0.6,
        ),
        # Bring the controllable onto the interactor (a touch / reach goal).
        "reach-interactor": GoalPattern(
            pattern_id="reach-interactor",
            relations=(("overlaps", "controllable", "interactor"),),
            confidence=0.4,
        ),
        # Make the controllable coincide with the target (a simple reach-the-goal).
        "reach-target": GoalPattern(
            pattern_id="reach-target",
            relations=(("inside", "controllable", "target"),),
            confidence=0.3,
        ),
    }


class GoalPatterns:
    """The PRIOR library of win-pattern templates + per-pattern confidence (CMP-10).

    ``instantiate(template, observed_objects)`` binds the template's role SLOTS to the roles the
    observed objects actually carry and returns a concrete :class:`GoalPredicate` tagged with the
    source ``pattern_id`` (traceable to its prior -- TS-08 (b)). ``confidence`` per pattern is a
    PRIOR that updates on evidence: :meth:`confirm` (a win-diff matched the pattern) raises it,
    :meth:`contradict` lowers it -- the TS-09 confidence-update oracle. No game literal lives in a
    template or a key (NFR-6); updates are deterministic (DP-10: fixed step, clamped to [0, 1])."""

    def __init__(self, patterns: Optional[Mapping[str, GoalPattern]] = None,
                 step: float = 0.2) -> None:
        self._patterns: dict = dict(patterns) if patterns is not None else _default_patterns()
        # Mutable per-pattern confidence, seeded from each template's prior.
        self._confidence: dict = {pid: p.confidence for pid, p in self._patterns.items()}
        self._step: float = float(step)

    # ---- access ----
    def ids(self) -> tuple:
        """Pattern ids in a DETERMINISTIC order (sorted), for stable iteration / tie-breaks."""
        return tuple(sorted(self._patterns.keys()))

    def get(self, pattern_id: str) -> GoalPattern:
        return self._patterns[pattern_id]

    def confidence(self, pattern_id: str) -> float:
        """Current confidence of ``pattern_id`` (its evolving abduction support)."""
        return self._confidence[pattern_id]

    def ranked(self) -> tuple:
        """Pattern ids ordered by DESCENDING confidence, ties broken by ascending id (a total,
        deterministic order -- DP-10). The planner/bootstrap prefers the highest-prior pattern."""
        return tuple(sorted(self._patterns.keys(), key=lambda pid: (-self._confidence[pid], pid)))

    # ---- confidence update (TS-09 oracle) ----
    def confirm(self, pattern_id: str) -> float:
        """Raise ``pattern_id``'s confidence (a win-diff MATCHED this pattern). Clamped to 1.0."""
        self._confidence[pattern_id] = min(1.0, self._confidence[pattern_id] + self._step)
        return self._confidence[pattern_id]

    def contradict(self, pattern_id: str) -> float:
        """Lower ``pattern_id``'s confidence (a win-diff CONTRADICTED this pattern). Clamped to
        0.0."""
        self._confidence[pattern_id] = max(0.0, self._confidence[pattern_id] - self._step)
        return self._confidence[pattern_id]

    # ---- instantiation ----
    def instantiate(self, template, observed_objects: Iterable) -> Optional[GoalPredicate]:
        """Bind a template's role SLOTS to the roles present among ``observed_objects`` and
        return a concrete :class:`GoalPredicate` (an AND over the template's relations), tagged
        with the source ``pattern_id``.

        ``template`` may be a :class:`GoalPattern` or a ``pattern_id`` string. ``observed_objects``
        is an iterable of ``(role_label, obj)`` pairs OR an object whose ``.role`` attribute gives
        the label OR a mapping ``{role_label: obj}`` -- only the role LABELS are consumed (the obj
        is along for grounding/debug; no colour/coord/id is read). Returns ``None`` iff a required
        slot is missing from the observed roles (the template does not apply to this board).

        The produced predicate is instance-invariant by construction: its atoms key on the role
        labels the slots bound to, never a literal.
        """
        pattern = template if isinstance(template, GoalPattern) else self._patterns[str(template)]
        observed_roles = _observed_role_set(observed_objects)
        # Every slot the template needs must be an observed role; else the template can't bind.
        if not pattern.required_slots() <= observed_roles:
            return None
        atoms = [Atom(desc=relation(*rel)) for rel in pattern.relations]
        if len(atoms) == 1:
            pred: GoalPredicate = atoms[0]
        else:
            pred = And(operands=tuple(atoms))
        return pred.with_source(pattern.pattern_id)


def _observed_role_set(observed_objects) -> frozenset:
    """Extract the set of role LABELS from the various accepted shapes of ``observed_objects``
    (a mapping {role: obj}, an iterable of (role, obj) pairs, or an iterable of objects carrying
    a ``.role`` attribute). Only labels are read -- never a surface literal (NFR-6)."""
    if isinstance(observed_objects, Mapping):
        return frozenset(str(k) for k in observed_objects.keys())
    roles: set = set()
    for item in observed_objects:
        if isinstance(item, tuple) and len(item) == 2:
            roles.add(str(item[0]))
        else:
            role = getattr(item, "role", None)
            if role is not None:
                roles.add(str(role))
    return frozenset(roles)


# ================================================================================ ModelGoal
class ModelGoal:
    """The goal model use-case (CMP-27): provisional-goal-from-prior before any win, and
    back-inference of an instance-invariant predicate from the FIRST win-diff (TERM-19).

    Bootstrap (``hypothesize``; SC-08 / TS-08): with NO win observed yet, pick the
    highest-confidence GoalPattern whose role slots the observed AbstractSituation offers, instantiate it
    onto those roles, and return the (low-confidence) provisional :class:`GoalPredicate` -- tagged
    with the source pattern id so it stays traceable to its prior. The planner then ranks moves by
    this predicate's :meth:`GoalPredicate.distance`.

    Refinement (``refine_from_win``; SC-09 / TS-09): on the first win, diff the pre-win and win
    AbstractSituations' HOLDING relations; the relations that newly hold in the win (and not before) ARE
    the goal, lifted to an instance-invariant predicate (role-keyed atoms only). Then update the
    library: every prior pattern whose relations are all satisfied by the win-diff is CONFIRMED
    (confidence up); a prior whose relations are contradicted (it expected a relation that the
    win-diff did not deliver, while another pattern did) is CONTRADICTED (confidence down).

    Holds the :class:`GoalPatterns` library (mutable confidences) but no per-board literal; all
    operations are deterministic (DP-10).
    """

    def __init__(self, patterns: Optional[GoalPatterns] = None) -> None:
        self.patterns: GoalPatterns = patterns if patterns is not None else GoalPatterns()

    # ---- bootstrap: provisional goal from the highest prior (no win yet) ----
    def hypothesize(self, situation: AbstractSituation) -> Optional[GoalPredicate]:
        """Return a provisional :class:`GoalPredicate` instantiated from the highest-confidence
        applicable GoalPattern onto ``situation``'s roles (SC-08 / TS-08). ``None`` iff NO baked
        pattern's slots are all present in the AbstractSituation (nothing to hypothesise yet).

        The observed roles come from ``situation.roles`` (the role LABELS the StateAbstraction
        projected). The returned predicate is traceable to its pattern via ``source_pattern``.
        Determinism: patterns are tried in :meth:`GoalPatterns.ranked` order (confidence desc,
        id asc), so the choice is unique."""
        observed = {role: role for role in situation.roles.keys()}  # role-label set, no literal
        for pattern_id in self.patterns.ranked():
            pred = self.patterns.instantiate(pattern_id, observed)
            if pred is not None:
                return pred
        return None

    # ---- back-inference: induce an instance-invariant goal from the first win-diff ----
    def refine_from_win(self, pre_win: AbstractSituation, win: AbstractSituation) -> GoalPredicate:
        """Induce an instance-invariant :class:`GoalPredicate` from the win-diff (SC-09 / TS-09).

        The relations that HOLD in ``win`` but did NOT hold in ``pre_win`` are the changed
        conditions that the win achieved -- the goal. Each is a role-keyed descriptor, so the
        conjunction over them is instance-invariant by construction (no coordinate / colour / id).
        If nothing newly holds (degenerate), fall back to the relations holding in ``win`` (still
        role-keyed). Side effect: update :class:`GoalPatterns` confidences from the win-diff.

        Returns the induced predicate (``source_pattern`` is ``None`` -- it is directly induced,
        not a prior instantiation, though a matching prior is separately confirmed)."""
        before = _holding(pre_win)
        after = _holding(win)
        gained = tuple(sorted(after - before, key=repr))   # deterministic order (DP-10)
        descs = gained if gained else tuple(sorted(after, key=repr))

        atoms = [Atom(desc=d) for d in descs]
        if not atoms:
            # No relations at all: an empty conjunction (vacuously true) -- still invariant.
            pred: GoalPredicate = And(operands=())
        elif len(atoms) == 1:
            pred = atoms[0]
        else:
            pred = And(operands=tuple(atoms))

        self._update_confidences(frozenset(descs))
        return pred.with_source(None)

    def _update_confidences(self, win_descs: frozenset) -> None:
        """Confirm every prior pattern whose relation schema is SATISFIED by the win-diff (all of
        its relations are among ``win_descs``); contradict a prior that is NOT satisfied while at
        least one OTHER prior is (so a genuinely-wrong prior is pushed down, but we never penalise
        every pattern when none matched). Deterministic (DP-10): pattern ids iterated in sorted
        order; fixed confidence step.

        Matching is on the relation TUPLES (name + role slots), which equal the win-diff role-keyed
        descriptors exactly when the slots coincide with the board roles -- the instance-invariant
        comparison the spec asks for."""
        satisfied: list = []
        unsatisfied: list = []
        for pattern_id in self.patterns.ids():
            schema = frozenset(tuple(rel) for rel in self.patterns.get(pattern_id).relations)
            if schema and schema <= win_descs:
                satisfied.append(pattern_id)
            else:
                unsatisfied.append(pattern_id)
        for pattern_id in satisfied:
            self.patterns.confirm(pattern_id)
        # Only contradict the others when SOMETHING matched (else the win used an unmodelled
        # pattern and penalising every prior would be noise).
        if satisfied:
            for pattern_id in unsatisfied:
                self.patterns.contradict(pattern_id)


# ================================================================================ Milestone
@dataclass(frozen=True)
class Milestone:
    """One ordered攻略 step (CMP-21): a named intermediate goal whose ``goal`` is a
    :class:`GoalPredicate` (the step's win condition). A固定実装 enforces the ORDER across a
    roadmap of Milestones -- the order is NOT delegated to an LLM (the belief-agent thrash
    lesson). This is the STRUCTURE only; the ordering ENFORCEMENT (PlanMoves planning each step
    in turn) is the solver wave's TS-19.

    ``name`` is a stable, game-literal-free label (e.g. "clear-board", "open-door"); ``order`` is
    the step's position in the roadmap (ascending). Frozen «value»-ish: two Milestones with the
    same name/order/goal compare equal (goal equality is by GoalPredicate.canonical)."""

    goal: GoalPredicate
    name: str = ""
    order: int = 0

    def is_met(self, situation: AbstractSituation) -> bool:
        """True iff this step's ``goal`` holds in ``situation`` (the step is complete)."""
        return self.goal.test(situation)

    def distance(self, situation: AbstractSituation) -> int:
        """Goal-distance to this step (delegates to the step's GoalPredicate -- the planner ranks
        moves toward the CURRENT milestone by this)."""
        return self.goal.distance(situation)


@dataclass
class Roadmap:
    """An ORDERED container of :class:`Milestone`s (Conception.roadmap, 1..* ordered). The order
    is the hard攻略 track a固定実装 must follow: a later milestone is never planned as the final
    target before its predecessors' goals hold (enforced by PlanMoves in TS-19; this class only
    HOLDS the ordered steps and exposes the next-unmet step).

    Milestones are kept sorted by ``order`` (then name) for a deterministic sequence (DP-10).
    """

    milestones: Tuple[Milestone, ...] = ()

    def __post_init__(self) -> None:
        # Deterministic, stable ordering by (order, name) so the roadmap sequence is total.
        self.milestones = tuple(sorted(self.milestones, key=lambda m: (m.order, m.name)))

    def ordered(self) -> Tuple[Milestone, ...]:
        """The milestones in roadmap order."""
        return self.milestones

    def current(self, situation: AbstractSituation) -> Optional[Milestone]:
        """The FIRST milestone (in roadmap order) whose goal does NOT yet hold -- the step the
        planner should currently aim at. ``None`` iff every milestone is met (roadmap complete)."""
        for m in self.milestones:
            if not m.is_met(situation):
                return m
        return None

    def final_goal(self) -> Optional[GoalPredicate]:
        """The last milestone's goal -- the overall win condition the roadmap culminates in."""
        return self.milestones[-1].goal if self.milestones else None

    def all_met(self, situation: AbstractSituation) -> bool:
        """True iff every milestone's goal holds (the whole roadmap is satisfied)."""
        return all(m.is_met(situation) for m in self.milestones)


# Helper retained for symmetry with the L1 helpers above: a no-op alias module-level marker so
# downstream imports can do `from agent.core import goal as G` and reach the whole cluster.
_GOAL_CLUSTER_LOADED = True
