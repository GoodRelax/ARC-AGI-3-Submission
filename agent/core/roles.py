"""AnalogizeRoles — assign a functional Role to each recognized object.

The role-classification use case (FR-R-1 / FR-168): for each TOP-LEVEL
:class:`agent.core.model.GameObject` it evaluates each catalog Role's
``recognized_by`` predicate (a :class:`agent.core.model.Relation` tree parsed by
the AssetLoader from ``agent/assets/roles.tsv``) over the object and assigns the
single best-matching role. It REUSES the existing operator interpreter
(``agent.core.goal.test`` / ``agent.core.goal.distance``) — NO new operator
machinery, NO appearance/position operators. Roles are FUNCTIONAL (controllable /
target / field / ...), never appearance/game names (no wall/avatar/goal-box).

Mechanism (mirrors ``goal.py``'s functional style):
  * For each candidate (deterministic ``sorted(by id)`` order) build a
    role-assignment :class:`RoleEnv` with the candidate as the SUBJECT.
  * Evaluate each role's predicate via the operator interpreter; the
    ``has(word)`` single-operand form reads ``env.subject()`` (rel_has already
    supports it), and any OTHER bare reference (a shape head-noun like ``box``)
    also resolves to the subject — the candidate is the object being recognized.
    The special name ``self`` resolves to the already-assigned controllable set.

ROLE PRECEDENCE BY AUTHORITY, realized as ORDERED WAVES (operators.md §2 (ii),
☆☆ DECIDED):  grounded(controllable) -> field -> relational(target). An object
assigned in an EARLIER wave is REMOVED from later-wave candidacy, so a later /
looser role can never STEAL an object an authoritative role already claimed:
  * Wave A — ``controllable`` (grounded, HIGHEST authority): the FR-168 motion-
    grounded pick stamped ``has(controllable)`` IS controllable, even if it also
    reads ``static`` (its motion rolled out of the affordance window). Assigned
    first, removed from later waves. (Fixes: ``field`` can no longer out-rank the
    grounded avatar.)
  * Wave B — ``field`` (background): ``has(background)`` / ``has(static)`` over the
    REMAINING objects.
  * Wave C — the RELATIONAL roles (``target`` / ``template`` / ``hazard`` / the
    deferred ``status-object``): over the REMAINING objects only (an object already ``field``
    cannot become one of these), with ``self`` ranging over the wave-A
    controllable set so ``inside(self, box)`` is evaluable. The box-shape
    constraint is REAL — roles.tsv target is
    ``or(has(marked), and(has(box), inside(self, box)))`` so ``has(box)`` filters
    the candidate to a box shape (the board background, a rect, does NOT match).

Wave membership is CATEGORY-AWARE (:func:`_is_relational`): a role is RELATIONAL
(Wave C) if its recognizer mentions ``self`` OR its asset ``category`` is
``referent``. This closes a syntactic-vs-semantic gap — ``template``
(``and(has(marked), not(has(controllable)))``) and ``hazard``
(``and(has(lethal), not(has(controllable)))``) name no ``self`` but ARE referents,
so without the category key they fell to Wave B and PRE-EMPTED the Wave-C
``target`` for a marked / lethal object. Keyed on category, every referent
(target / template / status-object / hazard) is in the relational wave, so ``target`` is
ranked against them by the SAME within-wave strict total order (distance ASC,
restrictiveness DESC, roles.tsv row order ASC) rather than being stolen from an
earlier wave. For TODAY's catalog ``target`` wins a shared marked / lethal object:
all referents tie on distance (0) and on restrictiveness (``target``'s ``or`` takes
its cheapest arm ``has(marked)`` = 1 leaf; ``template`` / ``hazard`` ``and`` = 1+0
= 1), so the row-order tie-break decides and ``target`` precedes them. NOTE this
last step is row-order ONLY because the referents tie through distance + restrictiveness
first — a FUTURE referent with a STRICTLY tighter recognizer (more required leaves
than ``target``'s cheapest ``or`` arm) could out-rank ``target`` on restrictiveness
DESC before row order is consulted; that is the intended "more specific wins"
behaviour, but it means row order is not a standalone guarantee. ``field`` (category
``ground``) stays Wave B.

Wave-A membership is POLARITY-AWARE (:func:`_references_word_positive`): only a
POSITIVE (non-negated) reference to the ``controllable`` marker makes a role
grounded-authority. A role that references it only under ``not(...)`` (``ref =
and(has(interactive-target), not(has(controllable)))`` — a "not the avatar"
filter) is NOT Wave A; it is a ``referent`` so it partitions into the relational
Wave C.

Known trade-off (LOW): a FILLED box-shaped goal container that ALSO reads
``static`` is masked as ``field`` (wave B claims it before wave C). Real
enter-the-box goals are usually HOLLOW (``ring``), not filled ``box`` — those are
unaffected; revisit if a filled-box goal appears.

The wave order is data-driven: each wave is the set of catalog roles whose
predicate's required referent set is satisfiable at that stage (controllable
needs nothing; field needs nothing; target needs the controllable set bound).
Within a wave, ONE role per object (argmax) by a SOUND strict total order
(DP-10): exact ``test == True``; among the predicates that hold, rank by
``distance`` ASC, then a disjunction-aware RESTRICTIVENESS (count required
positive leaves: ``and`` = sum, ``or`` = MIN over arms — a disjunction is LOOSER,
not more specific — leaf = 1, ``not`` = 0; MORE required leaves = more
restrictive = ranked first), then roles.tsv row order ASC. NOT raw AST node-count
(which wrongly treated a disjunction as more specific). If no wave's predicate
holds, fall through to ``situation.default_role_of`` so projection never yields an
un-roled object (no-regression).

DORMANT MVP roles (do NOT fire here): the roles.tsv rows ``held-state``
(has(pose-mutable)), ``interactor`` (has(interactive)), ``status-object``
(and(has(interactive-target), not(has(controllable)))), ``template``
(and(has(marked), not(has(controllable)))) and ``hazard``
(and(has(lethal), not(has(controllable)))) never match in the MVP, because their
``feat_afford`` sub-modifiers (pose-mutable / interactive / interactive-target /
marked / lethal) report ABSENT (their evidence — pose-carry / EffectSignature /
salience — is not wired). So the MVP effectively classifies controllable / field /
target only. They ARE in the catalog (and ``template`` / ``hazard`` / ``status-object`` /
``target`` partition into the relational Wave C by category), so the wave order is
correct the instant a detector lands.

Determinism (DP-10): no ``random``, no builtin ``hash()``; candidates and roles
are visited in a fixed total order and the within-wave tie-break is a strict
total order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.core import goal
from agent.core.model import Lexicon, Profile, Relation, Role

# The canonical "self" role label (agent/assets/roles.tsv: self/controllable).
# A wave-2 referent predicate names it (e.g. inside(self, box)).
_SELF = "self"
_CONTROLLABLE = "controllable"
# The roles.tsv `category` value that marks a RELATIONAL referent (target,
# reference, ref, hazard). A referent role lands in the relational wave EVEN when
# its recognizer does not name `self` (the syntactic-vs-semantic gap, see classify).
_REFERENT = "referent"


# --------------------------------------------------------------------------- #
# Object protocol (what classify reads off a candidate)
# --------------------------------------------------------------------------- #
# classify is written against the duck-typed surface a GameObject (model.py) and
# an ObjectRef.geometry already satisfy: ``.id``, ``.cells``, ``.profile`` and the
# pose geometry attrs read by the operator interpreter (orientation /
# symmetry_order / reflected / size). It does NOT import GameObject so a test
# double with the same surface works.


# --------------------------------------------------------------------------- #
# RoleEnv — the role-assignment evaluation env (operators.md §2 context (ii)).
# --------------------------------------------------------------------------- #

@dataclass
class RoleEnv:
    """The role-assignment :class:`agent.core.env.Env` (operators.md §2 (ii)).

    The SUBJECT is the candidate object being classified; ``has(word)`` (single
    operand) reads it via :meth:`subject`. ``object_for`` resolves the special
    name ``self`` to the salient representative of the already-assigned
    controllable set (the ``_self`` binding); ANY other bare reference (a shape
    head-noun like ``box``) resolves to the SUBJECT — in a role predicate the
    referent IS the candidate being recognized (e.g. ``inside(self, box)`` asks
    "is the controllable inside THIS box-shaped candidate"). ``objects_for``
    ranges a role over the controllable set for quantifiers; a bound name wins.

    Immutable per evaluation: :meth:`bind` returns a child env (quantifier
    bindings). The geometry accessors read straight off the candidate / ``self``
    object (the same surface the goal interpreter reads)."""

    lexicon: Lexicon
    _subject: Any
    _self_objs: Tuple[Any, ...] = ()
    _binds: Dict[str, Any] = field(default_factory=dict)

    # -- subject / resolution ---------------------------------------------- #

    def subject(self) -> Any:
        return self._subject

    def object_for(self, ref: str) -> Any:
        """A binding wins; ``self`` -> the salient controllable representative;
        anything else -> the candidate SUBJECT (the referent of a role predicate
        is the object being recognized)."""
        if ref in self._binds:
            return self._binds[ref]
        if ref == _SELF:
            return self._self_objs[0] if self._self_objs else self._subject
        return self._subject

    def objects_for(self, role: str) -> Tuple[Any, ...]:
        """The quantification range bound to ``role``. A binding wins (singleton);
        ``self`` ranges over the assigned controllable set; any other role over
        the subject (singleton). Deterministic order (the inputs are already
        ordered)."""
        if role in self._binds:
            return (self._binds[role],)
        if role == _SELF:
            return self._self_objs
        return (self._subject,)

    def bind(self, name: str, obj: Any) -> "RoleEnv":
        return RoleEnv(
            lexicon=self.lexicon,
            _subject=self._subject,
            _self_objs=self._self_objs,
            _binds={**self._binds, name: obj},
        )

    # -- geometry / profile accessors (read the candidate surface) --------- #

    def footprint(self, obj: Any) -> frozenset:
        return obj.cells

    def profile(self, obj: Any) -> Profile:
        return obj.profile

    def orientation(self, obj: Any) -> Optional[Tuple[float, float]]:
        return getattr(obj, "orientation", None)

    def symmetry_order(self, obj: Any) -> int:
        return getattr(obj, "symmetry_order", 1)

    def reflected(self, obj: Any) -> bool:
        return getattr(obj, "reflected", False)

    def size(self, obj: Any) -> Optional[Tuple[int, int]]:
        return getattr(obj, "size", None)


# --------------------------------------------------------------------------- #
# Disjunction-aware restrictiveness — the SOUND within-wave tie-break ranker.
# --------------------------------------------------------------------------- #

def _required_leaves(node: Any) -> int:
    """The count of positive leaves a predicate REQUIRES to be satisfied — a sound
    restrictiveness measure (replaces the unsound AST node-count).

    A conjunction needs ALL its arms (sum); a DISJUNCTION needs only its CHEAPEST
    arm (min — a disjunction is LOOSER, not more specific, which the old node-count
    got backwards); a positive leaf needs itself (1); a negation requires nothing
    positive (0 — it is a filter, not a positive requirement). MORE required leaves
    = more restrictive = ranked first on a distance tie."""
    if not isinstance(node, Relation):
        return 1  # a bare leaf reference (Role/Word) is one positive requirement.
    op = node.operator_word_id
    if op == "and":
        return sum(_required_leaves(a) for a in node.operands)
    if op == "or":
        return min((_required_leaves(a) for a in node.operands), default=0)
    if op == "not":
        return 0  # a negation imposes no POSITIVE leaf requirement.
    # A relation/quantifier leaf (has / inside / exists / ...): the operands are
    # references/bodies, not nested boolean structure — count it as one positive
    # requirement (its own truth), recursing into nested Relations it may carry.
    nested = sum(
        _required_leaves(a) for a in node.operands if isinstance(a, Relation)
    )
    return 1 + nested


def _mentions_self(node: Any) -> bool:
    """Whether the predicate references the ``self`` role (a relational/target
    predicate that needs the controllable set bound before it is evaluable)."""
    if isinstance(node, Relation):
        return any(_mentions_self(op) for op in node.operands)
    return node == _SELF


def _is_relational(role: Role) -> bool:
    """Whether ``role`` belongs in the RELATIONAL wave (Wave C).

    A role is relational if its recognizer mentions ``self`` (a structural
    relational predicate like ``inside(self, box)``) OR the asset marks it
    ``category == 'referent'`` (a relational referent the win predicate reads).

    This closes the syntactic-vs-semantic gap that caused the wave defect:
    ``template``'s recognizer ``and(has(marked), not(has(controllable)))`` and
    ``hazard``'s ``and(has(lethal), not(has(controllable)))`` name no ``self``, so
    a self-mention-only partition wrongly dropped them into Wave B where they
    PRE-EMPTED the Wave-C ``target`` for a marked / lethal object. Keying on the
    ``referent`` category puts every referent in the relational wave, so ``target``
    (also a referent) out-ranks them by the existing within-wave row-order
    tie-break instead of being stolen from an earlier wave. ``field`` stays Wave B
    (``category == 'ground'``); ``controllable`` stays Wave A (``category ==
    'self'``, references ``controllable`` positively). Determinism: a pure
    attribute / structural read (no RNG / hash)."""
    return _mentions_self(role.recognized_by) or role.category == _REFERENT


# --------------------------------------------------------------------------- #
# AnalogizeRoles.classify — the use case.
# --------------------------------------------------------------------------- #

def _references_word_positive(node: Any, word: str, *, positive: bool = True) -> bool:
    """Whether the predicate references the bare Word/leaf ``word`` in a POSITIVE
    (non-negated) position — i.e. NOT under an odd number of ``not(...)``.

    Polarity flips on each ``not`` operand. This is the Wave-A membership test: a
    role is grounded-authority only if it references the ``controllable`` marker
    POSITIVELY (``controllable = has(controllable)`` -> yes). A role that references
    ``controllable`` only under a negation (``ref =
    and(has(interactive-target), not(has(controllable)))``) does NOT belong in Wave
    A — the negated reference is a FILTER ("not the avatar"), not a claim of avatar
    authority, so ``status-object`` partitions by its other structure (it mentions no
    ``self`` -> Wave B). Determinism: a pure structural walk (no RNG / hash)."""
    if isinstance(node, Relation):
        if node.operator_word_id == "not":
            return any(
                _references_word_positive(op, word, positive=not positive)
                for op in node.operands
            )
        return any(
            _references_word_positive(op, word, positive=positive)
            for op in node.operands
        )
    return positive and node == word


def classify(
    objects: List[Any],
    roles: Dict[str, Role],
    lexicon: Lexicon,
    *,
    default_role_of=None,
) -> Dict[str, str]:
    """Assign ONE functional role to each top-level object (FR-R-1).

    ``objects`` are the top-level candidates (each exposing ``.id`` / ``.cells`` /
    ``.profile`` + pose geometry); ``roles`` the catalog (label -> Role with a
    parsed ``recognized_by``); ``lexicon`` the operator vocabulary. Returns
    ``{object.id -> role_label}`` for EVERY candidate (an object whose predicates
    all fail falls through to ``default_role_of`` so no object is left un-roled).

    ROLE PRECEDENCE BY AUTHORITY as ORDERED WAVES (see the module docstring): an
    object claimed in an earlier wave is REMOVED from later waves, so a looser role
    can never steal it.
      * Wave A = the grounded-authority roles (predicate references the
        ``controllable`` marker, e.g. ``has(controllable)``) — assigned first.
      * Wave B = the remaining NON-relational roles (no ``self`` reference and not
        ``category == 'referent'``) — field / the dormant self-only rows.
      * Wave C = the relational roles (mention ``self`` OR ``category ==
        'referent'``) — target / reference / hazard / ref — over the REMAINING
        objects only, with ``self`` ranging over the wave-A controllable set so
        ``inside(self, box)`` is evaluable.
    ``default_role_of`` (optional; the situation default) is called for any object
    no wave claimed; absent, such an object is simply omitted (the caller keeps its
    own default).
    """
    cands = sorted(objects, key=lambda o: o.id)
    # roles in catalog (roles.tsv) row order -> the lowest-priority tie-break.
    role_order = {label: i for i, label in enumerate(roles)}
    catalog = {
        label: r for label, r in roles.items() if r.recognized_by is not None
    }

    # Partition the catalog into the three authority waves (data-driven from each
    # role's predicate + asset category: grounded-authority -> field/other ->
    # relational). A role is RELATIONAL (Wave C) if it mentions `self` OR its asset
    # category is `referent` (:func:`_is_relational`) — this keeps every referent
    # (target / reference / ref / hazard) in the relational wave so a referent
    # whose recognizer happens not to name `self` cannot pre-empt `target` from an
    # earlier wave. `field` (category `ground`) stays Wave B.
    wave_a = {
        label: r for label, r in catalog.items()
        if not _is_relational(r)
        and _references_word_positive(r.recognized_by, _CONTROLLABLE)
    }
    wave_b = {
        label: r for label, r in catalog.items()
        if not _is_relational(r) and label not in wave_a
    }
    wave_c = {
        label: r for label, r in catalog.items()
        if _is_relational(r)
    }

    assigned: Dict[str, str] = {}

    # -- Wave A: grounded controllable (self range still empty). ---------- #
    for obj in cands:
        label = _best_role(obj, wave_a, role_order, lexicon, self_objs=())
        if label is not None:
            assigned[obj.id] = label
    controllable_objs = tuple(
        obj for obj in cands if assigned.get(obj.id) == _CONTROLLABLE
    )

    # -- Wave B: field / other self-only roles over the REMAINING objects. - #
    for obj in cands:
        if obj.id in assigned:
            continue  # an earlier (more authoritative) wave already claimed it.
        label = _best_role(obj, wave_b, role_order, lexicon, self_objs=())
        if label is not None:
            assigned[obj.id] = label

    # -- Wave C: relational (target) over the REMAINING objects, with `self`
    #    bound to the wave-A controllable set. ----------------------------- #
    for obj in cands:
        if obj.id in assigned:
            continue
        label = _best_role(
            obj, wave_c, role_order, lexicon, self_objs=controllable_objs
        )
        if label is not None:
            assigned[obj.id] = label

    # -- fall-through: default for any still-unclaimed object. ------------ #
    if default_role_of is not None:
        for obj in cands:
            if obj.id not in assigned:
                assigned[obj.id] = default_role_of(obj)

    return assigned


def _best_role(
    obj: Any,
    roles: Dict[str, Role],
    role_order: Dict[str, int],
    lexicon: Lexicon,
    *,
    self_objs: Tuple[Any, ...],
) -> Optional[str]:
    """The single best role for ``obj`` WITHIN ONE wave (argmax), or ``None`` when
    no role in that wave holds.

    Builds the role-assignment env (subject = obj; ``self`` ranges over
    ``self_objs``) and keeps every role whose predicate ``test`` is True, then
    ranks the true set by the SOUND DP-10 strict total order:
      1. ``distance`` ASC (closest to satisfaction first),
      2. disjunction-aware RESTRICTIVENESS DESC (more required positive leaves
         first; a disjunction counts its CHEAPEST arm — NOT raw AST node-count),
      3. roles.tsv row order ASC (the final deterministic tie-break).
    """
    env = RoleEnv(lexicon=lexicon, _subject=obj, _self_objs=self_objs)
    scored: List[Tuple[float, int, int, str]] = []
    for label, role in roles.items():
        pred = role.recognized_by
        if goal.test(pred, env):
            scored.append(
                (
                    goal.distance(pred, env),       # 1. distance ASC
                    -_required_leaves(pred),        # 2. restrictiveness DESC
                    role_order[label],              # 3. roles.tsv row order ASC
                    label,
                )
            )
    if not scored:
        return None
    scored.sort()
    return scored[0][3]
