"""v14 abstract-situation projection — values, entities, and the Env case B.

Concretizes the step-6-3.6 collaborators that ``agent.core.world_model`` declares
only as ``typing.Protocol`` stubs (AbstractSituation / ObjectTracker /
StateAbstraction) plus the goal-interpreter port ``agent.core.env.Env`` (case B:
a role binds to a SET of ObjectRef, ``object_for`` collapses it to ONE salient
representative). Faithful to the canonical domain model
(``docs/StrictDoc-specs/_assets/gr-arc-3-domain-model.json`` v031):

    ObjectRef{handle, profile, /geometry}                (value; identity by value)
    AbstractSituation{objects, relations, move_budget,
                      gauges}                            (value; hashable)
    ObjectTracker{objects}                               (entity)
    StateAbstraction{tracker, salience, role_of}         (function)
    ConcreteEnv{lexicon, situation, ...}                 (Env case B port impl)

Determinism discipline (DP-10): identity / memo keys NEVER use the builtin
``hash()`` and NEVER use an RNG. :func:`_blake2b_int` is the sole hash backend —
``blake2b`` over a deterministic, length-prefixed byte serialization of a
fully-primitive canonical form, reduced to an int via
``int.from_bytes(digest[:8], "big")``. The same canonical bytes produce the same
int in every process (no PYTHONHASHSEED dependence).

Scroll / world-map exclusion (terms.md TERM-51..58, domain v031): the
WorldModel.is_scrolling flag and the lazily-owned WorldMap are NEVER inputs to
:meth:`AbstractSituation.hash` or :meth:`AbstractSituation._canonical_tuple`. Nor
are an object's viewport ``cells`` / ``geometry`` (the abstract value is
position-blind so a scrolled frame hashes identically to its non-scrolled twin —
the no-regression invariant of the handoff). The [TrackViewport] hook in
:meth:`StateAbstraction.project` is INACTIVE: on a non-scrolling game there is no
WorldMap and camera_offset is ``(0, 0)`` — a true no-op (handoff §5 / NFR-6).

Mirrors the style of ``agent.core.model`` / ``agent.core.world_model`` (frozen
``@dataclass`` for «value», plain ``@dataclass`` for «entity», rich docstrings).
This module imports ONLY from ``agent.core.model`` and (for the inert TrackViewport
hook) ``agent.core.world_model``; ``world_model`` must NOT import this module (no
cycle).
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

from agent.core.model import GameObject, Lexicon, Profile

# --------------------------------------------------------------------------- #
# Canonical serialization + hash backend (DP-10).
# --------------------------------------------------------------------------- #

# Rounding decimals for ALL floats entering a canonical form (magnitudes /
# confidences / gauge values). Keeps float noise from flipping a hash.
_R = 6


def _canon_bytes(node: Any) -> bytes:
    """Deterministic, length-prefixed byte encoding of a fully-primitive nested
    structure (``str | int | float | bool | None | tuple``).

    The length prefixes make the encoding injective (no two distinct structures
    share a byte string), so the resulting hash is collision-stable across
    processes. ``bool`` is checked BEFORE ``int`` (``bool`` is an ``int``
    subclass) so ``True`` / ``1`` never alias. Any other type raises
    ``TypeError`` — the canonical form must be primitives only (DP-10: no object
    identity, no builtin ``hash``).
    """
    if node is None:
        return b"N;"
    if isinstance(node, bool):  # MUST precede int (bool is an int subclass).
        return b"B1;" if node else b"B0;"
    if isinstance(node, int):
        return b"I" + str(node).encode() + b";"
    if isinstance(node, float):
        # Normalize negative zero (and any value that rounded down to -0.0): in
        # Python ``-0.0 == 0.0`` is True, but ``format(-0.0, ".6f")`` is
        # "-0.000000" != "0.000000" — that would make two value-EQUAL situations
        # hash differently (eq/hash contract + DP-10 violation). Collapse the
        # sign of zero before formatting.
        if node == 0.0:
            node = 0.0
        return b"F" + format(node, ".6f").encode() + b";"
    if isinstance(node, str):
        b = node.encode("utf-8")
        return b"S" + str(len(b)).encode() + b":" + b + b";"
    if isinstance(node, tuple):
        return (
            b"T"
            + str(len(node)).encode()
            + b":"
            + b"".join(_canon_bytes(x) for x in node)
            + b";"
        )
    raise TypeError(f"_canon_bytes: non-primitive node {type(node).__name__}")


def _blake2b_int(data: bytes) -> int:
    """The sole hash backend (DP-10): ``blake2b`` digest of ``data``, first 8
    bytes as a big-endian unsigned int. NEVER the builtin ``hash()``, NEVER an
    RNG — the same bytes give the same int in every process."""
    return int.from_bytes(hashlib.blake2b(data).digest()[:8], "big")


# --------------------------------------------------------------------------- #
# ObjectRef (value)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, eq=False)
class ObjectRef:
    """One salient object reference in an AbstractSituation — a «value».

    Identity is BY VALUE: two ObjectRefs are equal iff their :meth:`canonical`
    forms (``handle`` + sorted/rounded Profile Characteristics) match. The
    ``geometry`` field is a NON-CANONICAL impl carrier (the same handedness as
    ``Affordance`` / ``ViewportDelta`` in ``world_model``): it carries the live
    :class:`agent.core.model.GameObject` (footprint / pose) so the Env can read
    cells / orientation / size, but it is EXCLUDED from :meth:`canonical` (and so
    from ``==`` and ``__hash__``). Excluding geometry is what makes the abstract
    value position-blind / scroll-stable: shifting every cell by a constant (a
    camera pan) does not change the ObjectRef's identity (DP-10 / NFR-6).

    ``frozen=True, eq=False``: the dataclass default ``__eq__`` / ``__hash__``
    would try to hash the mutable :class:`Profile` / :class:`GameObject` and
    crash, so both are defined manually over the canonical form.
    """

    handle: str
    profile: Profile
    geometry: GameObject

    def canonical(self) -> Tuple[Any, ...]:
        """The surface-free value key: ``(handle, sorted+rounded chars)``.

        Characteristics are sorted by ``word_id`` (a total order) and each rounded
        to :data:`_R` decimals so float noise cannot fork the key. ``geometry`` is
        intentionally NOT included (position-blindness / scroll-stability)."""
        chars = tuple(
            (c.word_id, round(c.magnitude, _R), round(c.confidence, _R))
            for c in sorted(self.profile.characteristics, key=lambda c: c.word_id)
        )
        return (self.handle, chars)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ObjectRef) and self.canonical() == other.canonical()

    def __hash__(self) -> int:
        # blake2b over the canonical bytes (NOT builtin hash) so an ObjectRef can
        # live in a frozenset deterministically across processes (DP-10).
        return _blake2b_int(_canon_bytes(self.canonical()))


# --------------------------------------------------------------------------- #
# AbstractSituation (value)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, eq=False)
class AbstractSituation:
    """The abstract board used for search / prediction — a «value» (domain v031).

    Satisfies the ``agent.core.world_model.AbstractSituation`` Protocol (it exposes
    :meth:`scalar`) and is a hashable «value» (value-equality + a deterministic
    :meth:`hash`).

    Fields:
      * ``objects``     — ``role -> frozenset[ObjectRef]`` (a role may bind to a
        SET; the Env case B fold collapses it for ``object_for``).
      * ``relations``   — ``frozenset[(operator_word_id, operand-handles-in-source-
        order)]`` (the salient relations between handles).
      * ``move_budget`` — ``(value, cap)`` or ``None``; the Markov move-budget
        gauge. :meth:`scalar` unwraps it to the VALUE.
      * ``gauges``      — ``name -> (value, cap)`` for other scalar gauges (health
        etc.).

    Determinism: ``is_scrolling`` / WorldMap / object geometry / viewport cells are
    NOT part of :meth:`_canonical_tuple` (position-blind, scroll-stable — NFR-6).
    ``frozen=True, eq=False`` for the same reason as :class:`ObjectRef` (the
    Mapping / frozenset fields are not hashable by the dataclass default).
    """

    objects: Mapping[str, FrozenSet[ObjectRef]] = field(default_factory=dict)
    relations: FrozenSet[Tuple[str, Tuple[str, ...]]] = field(default_factory=frozenset)
    move_budget: Optional[Tuple[float, float]] = None
    gauges: Mapping[str, Tuple[float, float]] = field(default_factory=dict)

    def scalar(self, name: str) -> Optional[float]:
        """The VALUE component of a Markov scalar gauge (NOT the ``(value, cap)``
        pair), or ``None`` if absent.

        Satisfies the loss-trigger substrate read by
        ``agent.core.world_model.WorldModel._terminal_outcome`` (which queries
        ``feature in {"presence", "scalar"}``): the special name ``"scalar"`` maps
        to the move_budget value; a named gauge maps to its own value; everything
        else (incl. ``"presence"`` when no such gauge exists) is ``None``. Always
        unwraps to a plain number so ``_terminal_outcome``'s ``isinstance`` guard
        sees a value, never a tuple."""
        if name in self.gauges:
            return float(self.gauges[name][0])
        if name == "scalar" and self.move_budget is not None:
            return float(self.move_budget[0])
        return None

    def _canonical_tuple(self) -> Tuple[Any, ...]:
        """The exact-ordered canonical form (the hash / equality basis).

        Ordering is fixed and total: roles sorted, each role's ObjectRefs sorted by
        their (sortable) :meth:`ObjectRef.canonical`, relations sorted, gauges
        sorted by name. ``is_scrolling`` / WorldMap / object geometry / viewport
        cells are INTENTIONALLY ABSENT — the abstract value is position-blind and
        scroll-stable (NFR-6 / DP-10). The move_budget contributes its VALUE ONLY
        (the ``cap`` is a static ceiling and is excluded; revisit if a game varies
        the cap mid-play)."""
        return (
            tuple(
                (role, tuple(sorted(ref.canonical() for ref in self.objects[role])))
                for role in sorted(self.objects)
            ),
            tuple(sorted(self.relations)),
            None if self.move_budget is None else round(self.move_budget[0], _R),
            tuple((name, round(self.gauges[name][0], _R)) for name in sorted(self.gauges)),
        )

    def hash(self) -> int:
        """Deterministic int hash over :meth:`_canonical_tuple` via
        :func:`_blake2b_int` (DP-10: not the builtin ``hash()``). This is the
        replay-hash the no-regression gate compares frame-by-frame."""
        return _blake2b_int(_canon_bytes(self._canonical_tuple()))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AbstractSituation)
            and self._canonical_tuple() == other._canonical_tuple()
        )

    def __hash__(self) -> int:
        return self.hash()


# --------------------------------------------------------------------------- #
# Frame helpers (the projector reads frames defensively).
# --------------------------------------------------------------------------- #

def _frame_objects(frame: Any) -> List[GameObject]:
    """Extract the GameObjects from ``frame`` whether it is a mapping
    ``id -> GameObject``, a struct carrying ``.objects``, or a plain iterable of
    GameObject. Returns them in ``sorted(by id)`` order (deterministic)."""
    if hasattr(frame, "objects"):
        source: Any = frame.objects
    else:
        source = frame
    if isinstance(source, Mapping):
        items: Iterable[GameObject] = source.values()
    else:
        items = source
    return sorted(items, key=lambda o: o.id)


# --------------------------------------------------------------------------- #
# ObjectTracker (entity)
# --------------------------------------------------------------------------- #

@dataclass
class ObjectTracker:
    """Frame-to-frame identity repository — a «entity» (domain v031).

    Holds ``objects : id -> GameObject`` (the latest known state per stable id).
    :meth:`associate` upserts the current frame's objects (existing id updated in
    place by replacement, new id inserted) and returns the merged view.

    Camera_offset / scroll compensation is DEFERRED (inert): this tracker TRUSTS
    the stable ids supplied in the frame and does no ego-motion re-anchoring
    (the active TrackViewport consensus body is the handoff 2-stage item). It is
    deterministic — objects are visited in ``sorted(by id)`` order."""

    objects: Dict[str, GameObject] = field(default_factory=dict)

    def associate(self, frame: Any) -> Dict[str, GameObject]:
        """Upsert the current ``frame``'s GameObjects (by id) into
        :attr:`objects` and return a copy of the merged view.

        ``frame`` is a loose carrier of GameObjects — a mapping ``id ->
        GameObject``, a struct with ``.objects``, or a plain iterable. Current-frame
        state wins on a colliding id (latest observation replaces the stored one).
        Deterministic: ids are processed in sorted order."""
        for obj in _frame_objects(frame):
            self.objects[obj.id] = obj
        return dict(self.objects)


# --------------------------------------------------------------------------- #
# StateAbstraction (function)
# --------------------------------------------------------------------------- #

# Tracking states that count as salient by default (visible or remembered-from-
# fog; an 'unknown' track is not projected). Mirrors world_model.TRACKING_STATES
# without importing it (this module stays import-light on world_model).
_SALIENT_TRACKING = frozenset({"visible", "remembered"})


def default_role_of(obj: GameObject) -> str:
    """The DEFAULT role bucket ``obj`` joins when no ``role_of`` is injected.

    Reads the MIN Characteristic word_id off the object's Profile (a minimal
    deterministic role proxy), falling back to a single shared ``"object"`` bucket
    when the Profile carries no Characteristic. This is a generic default — a real
    role classifier should be injected (NO game-specific hardcoding). Exposed at
    module level (DRY) so an injected ``role_of`` can delegate to it for the
    objects it does not reclassify (e.g. the avatar wiring in ``search_agent``)."""
    chars = obj.profile.characteristics
    if chars:
        return min(c.word_id for c in chars)
    return "object"


class StateAbstraction:
    """The projector raw-frame -> :class:`AbstractSituation` — a «function»
    (domain v031). Satisfies the ``agent.core.world_model.StateAbstraction``
    Protocol (it exposes :meth:`project`).

    Pipeline (verbalization §3): DivideFrame -> DetectFeatures ->
    [TrackViewport (INACTIVE no-op)] -> :meth:`ObjectTracker.associate` ->
    AbstractState. The three policy hooks are INJECTABLE so the projector stays
    game-literal-free (no board-specific hardcoding — NFR-6):
      * ``salience`` — ``Mapping[id, GameObject] -> Iterable[GameObject]`` (which
        tracked objects are salient this frame); default = tracks whose
        ``tracking_state`` is visible / remembered, deterministic.
      * ``role_of``  — ``GameObject -> str`` (the role bucket a salient object
        joins); default = read a role Word off the object's Profile, else a single
        shared bucket. Prefer injecting a real role classifier.
    The relations source and the gauge source are read off the frame defensively
    (a struct carrying ``.relations`` / ``.move_budget`` / ``.gauges``), so a plain
    iterable-of-GameObject frame projects to a no-relation / no-budget situation."""

    def __init__(
        self,
        tracker: ObjectTracker,
        *,
        salience: Optional[Callable[[Mapping[str, GameObject]], Iterable[GameObject]]] = None,
        role_of: Optional[Callable[[GameObject], str]] = None,
        classify: Optional[Callable[[List[GameObject]], Mapping[str, str]]] = None,
    ) -> None:
        self.tracker = tracker
        self._salience_fn = salience
        self._role_of_fn = role_of
        # SITUATION-AWARE role pre-pass (AnalogizeRoles): given the salient objects
        # this frame, return ``{object.id -> role_label}``. When injected it WINS
        # over the per-object ``role_of`` (a role can read the whole frame, e.g.
        # target's inside(self, box) referencing the controllable). When absent the
        # per-object ``role_of`` / default path runs unchanged (no regression).
        self._classify_fn = classify

    # -- public ------------------------------------------------------------- #

    def project(
        self, frame: Any, prev: Optional[AbstractSituation] = None
    ) -> AbstractSituation:
        """Project one raw ``frame`` into an :class:`AbstractSituation`.

        Flow: DivideFrame -> DetectFeatures -> [TrackViewport INACTIVE] ->
        ObjectTracker.associate -> AbstractState. ``prev`` (if given) is folded so a
        handle that vanished this frame is carried forward as ``remembered``
        (tracking continuity), while the current frame wins on a live handle."""
        objects = self.tracker.associate(frame)  # id -> GameObject
        # [TrackViewport] hook -- INACTIVE: non-scrolling => camera_offset (0, 0),
        #   world == viewport. We do NOT instantiate a WorldMap and do NOT call
        #   world_pos with a nonzero offset (a true no-op; NFR-6 / TERM-51..58).
        salient = self._salience(objects)               # deterministic, sorted
        refs_by_role = self._group_by_role(salient)     # role -> frozenset[ObjectRef]
        refs_by_role = self._carry_forward(prev, refs_by_role)
        relations = self._salient_relations(frame)      # frozenset[(op, handles)]
        move_budget, gauges = self._read_gauges(frame)
        return AbstractSituation(
            objects=refs_by_role,
            relations=relations,
            move_budget=move_budget,
            gauges=gauges,
        )

    # -- salience ----------------------------------------------------------- #

    def _salience(self, objects: Mapping[str, GameObject]) -> List[GameObject]:
        """The salient GameObjects this frame, deterministic (sorted by id).

        Default: tracks whose ``tracking_state`` is in :data:`_SALIENT_TRACKING`.
        An injected ``salience`` callable overrides the policy but the result is
        re-sorted by id so projection stays order-independent."""
        if self._salience_fn is not None:
            chosen = list(self._salience_fn(objects))
        else:
            chosen = [
                o for o in objects.values()
                if o.tracking_state in _SALIENT_TRACKING
            ]
        return sorted(chosen, key=lambda o: o.id)

    # -- roles -------------------------------------------------------------- #

    def _role_of(self, obj: GameObject) -> str:
        """The role bucket ``obj`` joins. An injected ``role_of`` wins; else the
        default reads the MIN Characteristic word_id off the Profile (a minimal
        deterministic role proxy), falling back to a single shared bucket when the
        Profile carries no Characteristic. This is a generic default — a real role
        classifier should be injected (NO game-specific hardcoding)."""
        if self._role_of_fn is not None:
            return self._role_of_fn(obj)
        return default_role_of(obj)

    def _group_by_role(
        self, salient: Iterable[GameObject]
    ) -> Dict[str, FrozenSet[ObjectRef]]:
        """Group salient GameObjects into ``role -> frozenset[ObjectRef]``. The
        handle is the object's id (deterministic-handle); the ObjectRef carries the
        live geometry as a non-canonical impl carrier.

        Role source: the injected situation-aware ``classify`` pre-pass
        (AnalogizeRoles) WINS when present — it sees the whole salient set, so a
        referent role (target's inside(self, box)) can read the controllable; an
        object the pre-pass leaves out falls to the per-object default. Absent a
        ``classify``, the per-object ``_role_of`` runs unchanged (no regression)."""
        salient = list(salient)
        roles = self._classify(salient)
        buckets: Dict[str, set] = {}
        for obj in salient:
            ref = ObjectRef(handle=obj.id, profile=obj.profile, geometry=obj)
            role = roles.get(obj.id) if roles is not None else None
            if role is None:
                role = self._role_of(obj)
            buckets.setdefault(role, set()).add(ref)
        return {role: frozenset(refs) for role, refs in buckets.items()}

    def _classify(
        self, salient: List[GameObject]
    ) -> Optional[Mapping[str, str]]:
        """The situation-aware role map (``object.id -> role_label``) from the
        injected ``classify`` pre-pass, or ``None`` when none is injected (the
        per-object path then runs). Deterministic: ``classify`` reads the salient
        set as already sorted by id."""
        if self._classify_fn is None:
            return None
        return self._classify_fn(salient)

    # -- tracking continuity ------------------------------------------------ #

    def _carry_forward(
        self,
        prev: Optional[AbstractSituation],
        refs_by_role: Dict[str, FrozenSet[ObjectRef]],
    ) -> Dict[str, FrozenSet[ObjectRef]]:
        """Fold ``prev`` so a handle present last frame but absent now is carried
        forward as ``remembered`` (the GameObject's ``tracking_state`` is set via
        :func:`dataclasses.replace`, keeping the value-typed ObjectRef immutable).
        The current frame WINS on a live handle: a handle present anywhere this
        frame is NOT carried forward, even if its role changed between frames
        (otherwise a re-bucketed live object would appear twice — once visible in
        its new role, once remembered in its stale role — double-counting the
        handle and corrupting objects_for / exists / forall ranges)."""
        if prev is None:
            return refs_by_role
        merged: Dict[str, set] = {
            role: set(refs) for role, refs in refs_by_role.items()
        }
        # Live handles are computed GLOBALLY across all roles, not per-role, so a
        # handle that merely migrated to a different role this frame is treated as
        # live (present) and not resurrected as a remembered ghost.
        live = {ref.handle for refs in merged.values() for ref in refs}
        for role, prev_refs in prev.objects.items():
            for ref in prev_refs:
                if ref.handle in live:
                    continue  # current frame wins on a live handle (any role)
                remembered_geom = dataclasses.replace(
                    ref.geometry, tracking_state="remembered"
                )
                merged.setdefault(role, set()).add(
                    ObjectRef(
                        handle=ref.handle,
                        profile=ref.profile,
                        geometry=remembered_geom,
                    )
                )
        return {role: frozenset(refs) for role, refs in merged.items()}

    # -- relations ---------------------------------------------------------- #

    def _salient_relations(
        self, frame: Any
    ) -> FrozenSet[Tuple[str, Tuple[str, ...]]]:
        """The salient relations between handles, read off the frame defensively.

        Default: NONE. If the frame carries a ``.relations`` iterable of
        ``(operator_word_id, (handle, ...))`` it is normalized into the canonical
        frozenset (operands kept in SOURCE order — relations may be directed). This
        keeps the generic path honest without inventing board-specific geometry
        (NO game-specific hardcoding); the fixture/test supplies relations
        explicitly via the frame."""
        raw = getattr(frame, "relations", None)
        if not raw:
            return frozenset()
        out = set()
        for op, operands in raw:
            out.add((op, tuple(operands)))
        return frozenset(out)

    # -- gauges ------------------------------------------------------------- #

    def _read_gauges(
        self, frame: Any
    ) -> Tuple[Optional[Tuple[float, float]], Dict[str, Tuple[float, float]]]:
        """Read ``(move_budget, gauges)`` off the frame defensively.

        A struct carrying ``.move_budget`` / ``.gauges`` is read; a plain
        iterable-of-GameObject frame (no such attrs) yields ``(None, {})``. Tolerant
        by design so simple fixtures drive the projector without a heavy Frame
        type."""
        move_budget = getattr(frame, "move_budget", None)
        gauges = getattr(frame, "gauges", None)
        return move_budget, dict(gauges) if gauges else {}


# --------------------------------------------------------------------------- #
# ConcreteEnv (Env case B) — the goal-interpreter port impl.
# --------------------------------------------------------------------------- #

@dataclass
class ConcreteEnv:
    """The concrete goal-interpreter environment over an :class:`AbstractSituation`
    — the Env case B port (``agent.core.env.Env``, env.py:36-40).

    A role binds to a SET ``frozenset[ObjectRef]``. SET-SEMANTIC predicates MUST be
    expressed with ``exists`` / ``forall`` over :meth:`objects_for` (the
    quantification range); :meth:`object_for` is the SALIENT SINGLE fold (case B) —
    it collapses a role's set to one representative via a deterministic total
    :meth:`_fold` and NEVER raises (an absent role folds to a degenerate empty
    ObjectRef). ``_binds`` (quantifier bindings) take precedence over the situation
    lookup. The geometry accessors read straight off the ObjectRef's non-canonical
    :class:`GameObject` carrier."""

    lexicon: Lexicon
    situation: AbstractSituation
    _subject: Any = None
    _binds: Dict[str, ObjectRef] = field(default_factory=dict)

    # -- subject / resolution ---------------------------------------------- #

    def subject(self) -> Any:
        return self._subject

    def object_for(self, ref: str) -> Any:
        """Resolve a bound name / Role label to ONE ObjectRef. A binding wins; else
        the role's set is collapsed by the salient fold (case B — NEVER raises)."""
        if ref in self._binds:
            return self._binds[ref]
        return self._fold(self.situation.objects.get(ref, frozenset()))

    def objects_for(self, role: str) -> Tuple[ObjectRef, ...]:
        """The quantification range bound to ``role`` (exists / forall), as a
        deterministic tuple sorted by the salient fold key."""
        return tuple(
            sorted(self.situation.objects.get(role, frozenset()), key=self._fold_key)
        )

    def bind(self, name: str, obj: Any) -> "ConcreteEnv":
        """A child env with ``name`` bound to ``obj`` (sharing lexicon + situation).
        Immutable: the parent env is unchanged (a new ConcreteEnv is returned)."""
        return ConcreteEnv(
            lexicon=self.lexicon,
            situation=self.situation,
            _subject=self._subject,
            _binds={**self._binds, name: obj},
        )

    # -- geometry / profile accessors (read the impl carrier) -------------- #

    def footprint(self, obj: ObjectRef) -> frozenset:
        return obj.geometry.cells

    def profile(self, obj: ObjectRef) -> Profile:
        return obj.profile

    def orientation(self, obj: ObjectRef) -> Optional[Tuple[float, float]]:
        return obj.geometry.orientation

    def symmetry_order(self, obj: ObjectRef) -> int:
        return obj.geometry.symmetry_order

    def reflected(self, obj: ObjectRef) -> bool:
        return obj.geometry.reflected

    def size(self, obj: ObjectRef) -> Optional[Tuple[int, int]]:
        return obj.geometry.size

    # -- the salient fold (case B) ----------------------------------------- #

    def _fold_key(self, ref: ObjectRef) -> Tuple[float, int, str]:
        """The total-order fold key (DP-10 deterministic): primary by DESCENDING
        confidence mass (-sum of Characteristic confidences), then DESCENDING
        footprint size (-cell count), then ASCENDING handle (the content-stable
        tie-break). The first element of the sorted set is the salient winner."""
        return (
            -sum(c.confidence for c in ref.profile.characteristics),
            -len(ref.geometry.cells),
            ref.handle,
        )

    def _fold(self, refs: FrozenSet[ObjectRef]) -> ObjectRef:
        """Collapse a role's ObjectRef set to its salient representative (case B).
        TOTAL — an empty set folds to a degenerate empty ObjectRef (empty handle /
        Profile / footprint) so predicates over an absent role never raise; a
        non-empty set yields the :meth:`_fold_key` minimum."""
        if not refs:
            return ObjectRef(handle="", profile=Profile(), geometry=GameObject(id=""))
        return sorted(refs, key=self._fold_key)[0]
