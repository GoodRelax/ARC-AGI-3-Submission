"""v14 World-View data model — entities and values.

Faithful in-memory port of the canonical domain model
(``docs/StrictDoc-specs/_assets/gr-arc-3-domain-model.json`` v031):

    Lexicon{words, relations}
    Word{id, part_of_speech, origin, /slot_rank}        (entity; identity = id)
    Relation{operator_word_id, operands, origin}        (entity; recursive)
    Role{label}                                          (value)
    Characteristic{word_id, magnitude, confidence}       (value)
    Profile{characteristics}                             (entity)
    GameObject{id, cells, tracking_state, centroid, orientation, reflected,
               size, symmetry_order, parts, profile}     (strong entity)
    Snapshot{(object_id, frame), cells, centroid, orientation, reflected,
             size, profile}                              (weak entity)
    Goal{predicate, status, confidence}                  (entity)

The shipped ``Word`` carries the asset columns (``category``/``impl_key``/
``params``/``slot``/``description``) on top of the abstract domain attributes;
the abstract ``Word`` is the projection {id, part_of_speech, origin, slot_rank}.
Operator semantics (``test``/``distance``/``canonical``) are NOT here — they
live in the goal interpreter (a later 段6 step) and follow the evaluation
contract in ``gr-arc-3-operators.md``. This module is pure data + lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

# --------------------------------------------------------------------------- #
# Controlled vocabularies (validated by the loader; kept as plain strings so
# the TSV stays the single source of the *values*).
# --------------------------------------------------------------------------- #

ORIGINS = frozenset({"builtin", "learned"})

# Naming-ladder slot order for composed_name (``Word.slot_rank``). Lower rank =
# earlier (leftmost) modifier; the head noun ("head") is last. Canonical order:
# ``naming-rule.md`` (OSASCOMP adaptation). Only content Words carry a slot;
# operators / transforms have none (slot_rank -> None). PROVISIONAL ordering —
# verbalization (Verbalize UC) owns the final ladder; widen here when slots are
# added to words.tsv.
LADDER_SLOTS = ("controllability", "size", "behavior", "color", "head")


# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #

@dataclass
class Word:
    """One concept/feature (red, color, dot, controllable) OR one operator Word
    (and, or, not, exists, forall, subset, inside, matches, has ...).

    Identity is ``id`` (an entity). ``impl_key`` is the dispatch key the
    AssetLoader binds to a callable (feat_* detector / op_*|rel_* evaluator /
    xf_* transform); axis/category Words have none. ``params`` holds the parsed
    ``k=v`` pairs from the TSV ``params`` column (e.g. {"index": "1"}).
    """

    id: str
    category: str
    part_of_speech: str
    description: str = ""
    slot: str = ""
    impl_key: str = ""
    params: Dict[str, str] = field(default_factory=dict)
    origin: str = "builtin"

    @property
    def slot_rank(self) -> Optional[int]:
        """Position in the naming ladder, or None for slot-less Words.

        Derived from the asset ``slot`` column (richer than ``part_of_speech``,
        which the domain model nominally cites). The asset-vs-POS derivation is a
        flagged reconciliation item; nothing depends on slot_rank until Verbalize.
        """
        try:
            return LADDER_SLOTS.index(self.slot)
        except ValueError:
            return None


# An operand is either a nested Relation or a bare reference (Role label / Word
# id) carried as a string. The interpreter resolves the string by the
# operator's operand convention (see gr-arc-3-operators.md).
Operand = Union["Relation", str]


@dataclass
class Relation:
    """A logical / quantified / relational predicate node.

    ``operator_word_id`` -> a (operator) ``Word.id``; ``operands`` is a list of
    nested ``Relation`` and/or bare string references. Recursive: ``and``/``or``
    nest Relations, ``exists``/``forall`` carry ``[Role, Relation]``, relation
    predicates carry ``[Role, Role]`` / ``[Word, Word]`` / ``[Word]`` (has).
    """

    operator_word_id: str
    operands: List[Operand] = field(default_factory=list)
    origin: str = "builtin"


@dataclass
class Lexicon:
    """The vocabulary: content/operator Words + builtin taxonomy Relations."""

    words: List[Word] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_id: Dict[str, Word] = {w.id: w for w in self.words}

    def word(self, word_id: str) -> Word:
        """The Word with this id (KeyError if absent)."""
        return self._by_id[word_id]

    def has_word(self, word_id: str) -> bool:
        return word_id in self._by_id

    def add_word(self, w: Word) -> None:
        """Append a Word (e.g. a learned one) and index it."""
        self.words.append(w)
        self._by_id[w.id] = w

    def add_relation(self, r: Relation) -> None:
        self.relations.append(r)


# --------------------------------------------------------------------------- #
# Roles
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Role:
    """A builtin-prior role label (controllable, target, field ...).

    ``recognized_by`` is the parsed membership predicate (a ``Relation`` tree
    over operator Words incl. ``has``), evaluated in the role-assignment env by
    AnalogizeRoles. None until the loader parses the roles.tsv expression.

    ``category`` is the roles.tsv ``category`` column (self | ground | referent |
    cause) — the wave-partition AUTHORITY key the classifier reads (AnalogizeRoles
    in ``agent/core/roles.py``): a ``referent``-category role is RELATIONAL (the
    relational wave) even when its recognizer does not syntactically name ``self``.
    Defaults to ``""`` (legacy / unknown) so any test double or non-asset
    ``Role(...)`` stays valid; the AssetLoader always sets it from the TSV.
    """

    label: str
    description: str = ""
    recognized_by: Optional[Relation] = None
    category: str = ""


# --------------------------------------------------------------------------- #
# Object description (values + entity)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Characteristic:
    """One graded feature on an object: ``word_id`` carries ``magnitude``
    (degree, [0,1]) with ``confidence`` ([0,1]). magnitude != confidence."""

    word_id: str
    magnitude: float
    confidence: float


@dataclass
class Profile:
    """A sparse feature vector: the Characteristics observed on one object."""

    characteristics: List[Characteristic] = field(default_factory=list)

    def of(self, word_id: str) -> Optional[Characteristic]:
        """The Characteristic for this Word, or None."""
        for c in self.characteristics:
            if c.word_id == word_id:
                return c
        return None

    def has_word(self, word_id: str, min_confidence: float = 0.0) -> bool:
        """Whether this Profile carries ``word_id`` at >= ``min_confidence``.

        This is the substrate the ``has`` operator (gr-arc-3-operators.md)
        evaluates over an env-bound subject.
        """
        c = self.of(word_id)
        return c is not None and c.confidence >= min_confidence


# --------------------------------------------------------------------------- #
# GameObject / Snapshot (board entities — domain-model.json v031)
# --------------------------------------------------------------------------- #
# pose geometry attributes (orientation / reflected / size / symmetry_order) are
# cells-derived geometry (the ``centroid`` pattern), MEASURED by DetectFeatures
# (attributes.compute_pose_geometry) and stored HERE as the VALUE — NOT as a
# Profile Characteristic (the Word stays 1-D). object-schema §3 / terms.md
# TERM-34/35/43..46 / ADR-016 (reflected = handedness) / ADR-015.

@dataclass
class GameObject:
    """A board object: the strong «entity» (identity = ``id``, kept stable by the
    ObjectTracker). Holds its CURRENT state (domain-model.json v031 GameObject).

    Position is carried at two grains: ``cells`` (the occupied footprint = the
    truth for collision / relations) and ``centroid`` (the cells-derived
    ``(row, col)`` mean, for arithmetic / heuristics). The pose geometry attributes
    ``orientation`` (unit-vector, norm=1), ``reflected`` (handedness bool),
    ``size`` (bbox extent ``(height, width)``) and ``symmetry_order``
    (rotational-symmetry order ``k``) are cells-derived geometry measured by
    DetectFeatures and live here (NOT in ``profile``). ``parts`` holds sub-objects
    (composite); ``profile`` holds the graded "qualities" (colour / shape /
    behaviour Characteristics). Past states live in :class:`Snapshot`.
    Structurally compatible with the world_model.GameObject Protocol (carries
    ``id`` + ``tracking_state``).

    ``cells`` are VIEWPORT coordinates (the footprint truth). The world position
    under scrolling is :meth:`world_pos` (= cells + camera_offset), a DERIVED
    projection in the ``centroid`` pattern: it is NOT persisted on the object
    (terms.md TERM-34/TERM-57). When not scrolling the camera_offset is
    ``(0, 0)`` and ``world_pos`` equals the plain ``cells`` set (true no-op).
    """

    id: str
    cells: FrozenSet[Tuple[int, int]] = field(default_factory=frozenset)
    tracking_state: str = "visible"
    centroid: Optional[Tuple[float, float]] = None
    orientation: Optional[Tuple[float, float]] = None
    reflected: bool = False
    size: Optional[Tuple[int, int]] = None
    symmetry_order: int = 1
    parts: List["GameObject"] = field(default_factory=list)
    profile: Profile = field(default_factory=Profile)

    def world_pos(
        self, camera_offset: Tuple[int, int] = (0, 0)
    ) -> FrozenSet[Tuple[int, int]]:
        """The object's footprint in WORLD coordinates: ``cells`` shifted by
        ``camera_offset`` (= :attr:`Viewport.origin`). TERM-57.

        DERIVED, not persisted (the ``centroid`` pattern): there is no stored
        ``world_pos`` attribute — this recomputes from ``cells`` each call so the
        viewport-coordinate footprint stays the single truth. At the default
        offset ``(0, 0)`` (non-scrolling games) it equals the plain ``(row, col)``
        set of ``cells`` — a true no-op. Deterministic (set-valued; no ordering
        escapes).
        """
        dr, dc = camera_offset
        return frozenset((row + dr, col + dc) for (row, col) in self.cells)


@dataclass
class Snapshot:
    """One object's state for ONE frame: a «weak entity» (history row), keyed by
    the composite ``(object_id, frame)`` (domain-model.json v031 Snapshot).

    ``object_id`` is the FK to :attr:`GameObject.id`; it has no independent id, so
    the "second independent id" problem structurally disappears. It carries the
    SAME pose geometry as :class:`GameObject` PER FRAME — ``cells`` / ``centroid``
    / ``orientation`` / ``reflected`` / ``size`` / ``profile`` — so EffectSignature
    can read pose deltas (rotation delta / log2 s) across the trajectory. It does
    NOT list ``symmetry_order`` or ``parts`` independently (the Snapshot node in
    the domain model omits them; parity is asserted by the pose-consistency test).
    """

    object_id: str
    frame: int
    cells: FrozenSet[Tuple[int, int]] = field(default_factory=frozenset)
    centroid: Optional[Tuple[float, float]] = None
    orientation: Optional[Tuple[float, float]] = None
    reflected: bool = False
    size: Optional[Tuple[int, int]] = None
    profile: Profile = field(default_factory=Profile)


# --------------------------------------------------------------------------- #
# Goal
# --------------------------------------------------------------------------- #

# Canonical GoalStatus enum (domain-model.json v031).
GOAL_STATUSES = frozenset({"provisional", "confirmed"})


@dataclass(frozen=True, eq=True)
class GoalPattern:
    """One row of the ``GoalPatternLibrary`` catalog: a reusable goal template.

    A «value» (frozen, by-value): the row type backing the domain v031
    ``GoalPatternLibrary`` entity, whose templates range over object roles &
    relations. ``predicate`` is the raw TSV string; for ``active`` rows the
    loader parses it into ``predicate_tree`` (a :class:`Relation` over the 14
    operator Words + role labels) and FK-validates the operators. ``deferred*``
    rows carry a ``TODO: …`` note (NOT parsed); ``predicate_tree`` stays None.

    ``goal_kind`` is the FK into :class:`GoalKind` (goal_kinds.tsv = the SSOT
    win-condition vocabulary; renamed from the old ``name``=goal_form column to
    avoid colliding with ``form``=the 12-thinking abstraction tier), ``form`` the
    abstraction tier, ``solver_kinds`` the (ungated, hint-only) solver ``id``s
    (FK into :class:`Solver`) that apply, ``distance_src`` the operator/heuristic
    the gradient reads, and ``origin`` mirrors :attr:`Word.origin`
    ({builtin, learned}).
    """

    id: str
    goal_kind: str
    predicate: str
    solver_kinds: Tuple[str, ...]
    form: str
    distance_src: str
    origin: str
    status: str
    remark: str = ""
    predicate_tree: Optional[Relation] = None


@dataclass(frozen=True, eq=True)
class GoalKind:
    """One win-condition archetype: a row of the ``goal_kinds.tsv`` vocabulary.

    A «value» (frozen, by-value). ``id`` is the kebab-case goal-kind name (the
    SSOT the v001 survey called ``goal_form``; renamed to avoid collision with
    ``GoalPattern.form``). ``category`` is the coarse bucket (spatial / structure /
    attribute / quantity / order / resource / survival). It is the FK target for
    ``GoalPattern.goal_kind`` and ``Solver.goal_kinds`` (the dispatch triangle).
    """

    id: str
    category: str
    description: str = ""
    remark: str = ""


@dataclass(frozen=True, eq=True)
class Solver:
    """One typed solver family: a row of the ``solvers.tsv`` catalog.

    A «value» (frozen, by-value) backing the ``SolverLibrary.prior`` catalog (the
    domain ``Solver`` entity, gr-arc-3-domain-model.json). Seeded from the v001
    2-round survey (47 structure-keyed families across 13 groups). ``category`` is
    the v001 ``group``; ``id`` the unique ``kind``; ``goal_kinds`` the FK list into
    :class:`GoalKind`; ``world_signature`` the world-structure half of
    applicability; ``verification_horizon`` the futility progress window
    (~0/low/med/high/max(parts)); ``backend`` the executing port
    ({SearchHeuristic, ConstrainedGenerator, Simulator}); ``parts`` the self-FK
    list of sub-solver ids (composites / Law E axis-factoring). ``goal_kinds`` and
    ``parts`` may be empty (polymorphic / runtime-bound solvers).
    """

    category: str
    id: str
    goal_kinds: Tuple[str, ...]
    world_signature: str
    verification_horizon: str
    backend: str
    parts: Tuple[str, ...]
    algorithm: str
    description: str = ""
    remark: str = ""


@dataclass
class Goal:
    """A win condition: a ``predicate`` (Relation tree) with a tracked status
    and confidence. test/distance over a situation are the goal interpreter's
    job (later step), per gr-arc-3-operators.md."""

    predicate: Relation
    status: str = "provisional"
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.status not in GOAL_STATUSES:
            raise ValueError(
                f"Goal.status {self.status!r} not in {sorted(GOAL_STATUSES)}"
            )


# --------------------------------------------------------------------------- #
# World map (scrolling) — domain-model.json v031 WorldMap «entity» / Viewport
# «value». INACTIVE on non-scrolling games: a WorldMap is created ONLY when
# WorldModel.is_scrolling becomes true (lazy 0..1 ownership); until then there is
# no WorldMap and camera_offset is (0, 0) = a true no-op. The active stitching /
# frontier logic is DEFERRED until a real scrolling game is observed (handoff
# 2-stage); the methods here are TOTAL safe-default stubs. terms.md TERM-51..58.
# --------------------------------------------------------------------------- #

# Returned by WorldMap.at for an unobserved world-cell (no colour known yet). A
# plain sentinel rather than a colour number keeps DP-10 (no game literal).
UNSEEN: Optional[int] = None


@dataclass(frozen=True)
class Viewport:
    """The current window rectangle within the world — a «value» (TERM-52).

    ``origin`` is the window's top-left world coordinate ``(row, col)``; it is the
    SINGLE home of ``camera_offset`` (TERM-53 — the cumulative pan). ``size`` is
    ``(height, width)`` = the observed frame extent (the 64x64 constant is NOT
    baked in — it is read from observation). Coordinate maps:
    world = cell + origin; cell = world - origin. Non-scrolling => origin is
    ``(0, 0)``. Frozen + integer tuples => deterministic value-equality (no float,
    avoiding GEOM-1-class sign flips).
    """

    origin: Tuple[int, int] = (0, 0)
    size: Tuple[int, int] = (0, 0)


@dataclass
class WorldMap:
    """A world stitched from windows as the camera pans — a «entity» (TERM-51).

    Lazily owned by :class:`agent.core.world_model.WorldModel` (``map 0..1``) and
    created ONLY when ``is_scrolling`` is true: on non-scrolling games there is no
    WorldMap at all (not even a degenerate grid — a true no-op). Fields:
      * ``bounds`` — the world rect ``(min_row, min_col, max_row, max_col)`` (int);
      * ``static_layer`` — ``world-cell -> colour`` for STATIC/field cells ONLY
        (dynamic objects stay in the ObjectTracker and are NOT baked into the
        raster — that would turn a moving object's trail into a wall, the
        ``occlude != destroy`` contamination; TERM-54);
      * ``seen_mask`` — observed world-cells (static union dynamic-seen); its
        complement is the spatial fog (TERM-55);
      * ``viewport`` — the current :class:`Viewport`.
    The origin is the initial frame's top-left ``(0, 0)`` and coordinates are
    integers (no float = GEOM-1 avoidance).

    The methods are TOTAL and never raise. ``stitch`` is a documented no-op for
    now — the real repaint+grow logic is DEFERRED (see below).
    """

    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
    static_layer: Dict[Tuple[int, int], int] = field(default_factory=dict)
    seen_mask: FrozenSet[Tuple[int, int]] = field(default_factory=frozenset)
    viewport: Viewport = field(default_factory=Viewport)

    def at(self, world_cell: Tuple[int, int]) -> Optional[int]:
        """The static colour at ``world_cell``, or :data:`UNSEEN` (``None``) for an
        unobserved cell. TOTAL — never raises (TERM-51). Dynamic objects are NOT
        in the raster, so ``at`` reports only the static layer."""
        return self.static_layer.get(world_cell, UNSEEN)

    def frontier(self) -> FrozenSet[Tuple[int, int]]:
        """The boundary between seen world-cells and spatial fog — the navigate
        Solver's exploration target (TERM-55). TOTAL: an empty / degenerate map
        returns ``frozenset()``.

        DEFERRED: the real frontier (4-neighbours of ``seen_mask`` that lie in the
        fog) lands after a real scrolling game is observed (handoff 2-stage). The
        stub returns the safe empty boundary so non-scrolling callers are inert.
        """
        return frozenset()

    def stitch(
        self,
        frame: object,
        static_cells: object,
        viewport: "Viewport",
    ) -> None:
        """Repaint the static layer and grow ``bounds`` / ``seen_mask`` from one
        window observation (TERM-51/TERM-54).

        # DEFERRED: real stitching lands after a real scrolling game is observed
        #           (handoff 2-stage). The real body does mode-repaint + hysteresis
        #           over recent observations of the STATIC cells only, grows bounds
        #           and seen_mask, and advances the viewport — it must NOT bake
        #           dynamic objects into the raster. Until then this is a NO-OP
        #           (records nothing) so the type+wiring is present but inert.
        """
        return None
