"""[Entity] GoalPredicate + the v1 GoalTemplate library (COINCIDE, REACH).

Pure, frozen Entity (spec §4.2). MUST NOT import the framework (`agents`/`arcengine`);
depends only on stdlib + the reused Phase A `segment` Entity (NFR-113).

A `GoalPredicate` is a CONJUNCTION of relations among object ROLES. There are two ways
a role id is bound to a concrete object:

  * **legacy shape-class rank** (`role_of`, FR-130 reactive path): the role is the RANK
    of the object's `shape_hash` among the distinct shape classes present. Used by the
    reactive win-diff verifier (`goal_inference.from_win`) and its tests. Kept unchanged.
  * **instance-invariant SELECTOR** (FR-147, ADR-0002 abduction path): the role is bound
    by a *relational / cardinality / controllability* feature — the **controllable**
    object, the **cardinality-unique** object (shape-class count == 1), the **largest**
    or **smallest** by size — NEVER a `shape_hash` rank (which is an instance identity
    that leaks across games, N-02). A `GoalPredicate` produced by a `GoalTemplate` carries
    its `selectors` so role binding is recomputed from structure each frame and is
    INVARIANT to `shape_hash` relabeling (behaviorally checked, NFR-111).

`holds()` is the planner's goal test; `distance()` is an admissible (never-overestimating)
distance-to-goal heuristic: the centroid Manhattan/Chebyshev lower bound, taken as the
**MAX (not sum)** over the predicate's independent sub-goals (N-03, SC-124).

`GoalTemplate`/`GoalHypothesisLibrary` hold the v1 abduction templates (FR-150). Only
COINCIDE + REACH are built; COVER/FILL, SYMMETRY, COUNT-EQUALIZE are Future (need a
tagged-union predicate) and are intentionally NOT implemented here (N-01).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from agent.segment import GRID_SIZE, GridObject, ObjectSet

__all__ = [
    "GoalPredicate",
    "GoalTemplate",
    "GoalHypothesisLibrary",
    "role_of",
    "RELATIONS",
    "SELECTORS",
    "select_role",
    "default_library",
    "Affordance",
    "SelectionContext",
    "Identity",
    "MIN_CONTROLLABILITY_SUPPORT",
    "BACKGROUND_SPAN",
]

# Supported relation names (over ROLE pairs). All are positional/topological and
# instance-invariant — none references a concrete coordinate or color value.
RELATIONS = ("overlaps", "adjacent", "left_of", "above", "same_row", "same_col")

# The instance-invariant role SELECTORS (FR-147). Each picks ONE object from an
# ObjectSet by a relational / cardinality / controllability feature — NEVER a
# `shape_hash` rank. A selector returns role-able OBJECTS; the binding is recomputed
# from structure every frame, so relabeling `shape_hash` cannot change it (NFR-111).
SELECTORS = ("controllable", "cardinality_unique", "largest", "smallest", "salient")
SelectorName = Literal[
    "controllable", "cardinality_unique", "largest", "smallest", "salient"
]

# A `salient` target must exclude the board BACKGROUND. An object is background when its
# bounding box spans most of BOTH board dimensions (a field/frame — solid or ring); a token
# spans at most one. A general geometric fraction of the fixed 64-board, no game value (NFR-122).
BACKGROUND_SPAN = 0.5


# --- v0.7 (ADR-012): grounded controllability — affordance map + SelectionContext -----
# An identity needs >= this much observed translate support before it can win the
# `controllable` binding (below it, the static fallback applies). No game-specific value.
MIN_CONTROLLABILITY_SUPPORT = 2

# Affordance-map key: (shape_hash, color) as plain ints. Instance-invariant under
# shape_hash relabeling because the SAME tuple relabels WITH the object (NFR-116).
Identity = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Affordance:
    """Per-identity observed responsiveness distilled from the WorldModel (FR-167)."""

    translate_support: int = 0   # positional controllability (translate under movement) -> `controllable`
    response_support: int = 0    # general responsiveness (non-noop under any action) -> future `responsive`
    vanish_support: int = 0      # disappear effects -> `vanisher`
    spawn_support: int = 0       # appear effects -> `spawner`
    autonomous: bool = False     # same translate vector under >=2 actions -> self-moving (`autonomous`)


@dataclass(frozen=True, slots=True)
class SelectionContext:
    """Optional dynamics context threaded through role binding (FR-170).

    Default-None at every call site -> static behaviour (FR-169), so the change is purely
    additive. `affordances` is built once per turn from the live WorldModel; the optional
    `pinned_controllable` identity is set by the policy's sticky binding (FR-171) and by the
    planner's plan-root fix (FR-170/H2) so the `controllable` role is STABLE across simulated
    states (no mid-search static fallback that could break admissibility).
    """

    affordances: dict[Identity, Affordance] = field(default_factory=dict)
    pinned_controllable: Optional[Identity] = None


def role_of(obj: GridObject, objset: ObjectSet) -> int:
    """Legacy instance-invariant role id for `obj` (shape-class RANK, FR-130 path).

    The role is the RANK of the object's `shape_hash` among the distinct shape classes
    present (sorted ascending), NOT the `shape_hash` value itself. Used by the reactive
    win-diff verifier (`goal_inference.from_win`); the ABDUCTION path uses SELECTORS
    (`select_role`) instead, never this rank (FR-147, N-02).
    """
    classes = sorted({o.shape_hash for o in objset.objects})
    return classes.index(obj.shape_hash)


def _roles_present(objset: ObjectSet) -> dict[int, list[GridObject]]:
    """Group objects by their legacy role id (shape-class rank)."""
    out: dict[int, list[GridObject]] = {}
    for o in objset.objects:
        out.setdefault(role_of(o, objset), []).append(o)
    return out


# --- instance-invariant role selectors (FR-147, ADR-0002) ----------------------


def _controllable(
    objset: ObjectSet, context: Optional["SelectionContext"] = None
) -> Optional[GridObject]:
    """The controllable object (the avatar), GROUNDED in observed dynamics (FR-168).

    v0.7 (ADR-012): the avatar is the object the WorldModel has OBSERVED to move when
    acted upon, not a static color/size guess. Resolution order:
      1. context with a PINNED identity (planner root-fix / policy sticky binding,
         FR-170/FR-171): return the object matching that `(shape_hash, color)`, else None
         (NO static fallback mid-search — keeps the heuristic admissible, H2).
      2. context with an affordance map: the present identity with the highest positional
         controllability (translate support) >= MIN_CONTROLLABILITY_SUPPORT (FR-168),
         tie-broken by a shape_hash-INVARIANT key (color, size, position) — NEVER
         shape_hash (H1).
      3. no context / no qualifying mover: the static structural fallback (FR-169).
    """
    if context is not None:
        if context.pinned_controllable is not None:
            sh, col = context.pinned_controllable
            for o in objset.objects:
                if o.shape_hash == sh and o.color == col:
                    return o
            return None  # pinned identity absent: unbound (admissible >= 1), no static fallback
        if context.affordances:
            movers = [
                (o, aff.translate_support)
                for o in objset.objects
                for aff in (context.affordances.get((o.shape_hash, o.color)),)
                if aff is not None
                and aff.translate_support >= MIN_CONTROLLABILITY_SUPPORT
            ]
            if movers:
                # shape_hash-INVARIANT tie-break (NFR-116, H1): NEVER key on shape_hash.
                return max(
                    movers,
                    key=lambda t: (t[1], -t[0].color, -t[0].size,
                                   -t[0].bbox[0], -t[0].bbox[1]),
                )[0]
    return _controllable_static(objset)  # cold-start / no-context fallback (FR-169)


def _controllable_static(objset: ObjectSet) -> Optional[GridObject]:
    """Static structural fallback for the controllable object (FR-169, pre-evidence).

    Heuristic, instance-invariant, framework-free: the rarest-color singleton of a
    small/medium size. We pick the object whose COLOR appears exactly once (a unique,
    interactive token) breaking ties toward the smaller object (avatars are small,
    not the background). Returns None when nothing is distinctive. This references
    only counts/sizes — no `shape_hash`, color value, or coordinate literal, so it is
    invariant to `shape_hash` relabeling (NFR-111).
    """
    if not objset.objects:
        return None
    color_counts: dict[int, int] = {}
    for o in objset.objects:
        color_counts[o.color] = color_counts.get(o.color, 0) + 1
    uniques = [o for o in objset.objects if color_counts[o.color] == 1]
    pool = uniques or list(objset.objects)
    # Smallest-by-size first (avatars are small); deterministic geometric tie-break.
    return min(pool, key=lambda o: (o.size, o.bbox[0], o.bbox[1], o.color))


def _cardinality_unique(
    objset: ObjectSet, context: Optional["SelectionContext"] = None
) -> Optional[GridObject]:
    """The object whose SHAPE-CLASS count is exactly 1 (the former SINGLETON).

    Instance-invariant because it is a COUNT, not an identity (N-02): an object is
    cardinality-unique iff no other object shares its `shape_hash`. If several shapes
    are unique we take the largest (most salient); None if every shape repeats.
    Relabeling `shape_hash` permutes which VALUE is unique but not WHETHER a count is
    1, so the selected object is unchanged (NFR-111).
    """
    shape_counts: dict[int, int] = {}
    for o in objset.objects:
        shape_counts[o.shape_hash] = shape_counts.get(o.shape_hash, 0) + 1
    singles = [o for o in objset.objects if shape_counts[o.shape_hash] == 1]
    if not singles:
        return None
    return max(singles, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1], o.color))


def _largest(
    objset: ObjectSet, context: Optional["SelectionContext"] = None
) -> Optional[GridObject]:
    """The largest object by cell count (size); deterministic geometric tie-break."""
    if not objset.objects:
        return None
    return max(objset.objects, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1], o.color))


def _smallest(
    objset: ObjectSet, context: Optional["SelectionContext"] = None
) -> Optional[GridObject]:
    """The smallest object by cell count (size); deterministic geometric tie-break."""
    if not objset.objects:
        return None
    return min(objset.objects, key=lambda o: (o.size, o.bbox[0], o.bbox[1], o.color))


def _is_background(obj: GridObject) -> bool:
    """True iff `obj` spans >= BACKGROUND_SPAN of BOTH board dimensions (FR-184).

    The board field/frame (solid or ring) spans most of the screen; a manipulable token spans
    at most one dimension. Pure geometry over the fixed GRID_SIZE board — relabel-invariant, no
    color/coordinate/shape_hash literal.
    """
    r0, c0, r1, c1 = obj.bbox
    span = BACKGROUND_SPAN * GRID_SIZE
    return (r1 - r0 + 1) >= span and (c1 - c0 + 1) >= span


def _salient(
    objset: ObjectSet, context: Optional["SelectionContext"] = None
) -> Optional[GridObject]:
    """The largest NON-background object — the most prominent manipulable token (FR-185).

    Excludes the board field/frame (`_is_background`) so goal TARGETS point at tokens, not the
    background (the trace-study Finding A vacuity). Largest by the existing deterministic,
    shape_hash-invariant tie-break; None if every object is background or the set is empty.
    """
    pool = [o for o in objset.objects if not _is_background(o)]
    if not pool:
        return None
    return max(pool, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1], o.color))


_SELECTOR_FNS: dict[str, Callable[..., Optional[GridObject]]] = {
    "controllable": _controllable,
    "cardinality_unique": _cardinality_unique,
    "largest": _largest,
    "smallest": _smallest,
    "salient": _salient,
}


def select_role(
    name: str, objset: ObjectSet, context: Optional["SelectionContext"] = None
) -> Optional[GridObject]:
    """Resolve a role SELECTOR name to a concrete object (FR-147), or None.

    The public, instance-invariant binding used by abduction. Returns the object the
    named selector picks; None when the selector cannot bind (e.g. no cardinality-
    unique object). The optional `context` (FR-170) grounds the `controllable` selector
    in observed dynamics (FR-168); the structural selectors ignore it. No
    `shape_hash`/color/coordinate literal influences the choice.
    """
    fn = _SELECTOR_FNS.get(name)
    return fn(objset, context) if fn is not None else None


# --- relation primitives -------------------------------------------------------


def _bbox_overlap(a: GridObject, b: GridObject) -> bool:
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _relation_holds(name: str, a: GridObject, b: GridObject) -> bool:
    """Whether the named relation holds between two concrete objects."""
    if name == "overlaps":
        return _bbox_overlap(a, b)
    if name == "adjacent":
        # Bounding boxes touch or overlap (Chebyshev gap <= 1 between boxes).
        ar0, ac0, ar1, ac1 = a.bbox
        br0, bc0, br1, bc1 = b.bbox
        row_gap = max(0, max(br0 - ar1, ar0 - br1))
        col_gap = max(0, max(bc0 - ac1, ac0 - bc1))
        return max(row_gap, col_gap) <= 1
    if name == "left_of":
        return a.centroid[1] < b.centroid[1]
    if name == "above":
        return a.centroid[0] < b.centroid[0]
    if name == "same_row":
        return abs(a.centroid[0] - b.centroid[0]) <= 1
    if name == "same_col":
        return abs(a.centroid[1] - b.centroid[1]) <= 1
    return False


# Chebyshev relations use a max-of-axes lower bound; the rest use Manhattan (N-03).
_CHEBYSHEV_RELATIONS = frozenset({"adjacent"})


def _bbox_gap(a: GridObject, b: GridObject) -> tuple[int, int]:
    """Per-axis empty-cell GAP between two bounding boxes (0 if they touch/overlap)."""
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox
    row_gap = max(0, br0 - ar1, ar0 - br1)
    col_gap = max(0, bc0 - ac1, ac0 - bc1)
    return int(row_gap), int(col_gap)


def _centroid_lower_bound(name: str, a: GridObject, b: GridObject) -> int:
    """Admissible lower bound on the MOVES to make `name` hold between a and b (N-03).

    We bound by the empty-cell GAP between the two bounding boxes, NOT the raw centroid
    distance: with EXTENDED objects two bboxes overlap before their centroids coincide,
    so a centroid bound would OVERESTIMATE (break admissibility, SC-124). The bbox gap is
    exactly the number of cells one object must close on each axis, and each move closes
    at most one cell per axis — so it is a provable lower bound on the action count.

      * overlap: Manhattan sum of the per-axis gaps (must close BOTH axes to overlap).
      * adjacent: Chebyshev max of the per-axis gaps (touching needs the larger axis
        closed to within 1 cell; gap already excludes the touching cell), floored at 0.

    Both satisfy `h(s) <= true_cost(s)`.
    """
    # same_row/same_col are centroid ±1-BAND relations (not bbox-touch): bound by the SINGLE-axis
    # centroid gap minus the band, NOT the bbox gap (FR-188; the bbox gap overestimates -> would
    # be inadmissible). ceil keeps it an admissible integer lower bound under the module's
    # unit-move convention (each move shifts the centroid by <= 1 per axis, as overlap/adjacent
    # also assume); tight when the target is static.
    if name == "same_row":
        return max(0, math.ceil(abs(a.centroid[0] - b.centroid[0]) - 1.0))
    if name == "same_col":
        return max(0, math.ceil(abs(a.centroid[1] - b.centroid[1]) - 1.0))
    row_gap, col_gap = _bbox_gap(a, b)
    if name in _CHEBYSHEV_RELATIONS:
        return max(0, max(row_gap, col_gap))
    return row_gap + col_gap


@dataclass(frozen=True, slots=True)
class GoalPredicate:
    """Conjunction of (relation_name, roleA, roleB) relations (FR-130, FR-147).

    Roles are bound in ONE of two ways:
      * `selectors` empty (legacy / reactive path): role ids are shape-class ranks,
        resolved by `role_of` (FR-130). Used by `goal_inference.from_win`.
      * `selectors` set (abduction path): `selectors[i]` is the SELECTOR name that binds
        role id `i` to a concrete object via `select_role` (FR-147). The binding is
        instance-invariant and recomputed from structure each frame, so it is INVARIANT
        to `shape_hash` relabeling (NFR-111). No coordinate/color/shape_hash literal.
    """

    relations: tuple[tuple[str, int, int], ...]
    # Selector name per role id (abduction). Empty => legacy shape-class-rank roles.
    selectors: tuple[str, ...] = ()

    # --- role resolution -------------------------------------------------------

    def _objs_for_role(
        self, role: int, objset: ObjectSet,
        context: Optional["SelectionContext"] = None,
    ) -> list[GridObject]:
        """All concrete objects bound to `role` for the active binding scheme.

        The optional `context` (FR-170) grounds selector binding in observed dynamics; it
        only affects the `controllable` selector and is ignored on the legacy rank path.
        """
        if self.selectors:
            obj = select_role(self.selectors[role], objset, context)
            return [obj] if obj is not None else []
        return _roles_present(objset).get(role, [])

    # --- predicate evaluation --------------------------------------------------

    def holds(
        self, objset: ObjectSet, context: Optional["SelectionContext"] = None
    ) -> bool:
        """True iff EVERY relation holds for SOME bound instance pair of its roles.

        Empty predicate => vacuously satisfied (degenerate goal). Existential over the
        objects each role binds to (a single object under a selector; possibly several
        under the legacy rank) so it generalizes across instance counts (L-3). `context`
        (FR-170) grounds the `controllable` binding when present.
        """
        for name, ra, rb in self.relations:
            objs_a = self._objs_for_role(ra, objset, context)
            objs_b = self._objs_for_role(rb, objset, context)
            if not any(
                _relation_holds(name, a, b)
                for a in objs_a
                for b in objs_b
                if a is not b
            ):
                return False
        return True

    def distance(
        self, objset: ObjectSet, context: Optional["SelectionContext"] = None
    ) -> int:
        """Admissible distance-to-goal heuristic (>= 0, never overestimates).

        Two regimes, both admissible (`h(s) <= true_cost(s)`, SC-124):

          * ABDUCTION (selectors set, N-03): for each UNSATISFIED relation take the best
            (minimum) centroid lower bound over its candidate pairs; the predicate
            heuristic is the **MAX over per-relation sub-goal distances, NOT the sum** —
            the sub-goals may be satisfiable concurrently, so the max stays a provable
            lower bound on the joint cost.
          * LEGACY (no selectors, FR-130 reactive path): the COUNT of unsatisfied
            relations — each unmet relation needs at least one transition, so the count
            never overestimates. Kept for the win-diff verifier and its tests.

        `context` (FR-170) grounds the `controllable` binding on the abduction path.
        """
        if self.selectors:
            return self._centroid_distance(objset, context)
        return self._unmet_count(objset)

    def _centroid_distance(
        self, objset: ObjectSet, context: Optional["SelectionContext"] = None
    ) -> int:
        """Max over per-relation centroid lower bounds (abduction path, N-03)."""
        worst = 0
        for name, ra, rb in self.relations:
            objs_a = self._objs_for_role(ra, objset, context)
            objs_b = self._objs_for_role(rb, objset, context)
            pairs = [(a, b) for a in objs_a for b in objs_b if a is not b]
            if not pairs:
                worst = max(worst, 1)  # role unbound: >= 1 move; admissible
                continue
            if any(_relation_holds(name, a, b) for a, b in pairs):
                continue  # sub-goal already holds: contributes 0
            sub = min(_centroid_lower_bound(name, a, b) for a, b in pairs)
            worst = max(worst, max(1, sub))  # unmet costs >= 1 even if bound rounds to 0
        return worst

    def _unmet_count(self, objset: ObjectSet) -> int:
        """Count of currently-unsatisfied relations (legacy admissible heuristic)."""
        if self.holds(objset):
            return 0
        unmet = 0
        for name, ra, rb in self.relations:
            objs_a = self._objs_for_role(ra, objset)
            objs_b = self._objs_for_role(rb, objset)
            if not any(
                _relation_holds(name, a, b)
                for a in objs_a
                for b in objs_b
                if a is not b
            ):
                unmet += 1
        return unmet


# --- ADR-0002: the v1 GoalTemplate library (COINCIDE, REACH) -------------------

TemplateName = Literal["coincide", "reach", "align"]


@dataclass(frozen=True, slots=True)
class GoalTemplate:
    """A GENERAL, parametric goal pattern from core-knowledge priors only (FR-150).

    `instantiate(objset) -> list[GoalPredicate]` binds parametric ROLES via instance-
    invariant SELECTORS (FR-147) and returns candidate predicates (top-k role pairs,
    bounded — N-05). `distance(objset, predicate) -> int` is the ADMISSIBLE centroid
    lower bound (max not sum over sub-goals, N-03); it delegates to `GoalPredicate.
    distance`. References ROLES/RELATIONS only — no color/coordinate/shape_hash literal.
    """

    name: TemplateName
    # Accept an optional trailing SelectionContext (FR-170); `...` keeps the hint valid
    # for both the legacy 1-arg call and the v0.7 context-threaded call.
    instantiate: Callable[..., list[GoalPredicate]]
    distance: Callable[..., int]


@dataclass(frozen=True, slots=True)
class GoalHypothesisLibrary:
    """The fixed, bounded set of the TWO v1 GoalTemplates (FR-150). Immutable data."""

    templates: tuple[GoalTemplate, ...]


# Candidate selector pairs, ordered by PRIOR STRENGTH (deterministic, N-05/N-10).
# Both operands instance-invariant; the cardinality-unique selector subsumes SINGLETON.
_COINCIDE_PAIRS: tuple[tuple[str, str], ...] = (
    # M1 (FR-186): the prominent NON-background token is the TOP target (placed first so it
    # survives the top-k cap rather than being crowded out by the background-binding pairs).
    ("controllable", "salient"),
    ("controllable", "cardinality_unique"),
    ("controllable", "largest"),
    ("smallest", "largest"),
    ("cardinality_unique", "largest"),
)
# REACH always binds operand A to the CONTROLLABLE object (FR-150); `salient` is the TOP target (FR-186).
_REACH_TARGETS: tuple[str, ...] = ("salient", "cardinality_unique", "largest", "smallest")

# Per-template instantiation is bounded to the top-k role pairs (N-05).
_INSTANTIATE_TOP_K = 3


def _distinct_binding(
    sa: str, sb: str, objset: ObjectSet,
    context: Optional["SelectionContext"] = None,
) -> bool:
    """True iff both selectors bind, to DISTINCT objects (drops vacuous pairs, N-11).

    `context` (FR-170) grounds the `controllable` operand so instantiation drops a pair
    whose grounded controllable coincides with the other role.
    """
    a = select_role(sa, objset, context)
    b = select_role(sb, objset, context)
    return a is not None and b is not None and a is not b


def _instantiate_coincide(
    objset: ObjectSet, context: Optional["SelectionContext"] = None
) -> list[GoalPredicate]:
    """COINCIDE: make role-A overlap role-B (FR-150). Top-k distinct selector pairs."""
    out: list[GoalPredicate] = []
    for sa, sb in _COINCIDE_PAIRS:
        if not _distinct_binding(sa, sb, objset, context):
            continue
        out.append(GoalPredicate(relations=(("overlaps", 0, 1),), selectors=(sa, sb)))
        if len(out) >= _INSTANTIATE_TOP_K:
            break
    return out


def _instantiate_reach(
    objset: ObjectSet, context: Optional["SelectionContext"] = None
) -> list[GoalPredicate]:
    """REACH: bring the CONTROLLABLE object adjacent to a target role (FR-150)."""
    out: list[GoalPredicate] = []
    for sb in _REACH_TARGETS:
        if not _distinct_binding("controllable", sb, objset, context):
            continue
        out.append(
            GoalPredicate(relations=(("adjacent", 0, 1),), selectors=("controllable", sb))
        )
        if len(out) >= _INSTANTIATE_TOP_K:
            break
    return out


def _template_distance(
    objset: ObjectSet, predicate: GoalPredicate,
    context: Optional["SelectionContext"] = None,
) -> int:
    """Admissible template distance: delegate to the predicate's max-bound (N-03)."""
    return predicate.distance(objset, context)


COINCIDE = GoalTemplate(
    name="coincide", instantiate=_instantiate_coincide, distance=_template_distance
)
REACH = GoalTemplate(
    name="reach", instantiate=_instantiate_reach, distance=_template_distance
)


# ALIGN targets non-background tokens ONLY (FR-187, review M-1): never `cardinality_unique`,
# which binds the board background in ls20 (trace-study Finding A).
_ALIGN_TARGETS: tuple[str, ...] = ("salient",)


def _instantiate_align(
    objset: ObjectSet, context: Optional["SelectionContext"] = None
) -> list[GoalPredicate]:
    """ALIGN: share a row OR a column between the CONTROLLABLE object and a target (FR-187).

    A predicate is a conjunction, so the OR is expressed as TWO candidate predicates per pair
    (one `same_row`, one `same_col`) — like REACH's multiple targets. Drops non-distinct/unbound
    pairs via `_distinct_binding` (an unbound controllable or a None salient yields nothing, no
    error — review F10).
    """
    out: list[GoalPredicate] = []
    for sb in _ALIGN_TARGETS:
        if not _distinct_binding("controllable", sb, objset, context):
            continue
        out.append(
            GoalPredicate(relations=(("same_row", 0, 1),), selectors=("controllable", sb))
        )
        out.append(
            GoalPredicate(relations=(("same_col", 0, 1),), selectors=("controllable", sb))
        )
        if len(out) >= _INSTANTIATE_TOP_K:
            break
    return out


ALIGN = GoalTemplate(
    name="align", instantiate=_instantiate_align, distance=_template_distance
)


def default_library() -> GoalHypothesisLibrary:
    """The fixed library: COINCIDE, REACH, ALIGN (prior-strength order, N-10)."""
    return GoalHypothesisLibrary(templates=(COINCIDE, REACH, ALIGN))
