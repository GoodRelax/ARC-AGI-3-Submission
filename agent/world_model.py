"""[Entity] Object-centric forward world model: WorldModel, TransitionRule, Hypothesis.

This is a pure, immutable domain core (functional core / imperative shell). Learning
returns a NEW `WorldModel` (FR-115); `predict()` is a pure function of
`(WorldModel, ObjectSet, ActionKey)` (NFR-105). Per the Clean-Architecture rule
(spec §3.1) this Entity module MUST NOT import the framework (`agents` / `arcengine`):
it depends only on stdlib + the reused Phase A Entity modules (`segment`,
`state_graph`).

Design notes (WHY):
  * **Roles, not coordinates (C-1, FR-110, FR-147).** A rule's precondition is keyed
    by an object's `shape_hash` (a translation-invariant SHAPE class), never by an
    absolute coordinate, `game_id`, or color-as-meaning. Effects are expressed as
    RELATIVE deltas (translation vector, target color, size delta) so a learned rule
    generalizes to a new instance of the same shape elsewhere on the grid.
  * **Deterministic-first (ADR-005, FR-113).** We assume one effect per (action,
    precondition). A conflicting effect under the same precondition flags the rule
    `nondeterministic` rather than modeling a distribution.
  * **Falsify, don't elaborate (ADR-006, FR-112).** Competing hypotheses per action
    are kept side by side; a hypothesis is pruned once its contradiction RATE exceeds
    `CONTRADICTION_PRUNE_TAU` (F-10). Prediction uses the highest-support consistent
    hypothesis.
  * **Compose across all matched objects (F-05, FR-114).** `predict()` applies each
    matched object's best effect simultaneously and assembles ONE successor
    `ObjectSet`, so multi-object successor states (needed by relational goals) are
    representable. The successor `node_hash` is read directly from the rebuilt
    `ObjectSet.signature` (F-11) — the hash lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional

from agent.goal import Affordance, Identity  # v0.7: affordance_map output types (goal never imports us)
from agent.segment import GridObject, ObjectSet, _build_object, _normalize, _signature
from agent.state_graph import ActionKey

__all__ = [
    "EffectKind",
    "Precondition",
    "Effect",
    "Hypothesis",
    "TransitionRule",
    "WorldModel",
    "CONTRADICTION_PRUNE_TAU",
    "correspond_and_diff",
    "affordance_map",
]

# Prune a hypothesis once contradictions/(support+contradictions) > tau (F-10). 0.5
# means "pruned once it has been wrong at least as often as right". Module-level
# constant: no concrete game value (NFR-103).
CONTRADICTION_PRUNE_TAU: float = 0.5

# RELATIVE effect kinds (no absolute coordinates). `noop` = the matched object did not
# change; kept explicit so a "this action does nothing to this shape" fact is learnable.
EffectKind = Literal["translate", "recolor", "resize", "appear", "disappear", "noop"]


@dataclass(frozen=True, slots=True)
class Precondition:
    """A pattern over object ATTRIBUTES/RELATIONS, never absolute coordinates (C-1, FR-110).

    `role` is a stable, instance-invariant object class — the object's `shape_hash`
    (translation-invariant geometry). `attrs` carries additional bucketed attribute
    constraints (e.g. color). We deliberately do NOT store position here, so a rule
    learned for one instance of a shape applies to another instance elsewhere.

    `guard` is a LOCAL relational guard over the object's NEIGHBOURHOOD (FR-160..FR-163,
    v0.6): e.g. `("dest_blocked", 1)` means the region the object would move into under
    this action is occupied or out of bounds in the BEFORE state. WHY a guard at all: it
    is what distinguishes a FREE move (`dest_blocked=0` -> translate) from a BLOCKED move
    (`dest_blocked=1` -> noop). The SAME shape at a wall vs. in open space therefore has a
    DIFFERENT precondition, so the two learn different effects WITHOUT colliding into one
    "non-deterministic" rule (FR-161). Derived from geometry/occupancy only — never a
    literal "wall" colour or coordinate (C-1, FR-110).
    """

    role: int  # shape_hash class — instance-invariant, NOT a coordinate
    attrs: frozenset[tuple[str, int]]  # (attribute_name, bucketed_value) pairs
    guard: frozenset[tuple[str, int]] = frozenset()  # local relational guard (FR-160)


@dataclass(frozen=True, slots=True)
class Effect:
    """A relative transformation applied to a matched object (FR-110).

    params are RELATIVE: translate -> (dr, dc); recolor -> (new_color,);
    resize -> (d_size,); appear/disappear/noop -> ().
    """

    kind: EffectKind
    params: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One candidate explanation of an action's effect on a precondition (Glossary).

    support / contradictions are the confirming / falsifying observation counts
    (FR-111). `nondeterministic` is latched on conflicting effects (FR-113).
    """

    action: ActionKey
    precondition: Precondition
    effect: Effect
    support: int = 0
    contradictions: int = 0
    nondeterministic: bool = False

    def contradiction_rate(self) -> float:
        """contradictions / (support + contradictions); 0.0 when never observed."""
        total = self.support + self.contradictions
        return self.contradictions / total if total else 0.0


@dataclass(frozen=True, slots=True)
class TransitionRule:
    """The competing hypotheses for one action class (FR-112)."""

    action: ActionKey
    hypotheses: tuple[Hypothesis, ...]

    def best_for(self, precondition: Precondition) -> Optional[Hypothesis]:
        """Highest-support CONSISTENT hypothesis matching `precondition` (FR-114).

        Consistent = not flagged non-deterministic and not over the prune threshold.
        Ties broken deterministically by `(precondition.role, effect.kind, params)`
        so prediction is reproducible (NFR-104).
        """
        candidates = [
            h
            for h in self.hypotheses
            if h.precondition == precondition
            and not h.nondeterministic
            and h.contradiction_rate() <= CONTRADICTION_PRUNE_TAU
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda h: (
                h.support,
                -h.contradictions,
                # deterministic tie-break (no randomness): fixed effect ordering
                h.effect.kind,
                str(h.effect.params),
            ),
        )

    def best_translate_vector(
        self, role: int, attrs: Optional[frozenset[tuple[str, int]]] = None
    ) -> Optional[tuple[int, int]]:
        """Displacement of the highest-support TRANSLATE hypothesis for the object (FR-162).

        WHY: the no-op guard needs the object's would-be displacement under this action
        to know whether the destination region is blocked. We take it from the
        highest-support translate hypothesis already learned for this object's identity
        (`role` = shape_hash AND `attrs` = bucketed color/etc.) — computed BEFORE
        selecting the noop-vs-translate rule (FR-162). Matching on `attrs` too keeps two
        objects that share a shape but differ in colour from merging their move vectors.
        `None` if no translate hypothesis exists yet (bootstrap) -> the guard is undefined
        and no no-op prediction is made (FR-160, H-2). Deterministic tie-break.
        """
        candidates = [
            h
            for h in self.hypotheses
            if h.precondition.role == role
            and (attrs is None or h.precondition.attrs == attrs)
            and h.effect.kind == "translate"
            and not h.nondeterministic
            and h.contradiction_rate() <= CONTRADICTION_PRUNE_TAU
        ]
        if not candidates:
            return None
        best = max(
            candidates,
            key=lambda h: (h.support, -h.contradictions, str(h.effect.params)),
        )
        dr, dc = best.effect.params
        return (int(dr), int(dc))

    def best(self) -> Optional[Hypothesis]:
        """Highest-support consistent hypothesis overall (Domain-model convenience)."""
        best: Optional[Hypothesis] = None
        for pre in {h.precondition for h in self.hypotheses}:
            cand = self.best_for(pre)
            if cand is not None and (best is None or cand.support > best.support):
                best = cand
        return best


@dataclass(frozen=True, slots=True)
class WorldModel:
    """Immutable set of competing rules; learn() returns a NEW model (FR-115)."""

    rules: tuple[TransitionRule, ...] = ()

    # --- learning (returns a new model) ---------------------------------------

    def learn(self, before: ObjectSet, action: ActionKey, after: ObjectSet) -> "WorldModel":
        """Attribute observed object deltas to `action` as hypotheses (FR-102).

        Pure functional update: builds and returns a NEW `WorldModel`. The body is the
        canonical learning algorithm (spec §4.3a). O(objects × active hypotheses) with
        no full-history rescan (NFR-102).
        """
        deltas = correspond_and_diff(before, after)

        # index_rules: ActionKey -> mutable list of hypotheses (rebuilt into a frozen
        # model at the end). Only the local lists are mutated; the input model is not.
        rules: dict[ActionKey, list[Hypothesis]] = {
            r.action: list(r.hypotheses) for r in self.rules
        }
        hyps = rules.setdefault(action, [])

        # Precondition-aware no-op learning (ADR-011, FR-160/161): enrich move-class
        # deltas (translate / noop) with a `dest_blocked` guard so a blocked move and a
        # free move are DISTINCT deterministic rules. The displacement comes from the
        # highest-support translate hypothesis for this (role, action) — taken from the
        # EXISTING model AND from this batch's own translate deltas, so a free move (the
        # translate we just saw) immediately seeds the vector that lets the next blocked
        # move at a wall be recognised (FR-162). Objects with no learned translate vector
        # stay guardless (FR-160): no no-op prediction, just learn.
        deltas = _guard_move_deltas(deltas, before, action, hyps)

        for pre, eff in deltas:
            match_idx = _find_idx(hyps, lambda h: h.precondition == pre)
            if match_idx is None:
                # New hypothesis for an unseen precondition (FR-110).
                hyps.append(Hypothesis(action, pre, eff, support=1))
                continue

            match = hyps[match_idx]
            if match.nondeterministic:
                # Already known stochastic: just record the alternative without
                # re-deriving a distribution (ADR-005). Keep/refresh an alt hyp.
                _ensure_alt(hyps, action, pre, eff)
                continue

            if match.effect == eff:  # confirmation (FR-111)
                hyps[match_idx] = replace(match, support=match.support + 1)
            else:  # contradiction (FR-111, FR-113)
                bumped = replace(
                    match,
                    contradictions=match.contradictions + 1,
                    nondeterministic=True,  # conflicting effects -> non-deterministic
                )
                if bumped.contradiction_rate() > CONTRADICTION_PRUNE_TAU:
                    # Falsify: drop the now-inconsistent hypothesis (FR-112). The
                    # alternative effect is recorded below so the model still learns.
                    del hyps[match_idx]
                else:
                    hyps[match_idx] = bumped
                _ensure_alt(hyps, action, pre, eff)

        new_rules = tuple(
            TransitionRule(act, tuple(hs)) for act, hs in rules.items()
        )
        return WorldModel(rules=new_rules)

    # --- prediction (pure) ----------------------------------------------------

    def predict(self, objset: ObjectSet, action: ActionKey) -> Optional[ObjectSet]:
        """Compose effects of ALL matched objects/rules (FR-114, F-05).

        Returns the successor `ObjectSet` (its `node_hash` is `successor.signature`,
        F-11) or `None` (unknown) if no object/rule matches — the caller treats
        `None` as maximally novel (FR-127).
        """
        rule = self._rule_for(action)
        if rule is None:
            return None

        effects_by_index: dict[int, Effect] = {}
        for i, obj in enumerate(objset.objects):
            eff = self._predict_object(rule, obj, objset)
            if eff is not None:
                effects_by_index[i] = eff

        if not effects_by_index:
            return None  # unknown (FR-127)

        return _apply_effects(objset, effects_by_index)

    def _predict_object(
        self, rule: TransitionRule, obj: GridObject, objset: ObjectSet
    ) -> Optional[Effect]:
        """Prediction-time guard resolution for one object (FR-162), deterministic.

        Order (BEFORE selecting noop-vs-translate): (1) find the highest-support
        translate displacement for this role; (2) if NONE exists yet, the guard is
        undefined for a move object — but a non-move object may still match a guardless
        rule (recolor/appear/etc.), so fall back to the guardless precondition; (3)
        otherwise compute the live `dest_blocked` guard from that vector vs `objset`
        geometry and select the precondition (free -> translate, blocked -> noop).

        WHY the bootstrap fallback: before any translate is learned, a blocked move's
        guard is undefined, so we make NO no-op prediction (H-2) — but we must still
        predict learned non-move effects, so we try the guardless rule too.
        """
        vec = rule.best_translate_vector(obj.shape_hash, _attrs_of(obj))
        if vec is None:
            # Bootstrap / non-move object: no translate vector -> guard undefined, so we
            # make NO no-op prediction (FR-160, FR-162 H-2). We still predict a guardless
            # NON-noop effect (recolor/appear/...) — those don't depend on a move guard.
            pre = _precondition_for(obj)
            hyp = rule.best_for(pre)
            if hyp is None or hyp.nondeterministic or hyp.effect.kind == "noop":
                return None  # undefined guard -> no no-op prediction; just learn (FR-160)
            return hyp.effect
        # Move object: resolve the guard, then select the matching guarded precondition.
        pre = _precondition_for(obj, objset, vec)
        hyp = rule.best_for(pre)
        if hyp is not None and not hyp.nondeterministic:
            return hyp.effect
        # The resolved (guarded) precondition has no learned rule. WHY fall back to the
        # FREE move rule: a `dest_blocked=1` resolution means the destination is merely
        # OCCUPIED/out-of-bounds by geometry, but unless a blocked-noop was actually
        # OBSERVED there we have no evidence the object stops — treating every untested
        # occupied destination as a wall would make legitimately reachable overlaps
        # (COINCIDE goals, FR-150) unpredictable. So we predict the learned translate; a
        # real wall-hit then teaches the blocked-noop rule (SC-128), after which this
        # branch selects it. Out-of-the-grid moves are still clamped by _transform_object.
        free_pre = Precondition(
            role=pre.role, attrs=pre.attrs, guard=frozenset({("dest_blocked", 0)})
        )
        free = rule.best_for(free_pre)
        return free.effect if free is not None and not free.nondeterministic else None

    def has_nondeterministic_match(self, objset: ObjectSet, action: ActionKey) -> bool:
        """True if any matched rule on this state/action is flagged non-deterministic.

        Used by the trust predicate (§1.8): a state touching a non-deterministic rule
        is not safely plannable (FR-113 -> FR-140).
        """
        rule = self._rule_for(action)
        if rule is None:
            return False
        for obj in objset.objects:
            # A state is unsafe to plan through if ANY hypothesis matching this object's
            # IDENTITY (role + attrs) is flagged stochastic — regardless of guard. We
            # match on identity, not the full guarded precondition, because a conflicting
            # effect is recorded as a nondeterministic hypothesis under the SAME guard,
            # and `best_translate_vector` (which skips nondeterministic hyps) would
            # otherwise hide it. Geometry-derived guards never make a genuinely stochastic
            # rule look safe (FR-113 -> FR-140).
            attrs = _attrs_of(obj)
            if any(
                h.precondition.role == obj.shape_hash
                and h.precondition.attrs == attrs
                and h.nondeterministic
                for h in rule.hypotheses
            ):
                return True
        return False

    def matches(self, objset: ObjectSet, action: ActionKey) -> bool:
        """True if at least one rule/object matches (start-state trust check, §1.8)."""
        return self.predict(objset, action) is not None

    def max_contradiction_rate(self) -> float:
        """Worst contradiction rate across all hypotheses (trust predicate, §1.8)."""
        rates = [
            h.contradiction_rate() for r in self.rules for h in r.hypotheses
        ]
        return max(rates) if rates else 0.0

    def _rule_for(self, action: ActionKey) -> Optional[TransitionRule]:
        for r in self.rules:
            if r.action == action:
                return r
        return None


# --- module-level pure helpers -------------------------------------------------


def _precondition_for(
    obj: GridObject,
    objset: Optional[ObjectSet] = None,
    translate_vec: Optional[tuple[int, int]] = None,
) -> Precondition:
    """Instance-invariant precondition for an object (FR-147, FR-160).

    Keyed by `shape_hash` (translation-invariant geometry) plus a bucketed color
    attribute. No coordinate, no game_id, no color-as-meaning literal.

    When `translate_vec` is given (the object has a learned move rule for this
    `(role, action)`), attach a definite `dest_blocked` guard (FR-161): every
    move-class precondition carries `("dest_blocked", 0|1)`, so a free move (translate)
    and a blocked move (noop) are DISTINCT deterministic rules that never collide.
    With no learned displacement the precondition is GUARDLESS (FR-160): the guard is
    undefined, so no no-op prediction is made — we simply learn from the observation.
    The guard is geometry/occupancy only (FR-163) — never a colour/coordinate literal.
    """
    if translate_vec is None or objset is None:
        guard: frozenset[tuple[str, int]] = frozenset()  # guardless (FR-160)
    else:
        blocked = 1 if _dest_blocked(obj, objset, translate_vec) else 0
        guard = frozenset({("dest_blocked", blocked)})  # always-guarded move (FR-161)
    return Precondition(
        role=obj.shape_hash,
        attrs=frozenset({("color", obj.color)}),
        guard=guard,
    )


def _dest_blocked(
    obj: GridObject, objset: ObjectSet, translate_vec: tuple[int, int]
) -> bool:
    """True iff the region `obj` would occupy after `translate_vec` is blocked (FR-163).

    Blocked == any destination cell falls OUT of the 0..63 grid bounds, OR overlaps
    another object's cells in `objset`. Pure geometry/occupancy — no colour/coordinate
    literal (C-1, NFR-115). Instance-invariant: depends only on the object's own cells,
    the displacement, and the occupancy of OTHER objects, so the same role meeting the
    same neighbourhood elsewhere resolves identically (FR-163).
    """
    dr, dc = translate_vec
    dest = {(r + dr, c + dc) for (r, c) in obj.cells}
    # (a) out of bounds.
    if any(r < 0 or r > 63 or c < 0 or c > 63 for (r, c) in dest):
        return True
    # (b) occupied by ANOTHER object (exclude this object's own current footprint so a
    #     partial self-overlap during a slide does not count as blocked).
    own = obj.cells
    others: set[tuple[int, int]] = set()
    for other in objset.objects:
        if other is obj or other.cells == own:
            continue
        others |= other.cells
    return bool(dest & others)


_Ident = tuple[int, frozenset[tuple[str, int]]]  # (role, attrs) object identity


def _attrs_of(obj: GridObject) -> frozenset[tuple[str, int]]:
    """The bucketed attribute set used in a precondition (kept in ONE place)."""
    return frozenset({("color", obj.color)})


def _known_translate_vectors(
    hyps: list[Hypothesis], batch: list[tuple[Precondition, Effect]]
) -> dict[_Ident, tuple[int, int]]:
    """Highest-support translate displacement per object identity (FR-162).

    Combines translate hypotheses already in the model with the translate deltas just
    observed this turn, so a FREE move immediately seeds the vector used to recognise a
    later BLOCKED move at a wall. Keyed by (role, attrs) so two objects sharing a shape
    but differing in colour do not merge their vectors. Deterministic: ties break by
    larger support then a fixed param order. WHY include the batch: bootstrapping the
    first transition, where the model has no rule yet but the observed translate IS the
    displacement.
    """
    scored: dict[_Ident, tuple[int, str]] = {}   # ident -> (support, params-str) tiebreak
    out: dict[_Ident, tuple[int, int]] = {}

    def consider(ident: _Ident, support: int, dr: int, dc: int) -> None:
        key = (support, f"{dr},{dc}")
        if ident not in scored or key > scored[ident]:
            scored[ident] = key
            out[ident] = (dr, dc)

    for h in hyps:
        if h.effect.kind == "translate" and not h.nondeterministic:
            dr, dc = h.effect.params
            ident = (h.precondition.role, h.precondition.attrs)
            consider(ident, h.support, int(dr), int(dc))
    for pre, eff in batch:
        if eff.kind == "translate":
            dr, dc = eff.params
            consider((pre.role, pre.attrs), 1, int(dr), int(dc))
    return out


def _guard_move_deltas(
    deltas: list[tuple[Precondition, Effect]],
    before: ObjectSet,
    action: ActionKey,
    hyps: list[Hypothesis],
) -> list[tuple[Precondition, Effect]]:
    """Attach a `dest_blocked` guard to move-class deltas (FR-160, FR-161).

    A delta is move-class iff its effect is `translate` OR `noop` AND its role has a
    known translate vector (from the model or this batch). For those, recompute the
    precondition with the geometry-derived guard so free (0) and blocked (1) become two
    distinct deterministic rules. Non-move objects (no learned translate vector) keep
    their guardless precondition (FR-160). Pure: returns a NEW delta list.
    """
    vectors = _known_translate_vectors(hyps, deltas)
    # Map identity (role, attrs) -> the before GridObject (for geometry). If several
    # objects share an identity we use the first (deterministic by normalized order).
    obj_by_ident: dict[_Ident, GridObject] = {}
    for o in before.objects:
        obj_by_ident.setdefault((o.shape_hash, _attrs_of(o)), o)

    out: list[tuple[Precondition, Effect]] = []
    for pre, eff in deltas:
        ident = (pre.role, pre.attrs)
        if eff.kind in ("translate", "noop") and ident in vectors:
            obj = obj_by_ident.get(ident)
            if obj is not None:
                pre = _precondition_for(obj, before, vectors[ident])
        out.append((pre, eff))
    return out


def _find_idx(hyps: list[Hypothesis], pred) -> Optional[int]:  # type: ignore[no-untyped-def]
    """Index of the first hypothesis satisfying `pred`, else None (deterministic)."""
    for i, h in enumerate(hyps):
        if pred(h):
            return i
    return None


def _ensure_alt(
    hyps: list[Hypothesis], action: ActionKey, pre: Precondition, eff: Effect
) -> None:
    """Record/refresh an alternative effect hypothesis for a precondition (FR-113).

    Marked non-deterministic from birth, because it co-exists with a conflicting
    effect under the same precondition. Idempotent on (pre, eff).
    """
    idx = _find_idx(hyps, lambda h: h.precondition == pre and h.effect == eff)
    if idx is None:
        hyps.append(
            Hypothesis(
                action, pre, eff, support=1, nondeterministic=True
            )
        )
    else:
        h = hyps[idx]
        hyps[idx] = replace(h, support=h.support + 1, nondeterministic=True)


def correspond_and_diff(
    before: ObjectSet, after: ObjectSet
) -> list[tuple[Precondition, Effect]]:
    """Correspond objects before->after and emit (precondition, effect) deltas (§4.3a).

    Correspondence heuristic (L-4): match each `before` object to the nearest-centroid
    `after` object of the SAME `shape_hash` first (a translate/recolor candidate),
    else the nearest `after` object regardless of shape (a recolor/resize candidate).
    Unmatched `after` objects are `appear`; unmatched `before` objects are `disappear`.

    All effects are RELATIVE (FR-110). The precondition is taken from the BEFORE
    object so a rule keys off the pre-state shape/color (instance-invariant, FR-147).
    """
    deltas: list[tuple[Precondition, Effect]] = []
    after_used: set[int] = set()

    for b in before.objects:
        j = _best_after_match(b, after, after_used)
        if j is None:
            # No correspondent: the object disappeared (FR-110).
            deltas.append((_precondition_for(b), Effect("disappear", ())))
            continue
        after_used.add(j)
        a = after.objects[j]
        deltas.append((_precondition_for(b), _effect_between(b, a)))

    for j, a in enumerate(after.objects):
        if j not in after_used:
            # New object appeared. Its precondition keys off the appeared shape so the
            # rule "this action makes shape X appear" is learnable.
            deltas.append((_precondition_for(a), Effect("appear", ())))

    return deltas


def _best_after_match(
    b: GridObject, after: ObjectSet, used: set[int]
) -> Optional[int]:
    """Nearest unused `after` object to `b`: same shape preferred, then any (L-4).

    Deterministic tie-break by (distance, index) so correspondence is reproducible.
    """
    same_shape: list[tuple[float, int]] = []
    any_shape: list[tuple[float, int]] = []
    br, bc = b.centroid
    for j, a in enumerate(after.objects):
        if j in used:
            continue
        ar, ac = a.centroid
        dist = (ar - br) ** 2 + (ac - bc) ** 2
        any_shape.append((dist, j))
        if a.shape_hash == b.shape_hash:
            same_shape.append((dist, j))
    pool = same_shape or any_shape
    if not pool:
        return None
    pool.sort(key=lambda t: (t[0], t[1]))
    return pool[0][1]


def _effect_between(b: GridObject, a: GridObject) -> Effect:
    """The relative effect turning `before` object `b` into `after` object `a`.

    Priority: translate (centroid moved, same shape) > recolor (color changed) >
    resize (size changed) > noop. Only ONE effect kind is emitted per object so the
    deterministic-first model stays simple (ADR-005).
    """
    dr = int(round(a.centroid[0] - b.centroid[0]))
    dc = int(round(a.centroid[1] - b.centroid[1]))
    if a.shape_hash == b.shape_hash and a.color == b.color:
        if (dr, dc) != (0, 0):
            return Effect("translate", (dr, dc))
        return Effect("noop", ())
    if a.color != b.color and a.size == b.size:
        return Effect("recolor", (a.color,))
    if a.size != b.size:
        return Effect("resize", (a.size - b.size,))
    # Shape changed but size/color same — treat as recolor-to-self fallback (noop-ish);
    # encode as translate of centroid so prediction at least moves the object.
    if (dr, dc) != (0, 0):
        return Effect("translate", (dr, dc))
    return Effect("noop", ())


def _apply_effects(
    objset: ObjectSet, effects_by_index: dict[int, Effect]
) -> ObjectSet:
    """Build ONE successor ObjectSet applying each matched object's effect (F-05).

    Pure: constructs new `GridObject`s (translating cells, recoloring, dropping
    disappeared objects) and re-normalizes + re-hashes via the reused Phase A helpers,
    so the successor `node_hash` is `result.signature` (F-11). Objects with no learned
    effect pass through unchanged.
    """
    survivors: list[GridObject] = []
    for i, obj in enumerate(objset.objects):
        eff = effects_by_index.get(i)
        if eff is None:
            survivors.append(obj)
            continue
        new_obj = _transform_object(obj, eff)
        if new_obj is not None:
            survivors.append(new_obj)
    normalized = _normalize(survivors)
    return ObjectSet(objects=normalized, signature=_signature(normalized))


def _transform_object(obj: GridObject, eff: Effect) -> Optional[GridObject]:
    """Apply a single relative effect to one object, returning a NEW GridObject.

    Returns None for `disappear` (object dropped from the successor set). Cells are
    clamped to the 0..63 grid so a translated object never leaves the legal range.
    """
    if eff.kind == "disappear":
        return None
    if eff.kind == "noop" or eff.kind == "appear":
        # `appear` has no pre-image object to transform here; pass through unchanged.
        return obj
    if eff.kind == "recolor":
        new_color = eff.params[0]
        return _build_object(new_color, sorted(obj.cells))
    if eff.kind == "translate":
        dr, dc = eff.params
        moved = [
            (min(63, max(0, r + dr)), min(63, max(0, c + dc)))
            for (r, c) in obj.cells
        ]
        return _build_object(obj.color, moved)
    if eff.kind == "resize":
        # Resize is approximated as identity geometry (we cannot invent cells without a
        # learned shape); the size delta is informational. Keep the object as-is so the
        # successor stays well-formed rather than fabricating cells.
        return obj
    return obj


# --- v0.7 (ADR-012): affordance map — observed dynamics for grounded controllability ---


def affordance_map(model: WorldModel) -> dict[Identity, Affordance]:
    """Distill the `WorldModel` into a per-identity affordance map (FR-167).

    Pure function of the model — emits NO real actions (NFR-117). Keyed by
    `(shape_hash, color)` so the key MATCHES `goal._controllable`'s `(o.shape_hash,
    o.color)` lookup (M1); the int color is extracted from each precondition's `attrs`
    frozenset. Only DETERMINISTIC hypotheses count (not flagged non-deterministic AND
    contradiction-rate <= tau — FR-167, M2):

      * `translate_support` sums `translate` effects under SIMPLE (movement-class,
        non-tuple) actions — the positional controllability that drives `controllable`.
      * `response_support` sums any non-`noop` effect under ANY action (incl. `ACTION6`
        clicks) — general responsiveness for the FUTURE responsive selector (FR-172).
    """
    translate: dict[Identity, int] = {}
    response: dict[Identity, int] = {}
    vanish: dict[Identity, int] = {}
    spawn: dict[Identity, int] = {}
    # Per identity: best translate vector PER action — for the `autonomous` (self-moving) test.
    vecs: dict[Identity, dict[ActionKey, tuple[int, int]]] = {}
    vsupp: dict[Identity, dict[ActionKey, int]] = {}
    for rule in model.rules:
        is_simple = not isinstance(rule.action, tuple)  # clicks are (6, x, y) tuples
        for h in rule.hypotheses:
            if h.nondeterministic or h.contradiction_rate() > CONTRADICTION_PRUNE_TAU:
                continue
            attrs = dict(h.precondition.attrs)
            if "color" not in attrs:
                continue
            ident: Identity = (h.precondition.role, int(attrs["color"]))
            kind = h.effect.kind
            if kind != "noop":
                response[ident] = response.get(ident, 0) + h.support
            if kind == "translate" and is_simple:
                translate[ident] = translate.get(ident, 0) + h.support
                s = vsupp.setdefault(ident, {})
                if rule.action not in s or h.support > s[rule.action]:
                    vecs.setdefault(ident, {})[rule.action] = (int(h.effect.params[0]), int(h.effect.params[1]))
                    s[rule.action] = h.support
            elif kind == "disappear":
                vanish[ident] = vanish.get(ident, 0) + h.support
            elif kind == "appear":
                spawn[ident] = spawn.get(ident, 0) + h.support
    out: dict[Identity, Affordance] = {}
    for ident in set(translate) | set(response) | set(vanish) | set(spawn):
        per_action = vecs.get(ident, {})
        # autonomous = moves the SAME way under >=2 DISTINCT actions (self-driven, not
        # action-controlled — the avatar moves DIFFERENTLY per action, so it is NOT flagged).
        autonomous = len(per_action) >= 2 and len(set(per_action.values())) == 1
        out[ident] = Affordance(
            translate_support=translate.get(ident, 0),
            response_support=response.get(ident, 0),
            vanish_support=vanish.get(ident, 0),
            spawn_support=spawn.get(ident, 0),
            autonomous=autonomous,
        )
    return out
