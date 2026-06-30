"""Observability + the connected-component bridge (goal-id / world-summary /
verbalization / inter-component dataflow log).

This module connects the implemented-but-unwired pieces of the per-turn pipeline
into a single, INERT-SAFE read layer that the observability sink and the LLM
briefing consume. It NEVER chooses the committed action (the baseline / LLM leg
still does that) — everything here is a pure, deterministic READ over the
artifacts the play loop already produces:

  * :func:`identify_goal` — the "察する = goal 同定" step. Match the loaded
    :class:`agent.core.model.GoalPattern` catalog against the current
    :class:`agent.core.situation.AbstractSituation` (its role buckets), via the
    SAME operator interpreter (:mod:`agent.core.goal`) and the SAME
    :class:`agent.core.situation.ConcreteEnv`. Returns the best-matching
    :class:`IdentifiedGoal` (pattern + Goal + distance) or ``None`` (graceful).
  * :func:`summarize_world` — surface the :class:`agent.core.world_model.WorldModel`
    state (induced rules / affordance evidence / tracking) as a structured
    summary.
  * :func:`verbalize_world` / :func:`verbalize_object` / :func:`verbalize_goal` —
    the agent's OWN natural-language rendering (the canonical verbalization,
    broader than the LLM briefing): a Naming-Ladder-style object name from its
    Profile, the world's move-effects / walls / scrolling, and the identified
    Goal.
  * :class:`DataflowLog` — a per-turn accumulator of structured stage events
    ``{stage, callee, input_summary, output_summary}`` (which component read which
    with what data and what it returned).
  * :func:`lexicon_growth` — the Lexicon size + the Words present (and the delta
    vs a previous snapshot), so the agent's vocabulary growth is observable.

Determinism (DP-10): no ``random``, no builtin ``hash()``; every iteration over a
mapping / set is sorted, and float magnitudes are rounded for stable strings.
This module imports only from the v14 core (model / goal / situation /
world_model) — it owns no IO and never mutates its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from agent.core import goal as goal_mod
from agent.core.model import Goal, GoalPattern, Lexicon
from agent.core.situation import AbstractSituation, ConcreteEnv, ObjectRef

# Rounding decimals for floats entering a verbalization string (stable output).
_R = 3


# --------------------------------------------------------------------------- #
# Goal identification (察する = goal 同定)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class IdentifiedGoal:
    """The goal the agent infers from the current situation — a «value».

    ``pattern`` is the matched :class:`GoalPattern` (its ``id`` / ``goal_kind`` /
    ``predicate`` name the inferred objective); ``goal`` wraps its parsed
    ``predicate_tree`` as a :class:`agent.core.model.Goal`; ``distance`` is
    ``goal.distance(situation)`` (a finite non-negative gradient, 0 iff already
    satisfied); ``satisfied`` mirrors ``distance == 0`` (the predicate tests true).
    Frozen + primitive fields => deterministic equality (DP-10)."""

    pattern: GoalPattern
    goal: Goal
    distance: float
    satisfied: bool


def _env_for(situation: AbstractSituation, lexicon: Lexicon) -> ConcreteEnv:
    """The goal-context :class:`ConcreteEnv` over ``situation`` (case B: a role
    binds to a SET; ``object_for`` folds it to the salient representative)."""
    return ConcreteEnv(lexicon=lexicon, situation=situation)


def _active_patterns(patterns: Tuple[GoalPattern, ...]) -> List[GoalPattern]:
    """The active (parsed) patterns in a deterministic id-ascending order."""
    return sorted(
        (p for p in patterns if p.predicate_tree is not None),
        key=lambda p: p.id,
    )


def identify_goal(
    situation: AbstractSituation,
    patterns: Tuple[GoalPattern, ...],
    lexicon: Lexicon,
) -> Optional[IdentifiedGoal]:
    """Identify the applicable Goal for ``situation`` from the GoalPattern catalog.

    Evaluate each ACTIVE pattern's parsed predicate over the situation's role
    buckets (the SAME operator interpreter the solver/roles use). A pattern is a
    CANDIDATE iff every role it references is present in the situation WITH >= 1
    object (so an orientation-deliver pattern does not match a board with no
    ``held-state``), AND it is not a DEGENERATE / vacuous match (CM-1):

      * a ``forall``/``exists`` quantifier whose bound domain is EMPTY (the
        quantifier is then vacuously true / its gradient meaningless -- e.g.
        ``forall(t, overlaps(controllable, t))`` on a board with NO ``target``
        object reads SATISFIED at distance 0 over an empty ``t``); or
      * a SATISFIED negation (top-level ``not(...)``) -- the GAP-3 "trivially
        satisfied ``not(overlaps)``" case, satisfied only because the thing it
        forbids merely is not happening, giving the agent no objective.

    Rejecting these stops the agent "believing it already won": on a board whose
    only recognizable predicate is degenerate, ``identify_goal`` returns ``None``
    (an honest "no objective yet") rather than a vacuously-satisfied goal.

    Among the surviving candidates, prefer the SMALLEST ``goal.distance`` (closest
    to satisfaction), then the smallest pattern ``id`` (deterministic tie-break).

    Cheap + deterministic (DP-10): one bounded pass over the ~15 active patterns;
    no RNG, no builtin ``hash()``."""
    env = _env_for(situation, lexicon)
    # A role "present" only if its bucket holds >= 1 object (an empty bucket keyed
    # but with no members is treated as ABSENT -- it would fold to a degenerate
    # empty ObjectRef just like a missing key).
    role_keys = frozenset(r for r, objs in situation.objects.items() if objs)
    best: Optional[IdentifiedGoal] = None
    best_key: Optional[Tuple[float, str]] = None
    for pattern in _active_patterns(patterns):
        tree = pattern.predicate_tree
        # Candidate gate: every bare ROLE reference the predicate reads must be a
        # PRESENT (non-empty) role bucket this turn (quantifier-bound vars excepted).
        needed = _referenced_roles(tree)
        if not needed.issubset(role_keys):
            continue
        # Degeneracy gate (CM-1): drop vacuous matches that would drown a real goal.
        if _empty_quantifier_domain(tree, env):
            continue
        try:
            g = Goal(predicate=tree)
            dist = float(goal_mod.goal_distance(g, env))
            satisfied = goal_mod.goal_test(g, env)
        except Exception:  # noqa: BLE001 - a malformed pattern must not crash goal-id
            continue
        if satisfied and _is_satisfied_negation(tree):
            continue
        key = (dist, pattern.id)
        if best_key is None or key < best_key:
            best_key = key
            best = IdentifiedGoal(
                pattern=pattern, goal=g, distance=dist, satisfied=satisfied
            )
    return best


def _empty_quantifier_domain(node: Any, env: ConcreteEnv) -> bool:
    """True iff ``node`` contains a ``forall``/``exists`` quantifier whose bound
    domain is EMPTY (``env.objects_for(var)`` has no members).

    A quantifier over an empty set is vacuous: ``forall`` is trivially true (no
    objective / distance 0) and ``exists`` is trivially false with a meaningless
    gradient. Either way the pattern names no real goal this turn, so it must not
    be selected. Structural (reads the var via the SAME env the interpreter uses);
    no game-specific role names, no RNG (DP-10)."""
    from agent.core.model import Relation

    if not isinstance(node, Relation):
        return False
    if node.operator_word_id in ("exists", "forall") and node.operands:
        var = node.operands[0]
        if isinstance(var, str) and len(env.objects_for(var)) == 0:
            return True
    return any(_empty_quantifier_domain(op, env) for op in node.operands)


def _is_satisfied_negation(node: Any) -> bool:
    """True iff ``node``'s top-level operator is ``not`` (a negation). Combined with
    a satisfied test, this flags the GAP-3 "trivially satisfied ``not(overlaps)``"
    case: the predicate holds only because the forbidden relation merely is not
    happening, which is no objective. Structural / deterministic."""
    from agent.core.model import Relation

    return isinstance(node, Relation) and node.operator_word_id == "not"


# Bound quantifier variables in the active patterns are NOT situation roles (they
# range over a role's set via exists/forall); they must be excluded from the
# candidate-gate role set. The set is the operands[0] of every exists/forall in
# the catalog (box / t / obj / r) — derived structurally, not hardcoded.
def _quantifier_vars(node: Any) -> frozenset:
    """The bound-variable names introduced by exists/forall in ``node``."""
    from agent.core.model import Relation

    if not isinstance(node, Relation):
        return frozenset()
    out: set = set()
    if node.operator_word_id in ("exists", "forall") and node.operands:
        var = node.operands[0]
        if isinstance(var, str):
            out.add(var)
    for op in node.operands:
        out |= _quantifier_vars(op)
    return frozenset(out)


def _referenced_roles(node: Any) -> frozenset:
    """The bare ROLE references a predicate reads (leaf strings that are not a
    nested Relation and not a bound quantifier variable). These must be present as
    situation role buckets for the pattern to be a candidate."""
    from agent.core.model import Relation

    bound = _quantifier_vars(node)
    out: set = set()

    def walk(n: Any) -> None:
        if isinstance(n, Relation):
            for i, op in enumerate(n.operands):
                # Skip the binder name (operands[0]) of a quantifier — it is the
                # bound var, not a role reference.
                if n.operator_word_id in ("exists", "forall") and i == 0:
                    continue
                walk(op)
        elif isinstance(n, str):
            if n not in bound:
                out.add(n)

    walk(node)
    return frozenset(out)


# --------------------------------------------------------------------------- #
# World summary
# --------------------------------------------------------------------------- #

def summarize_world(
    world: Any,
    *,
    affordance_evidence: Optional[Mapping[str, Any]] = None,
    controllable_id: Optional[str] = None,
) -> Dict[str, Any]:
    """A structured, JSON-able summary of the WorldModel state this turn.

    Surfaces what the agent KNOWS about dynamics: the induced
    :class:`agent.core.world_model.InteractionRule` count + a compact
    per-rule descriptor (trigger move + the EffectSignature canonical 5-tuples),
    the representation switch / scrolling flag / confidence, and the per-handle
    Affordance evidence (translate / vanish / spawn / response support +
    autonomous) that drives the controllable pick. Deterministic: rules in
    registration order, handles sorted."""
    rules = list(getattr(world, "rules", []) or [])
    rule_descs: List[Dict[str, Any]] = []
    for rule in rules:
        effects = [
            list(sig.canonical()) for sig in (getattr(rule, "effect", []) or [])
        ]
        rule_descs.append(
            {
                "move": getattr(rule, "move", None),
                "confidence": round(float(getattr(rule, "confidence", 0.0)), _R),
                "effects": effects,
            }
        )

    affordances: Dict[str, Dict[str, Any]] = {}
    for handle in sorted(affordance_evidence or {}):
        aff = (affordance_evidence or {})[handle]
        affordances[handle] = {
            "translate": round(float(getattr(aff, "translate_support", 0.0)), _R),
            "vanish": round(float(getattr(aff, "vanish_support", 0.0)), _R),
            "spawn": round(float(getattr(aff, "spawn_support", 0.0)), _R),
            "response": round(float(getattr(aff, "response_support", 0.0)), _R),
            "autonomous": bool(getattr(aff, "autonomous", False)),
        }

    return {
        "rule_count": len(rules),
        "rules": rule_descs,
        "representation": getattr(world, "representation", "Symbolic"),
        "is_scrolling": bool(getattr(world, "is_scrolling", False)),
        "confidence": round(float(getattr(world, "confidence", 0.0)), _R),
        "controllable_id": controllable_id,
        "affordances": affordances,
    }


# --------------------------------------------------------------------------- #
# Verbalization (the canonical NL rendering — Naming-Ladder style)
# --------------------------------------------------------------------------- #

# The naming-ladder axes (model.LADDER_SLOTS), in left-to-right modifier order.
# A Profile Characteristic on one of these axes contributes its Word id as a
# modifier; the head noun ("head" slot) is last.
_LADDER_AXES = ("controllability", "size", "behavior", "color", "head")


def _centroid_str(ref: ObjectRef) -> str:
    """The ``(row, col)`` centroid of an ObjectRef's footprint, or ``(?, ?)``."""
    cells = list(getattr(getattr(ref, "geometry", None), "cells", None) or [])
    if not cells:
        return "(?, ?)"
    rows = sum(int(r) for (r, _c) in cells)
    cols = sum(int(c) for (_r, c) in cells)
    n = len(cells)
    return "(%d, %d)" % (rows // n, cols // n)


def verbalize_object(role: str, ref: ObjectRef) -> str:
    """A deterministic Naming-Ladder-style sentence for one salient object.

    Renders ``<role>: <modifiers> object @<centroid> [size=N cells]`` where the
    modifiers are the object's Profile Characteristics (word id + rounded
    magnitude), so an object reads as e.g.
    ``controllable: controllable(1.0) red(0.8) object @(3, 3) [9 cells]``. Pure
    read; deterministic (characteristics sorted by word id)."""
    prof = getattr(ref, "profile", None)
    chars = list(getattr(prof, "characteristics", None) or [])
    mods = " ".join(
        "%s(%s)" % (c.word_id, round(float(c.magnitude), _R))
        for c in sorted(chars, key=lambda c: c.word_id)
    )
    cells = getattr(getattr(ref, "geometry", None), "cells", None) or frozenset()
    size = len(cells)
    handle = getattr(ref, "handle", "?")
    body = ("%s object" % mods) if mods else "object"
    return "%s: %s @%s - %s [%d cells]" % (
        role, handle, _centroid_str(ref), body, size
    )


def verbalize_objects(
    situation: AbstractSituation, max_objects: int = 8
) -> List[str]:
    """One verbalization line per salient object across all role buckets (roles
    sorted, handles sorted), bounded to ``max_objects`` lines. Deterministic."""
    lines: List[str] = []
    objects = getattr(situation, "objects", None) or {}
    for role in sorted(objects):
        for ref in sorted(objects[role], key=lambda r: getattr(r, "handle", "")):
            lines.append(verbalize_object(role, ref))
            if len(lines) >= max_objects:
                return lines
    return lines


def verbalize_world(
    world_summary: Mapping[str, Any], situation: AbstractSituation
) -> str:
    """The agent's NL rendering of the WORLD: move-effects (induced rules),
    walls/field presence, scrolling, and the salient-object count.

    Reads the :func:`summarize_world` dict + the situation role buckets — no IO,
    deterministic."""
    objects = getattr(situation, "objects", None) or {}
    n_objects = sum(len(refs) for refs in objects.values())
    n_rules = int(world_summary.get("rule_count", 0))
    scrolling = bool(world_summary.get("is_scrolling", False))
    has_field = any(
        role in ("field", "background") for role in objects
    )
    parts = [
        "World: %d salient object(s) in %d role(s)" % (n_objects, len(objects)),
        "%d induced move-rule(s)" % n_rules,
        "field/ground present" if has_field else "no field detected",
        "scrolling" if scrolling else "static viewport",
    ]
    return "; ".join(parts) + "."


def verbalize_goal(identified: Optional[IdentifiedGoal]) -> str:
    """The agent's NL rendering of the identified GOAL: the matched pattern, its
    predicate, and the distance-to-satisfaction. ``"no goal identified yet"`` when
    none matched (graceful). Deterministic."""
    if identified is None:
        return "no goal identified yet."
    p = identified.pattern
    state = "SATISFIED" if identified.satisfied else (
        "distance %s" % round(float(identified.distance), _R)
    )
    return "Goal [%s / %s]: %s - %s." % (
        p.id, p.goal_kind, p.predicate, state
    )


# --------------------------------------------------------------------------- #
# Lexicon growth
# --------------------------------------------------------------------------- #

def lexicon_growth(
    lexicon: Lexicon, prev_words: Optional[Tuple[str, ...]] = None
) -> Dict[str, Any]:
    """The Lexicon size + the Words present (sorted), and the delta vs a prior
    snapshot ``prev_words`` (added / removed word ids). Deterministic — word ids
    sorted so the snapshot is stable across processes."""
    words = tuple(sorted(w.id for w in getattr(lexicon, "words", []) or []))
    prev = frozenset(prev_words or ())
    cur = frozenset(words)
    added = tuple(sorted(cur - prev))
    removed = tuple(sorted(prev - cur))
    return {
        "size": len(words),
        "words": list(words),
        "added": list(added),
        "removed": list(removed),
    }


# --------------------------------------------------------------------------- #
# Inter-component dataflow log
# --------------------------------------------------------------------------- #

@dataclass
class DataflowLog:
    """A per-turn accumulator of inter-component stage events.

    Each :meth:`record` appends ``{stage, callee, input_summary, output_summary}``
    — which pipeline component ran, what it read (a compact summary), and what it
    returned (counts / ids / key values, NOT a full dump). :meth:`events` returns
    the ordered list (the order they ran), so the inspector can render the
    dataflow as a sequence. Order-preserving + deterministic (the caller passes
    already-compact summaries)."""

    _events: List[Dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        stage: str,
        callee: str,
        input_summary: Any,
        output_summary: Any,
    ) -> None:
        """Append one stage event (compact summaries; no full dumps)."""
        self._events.append(
            {
                "stage": stage,
                "callee": callee,
                "input": input_summary,
                "output": output_summary,
            }
        )

    def events(self) -> List[Dict[str, Any]]:
        """The ordered stage events recorded this turn."""
        return list(self._events)

    def clear(self) -> None:
        """Reset for the next turn."""
        self._events = []
