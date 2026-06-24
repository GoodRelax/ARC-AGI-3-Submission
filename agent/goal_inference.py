"""[Use Case] GoalInference: win-diff -> GoalPredicate, multi-win refine, novelty.

Use-Case layer (may import `arcengine`; it does not need to). Synthesizes a
generalizable `GoalPredicate` (Entity) by diffing the pre-win transition (FR-130),
refines it across wins by intersecting shared relations (FR-132), and scores
count-based novelty for the pre-win surrogate reward (FR-131).

CRITICAL (C-1, FR-147, NFR-103): the synthesized predicate references object ROLES and
RELATIONS only — never a concrete `shape_hash`, color value, or coordinate literal.
Roles come from `goal.role_of` (shape-class RANK within the ObjectSet), so the goal
transfers across instances and games.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

from agent.goal import (
    GoalHypothesisLibrary,
    GoalPredicate,
    RELATIONS,
    SelectionContext,
    _relation_holds,
    _roles_present,
    default_library,
)
from agent.segment import ObjectSet

__all__ = [
    "GoalInference",
    "GoalProposer",
    "GoalHypothesis",
    "MAX_HYPOTHESES",
    "TEMPLATE_PRECEDENCE",
]

# Bounded proposed-goal count (FR-151, NFR-112): propose() ranks the FULL candidate set
# first, THEN truncates to this cap so the cap keeps the BEST, not the first-enumerated.
MAX_HYPOTHESES = 16

# Explicit template precedence by PRIOR STRENGTH (N-10): COINCIDE before REACH — a
# documented tie-break, NOT alphabetical-by-accident. Canonical library order too.
TEMPLATE_PRECEDENCE: dict[str, int] = {"coincide": 0, "reach": 1, "align": 2}


@dataclass(frozen=True, slots=True)
class GoalHypothesis:
    """A GoalPredicate proposed from a template, with its rank key + reconsider status.

    Internal ranking record only; the proposer's PUBLIC return is list[GoalPredicate]
    (N-12). Immutable: an evidence update REBUILDS this frozen record (FR-152, N-06).
    """

    predicate: GoalPredicate
    template: str
    mdl_cost: int            # roles + relations count; lower = simpler (FR-152)
    evidence: int = 0        # int: novelty-count decreases + levels_completed deltas
                             # attributable to this hyp over the window (FR-152, N-06)
    down_weighted: bool = False  # set on surprise / no-progress this level (FR-154)

    def with_evidence(self, delta: int) -> "GoalHypothesis":
        """Immutable evidence bump: rebuild the frozen record (FR-152, N-06)."""
        return replace(self, evidence=self.evidence + delta)

    def down_weight(self) -> "GoalHypothesis":
        """Immutable down-weight: rebuild the frozen record (FR-154)."""
        return replace(self, down_weighted=True)


def _mdl(pred: GoalPredicate) -> int:
    """MDL simplicity key: distinct roles + relations. Lower = simpler (FR-152)."""
    roles = {r for (_n, a, b) in pred.relations for r in (a, b)}
    return len(roles) + len(pred.relations)


def _vacuous(pred: GoalPredicate) -> bool:
    """Empty predicate, or one with no relations, is vacuous and dropped (N-11)."""
    return len(pred.relations) == 0


def rank(hyps: list[GoalHypothesis]) -> list[GoalHypothesis]:
    """Deterministic rank of candidates (FR-152, NFR-104/112). No unseeded randomness.

    Key, ascending: (a) NOT down-weighted first; (b) MDL ascending (simpler first);
    (c) evidence DESCENDING (more evidence first, via `-evidence`); (d) template
    precedence (COINCIDE < REACH, N-10); (e) `(selectors, relations)` as a final stable
    tie-break. Given an identical ObjectSet AND pursuit history (evidence/down_weighted),
    the order is identical (N-06).
    """
    return sorted(
        hyps,
        key=lambda h: (
            h.down_weighted,
            h.mdl_cost,
            -h.evidence,
            TEMPLATE_PRECEDENCE.get(h.template, 99),
            (h.predicate.selectors, h.predicate.relations),
        ),
    )


class GoalProposer:
    """[Use Case] Proactive abductive goal acquisition over the v1 template library.

    ONE canonical public method: `propose(objset) -> list[GoalPredicate]` (N-12). It
    instantiates the two v1 templates, RANKS the full candidate set first, then truncates
    to MAX_HYPOTHESES so the cap keeps the best (N-05). Ranking over internal
    `GoalHypothesis` records is an implementation detail. Offline (NFR-113).
    """

    def __init__(self, library: GoalHypothesisLibrary | None = None) -> None:
        # The fixed library (COINCIDE, REACH); pure data, immutable, game-agnostic.
        self._library = library or default_library()

    def hypotheses(
        self, objset: ObjectSet, context: Optional[SelectionContext] = None
    ) -> list[GoalHypothesis]:
        """Instantiate + rank the candidate hypotheses (ranked, deduped, non-vacuous).

        Internal: returns ranked `GoalHypothesis` records (so the policy can track
        evidence/down-weight). `propose` wraps this and returns predicates only (N-12).
        `context` (FR-170) grounds the `controllable` operand during instantiation.
        """
        hyps: list[GoalHypothesis] = []
        for tmpl in self._library.templates:
            for pred in tmpl.instantiate(objset, context):  # roles/relations only; top-k pairs
                # Drop vacuous (empty) AND already-SATISFIED predicates (M1 spec 86, FR-190):
                # a goal that already holds gives directed exploration NO signal (distance 0),
                # so proposing it only burns the reconsider window before the agent reaches a
                # goal that can make progress. This is what surfaces `salient`/ALIGN instead of
                # burying them behind vacuous background-target goals (trace-study Finding A).
                if _vacuous(pred) or pred.holds(objset, context):
                    continue
                hyps.append(GoalHypothesis(pred, tmpl.name, mdl_cost=_mdl(pred)))
        ranked = rank(_dedup(hyps))
        return ranked[:MAX_HYPOTHESES]

    def propose(
        self, objset: ObjectSet, context: Optional[SelectionContext] = None
    ) -> list[GoalPredicate]:
        """Ranked, deduplicated, non-vacuous candidate goals (FR-151, FR-152, N-12)."""
        return [h.predicate for h in self.hypotheses(objset, context)]


def _dedup(hyps: list[GoalHypothesis]) -> list[GoalHypothesis]:
    """Drop hypotheses with an identical predicate (first occurrence wins, stable)."""
    seen: set[tuple[tuple[str, ...], tuple[tuple[str, int, int], ...]]] = set()
    out: list[GoalHypothesis] = []
    for h in hyps:
        key = (h.predicate.selectors, h.predicate.relations)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


class GoalInference:
    """Stateless service: pure functions of their inputs (NFR-105)."""

    def from_win(self, pre: ObjectSet, post: ObjectSet) -> GoalPredicate:
        """Synthesize a GoalPredicate from one pre-win transition (FR-130).

        We take the relations that became TRUE in `post` (the winning successor) and
        were not already trivially everywhere — i.e. the relations the winning action
        established between object roles. The predicate is the conjunction of those
        relations, expressed over ROLE ids (FR-147). Never coordinates/colors.

        Diff semantics: a relation is included if it holds in `post` for some role
        pair. `pre` is used only to prefer relations that the win MADE hold (held in
        post but not in pre), falling back to all post-relations if the win changed
        nothing structural (so we never synthesize an empty/vacuous goal when the win
        state has relations to describe).
        """
        post_rels = _relations_in(post)
        pre_rels = _relations_in(pre)
        established = post_rels - pre_rels
        chosen = established or post_rels
        # Deterministic order (NFR-104): sort the relation tuples.
        return GoalPredicate(relations=tuple(sorted(chosen)))

    def refine(
        self, goal: GoalPredicate, pre: ObjectSet, post: ObjectSet
    ) -> GoalPredicate:
        """Generalize the goal across wins by intersecting shared relations (FR-132).

        Keep only the relations common to the existing goal AND this new win, so a
        single win cannot overfit (L-3). Intersection monotonically shrinks the
        predicate toward the relations invariant across all observed wins.
        """
        new = self.from_win(pre, post)
        common = set(goal.relations) & set(new.relations)
        return GoalPredicate(relations=tuple(sorted(common)))

    def novelty(self, node_hash: int, counts: dict[int, int]) -> float:
        """Count-based novelty surrogate before any win (FR-131, 1606.01868).

        Higher for least-visited masked object-states. We use 1/sqrt(1 + visits) so an
        unseen state scores 1.0 and novelty decays smoothly with visits. A never-seen
        hash (count 0) is the most novel; an unknown-effect successor is handled by the
        caller as maximally novel (FR-127).
        """
        visits = counts.get(node_hash, 0)
        return 1.0 / math.sqrt(1.0 + visits)


def _relations_in(objset: ObjectSet) -> set[tuple[str, int, int]]:
    """All (relation, roleA, roleB) that hold between distinct-role object pairs.

    Only considers DISTINCT role pairs (roleA < roleB by id) for asymmetric relations
    handled by trying both orderings, so the relation set is well-defined and small.
    No coordinate/color literal enters the result — only role ids and relation names.
    """
    by_role = _roles_present(objset)
    roles = sorted(by_role)
    result: set[tuple[str, int, int]] = set()
    for i, ra in enumerate(roles):
        for rb in roles[i + 1 :]:
            for name in RELATIONS:
                # roleA, roleB ordering: test ra->rb and rb->ra for asymmetric rels.
                if any(
                    _relation_holds(name, a, b)
                    for a in by_role[ra]
                    for b in by_role[rb]
                ):
                    result.add((name, ra, rb))
                if any(
                    _relation_holds(name, b, a)
                    for a in by_role[ra]
                    for b in by_role[rb]
                ):
                    result.add((name, rb, ra))
    return result
