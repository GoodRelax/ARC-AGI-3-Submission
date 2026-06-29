"""Goal Relation interpreter — test / distance / canonical over a Relation tree.

Replaces the pre-v14 predicate-class hierarchy (Atom/And/Or/Not/Exists/Forall/
GoalPredicate, archived). v14: logic = builtin operator Words + a recursive
``Relation``; ``Goal`` owns ``predicate : Relation``. This module is the
evaluator: it ports the semantics in ``gr-arc-3-operators.md`` §2.

Dispatch: a Relation's ``operator_word_id`` -> the operator Word's ``impl_key``
(via ``env.lexicon``) -> the evaluator registered under that impl_key
(``agent.core.registry``). New operators plug in the same way (a words.tsv row +
an operators.md §2 row + a registration here), so the parallel Lexicon session
adds operators without touching the dispatch.

Invariants (operators.md §2): ``distance`` is total, non-negative, ``0 iff
test``, and monotone toward satisfaction. Combinators (and=Σ, or=min/empty=1,
forall=count-missing) preserve these as long as the leaves do. Determinism
(DP-10): ``canonical`` sorts commutative operands by ``repr`` — no builtin
``hash()`` / RNG.

Clean-room note: there is NO compatibility shim for the old GoalPredicate API —
the old consumers (solver, etc.) are archived and will be rebuilt against this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple, Union

from agent.core import registry
from agent.core.env import Env
from agent.core.model import Goal, Relation

# Operators whose operands are order-independent (canonical sorts them).
# MUST stay in sync with the commutative operators in gr-arc-3-operators.md §2
# (and / or / samekind / orientation-match — the last is commutative: its
# canonical is ("orientation-match", *sorted(A, B))). The Word id is the hyphen
# form ("orientation-match"), so that is what is listed here. A new commutative
# operator added via the plug contract must be added here too, or canonical()
# will treat it as directed.
_COMMUTATIVE = frozenset({
    "and", "or", "samekind", "orientation-match",
    # draft-promoted relational operators (all symmetric: equal extent / shared
    # centre / reflected-pair are mutual). Word ids are the hyphen forms.
    "concentric", "equal-size", "mirrored",
})

# Default confidence floor for `has` profile-membership.
_HAS_MIN_CONFIDENCE = 0.0

# orientation-match boolean-view tolerance: truth >= 1 - EPS reads as a match. EPS
# is a float-noise epsilon (NOT a real threshold) so the boolean view coincides
# with exact cells equality (operators.md §2: calibrated so the boolean view
# matches cells equality).
_ORIENT_MATCH_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Public evaluation entry points
# --------------------------------------------------------------------------- #

def test(node: Relation, env: Env) -> bool:
    """Whether the predicate ``node`` holds in ``env``."""
    return _eval(node, env).test(node, env)


def distance(node: Relation, env: Env) -> float:
    """Goal-distance heuristic for ``node`` (0 iff test; non-negative; monotone)."""
    return _eval(node, env).distance(node, env)


def canonical(node: Union[Relation, str]) -> Union[Tuple, str]:
    """Deterministic invariant key for a Relation tree (equality / memo basis)."""
    if not isinstance(node, Relation):
        return node  # bare leaf reference (Role label / Word id)
    kids = [canonical(o) for o in node.operands]
    if node.operator_word_id in _COMMUTATIVE:
        kids = sorted(kids, key=repr)
    return (node.operator_word_id, *kids)


def goal_test(goal: Goal, env: Env) -> bool:
    return test(goal.predicate, env)


def goal_distance(goal: Goal, env: Env) -> float:
    return distance(goal.predicate, env)


def goal_canonical(goal: Goal) -> Union[Tuple, str]:
    return canonical(goal.predicate)


def _eval(node: Relation, env: Env) -> "OperatorEval":
    if not isinstance(node, Relation):
        raise TypeError(f"test/distance expect a Relation, got {node!r}")
    word = env.lexicon.word(node.operator_word_id)
    ev = registry.resolve(word.impl_key)
    if ev is None:
        raise RuntimeError(
            f"no evaluator registered for operator {node.operator_word_id!r} "
            f"(impl_key {word.impl_key!r}); call register_operators()"
        )
    return ev


# --------------------------------------------------------------------------- #
# Operator evaluators
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OperatorEval:
    """One operator's test + distance (canonical is structural, handled above)."""

    test: Callable[[Relation, Env], bool]
    distance: Callable[[Relation, Env], float]


def _zero_one(test_fn: Callable[[Relation, Env], bool]) -> Callable[[Relation, Env], float]:
    """distance = 0 if test else 1 (the leaf / not / exists rule)."""
    return lambda node, env: 0.0 if test_fn(node, env) else 1.0


# -- logical ----------------------------------------------------------------- #

def _and_test(node, env):
    return all(test(op, env) for op in node.operands)


def _and_distance(node, env):
    return float(sum(distance(op, env) for op in node.operands))


def _or_test(node, env):
    return any(test(op, env) for op in node.operands)


def _or_distance(node, env):
    if not node.operands:
        return 1.0  # empty OR is unsatisfiable; positive unit (no min([]) crash)
    return float(min(distance(op, env) for op in node.operands))


def _not_test(node, env):
    return not test(node.operands[0], env)


# -- quantifiers ------------------------------------------------------------- #

def _quant_parts(node):
    role, body = node.operands  # [bound Role label, body Relation]
    return role, body


def _exists_test(node, env):
    role, body = _quant_parts(node)
    return any(test(body, env.bind(role, obj)) for obj in env.objects_for(role))


def _forall_test(node, env):
    role, body = _quant_parts(node)
    return all(test(body, env.bind(role, obj)) for obj in env.objects_for(role))


def _forall_distance(node, env):
    role, body = _quant_parts(node)
    return float(
        sum(1 for obj in env.objects_for(role) if not test(body, env.bind(role, obj)))
    )


# -- relation predicates (leaves) -------------------------------------------- #
# NOTE: the geometric / attribute definitions below are deterministic and satisfy
# 0-iff-test, but are PROVISIONAL — they will be reconciled with the archived
# attribute semantics + the step-3 attribute model. Flagged, not final.

def _inside_test(node, env):
    a, b = node.operands
    fa = env.footprint(env.object_for(a))
    fb = env.footprint(env.object_for(b))
    if not fa or not fb or fa == fb:
        return False
    rows = [c[0] for c in fb]
    cols = [c[1] for c in fb]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    return all(r0 <= r <= r1 and c0 <= c <= c1 for (r, c) in fa)


def _overlaps_test(node, env):
    a, b = node.operands
    return bool(env.footprint(env.object_for(a)) & env.footprint(env.object_for(b)))


def _adjacent_test(node, env):
    a, b = node.operands
    fa = env.footprint(env.object_for(a))
    fb = env.footprint(env.object_for(b))
    if fa & fb:
        return False  # overlapping is not "adjacent"
    for (r, c) in fa:
        for (dr, dc) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in fb:
                return True
    return False


def _centroid(cells):
    rs = [c[0] for c in cells]
    cs = [c[1] for c in cells]
    n = len(cells)
    return (sum(rs) / n, sum(cs) / n)


def _collinear_test(node, env):
    a, b = node.operands
    fa = env.footprint(env.object_for(a))
    fb = env.footprint(env.object_for(b))
    if not fa or not fb:
        return False
    ca, cb = _centroid(fa), _centroid(fb)
    return ca[0] == cb[0] or ca[1] == cb[1]


def _matches_test(node, env):
    a, b = node.operands
    pa = env.profile(env.object_for(a))
    pb = env.profile(env.object_for(b))
    wa = {c.word_id for c in pa.characteristics}
    wb = {c.word_id for c in pb.characteristics}
    return bool(wa & wb)


def _has_test(node, env):
    if len(node.operands) == 1:
        subject = env.subject()
        word = node.operands[0]
    else:
        subject, word = env.object_for(node.operands[0]), node.operands[1]
    return env.profile(subject).has_word(word, min_confidence=_HAS_MIN_CONFIDENCE)


def _subset_test(node, env):
    a, b = node.operands
    return any(
        r.operator_word_id == "subset" and list(r.operands) == [a, b]
        for r in env.lexicon.relations
    )


def _samekind_test(node, env):
    a, b = node.operands
    return any(
        r.operator_word_id == "samekind" and a in r.operands and b in r.operands
        for r in env.lexicon.relations
    )


# -- orientation-match (graded leaf — operators.md §2, first graded operator) -- #

def _orient_truth(node, env):
    """The graded truth in [0, 1] for ``orientation-match(A, B)`` (operators.md §2):
    ``0.5·(1 + cos(k·(θA − θB)))`` with ``k = min(symmetry_order(A),
    symmetry_order(B))`` folding rotational symmetry. ``θ = atan2(sinθ, cosθ)`` of
    each object's orientation unit-vector. If either orientation is ``None`` the
    truth is 0.0 (no match)."""
    oa, ob = node.operands
    a, b = env.object_for(oa), env.object_for(ob)
    va, vb = env.orientation(a), env.orientation(b)
    if va is None or vb is None:
        return 0.0
    ta = math.atan2(va[1], va[0])
    tb = math.atan2(vb[1], vb[0])
    k = min(env.symmetry_order(a), env.symmetry_order(b))
    return 0.5 * (1.0 + math.cos(k * (ta - tb)))


def _orientmatch_test(node, env):
    """Boolean view: ``truth >= 1 - EPS`` (calibrated so it coincides with cells
    equality mod translation)."""
    return _orient_truth(node, env) >= 1.0 - _ORIENT_MATCH_EPS


def _orientmatch_distance(node, env):
    """Graded distance ``1 - truth`` in [0, 1] (total, non-negative, 0 iff test)."""
    return 1.0 - _orient_truth(node, env)


# -- draft-promoted relational operators (concentric / equal-size / mirrored) -- #
# Small geometry leaves over the Env footprint/size surface, mirroring the
# inside/overlaps/collinear pattern (boolean, deterministic, 0-iff-test). They
# unlock the deferred gp-concentric / gp-equal-size-match / gp-mirror-copy
# patterns. PROVISIONAL like the other geometric leaves (axis-blind to colour;
# reconciled with the step-3 attribute model later). The matching backlog Word
# rel_aligned (lattice membership) is NOT here: it needs the cue_lattice
# perception threaded through Env, which is not available.

# concentric centre tolerance: centroids within half a cell read as co-centred
# (same centre cell). A tolerance (not exact float ==) is deliberate -- it is
# robust to centroid float-noise (GEOM-1 determinism lesson) and is the correct
# semantics for "share a centre".
_CONCENTRIC_TOL = 0.5


def _concentric_test(node, env):
    """``concentric(A, B)``: A and B share a centre (co-centred / nested rings /
    box-with-centre). True iff their footprint centroids coincide within half a
    cell. ``inside`` is containment, not co-centred -- this is the distinct
    relation re86/cd82 need. PROVISIONAL (like the other geometric leaves): the
    test is centroid-mean based, so two multi-lobe shapes can share a centroid
    without any visual nesting (a barbell and its midpoint). The grounding shapes
    are convex/contiguous so this rarely fires; reconciled with the step-3
    attribute model later."""
    a, b = node.operands
    fa = env.footprint(env.object_for(a))
    fb = env.footprint(env.object_for(b))
    if not fa or not fb:
        return False
    ca, cb = _centroid(fa), _centroid(fb)
    return abs(ca[0] - cb[0]) <= _CONCENTRIC_TOL and abs(ca[1] - cb[1]) <= _CONCENTRIC_TOL


def _equalsize_test(node, env):
    """``equal-size(A, B)``: A and B have equal bbox extent ``(h, w)``. Reads the
    GameObject ``size`` geometry attr via the Env (same pattern as
    orientation-match reading ``orientation``). ``matches`` is axis-blind and
    cannot assert equal SIZE specifically; this can. Equal bbox EXTENT, not equal
    cell-count/area -- a filled 3x3 and a hollow 3x3 ring both have size (3, 3)."""
    a, b = node.operands
    sa = env.size(env.object_for(a))
    sb = env.size(env.object_for(b))
    return sa is not None and sb is not None and tuple(sa) == tuple(sb)


def _normalize_cells(cells):
    """Translate a cell set so its bbox top-left is (0, 0) (translation-invariant
    shape key)."""
    r0 = min(c[0] for c in cells)
    c0 = min(c[1] for c in cells)
    return frozenset((r - r0, c - c0) for (r, c) in cells)


def _reflect_h(cells):
    """Mirror across a vertical axis (flip columns within the bbox)."""
    cmax = max(c[1] for c in cells)
    return frozenset((r, cmax - c) for (r, c) in cells)


def _reflect_v(cells):
    """Mirror across a horizontal axis (flip rows within the bbox)."""
    rmax = max(c[0] for c in cells)
    return frozenset((rmax - r, c) for (r, c) in cells)


def _mirrored_test(node, env):
    """``mirrored(A, B)``: A is the reflected copy of B across an axis (left/right
    or top/bottom mirror), modulo translation. Boolean, axis-blind to colour.
    ``orientation-match`` folds ROTATION; this is the distinct reflection-pair
    relation vc33/m0r0 need. A symmetric shape is its own mirror (True); an
    identical non-symmetric pair is NOT a mirror (False). Only AXIS (horizontal /
    vertical) reflection is matched; a diagonal/transpose reflection is
    intentionally NOT a mirror here (the grounding vc33/m0r0 are left/right
    copies)."""
    a, b = node.operands
    fa = env.footprint(env.object_for(a))
    fb = env.footprint(env.object_for(b))
    if not fa or not fb:
        return False
    nb = _normalize_cells(fb)
    return (_normalize_cells(_reflect_h(fa)) == nb
            or _normalize_cells(_reflect_v(fa)) == nb)


# --------------------------------------------------------------------------- #
# Registration (idempotent so tests that clear the registry can re-arm it)
# --------------------------------------------------------------------------- #

# impl_key -> (test_fn, distance_fn). distance defaults to 0/1 for leaves.
_SPECS = {
    "op_and": (_and_test, _and_distance),
    "op_or": (_or_test, _or_distance),
    "op_not": (_not_test, _zero_one(_not_test)),
    "op_exists": (_exists_test, _zero_one(_exists_test)),
    "op_forall": (_forall_test, _forall_distance),
    "rel_inside": (_inside_test, _zero_one(_inside_test)),
    "rel_overlaps": (_overlaps_test, _zero_one(_overlaps_test)),
    "rel_adjacent": (_adjacent_test, _zero_one(_adjacent_test)),
    "rel_collinear": (_collinear_test, _zero_one(_collinear_test)),
    "rel_matches": (_matches_test, _zero_one(_matches_test)),
    "rel_has": (_has_test, _zero_one(_has_test)),
    "rel_subset": (_subset_test, _zero_one(_subset_test)),
    "rel_samekind": (_samekind_test, _zero_one(_samekind_test)),
    "rel_orientmatch": (_orientmatch_test, _orientmatch_distance),
    "rel_concentric": (_concentric_test, _zero_one(_concentric_test)),
    "rel_equalsize": (_equalsize_test, _zero_one(_equalsize_test)),
    "rel_mirrored": (_mirrored_test, _zero_one(_mirrored_test)),
}


def register_operators() -> None:
    """Register the 17 builtin operator evaluators (idempotent): the 14 shipped +
    3 draft-promoted (concentric / equal-size / mirrored)."""
    for impl_key, (test_fn, distance_fn) in _SPECS.items():
        if not registry.is_registered(impl_key):
            registry.evaluator(impl_key)(
                OperatorEval(test=test_fn, distance=distance_fn)
            )


register_operators()
