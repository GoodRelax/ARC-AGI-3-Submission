"""Transform algebra + appliers (agent.core.transform) -- the SINGLE SOURCE OF
TRUTH for rotation / scale / cyclic geometry on coloured cell sets.

Two layers:

  * a small *detection algebra* over a coloured cell set
    ``frozenset[(row, col, color)]`` -- translation/rotation/scale primitives and
    the deterministic invariants ``canonical_pose`` (shape identity) and
    ``orientation_index`` (the pose label k in 0..3 under the C4 rotation group);
  * the six ``xf_*`` *appliers* (a ``TransformOperator`` evaluator each), registered
    under their ``xf_*`` impl_key in :mod:`agent.core.registry` so the AssetLoader
    wires them like any other plug (gr-arc-3-plug-architecture §2/§5, transform
    family).

Grounding (object-schema §3): ``orient(O)`` is a rotation INDEX, ``scale(O)`` is an
INTEGER factor read from block regularity, and ``shape`` is rotation/scale-invariant.

The POSE group is **C4 (4 rotations, NO reflection)** -- ``canonical_pose`` and
``orientation_index`` live in C4, so a glyph and its mirror are DISTINCT
orientations (a preserved pose is never collapsed onto its mirror). The larger
**D4** group (rotations + reflection, via ``d4_canonical``) is used ONLY to DERIVE
the ``reflected`` HANDEDNESS attribute (ADR-016): ``reflected`` is True iff the C4
canonical pose disagrees with the D4 canonical pose. This is NOT a reintroduction
of reflection into pose -- pose stays in C4; D4 is consulted solely to read off a
single handedness bit. Reflection-invariance of the head-noun still lives in
``attributes.shape_base``.

Determinism (DP-10): no ``random``, no Python builtin ``hash()`` in any returned
value. Canonical representatives (``canonical_pose`` / ``d4_canonical`` / the
``min(tuple(sorted(...)))`` reduction inside ``mirror``-based helpers) use
``sorted`` tuples; returned magnitudes are plain ints (``orientation_index`` in
0..3, ``scale`` >= 1, ``quarter_turns``) and ``reflected`` is a plain ``bool``.
``mirror`` / ``d4_canonical`` / ``reflected`` are pure functions of their input
(no RNG, no builtin ``hash()``). Magnitude/Characteristic encoding is a SEPARATE
step (6-3.5) -- not baked in here.

The geometry ops (rotate / scale / translate) take a coloured cell set; the value
ops (recolor / cycle / set_) take a scalar value. All appliers are pure and
stateless. ``register_transforms()`` runs at import and is idempotent, so tests can
re-arm after the conftest clears the registry.
"""

from __future__ import annotations

from typing import FrozenSet, List, Sequence, Tuple

from agent.core import registry

Colored = FrozenSet[Tuple[int, int, int]]


# --------------------------------------------------------------------------- #
# geometry primitives on coloured cells frozenset[(r, c, color)]
# --------------------------------------------------------------------------- #

def renorm(colored: Colored) -> Colored:
    """Translate ``colored`` so its bbox top-left sits at the origin (0, 0).

    The colour of each cell is preserved. An empty set renorms to empty. This is
    the translation-normal form every pose/scale comparison is taken modulo."""
    if not colored:
        return frozenset()
    mr = min(r for r, _, _ in colored)
    mc = min(c for _, c, _ in colored)
    return frozenset((r - mr, c - mc, col) for r, c, col in colored)


def rot90(colored: Colored) -> Colored:
    """Rotate ``colored`` 90 degrees CLOCKWISE (colour preserved), then renorm.

    A cell ``(r, c)`` in an ``h``-row bbox maps to ``(c, h - 1 - r)`` -- the
    standard CW rotation -- and the result is renormed so successive rotations
    stay translation-comparable (C4 closure)."""
    if not colored:
        return frozenset()
    h = max(r for r, _, _ in colored) + 1
    return renorm(frozenset((c, h - 1 - r, col) for r, c, col in colored))


def rotations(colored: Colored) -> List[Colored]:
    """The C4 orbit of ``colored``: ``[renorm, rot90, rot90^2, rot90^3]``."""
    out = [renorm(colored)]
    for _ in range(3):
        out.append(rot90(out[-1]))
    return out


def canonical_pose(colored: Colored) -> Tuple[Tuple[int, int, int], ...]:
    """The deterministic canonical representative of ``colored`` under C4: the
    lexicographically smallest ``tuple(sorted(rotation))`` over the four rotations.

    This is the rotation-invariant SHAPE identity. Because the orbit is only the 4
    rotations (no reflection), a chiral glyph and its mirror have DISTINCT canonical
    poses (proving the group is C4, not D4). Returns a sorted tuple (hashable,
    order-stable), never a set -- so equality is value-deterministic."""
    return min(tuple(sorted(rot)) for rot in rotations(colored))


# --------------------------------------------------------------------------- #
# reflection / D4 -- used ONLY to derive the `reflected` handedness bit (ADR-016).
# Pose itself stays in C4 (canonical_pose / orientation_index above).
# --------------------------------------------------------------------------- #

def mirror(colored: Colored) -> Colored:
    """The vertical reflection of ``colored``: ``(r, c, col) -> (r, -c, col)``, then
    renorm (colour preserved). An empty set mirrors to empty.

    This is the single mirror generator that, combined with the C4 rotations,
    closes the D4 orbit (see :func:`d4_canonical`)."""
    if not colored:
        return frozenset()
    return renorm(frozenset((r, -c, col) for r, c, col in colored))


def d4_canonical(colored: Colored) -> Tuple[Tuple[int, int, int], ...]:
    """The deterministic canonical representative of ``colored`` under D4 (the 8
    rotations-and-reflections): the lexicographically smallest
    ``tuple(sorted(m))`` over the four rotations of ``colored`` AND the four
    rotations of ``mirror(colored)``.

    D4 = C4 ∪ (C4 ∘ mirror). Because the orbit now includes reflection, a chiral
    glyph and its mirror share ONE D4 canonical (they collapse) -- which is exactly
    what :func:`reflected` exploits to read handedness off the C4-vs-D4
    disagreement. Returns a sorted tuple (hashable, order-stable)."""
    return min(
        tuple(sorted(m))
        for m in rotations(colored) + rotations(mirror(colored))
    )


def reflected(colored: Colored) -> bool:
    """The HANDEDNESS bit (ADR-016): ``canonical_pose(colored) != d4_canonical(colored)``.

    HANDEDNESS, NOT chirality, and NOT a relative comparison between two objects:
    it is a single intrinsic bit per object saying which of the two enantiomorphs
    the C4 lex-min canonicalization landed on. Equivalently it is
    ``canonical_pose(colored) > canonical_pose(mirror(colored))`` -- the C4 pose is
    "reflected" when its own canonical is the larger of {itself, its mirror}, i.e.
    the D4 canonical (always the min of the two) came from the mirror branch.

    CAVEAT: ``reflected`` operates on the cell set INCLUDING colour (the canonicals
    compare ``(r, c, col)`` triples), so a colour pattern can itself read as chiral.
    Callers wanting colour-BLIND handedness (as ``attributes.compute_pose_geometry``
    does) must stamp a single uniform colour on every cell first.

    Consequences:
      * an ACHIRAL glyph (its mirror is in its own C4 orbit, e.g. ELL/BOX/bar/cross/
        dot) is ALWAYS False -- C4 and D4 canonicals coincide;
      * a CHIRAL pair (e.g. the L- and J-tetrominoes) has EXACTLY ONE member True --
        the two share a D4 canonical but have distinct C4 canonicals, so precisely
        the one whose C4 canonical is the larger reads True.
    Empty -> False.

    NOT IMPLEMENTED NOW (future, when a chirality predicate / pairing is needed):
      * ``is_chiral := len(C4_orbit) < len(D4_orbit)`` -- whether the glyph HAS a
        distinct mirror at all (the achiral/chiral classifier);
      * ``mirror_of(a, b)`` -- whether two objects are each other's reflection.
    Those are deliberately out of scope here; this returns only the per-object
    handedness bit."""
    if not colored:
        return False
    return canonical_pose(colored) != d4_canonical(colored)


def orientation_index(colored: Colored) -> int:
    """The pose label: the unique ``k`` in {0, 1, 2, 3} such that rotating the
    canonical pose ``k`` quarter-turns CW reproduces ``renorm(colored)``.

    Computed at PRIMITIVE scale so orientation is scale-invariant: the input is
    downscaled first, hence a 2x-upscaled glyph and its 1x original yield the same
    index. A rotationally symmetric glyph matches at more than one ``k``; the first
    (smallest) match is returned, so symmetric glyphs collapse to a single
    deterministic index (0 for a fully symmetric box). Returns an int in 0..3
    (never ``None``); an empty set is index 0."""
    prim = downscale(renorm(colored))
    if not prim:
        return 0
    canon = canonical_pose(prim)
    cur: Colored = frozenset(canon)
    for k in range(4):
        if tuple(sorted(cur)) == tuple(sorted(prim)):
            return k
        cur = rot90(cur)
    # Unreachable: canon is the min over the closed C4 orbit of ``prim``, so ``prim``
    # is always one of the four rotations of ``canon`` -- a match must exist. If we
    # reach here the rotation orbit is broken; fail loudly rather than return a
    # plausible-but-wrong index.
    raise AssertionError(
        "orientation_index: no rotation of canonical_pose reproduced the input; "
        "C4 orbit invariant violated for %r" % (prim,)
    )


# --------------------------------------------------------------------------- #
# scale on coloured cells
# --------------------------------------------------------------------------- #

def detect_scale(colored: Colored) -> int:
    """The largest integer ``s`` (>= 1) such that the renormed bbox of ``colored``
    partitions into ``s x s`` blocks, each block fully empty OR fully filled with a
    single colour -- i.e. ``colored`` is an ``s``-times block up-scaling.

    Iterates ``s`` from ``min(h, w)`` down to 2, skipping any ``s`` that does not
    divide both ``h`` and ``w``, and returns the first valid factor; ``1`` if none.
    ``s > 1`` means the glyph is rendered upscaled.

    Caveat: a solid ``s x s`` single-colour block detects as scale ``s`` -- a solid
    square is mathematically indistinguishable from an upscaled single point."""
    norm = renorm(colored)
    if not norm:
        return 1
    h = max(r for r, _, _ in norm) + 1
    w = max(c for _, c, _ in norm) + 1
    cellmap = {(r, c): col for r, c, col in norm}
    for s in range(min(h, w), 1, -1):
        if h % s or w % s:
            continue
        ok = True
        for br in range(h // s):
            for bc in range(w // s):
                block = [(br * s + i, bc * s + j) for i in range(s) for j in range(s)]
                present = [p in cellmap for p in block]
                if any(present):
                    if not all(present) or len({cellmap[p] for p in block}) != 1:
                        ok = False
                        break
            if not ok:
                break
        if ok:
            return s
    return 1


def downscale(colored: Colored) -> Colored:
    """Reduce an ``s``-upscaled glyph to its primitive scale: keep one representative
    cell per ``s x s`` block (the block's top-left), colour preserved, renormed.

    If ``colored`` is already primitive (``detect_scale == 1``) it is returned
    unchanged (renormed). Inverse of :func:`scale` on a clean upscale."""
    norm = renorm(colored)
    s = detect_scale(norm)
    if s == 1:
        return norm
    return frozenset(
        (r // s, c // s, col) for r, c, col in norm if r % s == 0 and c % s == 0
    )


# --------------------------------------------------------------------------- #
# the six xf_* appliers (TransformOperator evaluators)
# --------------------------------------------------------------------------- #
# geometry ops take a coloured cell set; value ops take a scalar value. All pure.

def rotate(cells: Colored, quarter_turns: int) -> Colored:
    """Apply :func:`rot90` ``quarter_turns mod 4`` times (C4). ``quarter_turns`` may
    be any int (negative or > 3); only its residue mod 4 matters."""
    out = renorm(cells)
    for _ in range(quarter_turns % 4):
        out = rot90(out)
    return out


def scale(cells: Colored, factor: int) -> Colored:
    """Block-upscale: replace every cell with a ``factor x factor`` block of the same
    colour (inverse of :func:`downscale`). ``factor`` must be >= 1; ``factor == 1``
    is the renormed identity."""
    if factor < 1:
        raise ValueError("scale factor must be >= 1, got %r" % (factor,))
    norm = renorm(cells)
    return frozenset(
        (r * factor + i, c * factor + j, col)
        for r, c, col in norm
        for i in range(factor)
        for j in range(factor)
    )


def translate(cells: Colored, vec: Tuple[int, int]) -> Colored:
    """Shift every cell by ``vec = (dr, dc)`` (colour preserved). NOT renormed -- a
    translate is a real position change, so the result keeps its absolute coords."""
    dr, dc = vec
    return frozenset((r + dr, c + dc, col) for r, c, col in cells)


def recolor(value, color):
    """Value op: set the colour value to ``color`` (overwrite). The operand is the
    scalar colour value; the new colour is returned."""
    return color


def cycle(value, step: int, cycle_list: Sequence):
    """Value op: advance ``value`` ``step`` positions along ``cycle_list`` with MOD
    wrap -- ``cycle_list[(cycle_list.index(value) + step) % len(cycle_list)]``.
    ``step`` may be negative; wrap is modular either way."""
    idx = list(cycle_list).index(value)
    seq = list(cycle_list)
    return seq[(idx + step) % len(seq)]


def set_(value, new):
    """Value op: overwrite ``value`` with ``new`` (named ``set_`` to avoid shadowing
    the builtin; registered under ``xf_set``)."""
    return new


_TRANSFORMS = {
    "xf_rotate": rotate,
    "xf_scale": scale,
    "xf_translate": translate,
    "xf_recolor": recolor,
    "xf_cycle": cycle,
    "xf_set": set_,
}


def register_transforms() -> None:
    """Register the six ``xf_*`` appliers (idempotent). Each is guarded by
    :func:`registry.is_registered` so a re-call after a registry clear (or a double
    import) does not raise. Exposed so tests can re-arm after the conftest wipes the
    registry."""
    for impl_key, fn in _TRANSFORMS.items():
        if not registry.is_registered(impl_key):
            registry.transform(impl_key)(fn)


register_transforms()
