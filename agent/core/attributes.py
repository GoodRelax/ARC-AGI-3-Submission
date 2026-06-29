"""DetectFeatures — derive a Profile (Characteristics) from an object's geometry.

Attributes are derived functions over an object's cells (object-schema §3): each
is a ``Word``, its value a ``Characteristic`` on the ``Profile``. This module
hosts the static feature detectors and registers them under their ``feat_*``
impl_key in the dispatch registry, so the AssetLoader wires them and the parallel
Lexicon session can add features the same way (a words.tsv row + a registered
detector).

Scope so far: the STATIC detectors that read only the object's own cells +
colours — ``feat_color`` (verbalization §3 colour = dominant) and ``feat_shape``
(verbalization §3 shape bases, rotation/scale-invariant topology) — plus, from
step 6-3.3, ``feat_afford`` (verbalization §4 behaviour modifiers, derived from
the world_model :class:`Affordance` evidence + a per-object field flag carried on
:class:`FeatureContext`).
From step 6-3.5 (pose geometry / orientation representation):
  * ``feat_flipped`` is now LIVE — the derived handedness Word, a 1.0/absent view of
    :attr:`agent.core.model.GameObject.reflected` (relayed via
    ``FeatureContext.reflected``).
  * the pose geometry VALUES (orientation / reflected / size / symmetry_order) are
    MEASURED here by :func:`compute_pose_geometry` (the single pose-measurement
    point) but live on the GameObject (object-schema §3 / TERM-43..46), NOT in the
    Profile — the Word stays 1-D.
From step 6-3.7 (goal-marker wiring):
  * ``feat_afford`` sub-modifier ``mark`` is now LIVE — it emits the ``marked`` Word
    iff :attr:`FeatureContext.marked` is set. ``marked`` is a BOARD-CONTEXT signal
    (a rare-colour, non-field object), so the caller that holds the whole-frame
    colour census sets it (mirrors ``is_field``); the detector only relays it. This
    fires the ``has(marked)`` arm of the roles.tsv ``target`` recognizer.
Deferred to later sub-steps:
  * ``feat_afford`` sub-modifiers ``controllable`` / ``pose_mutable`` /
    ``interactive`` / ``interactive_target`` — their evidence (grounded pick /
    pose-carry / EffectSignature) is not wired yet; they report absent.
  * ``feat_scale`` / ``feat_orient`` — STAY deferred (NOT registered): they need the
    integer-magnitude representation settled (scale(O) is an integer factor, not a
    [0,1] magnitude — flagged); they name axes whose VALUES live on the GameObject.

DetectFeatures skips any feat_* Word whose detector is not yet registered, so the
Profile grows as detectors come online. Deterministic (DP-10): Words are visited
in Lexicon (TSV) order; ties broken by index.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

from agent.core import registry, transform
from agent.core.model import Characteristic, Lexicon, Profile
from agent.core.world_model import Affordance

Cell = Tuple[int, int]
# A detector returns (magnitude, confidence) if the feature is present, else None.
Detection = Optional[Tuple[float, float]]

_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class FeatureContext:
    """The object view a detector reads: its cells, per-colour cell counts, and
    (for the behaviour detectors) the world_model affordance evidence + a field
    flag.

    ``affordance`` is this object's :class:`agent.core.world_model.Affordance`
    evidence (from ``world_model.affordance_evidence``); ``None`` means no
    evidence was supplied (the object reads as ``static``). ``is_field`` is the
    per-object isField flag (object-schema field/地の場). Importing ``Affordance``
    from world_model is a one-way, acyclic core import (attributes -> world_model;
    world_model never imports attributes). Tests construct this directly.
    """

    cells: FrozenSet[Cell]
    color_counts: Mapping[int, int] = field(default_factory=dict)
    affordance: Optional[Affordance] = None
    is_field: bool = False
    reflected: bool = False
    marked: bool = False
    """Whether this (non-field) object is a goal-marker per the BOARD-CONTEXT
    rare-colour rule (a salient/goal-marker signal). Set by the caller that holds
    the whole-frame colour census (search_agent), never by a detector reading only
    this object's own cells -- ``marked`` needs per-colour totals over the WHOLE
    board, not this object's footprint. Mirrors :attr:`is_field` (a per-object flag
    the caller computes once and relays). ``feat_afford(mark)`` emits the ``marked``
    Word iff this is set, so the ``has(marked)`` arm of the roles.tsv ``target``
    recognizer fires. Default ``False`` -> byte-identical to the unwired baseline."""


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #

def _bbox(cells: FrozenSet[Cell]) -> Tuple[int, int, int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    return r0, r1, c0, c1, (r1 - r0 + 1), (c1 - c0 + 1)


def _degree(cell: Cell, cells: FrozenSet[Cell]) -> int:
    r, c = cell
    return sum(((r + dr, c + dc) in cells) for dr, dc in _NEIGHBOURS)


def _is_connected(cells: FrozenSet[Cell]) -> bool:
    """4-connectivity over the cell set. The §3 shape bases describe connected
    glyphs; a disconnected set (object-schema D3 permits these) is a ``blob``."""
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in _NEIGHBOURS:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(cells)


def _has_enclosed_hole(cells: FrozenSet[Cell]) -> bool:
    """True if some background cell inside the bbox is NOT reachable from the bbox
    border through background (4-conn) — i.e. the object encloses a hole (ring)."""
    r0, r1, c0, c1, _, _ = _bbox(cells)
    bg = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in cells
    }
    if not bg:
        return False
    border = {
        (r, c) for (r, c) in bg if r in (r0, r1) or c in (c0, c1)
    }
    seen = set(border)
    q = deque(border)
    while q:
        r, c = q.popleft()
        for dr, dc in _NEIGHBOURS:
            nb = (r + dr, c + dc)
            if nb in bg and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return bool(bg - seen)


def shape_base(cells: FrozenSet[Cell]) -> str:
    """The rotation/scale-invariant shape base (verbalization §3), first match in
    priority order; ``blob`` is the fallback."""
    if not cells:
        return "blob"
    r0, r1, c0, c1, h, w = _bbox(cells)
    size = len(cells)
    skeletal = size == h + w - 1
    endpoints = [cell for cell in cells if _degree(cell, cells) == 1]
    junctions = [cell for cell in cells if _degree(cell, cells) >= 3]

    if size == 1:
        return "dot"
    if not _is_connected(cells):
        return "blob"  # disconnected sets are not §3 glyphs (object-schema D3)
    if skeletal and (h == 1 or w == 1):
        return "bar"
    if size == h * w and h == w:
        return "box"
    if size == h * w and h != w:
        return "rect"
    if skeletal and len(junctions) == 1 and _degree(junctions[0], cells) == 4 and len(endpoints) == 4:
        return "cross"
    if skeletal and len(junctions) == 0 and len(endpoints) == 2 and h > 1 and w > 1:
        return "ell"
    if skeletal and len(junctions) == 1 and _degree(junctions[0], cells) == 3 and len(endpoints) == 3:
        return "tee"
    if _has_enclosed_hole(cells):
        return "ring"
    return "blob"


def dominant_color(color_counts: Mapping[int, int]) -> Optional[int]:
    """The most frequent colour over the object's cells (ties -> lowest index)."""
    if not color_counts:
        return None
    return min(color_counts, key=lambda c: (-color_counts[c], c))


# --------------------------------------------------------------------------- #
# pose geometry (TRS-pose: orientation / reflected / size / symmetry_order).
# The SINGLE pose-measurement point: VALUES live on GameObject (object-schema §3),
# NOT in the Profile. Colour-blind by construction.
# --------------------------------------------------------------------------- #

def compute_pose_geometry(
    cells: FrozenSet[Cell],
) -> Tuple[Optional[Tuple[float, float]], bool, Optional[Tuple[int, int]], int]:
    """Measure the four cells-derived pose geometry attributes of one object
    (object-schema §3 / terms.md TERM-43..46). Returns
    ``(orientation, reflected, size, symmetry_order)``:

      * ``size`` = bbox extent ``(h, w)`` (via :func:`_bbox`); ``None`` if empty.
      * pose geometry is COLOUR-BLIND: a single synthetic colour ``0`` is stamped
        on every cell (``colored = {(r, c, 0)}``) so the transform algebra's
        coloured-cell API is reused without colour influencing the result.
      * ``reflected`` = :func:`transform.reflected` of that colour-blind set (the
        handedness bit, ADR-016); ``False`` for empty.
      * ``symmetry_order`` = the count of C4 rotations of the colour-blind set whose
        ``tuple(sorted(rot))`` equals that of the renormed set — 1 (no symmetry) /
        2 (180°) / 4 (full C4, incl. the degenerate dot/box); 1 for empty.
      * ``orientation`` = the unit vector of the moment principal axis, ``None`` for
        empty. Centroid ``(r̄, c̄)``; second moments ``Ixx = Σ(r-r̄)²``,
        ``Iyy = Σ(c-c̄)²``, ``Ixy = Σ(r-r̄)(c-c̄)``; principal angle
        ``θ = 0.5·atan2(2·Ixy, Ixx-Iyy)`` giving ``(cosθ, sinθ)``. Orientation is a
        cells-derived SHAPE attribute and so MUST be translation-invariant: the
        moments are taken on the bbox-renormed cell set (see :func:`_principal_axis`),
        making the result byte-identical for every translation of the same shape.

    Two determinism conventions (DP-10, no RNG / builtin hash):
      * DEGENERATE inertia (``Ixx == Iyy`` AND ``Ixy == 0`` — an isotropic blob such
        as a dot, a box, or a plus) has no principal axis, so a fixed
        ``(1.0, 0.0)`` is returned.
      * 180° RAY AMBIGUITY: the principal axis is a line, so ``θ`` and ``θ+π`` both
        satisfy it. The ray is fixed deterministically toward the heavier tail using
        the third moment (skew) PROJECTED along the axis ``u = (cosθ, sinθ)``:
        ``skew = Σ (Δ·u)³`` with ``Δ = (r-r̄, c-c̄)``; if ``skew < -eps`` the vector is
        negated. When ``abs(skew) <= eps`` (symmetric mass, where the exact value is 0
        but float rounding leaves a tiny ±residue) the half-plane rule ``sinθ >= 0``
        (then ``cosθ >= 0`` on the boundary) picks the representative.

    Uses :mod:`agent.core.transform` (acyclic core import) for the C4/D4 algebra.
    """
    if not cells:
        return None, False, None, 1

    _, _, _, _, h, w = _bbox(cells)
    size = (h, w)

    colored = frozenset((r, c, 0) for (r, c) in cells)
    reflected = transform.reflected(colored)

    norm_key = tuple(sorted(transform.renorm(colored)))
    symmetry_order = sum(
        1 for rot in transform.rotations(colored) if tuple(sorted(rot)) == norm_key
    )

    orientation = _principal_axis(cells)
    return orientation, reflected, size, symmetry_order


def _principal_axis(cells: FrozenSet[Cell]) -> Tuple[float, float]:
    """The deterministic unit principal-axis vector of ``cells`` (see
    :func:`compute_pose_geometry` for the conventions). ``cells`` is non-empty.

    TRANSLATION-INVARIANT by construction: the moments are computed on the
    bbox-renormed cell set (top-left translated to the origin) rather than on the
    raw board coordinates. The float centroid is then ``O(1)`` regardless of the
    object's absolute position, so the moment sums — and the skew tiebreak below —
    are byte-identical for every translation of the same shape (GEOM-1). Renorm is
    integer (subtract min row / min col), so it adds no rounding of its own."""
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    norm = [(r - r0, c - c0) for r, c in cells]

    n = len(norm)
    rbar = sum(r for r, _ in norm) / n
    cbar = sum(c for _, c in norm) / n
    ixx = sum((r - rbar) ** 2 for r, _ in norm)
    iyy = sum((c - cbar) ** 2 for _, c in norm)
    ixy = sum((r - rbar) * (c - cbar) for r, c in norm)

    # Isotropic / degenerate: no principal axis (dot / box / plus). Fixed value.
    if ixx == iyy and ixy == 0:
        return (1.0, 0.0)

    theta = 0.5 * math.atan2(2 * ixy, ixx - iyy)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Resolve the 180-degree ray ambiguity toward the heavier tail (third moment
    # projected on the axis). For a shape whose principal axis is a symmetry line the
    # skew is EXACTLY 0 in exact arithmetic, but the float sum of cubed projections
    # rounds to a tiny ±residue whose SIGN is otherwise unstable. Use a tolerance band
    # so such shapes deterministically take the half-plane fallback instead of an
    # arbitrary signed ray. eps scales with the moment magnitude: the cubed projection
    # sum has O(n) terms each up to ~(bbox extent)^3, so a relative band
    # 1e-9 * (sum of |projection|^3 + 1) tracks the scale; the +1 keeps it nonzero for
    # tiny shapes. (An absolute 1e-6 would also work for our small glyphs, but the
    # relative form stays correct if extents grow.)
    proj = [(r - rbar) * cos_t + (c - cbar) * sin_t for r, c in norm]
    skew = sum(p ** 3 for p in proj)
    eps = 1e-9 * (sum(abs(p) ** 3 for p in proj) + 1.0)
    if skew < -eps:
        cos_t, sin_t = -cos_t, -sin_t
    elif abs(skew) <= eps:
        if sin_t < 0 or (sin_t == 0 and cos_t < 0):
            cos_t, sin_t = -cos_t, -sin_t
    return (cos_t, sin_t)


# --------------------------------------------------------------------------- #
# detectors (registered under feat_* impl_keys)
# --------------------------------------------------------------------------- #

def _detect_color(ctx: FeatureContext, params: Mapping[str, str]) -> Detection:
    raw = params.get("index")
    if raw is None:
        return None
    idx = int(raw)
    cc = ctx.color_counts
    dom = dominant_color(cc)
    if dom is None or dom != idx:
        return None
    total = sum(cc.values())
    if total <= 0:
        return None
    return (1.0, cc[idx] / total)


def _detect_shape(ctx: FeatureContext, params: Mapping[str, str]) -> Detection:
    topo = params.get("topo")
    if topo is None or shape_base(ctx.cells) != topo:
        return None
    # 'blob' is the fallback base -> lower confidence than a positive topology match.
    confidence = 0.5 if topo == "blob" else 1.0
    return (1.0, confidence)


def _detect_afford(ctx: FeatureContext, params: Mapping[str, str]) -> Detection:
    """Behaviour modifier detector (verbalization §4). Dispatches on the
    ``affordance`` param to one of the §4 derivations over ``ctx.affordance``
    (the world_model evidence) and ``ctx.is_field``.

    Magnitude is categorical presence (always ``1.0`` when present); confidence is
    the backing support ratio for the dynamics channels, else ``1.0`` for the
    boolean flags. Mirrors :func:`_detect_color` (presence 1.0, confidence =
    fraction). Returns ``None`` when the modifier is absent or its evidence source
    is not yet wired.
    """
    affordance = params.get("affordance")
    if affordance is None:
        return None
    af = ctx.affordance

    if affordance == "movable":
        # §4: translate_support > 0.
        if af is not None and af.translate_support > 0:
            return (1.0, af.translate_support)
        return None
    if affordance == "vanishing":
        # §4: vanish_support > 0.
        if af is not None and af.vanish_support > 0:
            return (1.0, af.vanish_support)
        return None
    if affordance == "spawning":
        # §4: spawn_support > 0.
        if af is not None and af.spawn_support > 0:
            return (1.0, af.spawn_support)
        return None
    if affordance == "recolorable":
        # §4: response_support > 0 AND translate_support == 0.
        if af is not None and af.response_support > 0 and af.translate_support == 0:
            return (1.0, af.response_support)
        return None
    if affordance == "autonomous":
        # §4: same displacement under >= 2 distinct actions (boolean flag).
        if af is not None and af.autonomous:
            return (1.0, 1.0)
        return None
    if affordance == "field":
        # Per-object isField flag (boolean).
        if ctx.is_field:
            return (1.0, 1.0)
        return None
    if affordance == "mark":
        # Goal-marker salience (verbalization §4 `mark`). NOT derived from the
        # world_model affordance evidence: ``marked`` is a BOARD-CONTEXT signal
        # (a rare-colour, non-field object) that the caller computes once over the
        # whole frame and relays via ``ctx.marked`` (mirrors ``ctx.is_field``).
        # Present (1.0, 1.0) iff that flag is set; this is the ``has(marked)`` arm
        # of the roles.tsv `target` recognizer.
        if ctx.marked:
            return (1.0, 1.0)
        return None
    if affordance == "static":
        # §4: no dynamics evidence => static is detectable (roles.tsv `field`
        # reads has(static)); naming omits it but detection emits it. Present iff
        # there is no affordance evidence at all: None, OR all four supports == 0
        # and not autonomous.
        #
        # MARKED SUPPRESSES STATIC: a `marked` object is a FOREGROUND goal-marker,
        # not background. Because the `field` recognizer or(has(is_field),
        # has(static)) runs in the non-relational wave (BEFORE the relational
        # `target`), a marked-but-motionless goal object would otherwise be claimed
        # as `field` and never reach `target`. Withholding `static` when ctx.marked
        # keeps the field-precedence invariant intact for genuine background (which
        # is never marked) while letting the marked goal flow to `target`. This is
        # the single-frame unblock; multi-frame `static` refinement stays deferred.
        # OFF byte-identity: ctx.marked is False when ARC_MARKED is off, so this
        # branch is unchanged from the baseline.
        if ctx.marked:
            return None
        if af is None or (
            af.translate_support == 0
            and af.vanish_support == 0
            and af.spawn_support == 0
            and af.response_support == 0
            and not af.autonomous
        ):
            return (1.0, 1.0)
        return None

    # DEFERRED (evidence source not wired yet, lands in a later step):
    #   controllable      -> FR-168 grounded controllable (grounded pick, not in
    #                        the affordance evidence);
    #   pose_mutable      -> pose-carry signal;
    #   interactive / interactive_target -> EffectSignature (function naming §5);
    #   clickable / gauge / blocking / lethal -> their evidence is not wired.
    # These must not crash; they simply report absent for now.
    return None


def _detect_flipped(ctx: FeatureContext, params: Mapping[str, str]) -> Detection:
    """Derived handedness Word ``flipped`` (verbalization §3 / TERM-48): the
    surface view of :attr:`GameObject.reflected`. The pose VALUE is measured once by
    :func:`compute_pose_geometry` and carried on the object; the FeatureContext
    relays it via ``ctx.reflected``. Present (``(1.0, 1.0)``) iff that bit is set.

    Guards on ``params['derived'] == 'reflected'`` (the words.tsv ``derived=reflected``
    convention) so a mis-routed param does not silently emit."""
    if params.get("derived") != "reflected":
        return None
    return (1.0, 1.0) if ctx.reflected else None


_DETECTORS = {
    "feat_color": _detect_color,
    "feat_shape": _detect_shape,
    "feat_afford": _detect_afford,
    "feat_flipped": _detect_flipped,
}


def register_detectors() -> None:
    """Register the static feature detectors (idempotent)."""
    for impl_key, fn in _DETECTORS.items():
        if not registry.is_registered(impl_key):
            registry.feature(impl_key)(fn)


register_detectors()


# --------------------------------------------------------------------------- #
# DetectFeatures use case
# --------------------------------------------------------------------------- #

def detect_features(ctx: FeatureContext, lexicon: Lexicon) -> Profile:
    """Build a Profile by running every registered ``feat_*`` detector over the
    object. Words whose detector is not (yet) registered are skipped, so the
    Profile grows as detectors come online."""
    chars = []
    for w in lexicon.words:
        if not w.impl_key.startswith("feat_"):
            continue
        fn = registry.resolve(w.impl_key)
        if fn is None:
            continue
        result = fn(ctx, w.params)
        if result is not None:
            magnitude, confidence = result
            chars.append(Characteristic(w.id, magnitude, confidence))
    return Profile(chars)


def _color_counts_from_grid(cells: FrozenSet[Cell], grid: Dict[Cell, int]) -> Dict[int, int]:
    """Helper: colour histogram over ``cells`` given a cell->colour map (the frame
    coloring g of object-schema §1). Convenience for callers that hold a grid."""
    counts: Dict[int, int] = {}
    for cell in cells:
        color = grid.get(cell)
        if color is not None:
            counts[color] = counts.get(color, 0) + 1
    return counts
