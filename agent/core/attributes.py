"""[agent/core] The four FOUNDATIONAL ARC attributes as separately-managed axes.

Per the star-star directive: colour, shape, orientation, size are the root attributes of
ARC objects, and ORIENTATION is a first-class axis -- never folded into position or shape.

  colour      -> dom_color (+ the per-cell colour map in Obj.colored)
  size        -> cell count; plus the detected integer SCALE of an up-scaled rendering
  shape       -> rotation- and scale-INVARIANT canonical of the colour pattern (identity)
  orientation -> the POSE: the scale-normalised, translation-normalised colour pattern.
                 Its rotation index relative to the shape canonical is the orientation label.

This is general (no game literals). It is assumption-light about *how* a game renders an
orientation: ``pose`` is just the observed colour pattern at primitive scale, and two poses
match iff their patterns are equal -- we never assume a render is a clean pixel rotation.
``shape`` additionally quotients out rotation so the same object in different poses shares
one identity.

MATCH(a, b, attrs) compares only the requested axes, so a goal can demand
"same colour + shape + orientation, ANY scale" (the ls20 carried-state vs goal-mark case,
where the carried state is drawn at 2x and the goal mark at 1x).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Optional

from .perceive import Obj


# --------------------------------------------------------------------- pattern helpers
def _renorm(colored) -> frozenset:
    """Translate a {(r, c, color)} set so its bbox top-left is the origin."""
    if not colored:
        return frozenset()
    mr = min(r for r, _, _ in colored)
    mc = min(c for _, c, _ in colored)
    return frozenset((r - mr, c - mc, col) for r, c, col in colored)


def _detect_scale(norm) -> int:
    """Largest integer s such that ``norm`` is an s-times up-scaling: partition the bbox into
    s x s blocks; every block must be empty OR fully filled with a single colour."""
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


def downscale(norm) -> frozenset:
    """Reduce an s-up-scaled pattern to primitive scale (one cell per filled block)."""
    s = _detect_scale(norm)
    if s == 1:
        return frozenset(norm)
    return frozenset((r // s, c // s, col) for r, c, col in norm if r % s == 0 and c % s == 0)


def rot90cw(norm) -> frozenset:
    """Rotate a normalised {(r, c, color)} pattern 90 degrees clockwise (colour preserved)."""
    if not norm:
        return frozenset()
    h = max(r for r, _, _ in norm) + 1
    return _renorm(frozenset((c, h - 1 - r, col) for r, c, col in norm))


def rotations(norm) -> list:
    out = [frozenset(norm)]
    for _ in range(3):
        out.append(rot90cw(out[-1]))
    return out


# --------------------------------------------------------------------- the four axes
def color(obj: Obj) -> int:
    return obj.dom_color


def size(obj: Obj) -> int:
    return obj.size


def pose(obj: Obj) -> frozenset:
    """ORIENTATION axis: the scale-normalised, translation-normalised colour pattern.
    Distinct poses (orientations) of the same shape are distinct values."""
    return downscale(_renorm(obj.colored))


def shape(obj: Obj) -> frozenset:
    """SHAPE axis: rotation- and scale-invariant identity = the canonical (min) rotation of
    the pose. The same object in any orientation maps to one shape value."""
    return min((tuple(sorted(p)) for p in rotations(pose(obj))))


def orientation_index(obj: Obj):
    """The orientation label: the rotation index k in 0..3 with rot^k(shape) == pose, or None."""
    p = pose(obj)
    canon = shape(obj)
    cur = frozenset(canon)
    for k in range(4):
        if cur == p:
            return k
        cur = rot90cw(cur)
    return None


def match(a: Obj, b: Obj, attrs=("color", "shape", "orientation")) -> bool:
    """True iff ``a`` and ``b`` agree on every requested attribute axis.

    ``color`` -> same dominant colour. ``shape`` -> same rotation/scale-invariant identity.
    ``orientation`` -> same pose (which, given equal shape, pins the rotation). ``size`` ->
    same raw cell count (omit it to match across scales, as ls20's 2x carried vs 1x goal does).
    """
    if "color" in attrs and a.dom_color != b.dom_color:
        return False
    if "size" in attrs and a.size != b.size:
        return False
    if "shape" in attrs and shape(a) != shape(b):
        return False
    if "orientation" in attrs and pose(a) != pose(b):
        return False
    return True


# =====================================================================================
# Attributes / verbalization cluster
# -------------------------------------------------------------------------------------
# Object naming (Verbalize, CMP-24) + effect-signature role keys (EffectSignature,
# TERM-17). Canon (cite, never duplicate):
#   - docs/StrictDoc-specs/_assets/gr-arc-3-verbalization.md  (Naming Ladder + render;
#     EffectSignature := {event, target, attribute, operator, params_class}; honest limit)
#   - docs/StrictDoc-specs/_assets/gr-arc-3-terms.md          (TERM-27 DerivedNaming,
#     tau default 0.8; TERM-17 EffectSignature; TERM-18 Lexicon; TERM-25 Profile;
#     TERM-26 Dimension)
#   - docs/StrictDoc-specs/_assets/gr-arc-3-domain-model.md   (Profile / Lexicon / Dimension)
#
# Hard rules honoured here:
#   * No game-specific literals in any matching/naming KEY (NFR-6): an EffectSignature key
#     is built ONLY from {event, target, attribute, operator, params_class}; never a colour
#     number, coordinate, or glyph. (A palette colour-NAME reference list is allowed for
#     RENDER vocabulary only -- it is never part of a role/signature key.)
#   * Determinism (DP-10): no `random`; no Python builtin `hash()` for stable identity
#     (per-process salted). Stable identity uses sorted tuples / hashlib only. Tie-breaks
#     are deterministic (ascending slot rank, then ascending Dimension name).
# =====================================================================================

# Default confidence threshold for the Naming Ladder (TERM-27 DerivedNaming, tau=0.8).
DEFAULT_TAU: float = 0.8
# Default cap on the number of modifiers in a short name (verbalization v001 sec.2.2, N=5).
DEFAULT_MAX_SLOTS: int = 5

# The Naming Ladder slot order (verbalization v001 sec.2.1). Lower rank = closer to the
# LEFT (most role-determining / most stable); the head-noun (shape base) is the rightmost,
# grammatical noun. These are slot NAMES (axes), not game literals.
LADDER_SLOTS: tuple = (
    "controllability",      # controllability / capability  (e.g. controllable, uncontrollable)
    "size",                 # size                          (e.g. small, large)
    "behavior",             # behavior / dynamics           (e.g. moving, static)  -- or a
    #                         learned function-modifier, which supersedes the coarse one
    "shape",                # shape-as-modifier (rare; head-noun is the usual shape slot)
    "color",                # surface colour                (e.g. red, blue)
    "origin",              # origin / provenance
    "texture",              # texture
    "function_modifier",    # most-specific learned interaction role (just before the noun)
    "head_noun",            # the grammatical noun = shape base (NEVER dropped by tau)
)
# Rank lookup: slot name -> its ladder position (ascending = left-to-right).
_SLOT_RANK: dict = {name: i for i, name in enumerate(LADDER_SLOTS)}
# The head-noun slot is the grammatical noun: a name must keep it even at low confidence,
# otherwise an object could render to the empty string. (verbalization v001 sec.2.1.)
HEAD_NOUN_SLOT: str = "head_noun"


@dataclass(frozen=True)
class Dimension:
    """An attribute axis (a Lexicon "word" definition) -- TERM-26.

    A Dimension names ONE measurable axis (controllability, size, behavior, shape, colour,
    ...) and says which ladder ``slot`` its value occupies when a name is composed. It is
    general: the axis is an observable property, never a game-specific meaning. ``dim_id``
    is a stable, version-pinned identifier; ``salient`` lets a value carry "present but not
    worth naming" (e.g. a neutral/default reading) so it can be suppressed independently of
    confidence.
    """

    dim_id: str                       # stable id (e.g. "controllability"); version-pinned
    slot: str                         # which LADDER_SLOTS entry this value occupies
    rank: int = 0                     # intra-slot tie-break / detector precedence (ascending)

    def slot_rank(self) -> int:
        """Ladder position of this Dimension's slot (ascending = closer to the left).
        Unknown slots sort AFTER every known slot (deterministically), never crashing."""
        return _SLOT_RANK.get(self.slot, len(LADDER_SLOTS))


@dataclass(frozen=True)
class Lexicon:
    """Base vocabulary: (Dimension value) -> word, plus the Dimension definitions -- TERM-18.

    Two layers (terms v003): a read-only ``base`` (shipped, version-pinned) and an optional
    ``overlay`` (in-game additions only; it never OVERWRITES base -- base wins on conflict,
    so a name is stable within a game). A word is looked up by ``(dim_id, value)``. Naming is
    DERIVED from this table by :func:`render`; there is no per-object hand-saved name here.

    The optional ``colour_names`` map is a fixed palette colour-NAME reference for RENDER
    only (so a colour value reads as e.g. "red" in a log). It is vocabulary, NOT a matching
    key: no signature or role key ever consults it (NFR-6).
    """

    dims: dict = field(default_factory=dict)          # dim_id -> Dimension
    base: dict = field(default_factory=dict)          # (dim_id, value) -> word   (read-only)
    overlay: dict = field(default_factory=dict)       # (dim_id, value) -> word   (additive)
    colour_names: dict = field(default_factory=dict)  # colour value -> name  (render only)

    def dimension(self, dim_id: str) -> Optional[Dimension]:
        return self.dims.get(dim_id)

    def word(self, dim_id: str, value) -> Optional[str]:
        """Resolve a Dimension value to its word. ``base`` takes precedence over ``overlay``
        (terms v003: overlay is additive-only and may not shadow base). Returns ``None`` when
        no word is known (the value is then simply omitted from the name)."""
        key = (dim_id, value)
        if key in self.base:
            return self.base[key]
        if key in self.overlay:
            return self.overlay[key]
        return None


@dataclass(frozen=True)
class Profile:
    """A sparse property vector for ONE object -- TERM-25.

    ``entries`` maps a Dimension id to ``(value, confidence)``. The definition (axis, slot)
    is frozen in the :class:`Lexicon`; only the value/confidence update over time. The name
    is DERIVED from this Profile via :func:`render` -- a Profile holds no saved name.
    """

    entries: dict = field(default_factory=dict)       # dim_id -> (value, confidence)

    def get(self, dim_id: str):
        """Return ``(value, confidence)`` for ``dim_id`` or ``None`` if absent."""
        return self.entries.get(dim_id)


def render(
    profile: Profile,
    lexicon: Lexicon,
    tau: float = DEFAULT_TAU,
    max_slots: int = DEFAULT_MAX_SLOTS,
) -> str:
    """The Naming Ladder short name -- DerivedNaming (TERM-27); rule SSOT = verbalization
    v001 sec.2.

    Algorithm (verbalization v001 sec.2.2), made fully deterministic:
      1. keep entries with ``confidence >= tau`` (drop uncertain modifiers) AND whose value
         resolves to a Lexicon word -- EXCEPT the head-noun, which is always kept (the name
         needs a grammatical noun; verbalization sec.2.1);
      2. order by (ladder slot rank, intra-slot rank, dim_id) -- a TOTAL deterministic order,
         independent of dict insertion order, call context, and game id;
      3. cap modifiers at ``max_slots`` (head-noun is reserved and not counted out);
      4. hyphen-compose the words in ladder order.

    The output is a pure function of (profile, lexicon, tau, max_slots): identical across
    repeated calls, contexts, and game ids (no RNG, no ``hash()``, no clock). When the
    hand-saved-name table is empty/unset, a derived name is still produced -- there is no
    saved-name table to consult.
    """
    if tau is None:
        tau = DEFAULT_TAU

    head: Optional[tuple] = None              # (rank-key, word) for the head-noun
    mods: list = []                           # [(rank-key, word)] for confident modifiers
    for dim_id, ve in profile.entries.items():
        value, conf = ve
        dim = lexicon.dimension(dim_id)
        slot = dim.slot if dim is not None else dim_id
        intra = dim.rank if dim is not None else 0
        slot_rank = dim.slot_rank() if dim is not None else len(LADDER_SLOTS)
        rank_key = (slot_rank, intra, dim_id)   # total, deterministic ordering key
        word = lexicon.word(dim_id, value)
        if slot == HEAD_NOUN_SLOT:
            # The grammatical noun is mandatory and tau-exempt; keep the lowest-rank one.
            if word is not None and (head is None or rank_key < head[0]):
                head = (rank_key, word)
            continue
        if conf < tau or word is None:
            continue                             # drop uncertain / unword-able modifiers
        mods.append((rank_key, word))

    mods.sort(key=lambda t: t[0])               # ladder order (deterministic)
    if max_slots is not None and len(mods) > max_slots:
        mods = mods[:max_slots]                  # cap modifier count (head-noun is separate)

    parts = [w for _, w in mods]
    if head is not None:
        parts.append(head[1])
    return "-".join(parts)


# ---------------------------------------------------------------- EffectSignature (TERM-17)
# The five role-key axes (verbalization v001 sec.5.2):
#   signature := (event, target, attribute, operator, params_class)
# NOT colour / position / glyph. ``params_class`` is the *class* of the operator's params
# (e.g. a step magnitude bucket, a cycle length, a rotation quarter-count) -- a structural
# descriptor, never a concrete colour value or coordinate.
_SIG_FIELDS: tuple = ("event", "target", "attribute", "operator", "params_class")


@dataclass(frozen=True, order=True)
class EffectSignature:
    """A hashable role key built ONLY from functional axes -- EffectSignature (TERM-17).

    Two objects with the SAME signature are *analogous* (same functional role); two with a
    similar SURFACE but a different signature are *contrasted* (different role) -- e.g.
    a white-plus ``rotate+90(ref.orientation)`` vs a white-4-dot ``cycle+1(ref.form)`` look
    alike but separate (verbalization v001 sec.5.2). The key deliberately EXCLUDES colour,
    position, and glyph, so permuting an object's surface colour leaves its signature
    unchanged (NFR-6).

    Equality / hashing / ordering derive from the five fields in a fixed order, so the key is
    deterministic and stable across processes (frozen dataclass tuple identity -- no salted
    builtin ``hash()`` of mutable surface data).
    """

    event: str                  # geometric trigger: overlap / enter / adjacent / click ...
    target: str                 # whom the effect lands on: self / this / ref / board
    attribute: str              # typed state axis: position / orientation / form / colour /
    #                             count / presence / scalar  (the NAME of the axis, not a value)
    operator: str               # the verb: set / inc / dec / cycle / rotate / toggle / block ...
    params_class: str = ""      # STRUCTURAL class of the operator params (bucket / arity);
    #                             never a concrete colour number or coordinate

    def key(self) -> tuple:
        """The bare comparison tuple (event, target, attribute, operator, params_class).
        Deterministic and free of any surface (colour/coords/glyph) information."""
        return (self.event, self.target, self.attribute, self.operator, self.params_class)

    def stable_id(self) -> str:
        """A process-stable short id for logs/handles -- hashlib over the key (NOT builtin
        ``hash()``), so it is identical across runs, contexts, and game ids.

        The key is UTF-8 encoded (NOT ASCII): the canonical operator vocabulary itself uses
        non-ASCII glyphs (verbalization v001 sec.5.1 lists ``rotate+theta`` as ``rotate+`` +
        Greek theta), so an ASCII-only encode would raise on a perfectly valid role label.
        UTF-8 is deterministic and lossless, so the id stays identical across runs.
        """
        digest = hashlib.sha1("\x1f".join(self.key()).encode("utf-8")).hexdigest()
        return digest[:12]


def effect_signature(
    event: str,
    target: str,
    attribute: str,
    operator: str,
    params_class: str = "",
) -> EffectSignature:
    """Build an :class:`EffectSignature` from the five functional axes only (TERM-17).

    Surface inputs (colour, coordinate, glyph) are intentionally NOT parameters here: the
    role key cannot read them by construction (NFR-6 -- no game-specific literal in a key).
    """
    return EffectSignature(
        event=event,
        target=target,
        attribute=attribute,
        operator=operator,
        params_class=params_class,
    )


def analogous(a: EffectSignature, b: EffectSignature) -> bool:
    """True iff ``a`` and ``b`` denote the SAME functional role (identical signature key).
    Same-role recognition is by function, not appearance (recognition-before-probe, AP-4)."""
    return a.key() == b.key()


def contrasted(a: EffectSignature, b: EffectSignature) -> bool:
    """True iff ``a`` and ``b`` denote DIFFERENT roles (signature keys differ). Surface
    similarity is irrelevant -- two surface-alike objects with different signatures contrast."""
    return a.key() != b.key()


# =====================================================================================
# World render / Goal render (verbalization v001 sec.6 / sec.7) -- TS-06 / SC-06.
# -------------------------------------------------------------------------------------
# A WorldModel (its InteractionRule set) -> a ONE-LINE NL string of the salient rules as
# plain verb phrases; a GoalPredicate tree -> a ONE-LINE win-condition string with AND/OR/
# NOT joined recursively. Objects are referred to by their DERIVED short name (``render``).
#
# These helpers are DUCK-TYPED (structural), never importing goal.py / world_model.py:
#   - those modules import situation.py, which imports THIS module, so importing them back
#     would be circular. Like ``classify_move_effect`` (world_model.py), the contract is the
#     SHAPE of the argument, not its concrete class.
#   - a "rule" is anything exposing ``.name`` (a game-literal-free label, e.g. "move" /
#     "loss:move_budget" / "block:box") and optionally ``.confidence`` (a float in [0,1]).
#   - a "goal" is anything exposing ``.describe() -> str`` (a one-line, ASCII AND/OR/NOT
#     rendering whose leaves name role LABELS) -- exactly GoalPredicate.describe (goal.py).
#
# Honest limit (verbalization v001 sec.1): the rendered words are OBSERVATION-DERIVED only.
# A fixed list of game-SEMANTIC words (``key`` / ``door`` / ``enemy`` ...) must NEVER appear:
# claiming them would assert a latent game meaning the agent cannot observe (NFR-6). The
# render vocabulary is structural verbs + the derived short names; nothing here consults a
# colour number, coordinate, or glyph.
# =====================================================================================

# Game-SEMANTIC words a render must never emit (verbalization v001 sec.1 honest-limit). These
# are meanings you only learn by PLAYING a specific game; asserting one from observation alone
# is hardcoding. The list is the canonical sec.1 set plus its closest synonyms. Used by the
# TS-06 oracle (assert absence) and by callers that want to self-check a rendered line.
FORBIDDEN_WORDS: frozenset = frozenset({
    "key", "door", "lock", "wall", "enemy", "hazard", "player", "avatar",
    "goal", "exit", "win", "lose", "death", "fuel", "health", "coin", "gem",
    "ball", "paddle", "bullet", "weapon", "treasure", "monster", "trap", "spike",
})


def _tokens(text: str) -> list:
    """Split ``text`` into role-label-style tokens (maximal runs of label characters), so a
    word check matches WHOLE tokens. This keeps a structural verb like ``block`` from tripping
    a substring match on the game-word ``lock`` (the two are different tokens)."""
    out: list = []
    cur: list = []
    for ch in text:
        if ch in _LABEL_CHARS:
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def forbidden_words_in(text: str) -> list:
    """The game-SEMANTIC words (:data:`FORBIDDEN_WORDS`) that appear as WHOLE tokens in
    ``text`` -- empty iff the line is honest-limit clean (verbalization v001 sec.1). Matching is
    case-insensitive and token-boundary aware (so ``block`` does not match ``lock``). The TS-06
    oracle asserts this is empty for every rendered World / Goal line."""
    present = {t.lower() for t in _tokens(text)} & FORBIDDEN_WORDS
    return sorted(present)

# How a rule's game-literal-free NAME maps to a plain English verb phrase. The name encodes a
# learned interaction (verbalization v001 sec.5 operator vocab); we surface it as a readable
# clause WITHOUT introducing any game meaning. A name is typically ``"<op>"`` or
# ``"<op>:<arg>"`` where ``<arg>`` is a role label / Dimension name (never a literal).
_RULE_VERB: dict = {
    "move": "moves on input",
    "loss": "ends the play when %s reaches zero",
    "block": "blocks movement at %s",
    "translate": "is pushed along by a move",
    "recolor": "changes colour on contact",
    "rotate": "turns on contact",
    "cycle": "advances %s on contact",
    "reach": "is reached by moving",
    "deform": "changes form on contact",
}


def _rule_phrase(rule, lexicon: Optional[Lexicon] = None) -> str:
    """One plain verb phrase for an InteractionRule-like ``rule`` (sec.6 template).

    Derives the clause from ``rule.name`` only -- a game-literal-free label of the form
    ``"<op>"`` or ``"<op>:<arg>"`` (the ``<arg>`` is a role label or Dimension NAME). Unknown
    ops fall back to a generic "<op> rule" phrase, so a new operator never crashes and never
    invents a game meaning. No colour/coordinate/glyph is read (NFR-6)."""
    name = getattr(rule, "name", "") or "rule"
    head, _, arg = name.partition(":")
    template = _RULE_VERB.get(head)
    if template is None:
        return ("%s rule" % head) if not arg else ("%s rule on %s" % (head, arg))
    if "%s" in template:
        return template % (arg if arg else "it")
    return template


def verbalize_world(
    rules,
    lexicon: Optional[Lexicon] = None,
    max_rules: int = DEFAULT_MAX_SLOTS,
) -> str:
    """Render a WorldModel's rule set to a ONE-LINE NL summary of its SALIENT rules
    (verbalization v001 sec.6); TS-06 / SC-06.

    ``rules`` is any iterable of InteractionRule-like objects (duck-typed: ``.name`` +
    optionally ``.confidence``). Salience = highest confidence first; ties broken by ascending
    name (a TOTAL deterministic order -- DP-10, no RNG / no ``hash()``). At most ``max_rules``
    clauses are kept, joined by `` / ``. Each clause is a plain verb phrase derived ONLY from
    the rule's game-literal-free name (:func:`_rule_phrase`), so no game-semantic word
    (:data:`FORBIDDEN_WORDS`) can appear. Returns ``""`` for an empty rule set.

    The output is a pure function of the rules' (name, confidence): identical across repeated
    calls, contexts, and game ids."""
    items = list(rules)
    # Deterministic salience order: confidence desc, then name asc (total order).
    items.sort(key=lambda r: (-float(getattr(r, "confidence", 1.0)), getattr(r, "name", "")))
    if max_rules is not None and len(items) > max_rules:
        items = items[:max_rules]
    phrases = [_rule_phrase(r, lexicon) for r in items]
    return " / ".join(phrases)


def verbalize_goal(
    goal,
    lexicon: Optional[Lexicon] = None,
    names: Optional[Mapping[str, str]] = None,
) -> str:
    """Render a GoalPredicate tree to a ONE-LINE win-condition string (verbalization v001
    sec.7); TS-06 / SC-06.

    ``goal`` is any object exposing ``describe() -> str`` (duck-typed) -- exactly
    ``GoalPredicate.describe`` (goal.py), which already joins AND/OR/NOT recursively into one
    ASCII line whose atom terms are ROLE LABELS. When a ``names`` map (role label -> derived
    short name) is given, each whole-word role label in the line is substituted by its short
    name (so the line refers to objects by the SAME derived name :func:`render` produces). The
    substitution is plain word-boundary replacement; it never introduces a literal.

    Determinism: ``describe`` is itself deterministic (canonical tree order), and the
    substitution is a fixed-order pass over ``sorted(names)`` longest-first (so a label that is
    a prefix of another is replaced correctly). No game-semantic word is added."""
    line = goal.describe()
    if names:
        # Replace longest labels first so "carried-state" is handled before "state" etc.; do a
        # word-boundary-aware replace so we never corrupt a substring of another token.
        for label in sorted(names, key=lambda s: (-len(s), s)):
            short = names[label]
            line = _replace_word(line, label, short)
    return line


def _replace_word(text: str, word: str, repl: str) -> str:
    """Replace whole-token occurrences of ``word`` in ``text`` with ``repl``. A token boundary
    is the start/end of string or any character that is not part of a role label (role labels
    are ``[A-Za-z0-9_-]+``). Deterministic and literal-free."""
    if not word:
        return text
    out: list = []
    i = 0
    n = len(text)
    wlen = len(word)
    label_chars = _LABEL_CHARS
    while i < n:
        if text[i : i + wlen] == word:
            before_ok = i == 0 or text[i - 1] not in label_chars
            after = i + wlen
            after_ok = after >= n or text[after] not in label_chars
            if before_ok and after_ok:
                out.append(repl)
                i = after
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


# Characters that may appear inside a role label / short name (so a replace respects token
# boundaries). Role labels are hyphen/underscore-joined alphanumerics; short names too.
_LABEL_CHARS: frozenset = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)
