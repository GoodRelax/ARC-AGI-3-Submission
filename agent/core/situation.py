"""[agent/core] AbstractSituation + StateAbstraction -- the salient-only abstract state (CMP-06 /
CMP-25; SC-18 / FR-C-4).

Canon (cite, never duplicate):
  - docs/StrictDoc-specs/_assets/gr-arc-3-terms.md
      TERM-31 AbstractSituation       -- minimal sufficient abstract state: role -> salient Profile,
                                 holding salient Relations, plus scalar components
                                 (Move Budget, TERM-32); hash() collapses equal configs so
                                 search is memoised. A «value» (no identity).
      TERM-24 StateAbstraction -- map (frame, prev) -> AbstractSituation, selecting salient and
                                 dropping non-salient, carrying the previous state forward.
  - docs/StrictDoc-specs/_assets/gr-arc-3-domain-model.md
      AbstractSituation = salient role -> Profile + holding Relations + observed scalars;
      StateAbstraction.project selects salient objects/dimensions/relations, carry-forward.
  - docs/StrictDoc-specs/04-specification.sdoc  SC-18 (the acceptance scenario)
  - docs/StrictDoc-specs/05-test-strategy.sdoc  TS-18 (the executable oracle)

Hard rules honoured here:
  * No game literals (NFR-6). Salience is decided from OBSERVATION ONLY: an object is
    salient iff it carries a *participating* recognised role (controllable / interactor /
    goal-target / carried-state ...) -- plus any observed scalar gauge. Background
    (``is_field``, object-schema D1 / invariant C4) and non-participating (un-roled)
    objects are non-salient and never enter the AbstractSituation. The projector never reads a
    colour number, coordinate, or glyph to decide salience.
  * Determinism (DP-10). Equality and hash come from a fully-sorted CANONICAL form built
    from primitives; they are ORDER-INDEPENDENT (no reliance on dict/set iteration order)
    and use NO RNG. Equality is defined from the canonical content -- NOT from the builtin
    ``hash()`` -- so two independently constructed equal configurations compare equal; the
    in-process ``hash()`` is kept consistent with it so a visited-set/memo treats them as
    one node.

This evolves the former L1-slice ``AbstractSituation(avatar_pos, carried_pose)`` (a BFS state-key
idea) into the general role->Profile map. ``agent/core/solver.py`` and ``agent/core/play.py``
use raw tuple state keys and do NOT import this class, so they are unaffected; ``AbstractSituation``
can be adopted by them later as the search key (see ``canonical()`` / ``hash()``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional, Sequence

from .attributes import Profile
from .perceive import Obj

# ----------------------------------------------------------------------- salience policy
# The participating roles. An object with one of these recognised roles is part of the
# task and is kept; everything else (background / un-roled) is dropped. These are role
# AXES produced by upstream recognition (see play.py `_role`), not game-specific literals:
# "controllable" = the avatar, "interactor" = a thing whose touch changes state,
# "target"/"goal-target" = the goal mark, "carried-state" = the agent's carried pose.
PARTICIPATING_ROLES: frozenset = frozenset(
    {"controllable", "interactor", "target", "goal-target", "carried-state"}
)
# The label given to an object that carries no participating role (play.py uses this too).
UNCLASSIFIED_ROLE: str = "unclassified"


def _canon_value(value):
    """Canonicalise a Profile/Relation value into an ORDER-INDEPENDENT, hashable primitive.

    Sets/frozensets become sorted tuples and dicts become sorted item tuples, so equality
    and hash never depend on iteration order (DP-10). Other hashable values pass through.
    Sorting falls back to ``repr`` when elements are not mutually orderable, which is still
    deterministic and order-independent.
    """
    if isinstance(value, (frozenset, set)):
        return ("set", _sorted_tuple(_canon_value(v) for v in value))
    if isinstance(value, dict):
        return ("map", _sorted_tuple((_canon_value(k), _canon_value(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        # Order IS significant for an explicit sequence; keep it, but canonicalise elements.
        return tuple(_canon_value(v) for v in value)
    return value


def _sorted_tuple(items) -> tuple:
    """Deterministically sort an iterable of canonical values into a tuple."""
    return tuple(sorted(items, key=lambda x: (type(x).__name__, repr(x))))


def _canon_profile(profile: Profile) -> tuple:
    """Canonical, order-independent form of a Profile: sorted (dim_id, value, confidence)."""
    out = []
    for dim_id, ve in profile.entries.items():
        value, conf = ve
        out.append((dim_id, _canon_value(value), conf))
    return _sorted_tuple(out)


@dataclass(frozen=True)
class AbstractSituation:
    """The salient-only abstract state -- a «value» with deterministic, order-independent
    equality and hash (TERM-31).

    Content (all salient; non-salient board/field/un-roled objects are excluded by
    construction in :meth:`StateAbstraction.project`):
      * ``roles``   -- role label -> :class:`Profile` for each participating object.
      * ``relations`` -- the salient Relations that HOLD in this state (each a hashable
        descriptor; a frozenset of them, so order is irrelevant).
      * ``scalars`` -- observed scalar gauges (e.g. ``{"move_budget": 7}``) that keep the
        state Markov (Move Budget, TERM-32).

    Two AbstractSituations are equal iff their :meth:`canonical` forms are equal; ``__hash__`` is
    consistent with that. Equality does NOT consult the builtin ``hash()`` (DP-10), so
    independently constructed equal configurations compare equal, and an in-process
    visited-set/memo collapses them to one node.
    """

    roles: Mapping[str, Profile] = field(default_factory=dict)
    relations: frozenset = field(default_factory=frozenset)
    scalars: Mapping[str, object] = field(default_factory=dict)

    def canonical(self) -> tuple:
        """The fully-sorted canonical tuple equality and hash derive from.

        Built from primitives only and ORDER-INDEPENDENT: roles sorted by label, each
        Profile canonicalised, relations sorted, scalars sorted by name. This is the
        intended cross-construction identity (and a stable search key for solver/play)."""
        roles = _sorted_tuple(
            (label, _canon_profile(prof)) for label, prof in self.roles.items()
        )
        relations = _sorted_tuple(_canon_value(rel) for rel in self.relations)
        scalars = _sorted_tuple((name, _canon_value(val)) for name, val in self.scalars.items())
        return (roles, relations, scalars)

    def salient_roles(self) -> frozenset:
        """The set of role labels present -- a convenience for assertions/debug."""
        return frozenset(self.roles.keys())

    def __eq__(self, other) -> bool:
        if not isinstance(other, AbstractSituation):
            return NotImplemented
        return self.canonical() == other.canonical()

    def __hash__(self) -> int:
        return hash(self.canonical())


def _default_profile(role: str) -> Profile:
    """A minimal structural Profile when none is supplied: it records only the recognised
    role on a single ``role`` Dimension. No surface literal (colour/coord/glyph) is read,
    so the AbstractSituation stays NFR-6-clean even without a full vocabulary pass."""
    return Profile(entries={"role": (role, 1.0)})


@dataclass(frozen=True)
class StateAbstraction:
    """Map a perceived frame (+ previous AbstractSituation) to a salient-only :class:`AbstractSituation`
    (TERM-24).

    Salience policy (general, observation-driven): an object is salient iff it is NOT a
    field/background object AND its recognised role is participating
    (:data:`PARTICIPATING_ROLES`). A custom ``participating`` set or an ``is_salient``
    predicate can override the policy without introducing game literals.
    """

    participating: frozenset = PARTICIPATING_ROLES

    def is_salient(self, obj: Obj, role: str) -> bool:
        """Observation-driven salience: drop background (``is_field``) and any object whose
        recognised ``role`` is not participating. Reads only the role + the field flag --
        never a colour/coordinate/glyph (NFR-6)."""
        if getattr(obj, "is_field", False):
            return False
        return role in self.participating

    def project(
        self,
        objects: Sequence[Obj],
        prev: Optional[AbstractSituation] = None,
        roles: Optional[Mapping[Obj, str] | Callable[[Obj], str]] = None,
        profiles: Optional[Mapping[Obj, Profile]] = None,
        relations: Optional[Iterable] = None,
        scalars: Optional[Mapping[str, object]] = None,
        is_salient: Optional[Callable[[Obj, str], bool]] = None,
    ) -> AbstractSituation:
        """Build a :class:`AbstractSituation` from ``objects``, selecting salient and dropping the
        rest, with carry-forward from ``prev``.

        Parameters
        ----------
        objects:
            The perceived frame -- a sequence of :class:`agent.core.perceive.Obj` (which now
            carry ``parts`` / ``is_field``).
        prev:
            The previous AbstractSituation, for carry-forward (TERM-24): a salient role present in
            ``prev`` but absent from the current frame (e.g. momentarily occluded / fogged)
            is carried forward so identity/continuity persists. Current observations win.
        roles:
            Recognised role per object, as a mapping or a callable ``obj -> role``. Objects
            with no entry are treated as :data:`UNCLASSIFIED_ROLE` (non-participating).
        profiles:
            :class:`Profile` per object (from the vocabulary/attributes layer). When an
            object has no Profile, a minimal role-only Profile is used (still NFR-6-clean).
        relations:
            Salient Relations that HOLD in this state (each a hashable descriptor). Only the
            holding relations are passed in; the projector stores them order-independently.
        scalars:
            Observed scalar gauges (e.g. ``{"move_budget": 7}``) -- the Markov scalars.
        is_salient:
            Optional override of the salience predicate ``(obj, role) -> bool``.

        Returns
        -------
        AbstractSituation
            A salient-only value: role -> Profile, holding Relations, observed scalars.
            Non-salient content (background / un-roled objects, their positions) is ABSENT.
        """
        role_of = _role_lookup(roles)
        salient_pred = is_salient if is_salient is not None else self.is_salient

        role_map: dict = {}
        for obj in objects:
            role = role_of(obj)
            if not salient_pred(obj, role):
                continue
            prof = profiles.get(obj) if profiles is not None else None
            if prof is None:
                prof = _default_profile(role)
            # Deterministic conflict policy: if two objects claim the same role, keep the
            # one whose canonical Profile sorts first (order-independent, not iteration-order).
            if role in role_map:
                if _canon_profile(prof) >= _canon_profile(role_map[role]):
                    continue
            role_map[role] = prof

        # Carry-forward: fill salient roles missing this frame from prev (current wins).
        if prev is not None:
            for label, prof in prev.roles.items():
                role_map.setdefault(label, prof)

        rel_set = frozenset(relations) if relations is not None else frozenset()
        scalar_map = dict(scalars) if scalars is not None else {}
        return AbstractSituation(roles=role_map, relations=rel_set, scalars=scalar_map)


def _role_lookup(roles) -> Callable[[Obj], str]:
    """Normalise the ``roles`` argument (mapping | callable | None) to a callable that
    returns a role label, defaulting to :data:`UNCLASSIFIED_ROLE`."""
    if roles is None:
        return lambda _obj: UNCLASSIFIED_ROLE
    if callable(roles):
        return lambda obj: roles(obj) or UNCLASSIFIED_ROLE
    return lambda obj: roles.get(obj, UNCLASSIFIED_ROLE)
