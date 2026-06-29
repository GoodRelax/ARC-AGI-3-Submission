"""Evaluation environment (port) for the goal Relation interpreter.

A predicate needs a *subject* and a way to resolve Role references to concrete
objects; the ``Env`` supplies them (gr-arc-3-operators.md §2 preamble). Two
binding contexts share one interpreter:

  (i) Goal context        — env binds AbstractSituation role -> object.
  (ii) role-assignment    — env binds the candidate object as the subject and
                            the already-assigned roles (e.g. 'self').

This is a Protocol (a port): the interpreter depends only on this surface. The
real implementation arrives with AbstractSituation / ObjectRef (a later 段6
step); tests use a fake. Object handles are opaque to the interpreter — the env
knows how to turn one into a footprint / Profile.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol, Tuple

from agent.core.model import Lexicon, Profile


class Env(Protocol):
    """What the operator evaluators read. Object handles are opaque (``Any``)."""

    lexicon: Lexicon  # for subset / samekind taxonomy lookups

    def subject(self) -> Any:
        """The implicit subject for ``has`` (the candidate in role-assignment)."""

    def object_for(self, ref: str) -> Any:
        """Resolve a Role label / 'self' / bound name to ONE object handle
        (a binding takes precedence over the role's salient/sole object).

        Case B note (for the step-3 AbstractSituation impl): a role may bind to a
        SET {ObjectRef}; ``object_for`` must collapse it to a single representative
        (the salient one). Relation predicates that need every element should be
        wrapped in a quantifier (exists/forall) rather than relying on this.
        """

    def objects_for(self, role: str) -> Iterable[Any]:
        """The quantification range bound to ``role`` (exists / forall)."""

    def bind(self, name: str, obj: Any) -> "Env":
        """Return a child env with ``name`` bound to ``obj`` (quantifiers)."""

    def footprint(self, obj: Any) -> frozenset:
        """The object's occupied cells as a set of ``(row, col)``."""

    def profile(self, obj: Any) -> Profile:
        """The object's Profile."""

    def orientation(self, obj: Any) -> Optional[Tuple[float, float]]:
        """The object's pose orientation unit-vector ``(cosθ, sinθ)`` (norm=1), or
        ``None`` when undefined (e.g. an empty/degenerate footprint). The
        ``orientation-match`` operator reads this (GameObject geometry attr)."""

    def symmetry_order(self, obj: Any) -> int:
        """The object's rotational-symmetry order ``k`` (1 / 2 / 4) — the
        symmetry-fold used by ``orientation-match`` (GameObject geometry attr)."""

    def reflected(self, obj: Any) -> bool:
        """The object's handedness bit (GameObject geometry attr, ADR-016)."""

    def size(self, obj: Any) -> Optional[Tuple[int, int]]:
        """The object's bbox extent ``(h, w)``, or ``None`` when empty (GameObject
        geometry attr)."""
