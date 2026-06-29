"""ConstrainedGenerator port (API-04) — the Local-LLM consultation seam.

The classical core consults the Local LLM ONLY through this port: a stateless,
schema-constrained, PROPOSE-ONLY generator. The master is classical — it
disposes (ModelGoal / SelectSolver verify proposals against observation); the LLM
only proposes. The concrete backend (Qwen3-4B/8B + vLLM + xgrammar, offline) is a
drop-in Adapter conforming to this shape (see spikes/llm-kaggle/), NEVER imported
by the core (NFR-1 offline, NFR-9 swappable).

Wiring it in early (before the solver/goal consult points exist) de-risks the
seam: the core is built against the port from the start, defaulting to
``NullGenerator`` so it solves with NO LLM (DP-20 bounded classical fallback).
The real backend is validated separately on Kaggle (T3) and dropped in by shape.

``consult()`` is the single place the propose -> classical-fallback discipline
lives: any decline / error / unparsable output returns None, and the caller
takes its classical path. Deterministic (DP-10): NullGenerator always declines;
ScriptedGenerator replays canned replies in order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, Union

_log = logging.getLogger(__name__)

T = TypeVar("T")


class GeneratorUnavailable(Exception):
    """No LLM backend is wired (the classical-only / NullGenerator case). Expected."""


class GeneratorError(Exception):
    """A wired backend failed or produced unusable output. Caller falls back."""


class ConstrainedGenerator(Protocol):
    """A propose-only, schema-constrained, stateless typed generator (API-04)."""

    name: str

    def propose(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        *,
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Re-grounded from ``messages`` each call (no memory): a system+user chat
        pair (the rendered move-proposer prompt) consumed via ``apply_chat_template``.
        Returns a dict parsed from the schema-constrained generation. Raises
        ``GeneratorUnavailable`` if there is no backend, ``GeneratorError`` if the
        backend output is unusable. PROPOSE-ONLY — the classical master disposes.
        """
        ...


@dataclass
class NullGenerator:
    """The default: no backend. Always declines, so the core stays classical-only
    (DP-20). Deterministic."""

    name: str = "null"

    def propose(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        *,
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        raise GeneratorUnavailable("no LLM backend wired (classical-only)")


@dataclass
class ScriptedGenerator:
    """A test fake: replays ``replies`` in order. A reply that is an Exception is
    raised (to simulate decline / backend error); otherwise it is returned."""

    replies: List[Union[Dict[str, Any], Exception]] = field(default_factory=list)
    name: str = "scripted"
    _i: int = 0

    def propose(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        *,
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        if self._i >= len(self.replies):
            raise GeneratorError("ScriptedGenerator exhausted")
        reply = self.replies[self._i]
        self._i += 1
        if isinstance(reply, Exception):
            raise reply
        return reply

    def reset(self) -> None:
        """Rewind the reply cursor (the fake is single-use per scenario otherwise)."""
        self._i = 0


def consult(
    generator: ConstrainedGenerator,
    messages: List[Dict[str, str]],
    schema: Dict[str, Any],
    parse: Callable[[Dict[str, Any]], T],
    *,
    max_new_tokens: int = 256,
) -> Optional[T]:
    """Propose + parse, returning None on ANY decline / error / parse failure.

    The one place the propose -> classical-fallback contract lives: consult points
    (ModelGoal / SelectSolver) call this and treat None as "no proposal — use the
    classical path". The core therefore never blocks on the LLM (DP-20).
    """
    name = getattr(generator, "name", "?")
    try:
        raw = generator.propose(messages, schema, max_new_tokens=max_new_tokens)
    except GeneratorUnavailable:
        return None  # expected (no backend) — classical path, no log noise
    except Exception as exc:  # noqa: BLE001 — load-bearing backstop: the LLM must
        # NEVER crash the classical master (a real offline backend can raise
        # RuntimeError/OSError/etc.). Adapters SHOULD raise only GeneratorError,
        # but we defend broadly and make the fallback observable.
        _log.warning("LLM consult fell back: backend %s raised %r", name, exc)
        return None
    try:
        return parse(raw)
    except (KeyError, ValueError, TypeError) as exc:
        # The proposal did not fit the expected shape — fall back, but log it so a
        # buggy parser is not invisible. (Other exception types propagate as bugs.)
        _log.warning("LLM proposal did not parse (backend %s): %r", name, exc)
        return None
