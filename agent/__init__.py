"""Our code — the Processing function.

The ONLY code we own in the play loop. Everything else (fetching frames,
submitting actions, recording, scoring) is the vendored framework's Input/Output
ports. See docs/specs/10-io-ports.md.

NOTE: we deliberately do NOT eagerly import `search_agent` here. `search_agent`
imports the vendored `agents.agent`, whose package `__init__` in turn imports
`agent.search_agent` for registration — importing it from this package `__init__`
would create a circular import the moment any `agent.*` submodule is imported
first (e.g. tests importing `agent.policy`). Consumers that need the class import
it explicitly: `from agent.search_agent import OurSearchAgent`.
"""

__all__ = ["OurSearchAgent"]


def __getattr__(name: str) -> object:
    # Lazy access so `agent.OurSearchAgent` still works for callers that expect it,
    # without importing the framework at package-import time (avoids the cycle).
    if name == "OurSearchAgent":
        # Import the framework registry first to reproduce the production import
        # order (agents/__init__.py defines `Agent`, then imports us). This breaks
        # the cycle no matter who triggers the access.
        import agents  # noqa: F401
        _ = agents  # imported for registration side-effect

        from agent.search_agent import OurSearchAgent

        return OurSearchAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
