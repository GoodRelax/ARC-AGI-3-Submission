"""GR-ARC-3 agent package (v14 clean-room rebuild).

The pre-v14 implementation was archived to ``_archive/agent-pre-v14/`` on
2026-06-27; the specification (``docs/StrictDoc-specs/``) is the source of truth
and this package is rebuilt against it (handoff 2026-06-27-implementation-handoff).

This ``__init__`` is a bare package marker on purpose: it imports nothing at
package-import time so that ``agent.core.*`` unit tests stay free of the vendored
framework. The live play entry point (``OurSearchAgent``) is re-established in a
later 段6 step; until then ``import agents`` (the vendored registry) is expected
to fail because ``agent.search_agent`` is archived.
"""
