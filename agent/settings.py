"""[agent] Settings -- boundary loader for ``gr-arc-3-settings.json`` (config; ADR-014, DP-20).

``agent/core`` stays pure (no file I/O -- NFR-6 / DP-10): this BOUNDARY module reads the JSON
ONCE and produces a plain :class:`Settings` VALUE handed to ``SolveGame(settings=...)``. Every
field is a configurable GUARDRAIL toggle; the core must solve with all guardrails OFF (DP-20),
so the default is ON (an absent / partial file is safe).

Schema + human doc: ``agent/gr-arc-3-settings.schema.json`` /
``docs/StrictDoc-specs/_assets/gr-arc-3-settings.md``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "gr-arc-3-settings.json")
_ENABLED = "enabled"
_DISABLED = "disabled"


@dataclass(frozen=True)
class Settings:
    """Resolved agent settings -- a «value» (no I/O). Each field toggles one guardrail; the
    default is ON (guardrail enabled) so a missing file / key is safe."""

    futility_check: bool = True


DEFAULT_SETTINGS = Settings()


def _as_bool(value: object, default: bool) -> bool:
    """Map the schema's enum string (``"enabled"`` / ``"disabled"``) -- or a bool -- to a bool.
    Unknown values fall back to ``default`` (safe)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v == _ENABLED:
            return True
        if v == _DISABLED:
            return False
    return default


def load_settings(path: str = _DEFAULT_PATH) -> Settings:
    """Read ``gr-arc-3-settings.json`` at the boundary and return a :class:`Settings` value.

    The ONLY settings I/O point (``agent/core`` never reads files). A missing file, malformed
    JSON, or an absent key falls back to the safe default (guardrail ON). Deterministic."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return DEFAULT_SETTINGS
    if not isinstance(raw, dict):
        return DEFAULT_SETTINGS
    return Settings(
        futility_check=_as_bool(raw.get("futility_check", _ENABLED), True),
    )
