"""impl_key -> callable dispatch registries (core-side, decorator-based).

The AssetLoader (adapter) loads ``words.tsv`` data; the *semantics* behind each
``impl_key`` live in core impl modules, which register their functions here by
decorating them. The loader then resolves each Word's ``impl_key`` against the
merged registry and can FK-validate completeness once all impl modules import.

Four registries over five ``impl_key`` prefixes (``op_`` and ``rel_`` share the
evaluator registry):

    feat_*        feature detectors   (DetectFeatures: object/frame -> magnitude/confidence)
    op_* / rel_*  operator evaluators (goal Relation interpreter: gr-arc-3-operators.md)
    xf_*          transform appliers  (apply a TransformOperator)
    cue_*         parse cues          (DivideFrame: object grouping cost models -- the
                                       connectivity / similarity / enclosure / continuation
                                       / lattice cost models registered by agent.core.perceive)

Usage in an impl module (a later 段6 step)::

    from agent.core.registry import feature, evaluator, transform

    @feature("feat_color")
    def detect_color(...): ...

    @evaluator("rel_inside")
    def eval_inside(...): ...

This module owns NO semantics — only the registration plumbing. It is the frozen
plug point for the parallel enrichment sessions: add a words.tsv row + a
decorated function, nothing in the loader changes.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

_FEATURES: Dict[str, Callable] = {}
_EVALUATORS: Dict[str, Callable] = {}
_TRANSFORMS: Dict[str, Callable] = {}
_CUES: Dict[str, Callable] = {}

# impl_key prefix -> the registry it belongs in.
_PREFIXES = (
    ("feat_", _FEATURES),
    ("op_", _EVALUATORS),
    ("rel_", _EVALUATORS),
    ("xf_", _TRANSFORMS),
    ("cue_", _CUES),
)


def _registry_for(impl_key: str) -> Optional[Dict[str, Callable]]:
    for prefix, reg in _PREFIXES:
        if impl_key.startswith(prefix):
            return reg
    return None


def _register(reg: Dict[str, Callable], expected_prefixes, impl_key: str):
    if not any(impl_key.startswith(p) for p in expected_prefixes):
        raise ValueError(
            f"impl_key {impl_key!r} does not match expected prefix(es) "
            f"{expected_prefixes} for this registry"
        )
    if impl_key in reg:
        raise ValueError(f"duplicate registration for impl_key {impl_key!r}")

    def deco(fn: Callable) -> Callable:
        reg[impl_key] = fn
        return fn

    return deco


def feature(impl_key: str):
    """Register a feature detector (``feat_*``)."""
    return _register(_FEATURES, ("feat_",), impl_key)


def evaluator(impl_key: str):
    """Register an operator evaluator (``op_*`` / ``rel_*``)."""
    return _register(_EVALUATORS, ("op_", "rel_"), impl_key)


def transform(impl_key: str):
    """Register a transform applier (``xf_*``)."""
    return _register(_TRANSFORMS, ("xf_",), impl_key)


def cue(impl_key: str):
    """Register a parse cue (``cue_*``)."""
    return _register(_CUES, ("cue_",), impl_key)


def resolve(impl_key: str) -> Optional[Callable]:
    """The callable registered for ``impl_key``, or None if unregistered."""
    reg = _registry_for(impl_key)
    if reg is None:
        return None
    return reg.get(impl_key)


def is_registered(impl_key: str) -> bool:
    return resolve(impl_key) is not None


def registered_keys() -> List[str]:
    """All impl_keys with a registered callable (sorted, deterministic)."""
    return sorted({*_FEATURES, *_EVALUATORS, *_TRANSFORMS, *_CUES})


def all_dispatch() -> Dict[str, Callable]:
    """Merged impl_key -> callable map (a fresh dict)."""
    merged: Dict[str, Callable] = {}
    merged.update(_FEATURES)
    merged.update(_EVALUATORS)
    merged.update(_TRANSFORMS)
    merged.update(_CUES)
    return merged


def _clear_for_tests() -> None:
    """Drop all registrations (test isolation only)."""
    _FEATURES.clear()
    _EVALUATORS.clear()
    _TRANSFORMS.clear()
    _CUES.clear()
