"""Shippable offline Qwen backend (API-04 concrete Adapter) — a real
``ConstrainedGenerator``.

This is the drop-in backend the agent mounts on Kaggle at
``/kaggle/input/models/...``. It conforms to the ``agent.core.llm``
``ConstrainedGenerator`` Protocol (a ``name`` attribute + ``propose(messages,
schema, *, max_new_tokens=256) -> dict``, where ``messages`` is the rendered
system+user chat pair) and is consulted ONLY through ``agent.core.llm.consult``
(propose -> classical-fallback). The classical master disposes; this only
proposes (NFR-1 offline, NFR-9 swappable).

IMPORT-SAFE WITHOUT TORCH (load-bearing): ``import agent.core.llm_qwen`` must NOT
require ``torch`` / ``transformers``. They are imported lazily INSIDE
:meth:`QwenGenerator.load` (the only place a model is needed), so a machine
without torch can still import this module (and the agent still boots with a
NullGenerator). Instantiating + loading the generator is where torch becomes a
hard requirement.

Offline + deterministic (greedy decode, ``do_sample=False``) to honour DP-10 /
NFR-1: same briefing -> same proposal. No ``random``, no builtin ``hash()``.
English / ASCII.

Adapted from ``spikes/llm-kaggle/constrained_generator_adapter.py`` (the proven
T3-feasibility adapter): the CPU-fallback arch check, the ``local_files_only``
offline load, the greedy generate, the best-effort JSON slice, and
``_resolve_model_path`` (auto-discover ``config.json`` under a mount root). The
spike's separate LlmBackend/Adapter split is collapsed into ONE class here
because the agent seam only needs ``propose`` (the feasibility probe's timing /
vram accounting is dropped — that belonged to the spike's measurement, not the
shipped agent).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from agent.core.llm import GeneratorError

_log = logging.getLogger(__name__)

# Process-level cache of a loaded ``(tokenizer, model, device_used, fallback_reason)``
# keyed by the RESOLVED model dir. The framework reconstructs the agent (and thus a
# fresh QwenGenerator) PER GAME, so without this every game would reload ~15 GB of 7B
# weights from disk -- ~40 min wasted over a 55-game run plus repeated VRAM churn (OOM
# risk). Populated on the first successful load; later instances reuse it. Same-process
# only (a fresh process / kernel starts empty, which is correct).
_MODEL_CACHE: Dict[str, Any] = {}


def _read_max_time_env(default: float) -> float:
    """Per-consult generation time cap (seconds) from ``ARC_LLM_MAX_TIME``; a
    non-float value -> ``default``; ``<= 0`` -> ``0.0`` = cap DISABLED. Deterministic."""
    raw = os.getenv("ARC_LLM_MAX_TIME")
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except (ValueError, AttributeError):
        return default
    return value if value > 0 else 0.0


def _resolve_model_path(path: str) -> str:
    """Return a local dir that actually holds the model (contains ``config.json``).

    A Kaggle Model mount nests differently per framework/variation/version
    (``/kaggle/input/models/<owner>/<model>/<framework>/<variation>/<version>``),
    so a hardcoded path can miss. If ``path`` already has ``config.json``, keep
    it; else walk ``path`` (when it is a directory) for the first dir containing
    ``config.json``. Deterministic: shallowest dir first, then lexicographic (no
    RNG, no builtin ``hash()`` — DP-10). Returns ``path`` unchanged when nothing
    is found (the caller then fails loudly on load, which is the honest signal)."""
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    if not os.path.isdir(path):
        return path
    hits: List[str] = []
    for root, _dirs, files in os.walk(path):
        if "config.json" in files:
            hits.append(root)
    hits.sort(key=lambda r: (r.count(os.sep), r))
    return hits[0] if hits else path


def _slice_json(text: str) -> str:
    """Best-effort: extract the first balanced ``{...}`` object from ``text``.

    Used when no schema constrainer is installed, so a plain-greedy decode that
    wraps the JSON in prose still yields a parseable object. Deterministic."""
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


class QwenGenerator:
    """An offline Qwen ``ConstrainedGenerator`` (transformers backend).

    Lazily loads a Qwen model OFFLINE from ``model_path`` (``local_files_only=
    True``) on first :meth:`propose` (or an explicit :meth:`load`), GPU if the
    device arch is supported else CPU+float32 (the spike's P100 fallback). Greedy
    decode (deterministic). Constrained decoding via ``lm-format-enforcer`` IF it
    is installed (records ``constrained``), else plain greedy + a JSON slice.

    Raises :class:`agent.core.llm.GeneratorError` on any unusable output so
    ``agent.core.llm.consult`` catches it and the agent falls back to classical
    (the load-bearing fallback contract). Construction does NOT load — it only
    records config — so importing + constructing is cheap and torch-free; the
    weights are pulled in on the first :meth:`load`."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "float16",
        name: str = "qwen",
        max_time: Optional[float] = None,
    ) -> None:
        self.name = name
        self.model_path = model_path
        self.dtype = dtype
        # Per-consult GENERATION time cap (seconds) passed to generate() as
        # ``max_time`` -> a transformers MaxTimeCriteria that stops decoding after
        # the bound. Bounds the generation loop ONLY (NOT prefill -- a large prompt's
        # prefill is bounded upstream by the observation/prompt size). A time-stop can
        # truncate the JSON; propose() then raises GeneratorError -> consult() falls
        # back to classical (safe). Default from ARC_LLM_MAX_TIME (env), else 20 s.
        if max_time is None:
            max_time = _read_max_time_env(20.0)
        self.max_time = max_time
        self.constrained = False
        self.device_used: Optional[str] = None  # "cuda" | "cpu" (set in load)
        self.fallback_reason: Optional[str] = None
        self._tok: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._loaded = False

    # -- load (the only torch-requiring step) ------------------------------- #

    def load(self) -> None:
        """Load tokenizer + weights OFFLINE (once). Imports ``torch`` /
        ``transformers`` HERE (lazy) so the module import stays torch-free.

        Uses the GPU only if torch was compiled for THIS device's arch (a Kaggle
        P100 sm_60 is not in modern torch's arch_list -> matmul raises "no kernel
        image"); otherwise falls back to CPU + float32 so the offline pipeline
        still completes. Raises :class:`GeneratorError` on any load failure (so an
        unusable backend degrades to classical, never crashes the agent)."""
        if self._loaded:
            return
        try:
            import torch  # lazy: offline, in the Kaggle image — NOT a module-level dep
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch
            resolved = _resolve_model_path(self.model_path)

            # Reuse a model already loaded in THIS process (per-game agents share it).
            cached = _MODEL_CACHE.get(resolved)
            if cached is not None:
                self._tok, self._model, self.device_used, self.fallback_reason = cached
                self._loaded = True
                return

            use_gpu, reason = False, "no cuda"
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(0)
                sm = "sm_%d%d" % (cap[0], cap[1])
                archs = list(torch.cuda.get_arch_list())
                if sm in archs:
                    use_gpu = True
                    reason = "gpu %s supported (arch_list=%s)" % (sm, archs)
                else:
                    reason = "gpu %s NOT in torch arch_list %s -> CPU fallback" % (
                        sm,
                        archs,
                    )
            self.device_used = "cuda" if use_gpu else "cpu"
            self.fallback_reason = reason
            if use_gpu:
                td = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
                    self.dtype, torch.float16
                )
                device_map = "auto"
            else:
                td = torch.float32  # fp16 matmul is unsupported/slow on CPU
                device_map = "cpu"

            self._tok = AutoTokenizer.from_pretrained(resolved, local_files_only=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                resolved,
                torch_dtype=td,
                device_map=device_map,
                local_files_only=True,
            )
            _MODEL_CACHE[resolved] = (
                self._tok, self._model, self.device_used, self.fallback_reason
            )
            self._loaded = True
        except Exception as exc:  # noqa: BLE001 - any load failure -> GeneratorError
            raise GeneratorError("Qwen offline load failed: %r" % (exc,)) from exc

    # -- propose (the ConstrainedGenerator contract) ------------------------ #

    def propose(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        *,
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Re-grounded from ``messages`` each call (stateless): a system+user chat
        pair (the rendered move-proposer prompt) greedy-decoded into a JSON object
        matching ``schema`` (the per-turn narrowed grammar), parsed to a dict.

        Raises :class:`GeneratorError` on any failure (load error, generation
        error, or non-JSON output) so ``consult`` catches it and the classical
        path runs. PROPOSE-ONLY — the classical master disposes."""
        if not self._loaded:
            self.load()  # raises GeneratorError on failure
        try:
            raw = self._generate(messages, schema, max_new_tokens=max_new_tokens)
        except GeneratorError:
            raise
        except Exception as exc:  # noqa: BLE001 - any generation error -> fallback
            raise GeneratorError("Qwen generation failed: %r" % (exc,)) from exc
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise GeneratorError(
                "Qwen produced non-JSON output: %r" % (raw,)
            ) from exc

    def _generate(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        *,
        max_new_tokens: int,
    ) -> str:
        """Apply the chat template to the system+user pair, greedy-decode with the
        per-turn schema enforced via ``lm-format-enforcer``'s stable
        ``prefix_allowed_tokens_fn`` integration (when installed), else plain greedy
        + a JSON slice. Mirrors the proven stage-2 harness
        ``docs/llm-prompt-design/run_proposer_test.py``."""
        torch = self._torch
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        prefix_fn = self._prefix_fn(schema)
        self.constrained = bool(prefix_fn)
        gkw = {"prefix_allowed_tokens_fn": prefix_fn} if prefix_fn else {}
        # Per-consult generation time cap (transformers MaxTimeCriteria). Bounds the
        # decode loop only; truncation -> non-JSON -> GeneratorError -> classical.
        if self.max_time and self.max_time > 0:
            gkw["max_time"] = self.max_time
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy = deterministic (DP-10)
                num_beams=1,
                pad_token_id=self._tok.eos_token_id,
                **gkw,
            )
        gen = self._tok.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return gen if self.constrained else _slice_json(gen)

    def _prefix_fn(self, schema: Dict[str, Any]) -> Optional[Any]:
        """A ``prefix_allowed_tokens_fn`` enforcing ``schema`` via
        lm-format-enforcer's STABLE transformers integration
        (``build_transformers_prefix_allowed_tokens_fn`` + ``JsonSchemaParser``),
        or ``None`` (records ``constrained=False`` and falls back to a JSON slice).

        Uses the prefix-fn integration, NOT the removed ``LogitsProcessor`` import
        (which does not exist in lmfe 0.10.9 and broke the stale path). All-or-
        nothing and guarded: a missing/incompatible constrainer must never crash
        generation -- the agent then degrades to plain greedy + slice, same as the
        harness."""
        try:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import (
                build_transformers_prefix_allowed_tokens_fn,
            )

            return build_transformers_prefix_allowed_tokens_fn(
                self._tok, JsonSchemaParser(schema)
            )
        except Exception:  # noqa: BLE001 - no constrainer -> plain greedy + JSON slice
            return None
