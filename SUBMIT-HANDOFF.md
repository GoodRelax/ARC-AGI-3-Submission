# SUBMIT-HANDOFF — 7B combine score attempt (LLM-ON, Qwen2.5-7B, T4×2)

**Purpose of THIS submission:** a **score attempt** with the LLM combine =
**classical floor + Qwen2.5-7B move-proposer × improved input, time-capped**.
It is a SAFE upside bet over the classical floor: the LLM only fires when the
classical side is genuinely stuck/stalled, classical navigate/click keeps move
precedence (so it never displaces a classical win), and it is wall-clock-capped
so it can never time out — worst case it degrades to the classical floor.

**Why 7B (not the earlier 1.5B that scored 0.01):** the 0.01 run was 1.5B +
ALWAYS-ON aiming. Two fixes since: (1) the **input** was de-bloated + grounded
(observation ~12 KB→~3.6 KB, robust button-effect directions, 4-state) — which
took the proposer from 1 identical degenerate proposal to board-responsive ones;
(2) the **model** is now 7B — same Qwen2 arch, runs on the bundled
`transformers==4.44.2`, no wheels rebuild. Offline-verified on 2×T4: 7B picks real
goals/targets/moves; cd82's 82 min/game → ~9 min after the de-bloat + caps; classical
wins (cd82/sp80/lp85/vc33) are NOT regressed. Honest expectation: on the 25 PUBLIC
games the LLM adds no win beyond classical (they are either classical wins or need
rule/plan/hidden-state beyond a per-move proposer) — its value is a bet on the 55
HIDDEN games having reasoning-shaped levels a 7B can pick. **Floor stays available
(see below) if you prefer the safe ~0.08–0.16.**

## State (built & verified — 2026-06-30, 7B combine LLM-ON build)
`notebooks/submission.ipynb` + `agent/my_agent.py` freshly built from the latest
agent and checked:
- bundle = 30 files (v14 `agent/assets/*.tsv` + prompts) with the full solving layer:
  **marked**/**navigate(robust displacement)**/**click-navigate** + the **stuck/stall-gated**
  LLM proposer. Gate-green (pytest 529). Verified the bundle embeds: the wall-clock caps
  (`ARC_LLM_DEADLINE_EPOCH` global 9 h deadline + per-game time budget), the per-process
  **model cache** (loads 7B ONCE, not per game), per-consult **`max_time`** + the prefill
  guard, and the **click-game prompt rule**.
- Competition cell exports `ARC_LLM=1`, `ARC_LLM_MODEL=<7B mount>`, and stamps
  `ARC_LLM_DEADLINE_EPOCH=$(python -c 'import time;print(time.time())')` so the global
  9 h deadline (12 h cap − 3 h safety) is measured from agent start.
- offline-deps cell installs `transformers==4.44.2` + `lm-format-enforcer==0.10.9`
  (constrained decoding) from the wheels dataset (guarded → graceful classical fallback).
- `notebooks/kernel-metadata.json`: `enable_gpu:true`, `dataset_sources:["goodrelax/arc-agi3-llm-wheels"]`,
  `model_sources:["qwen-lm/qwen2.5/transformers/7b-instruct/1"]`.
- Build constants (`scripts/build_notebook.py`): **`ACCELERATOR="t4"` (T4×2 = 32 GB → 7B fp16 fits via `device_map="auto"`), `ENABLE_LLM=True`**.

## Prerequisites (must be accessible to goodrelax)
- Kaggle model **`qwen-lm/qwen2.5/transformers/7b-instruct/1`** (~15.2 GB, v1 — verified).
- Kaggle dataset **`goodrelax/arc-agi3-llm-wheels`** (transformers 4.44.2 + lm-format-enforcer wheels).
Both are declared in `kernel-metadata.json`, so `kernels push` wires them.

## Open risks to watch in the Phase-B log
- **7B VRAM**: needs T4×2 (32 GB). If the scored rerun gives a single GPU, 7B fp16 (~15 GB)
  may OOM on load → the agent falls back to NullGenerator (classical) — SAFE (no crash) but
  the LLM is then inert. Check the log for `device_used:"cuda"` + `qwen_loaded:true`.
- **Wall-clock**: the caps should keep it ≈3–4 h (most games never trigger the LLM). Confirm it finishes < 12 h.

## To fall back to the classical FLOOR (the safe ~0.08–0.16 build)
Flip `scripts/build_notebook.py` to `ACCELERATOR="cpu"` + `ENABLE_LLM=False`, set
`model_sources: []` in `notebooks/kernel-metadata.json` (hand-maintained), then rebuild.
That is the proven aim-when-stuck classical build (no Qwen, no wheels, self-contained).

## Auth (global kaggle CLI at C:\Python313\Scripts\kaggle.exe)
```
export KAGGLE_USERNAME=goodrelax
export KAGGLE_KEY="$(tr -d '\r\n' < /c/Users/good_/.kaggle/access_token)"
export PYTHONUTF8=1
```

## Steps
1. **(optional) rebuild** — only if `agent/` changed since 2026-06-30. Uses the main repo's venv python (no submission venv needed for build):
   ```
   PY="/c/Users/good_/OneDrive/Documents/GitHub/Kaggle/ARC-AGI-3/ARC-AGI-3-Agents/.venv/Scripts/python.exe"
   "$PY" scripts/bundle_agent.py && "$PY" scripts/build_notebook.py
   ```
2. **Push the kernel** (this is NOT yet the competition submit — it creates a kernel version):
   ```
   kaggle kernels push -p notebooks/
   ```
3. **Track the run**:
   ```
   kaggle kernels status goodrelax/arc-prize-2026-arc-agi-3-starter
   # logs (PYTHONUTF8=1): kaggle kernels output goodrelax/arc-prize-2026-arc-agi-3-starter -p out/
   ```
4. **Submit to Competition** — *USER action* (Claude must not do this): on kaggle.com open the kernel → **Submit to Competition** (triggers the scored Phase-B rerun). Limit **1/day**.

## What to watch for
- **Phase A (the plain push)** only writes a dummy `submission.parquet`; the agent runs ONLY in the scored **Phase-B rerun**. So a clean push proves nothing — the real signal is the **Phase-B log + score** after Submit.
- In the Phase-B log, check: the official `arc-agi` framework wheel installs (rc 0); the agent loads (no import crash); games play without timeout/exception. The classical build installs **no** transformers / lm-format-enforcer (none expected in the log).
- **Score target:** beat 0.08 (the v1 classical baseline). Local proxy = 4 levels {cd82, sp80, lp85, vc33}; ~0.04 score/level ⇒ hope for ≈0.12–0.16. Capture the full Phase-B log if anything errors.

## Note
`notebooks/submission.ipynb` is a built artifact (gitignored). The submission repo
has uncommitted changes (the classical aim-when-stuck re-sync) — committing is
optional for `kernels push` (it pushes local files, not git) but recommended for
record. Commit message: `Build classical aim-when-stuck submission (LLM-OFF)`.
