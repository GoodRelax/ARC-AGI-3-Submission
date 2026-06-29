# SUBMIT-HANDOFF — one validation submission (LLM-linked, offline)

**Purpose of THIS submission:** a single *pipeline-validation* run — confirm the
LLM-linked agent loads and runs end-to-end on the scored Kaggle infra with **no
unexpected errors**. **NOT a score attempt.** Honest expectation: score ~0 — the
proposer FIRES but does not yet SOLVE any game (verified: 3 games tested offline on
practice, 0 solved). Treat a clean Phase-B run (no crash) as success.

## State (already built & verified — 2026-06-29, final build)
Everything here is ready. `notebooks/submission.ipynb` + `agent/my_agent.py` were
freshly built from the latest agent and checked:
- bundle = 30 files incl. `agent/assets/prompts/{prompt-template.md,*-schema.json}` and the
  full solving layer in `search_agent.py`: **marked** goal-marker detector (fires the `target`
  role) + greedy/**BFS navigate** (routes the controllable around walls) + **click-navigate**
  (clicks a target centroid on click games) + the offline-Qwen LLM proposer. Gate-green
  (pytest 527). Offline-verified on T4: runs end-to-end, LLM drives, no crash; **0 levels
  solved** (maze/routing/which-target win-conditions are beyond greedy+BFS+1.5B today — expect
  ~0 score; this is the no-crash pipeline validation).
- notebook offline-deps cell: glob-auto-discovers the wheels dir, then `pip install --no-index --find-links <dir> transformers==4.44.2 lm-format-enforcer==0.10.9` (guarded → graceful classical fallback if it fails).
- `notebooks/kernel-metadata.json`: id `goodrelax/arc-prize-2026-arc-agi-3-starter`, `enable_internet:false`, `enable_gpu:true`, `dataset_sources:["goodrelax/arc-agi3-llm-wheels"]`, `model_sources:["qwen-lm/qwen2.5/transformers/1.5b-instruct/1"]`.

## Prerequisites (must be accessible to goodrelax)
- Kaggle dataset **`goodrelax/arc-agi3-llm-wheels`** (24 wheels — already created).
- Kaggle model **`qwen-lm/qwen2.5/transformers/1.5b-instruct/1`** (already mounted in prior practice runs).
Both are declared in `kernel-metadata.json`, so `kernels push` wires them automatically.

## Auth (global kaggle CLI at C:\Python313\Scripts\kaggle.exe)
```
export KAGGLE_USERNAME=goodrelax
export KAGGLE_KEY="$(tr -d '\r\n' < /c/Users/good_/.kaggle/access_token)"
export PYTHONUTF8=1
```

## Steps
1. **(optional) rebuild** — only if `agent/` changed since 2026-06-29. Uses the main repo's venv python (no submission venv needed for build):
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

## What to watch for (the whole point — "unexpected errors")
- **Phase A (the plain push)** only writes a dummy `submission.parquet`; the agent runs ONLY in the scored **Phase-B rerun**. So a clean push proves nothing about the LLM — the real validation is the **Phase-B log** after Submit.
- In the Phase-B log, check: the offline `pip --no-index` installs `transformers 4.44.2` (rc 0); `lm-format-enforcer` imports; the agent loads (no import crash); games play without timeout/exception; ideally `move_source` shows some `llm`. If `dataset_sources` ever detaches, the enforcer fails **silently** → unconstrained → inert (no crash, but no constrained LLM) — check the install log, not just "did it crash".
- Capture the full Phase-B log if anything errors; that's the artifact to bring back.

## Classical fallback (if the LLM path errors and you just want a clean run)
One-switch classical-only in `notebooks/kernel-metadata.json` + the build: set the build's `ENABLE_LLM=False` (drops the wheels install + `dataset_sources`; agent runs classical via NullGenerator). Rebuild + push. Use this only if the LLM path blocks the validation.

## Note
`notebooks/submission.ipynb` is a built artifact (gitignored). The submission repo
has uncommitted changes (the agent re-sync + offline wiring) — committing is
optional for `kernels push` (it pushes local files, not git) but recommended for
record. Commit message: `Wire offline Qwen proposer into submission notebook`.
