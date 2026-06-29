# SUBMIT-HANDOFF — classical "aim-when-stuck" score attempt (LLM-OFF, CPU)

**Purpose of THIS submission:** a real **score attempt**. The recommended config:
**LLM OFF + navigate/marked ON + stuck-gate** (the agent runs the proven classical
round-robin by default and only DEVIATES to aiming when genuinely STUCK).

**Why LLM-OFF:** submitted history is decisive — **v1 classical = 0.08** >> **v3
(LLM + always-on aiming) = 0.01**. Always-on smart moves REGRESSED the score (chased
wrong targets, broke games the blind round-robin solved). The fix = `aim-when-stuck`:
navigate/click-navigate fire ONLY after the masked board has not changed for
`>= ARC_AIM_STUCK` turns (default K=6). Local 25-game offline A/B (LLM off) proved
**4 levels {cd82, sp80, lp85, vc33}** vs classical's 2 {cd82, sp80} — keeps classical's
wins AND adds 2 click games (~2x). Dropping the LLM also removes the wheels/Qwen/
constrained-decoding risk surface — simpler, lower-risk, and empirically better.

## State (built & verified — 2026-06-30, classical LLM-OFF build)
Everything here is ready. `notebooks/submission.ipynb` + `agent/my_agent.py` were
freshly built from the latest agent and checked:
- bundle = 30 files incl. the v14 `agent/assets/*.tsv` catalogs and the full classical
  solving layer in `search_agent.py`: **marked** goal-marker detector (fires the `target`
  role) + greedy/**BFS navigate** (routes the controllable around walls) + **click-navigate**
  (clicks a target centroid on click games), all gated behind the **stuck-gate**
  (`_no_progress_streak` / `ARC_AIM_STUCK`, default K=6). Gate-green (pytest 527).
  Verified: bundle self-extracts and `OurSearchAgent` imports & registers via the Kaggle
  (`agents`-first) load path; the stuck-gate is present in the bundled source.
- **NO offline-deps cell** (classical build drops it) — only the official competition
  framework wheel (`arc-agi`) is installed, exactly as the starter requires.
- `notebooks/kernel-metadata.json`: id `goodrelax/arc-prize-2026-arc-agi-3-starter`,
  `enable_internet:false`, **`enable_gpu:false`**, **`dataset_sources:[]`**, **`model_sources:[]`**.
- Build constants (`scripts/build_notebook.py`): **`ACCELERATOR="cpu"`, `ENABLE_LLM=False`**.

## Prerequisites
None beyond the competition itself — the classical build needs **no** external dataset
or model (no `arc-agi3-llm-wheels`, no Qwen mount). Self-contained.

## To rebuild the LLM-ON variant (one toggle back)
Flip `scripts/build_notebook.py` to `ACCELERATOR="t4"` + `ENABLE_LLM=True`, re-add
`"qwen-lm/qwen2.5/transformers/1.5b-instruct/1"` to `model_sources` in
`notebooks/kernel-metadata.json` (hand-maintained), then rebuild. Not recommended
(scored 0.01).

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
