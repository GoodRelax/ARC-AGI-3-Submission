# Submissions — ARC Prize 2026 / ARC-AGI-3

One row per Kaggle submission. Each is built from this repo
(`scripts/bundle_agent.py` → `scripts/build_notebook.py`; see README), pushed
with `kaggle kernels push`, then "Submit to Competition". Newest on top.

| # | Date | Notebook (version) | Accel | Public score | Public LB rank | Notebook public? | Notes |
|--:|------|--------------------|-------|-------------:|----------------|------------------|-------|
| **001** | 2026-06-25 | [arc-prize-2026-arc-agi-3-starter](https://www.kaggle.com/code/goodrelax/arc-prize-2026-arc-agi-3-starter) · v1 | cpu | **0.08** | 1161 / 1412 | yes — Apache 2.0 | Pipeline-validation baseline. Our search agent (`OurSearchAgent`) shipped as a self-extracting `my_agent.py`. First leaderboard entry. |

## submit-001 — what it established
- **End-to-end path works**: `push` → **Phase A** commit (dummy output) →
  **Submit to Competition** → **Phase B** competition rerun (the agent actually
  plays via the gateway) → scored. Our multi-file `agent/` package travels inside
  the single-file notebook as a zip+base64 self-extracting payload.
- **No extra deps needed**: `numpy` / `scipy` / `scikit-image` resolve from
  Kaggle's base image; only `arc-agi` comes from the offline competition wheel.
- **Eligibility**: notebook published (public) before the 2026-06-30 milestone.
- **Score 0.08 is a baseline** (the agent does not yet self-solve levels on the
  real engine) — above the 0.00 entries, well below the contenders (~0.5–1.2;
  #1 was 1.21 on 2026-06-25).

## Next submission
Improve the agent (self-solve L1+ on the real engine) → re-run
`bundle_agent.py` + `build_notebook.py` → `kaggle kernels push` (creates v2) →
Submit. Limit: **1 submission/day**. Tag each release commit `submission-NNN`.
