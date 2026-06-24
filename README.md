# ARC-AGI-3 Agent — Competition Submission

Public, prize-eligible submission for **ARC Prize 2026 — ARC-AGI-3** (Kaggle
`arc-prize-2026-arc-agi-3`). Licensed **CC BY 4.0** — the winner license required
by the Official Competition Rules (§1.6 / §2.5).

## What this repo is
A **derived export**. The agent's source of truth is a separate private dev repo;
the `agent/` package here is exported from it **as-is** — don't hand-edit `agent/`
here, change it upstream and re-export. The Kaggle starter scaffold and the build
scripts (`scripts/`) are this repo's own tooling.

## Layout
```
agent/                  our search-based agent (exported package)
  my_agent.py           GENERATED self-extracting entry point (do not hand-edit)
scripts/
  bundle_agent.py       packs agent/  ->  self-extracting agent/my_agent.py
  build_notebook.py     splices my_agent.py  ->  notebooks/submission.ipynb
notebooks/
  kernel-metadata.json  kernel id + settings (accelerator / internet off / competition)
  submission.ipynb      GENERATED + git-ignored — the notebook pushed to Kaggle
Makefile                starter targets (Unix only; on Windows run the scripts)
LICENSE                 CC BY 4.0
```

## How `submission.ipynb` is built
This is a Kaggle **code competition**: we ship a notebook and Kaggle runs it. The
notebook is generated from our `agent/` package in two steps:

1. **`scripts/bundle_agent.py`** zips the whole `agent/` package (every `.py` plus
   the settings `.json`; it **excludes** `my_agent.py` itself and `__pycache__`),
   base64-encodes it, and writes **`agent/my_agent.py`** — a *self-extracting*
   file. When imported on Kaggle it unpacks the package onto `sys.path` and exposes
   `MyAgent` (= our `OurSearchAgent`).
2. **`scripts/build_notebook.py`** writes **`notebooks/submission.ipynb`** (5 cells):

   | cell | content |
   |------|---------|
   | 1 markdown | header |
   | 2 code | `pip install --no-index … arc-agi` — offline game engine |
   | 3 code | `%%writefile /tmp/my_agent.py` + our self-extracting `my_agent.py` (~260 KB = the entire agent) |
   | 4 code | competition rerun: wait for the gateway, register `MyAgent`, run `main.py --agent myagent` |
   | 5 code | commit mode only: write a dummy `submission.parquet` |

   The accelerator is the `ACCELERATOR` constant in `build_notebook.py` (currently
   `cpu` — classical search, no GPU).

Run-time deps: `numpy` / `scipy` / `scikit-image` (already in Kaggle's base image);
`arc-agi` from the offline wheel installed in cell 2. No extra wheels dataset needed.

## Rebuild & submit (Windows — no `make`)
The `Makefile` assumes a Unix venv layout, so on Windows run the scripts directly:

```sh
# 1) after editing agent/, regenerate the bundle and the notebook
python scripts/bundle_agent.py
python scripts/build_notebook.py

# 2) push the notebook to Kaggle (needs the kaggle CLI + an API token in ~/.kaggle/)
kaggle kernels push -p notebooks
kaggle kernels status goodrelax/arc-prize-2026-arc-agi-3-starter   # wait for COMPLETE

# 3) on kaggle.com: open the kernel -> Submit to Competition -> output: submission.parquet
#    then Make the notebook Public (required for prize / milestone eligibility)
```

Notes:
- `kernel-metadata.json` `id` = `goodrelax/...` — the Kaggle **username must be
  lowercase**, matching the account.
- **Identity verification** (Kaggle / Persona) is required before *Submit to Competition*.
- Submission limit: **1 / day**. Internet is **off** in the scored kernel.

## Design constraints (from the dev repo)
- Generalize to unseen games — no game-specific hardcoding.
- Run fully offline at evaluation time (no internet, no external services).
