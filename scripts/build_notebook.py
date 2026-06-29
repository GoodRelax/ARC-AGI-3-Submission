"""Splice the current `agent/my_agent.py` into `notebooks/submission.ipynb`.

The notebook follows the exact pattern used by Kaggle's official sample
("ARC3 Sample Submission - Stochastic Goose"):

  Cell 1: install the `arc-agi` wheel from the offline competition dataset.
  Cell 2: write `my_agent.py` to /kaggle/working/ — its body is THIS file.
  Cell 3: if running inside the Kaggle competition rerun, wait for the
          gateway sidecar, copy the framework into /kaggle/working/, register
          MyAgent, and run `python main.py --agent myagent`.
  Cell 4: otherwise (during commit / save-and-run-all), write a dummy
          submission.parquet so Kaggle accepts the commit.

You don't normally need to call this directly — `make submit` runs it for you.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THIS ONE LINE TO PICK YOUR KAGGLE ACCELERATOR
# Options:
#   "cpu"      — no GPU. Good for the random starter or any non-ML agent.
#   "t4"       — Nvidia T4 ×2 (default; matches Kaggle's sample submission).
#   "p100"     — Nvidia P100 (single big-memory GPU).
#   "rtx6000"  — Nvidia RTX 6000 (g4-standard-48). ARC-AGI-3 exclusive,
#                burns GPU quota faster — use only when you're confident.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "t4"

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL-LLM (Qwen) move proposer. When ENABLE_LLM is True the competition cell
# exports ARC_LLM=1 + ARC_LLM_MODEL=<mount> so the agent's _build_llm activates
# the offline QwenGenerator; the Qwen weights are mounted via model_sources in
# notebooks/kernel-metadata.json. The agent gracefully degrades to the classical
# NullGenerator if the mount/load fails, so this is low-risk.
#
# TO BUILD THE CLASSICAL v14 VARIANT (CPU, no model, no LLM): flip the two
# constants below, then re-run `python scripts/build_notebook.py`:
#   1. ACCELERATOR = "cpu"   -> metadata enable_gpu=false (synced in main()).
#   2. ENABLE_LLM   = False  -> no ARC_LLM env exported (pure classical), the
#                               offline-deps cell is dropped, AND dataset_sources
#                               is cleared automatically (synced in main()).
# Then MANUALLY remove the Qwen entry from "model_sources" in
# notebooks/kernel-metadata.json (model_sources is hand-maintained, not synced).
# That is the whole classical recipe: ACCELERATOR=cpu, ENABLE_LLM=False, drop
# model_sources (manual), drop dataset_sources (automatic).
# ─────────────────────────────────────────────────────────────────────────────
ENABLE_LLM = True
# Offline constrained-decoding wheels (transformers 4.44.2 + lm-format-enforcer
# 0.10.9 + transitive deps) live in this Kaggle Dataset; attached via
# dataset_sources when ENABLE_LLM. Cleared automatically for the classical build.
WHEELS_DATASET_SLUG = "goodrelax/arc-agi3-llm-wheels"
# Kaggle mounts a model_sources slug "<owner>/<model>/<framework>/<variation>/
# <version>" at /kaggle/input/<owner>/<model>/<framework>/<variation>/<version>.
# The agent's _resolve_model_path auto-discovers config.json under this root, so
# a nested version dir still resolves even if the exact leaf differs.
LLM_MODEL_MOUNT = "/kaggle/input/qwen-lm/qwen2.5/transformers/1.5b-instruct/1"

# Internal mapping; don't edit unless Kaggle adds new options.
_ACCELERATORS = {
    "cpu":     {"name": "none",            "gpu": False},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text()

    install_cell = code_cell(
        "!pip install --no-index --find-links \\\n"
        "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv"
    )

    # OFFLINE constrained-decoding deps. enable_internet=false forbids internet
    # pip, so transformers==4.44.2 + lm-format-enforcer==0.10.9 (+ transitive
    # deps) must come from an attached wheels dataset installed with --no-index.
    # The Kaggle base image ships transformers 5.0.0, which removed LogitsWarper /
    # moved PreTrainedTokenizerBase and breaks lmfe's import; this DOWNGRADE to
    # 4.44.2 must run BEFORE any transformers import (i.e. before the run cell's
    # `python main.py` subprocess lazily imports torch/transformers).
    #
    # The wheels dataset mounts under a NESTED path
    # (/kaggle/input/datasets/<owner>/<name>/), so we AUTO-DISCOVER the wheel dir
    # at runtime via a recursive glob rather than hard-coding the mount. The whole
    # step is guarded (try/except + check): if the dataset is absent (e.g. the
    # classical build) it is a harmless no-op and the agent degrades to the
    # classical NullGenerator — it never crashes the scored run.
    offline_deps_cell = code_cell(
        dedent(
            """\
            # Offline install of the constrained-decoding deps from the wheels
            # dataset (auto-discovered; no-op + graceful if the dataset is absent).
            import glob, os, subprocess, sys

            _whls = glob.glob('/kaggle/input/**/*.whl', recursive=True)
            if _whls:
                _wheel_dir = os.path.dirname(sorted(_whls)[0])
                print(f'[offline-deps] wheel dir: {_wheel_dir} ({len(_whls)} wheels)')
                try:
                    subprocess.run(
                        [sys.executable, '-m', 'pip', 'install', '--no-index',
                         '--find-links', _wheel_dir,
                         'transformers==4.44.2', 'lm-format-enforcer==0.10.9'],
                        check=True,
                    )
                    print('[offline-deps] installed transformers==4.44.2 + '
                          'lm-format-enforcer==0.10.9')
                except Exception as exc:  # degrade to classical, never crash
                    print(f'[offline-deps] install FAILED ({exc}); agent will '
                          'run classically (unconstrained/NullGenerator).')
            else:
                print('[offline-deps] no wheels dataset mounted; skipping '
                      '(classical build or detached dataset).')
            """
        )
    )

    # We write the agent to /tmp/ (not /kaggle/working/) so it does NOT appear
    # as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer it as a candidate submission file alongside submission.parquet,
    # and an unlucky default selection rejects the submission.
    write_agent_cell = code_cell(
        "%%writefile /tmp/my_agent.py\n" + agent_body
    )

    run_cell_source = dedent(
        """\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            # Drop our agent in as a framework template.
            !cp /tmp/my_agent.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

            # Register MyAgent in the framework's agent registry. We rewrite
            # __init__.py because the upstream version eagerly imports
            # templates with deps we don't ship (langgraph, smolagents, etc.).
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }
        \"\"\")

            # Point the framework at the gateway sidecar.
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        \"\"\")

            # Run it. The gateway records every action and emits submission.parquet.
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
__LLM_ENV__                python main.py --agent myagent
        """
    )
    # LOCAL-LLM env: when ENABLE_LLM, export ARC_LLM=1 + ARC_LLM_MODEL=<mount> so
    # the agent's _build_llm activates the offline Qwen backend. If the model
    # doesn't mount / fails to load, the agent degrades to NullGenerator
    # (classical) — never crashes. Each line is a shell `VAR=val \` continuation
    # spliced INTO the `!cd ... && ...` command above (so they apply to main.py).
    # We use a literal-token replace (not str.format) because the run cell body
    # contains unescaped { } braces (the rewritten agents/__init__.py dict).
    if ENABLE_LLM:
        llm_env = (
            "                ARC_LLM=1 \\\n"
            f"                ARC_LLM_MODEL={LLM_MODEL_MOUNT} \\\n"
        )
    else:
        llm_env = ""
    run_cell_source = run_cell_source.replace("__LLM_ENV__", llm_env)
    run_cell = code_cell(run_cell_source)

    dummy_submission_cell = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Save-and-run-all (commit) mode: emit a dummy submission so the
                # commit succeeds. The real submission.parquet is produced by the
                # gateway during competition rerun.
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'])
                submission.to_parquet('/kaggle/working/submission.parquet', index=False)
                submission.head()
            """
        )
    )

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 Submission\n\n"
                "Built from `agent/my_agent.py` via `scripts/build_notebook.py`. "
                "Do not edit cells directly — edit the source file and re-run "
                "`make submit`."
            ),
            install_cell,
            # Offline LLM deps only when ENABLE_LLM (the classical build omits the
            # wheels dataset entirely, so the cell would be a pure no-op anyway;
            # we drop it for a cleaner classical notebook).
            *( [offline_deps_cell] if ENABLE_LLM else [] ),
            write_agent_cell,
            run_cell,
            dummy_submission_cell,
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
          f"(accelerator: {ACCELERATOR})")

    # Keep notebooks/kernel-metadata.json in sync so the user never has to
    # edit it just to flip CPU <-> GPU or LLM on/off. We sync both enable_gpu
    # (from ACCELERATOR) and dataset_sources (the wheels, from ENABLE_LLM); the
    # Qwen model_sources entry stays hand-maintained.
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        changed = False

        wanted_gpu = _ACCELERATORS[ACCELERATOR]["gpu"]
        if meta.get("enable_gpu") != wanted_gpu:
            meta["enable_gpu"] = wanted_gpu
            changed = True
            print(f"[build_notebook] Synced enable_gpu={wanted_gpu}")

        wanted_datasets = [WHEELS_DATASET_SLUG] if ENABLE_LLM else []
        if meta.get("dataset_sources") != wanted_datasets:
            meta["dataset_sources"] = wanted_datasets
            changed = True
            print(f"[build_notebook] Synced dataset_sources={wanted_datasets}")

        if changed:
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"[build_notebook] Updated {METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
