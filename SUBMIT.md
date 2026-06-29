# SUBMIT — quick steps

Current build: **classical aim-when-stuck (LLM-OFF, CPU)**. Self-contained
(no external dataset/model). Already built & verified — just push and submit.

> Rebuild only if `agent/` changed:
> ```bash
> PY="/c/Users/good_/OneDrive/Documents/GitHub/Kaggle/ARC-AGI-3/ARC-AGI-3-Agents/.venv/Scripts/python.exe"
> "$PY" scripts/bundle_agent.py && "$PY" scripts/build_notebook.py
> ```

## 1. Auth (once per shell)
```bash
export KAGGLE_USERNAME=goodrelax
export KAGGLE_KEY="$(tr -d '\r\n' < /c/Users/good_/.kaggle/access_token)"
export PYTHONUTF8=1
```

## 2. Push the kernel (NOT yet the competition submit)
```bash
kaggle kernels push -p notebooks/
```

## 3. Submit in the browser
Open the kernel on kaggle.com → **Submit to Competition**.
(Limit **1/day**. The scored run happens in the Phase-B rerun after Submit.)

- Kernel: https://www.kaggle.com/code/goodrelax/arc-prize-2026-arc-agi-3-starter

## (optional) Track the push
```bash
kaggle kernels status goodrelax/arc-prize-2026-arc-agi-3-starter
kaggle kernels output goodrelax/arc-prize-2026-arc-agi-3-starter -p out/
```
