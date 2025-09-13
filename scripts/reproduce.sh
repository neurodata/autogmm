#!/usr/bin/env bash
set -euo pipefail

# Where to drop figures & executed notebook
mkdir -p out
export AGMM_OUT="${AGMM_OUT:-out}"

# Ensure in the repo root
if [[ ! -f pyproject.toml ]]; then
  echo "ERROR: Run this from the repository root (pyproject.toml not found)." >&2
  exit 1
fi

python -m pip install --upgrade pip
pip install -e '.[experiments]'

# Fail fast if R/mclust/rpy2 are not available
python - <<'PY'
import sys
try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    importr("mclust")
    print("R OK:", ro.r('R.version.string')[0])
except Exception as e:
    sys.stderr.write(
        "\nERROR: Full reproduction requires R + mclust + rpy2.\n"
        "Activate your experiments env, e.g.:\n"
        "  conda activate autogmm-exp-full\n"
    )
    raise
PY

# Locate the notebook
if [[ -f examples/experiments.ipynb ]]; then
  NB="examples/experiments.ipynb"
else
  echo "ERROR: experiments notebook not found." >&2
  exit 1
fi

# Execute and tee a log
python scripts/run_notebook.py "$NB" out/experiments_executed.ipynb | tee out/execute.log
echo "Re-executed notebook saved to out/experiments_executed.ipynb"
echo "Figures saved under out/"
