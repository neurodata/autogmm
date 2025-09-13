# AutoGMM Experiments

This folder contains the notebook and data used to reproduce the figures/tables in the AutoGMM paper.

## Quick start

```bash
# 1) Create a clean environment
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# 2) Install package + experiments extras
pip install -e .[experiments]

# 3) (If needed) fetch larger external data
# (No-op if all real-data CSVs are shipped in experiments/data/real)
python experiments/download_data.py || true

# 4) Re-run the master notebook
bash scripts/reproduce.sh
