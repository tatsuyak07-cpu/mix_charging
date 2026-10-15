#!/usr/bin/env bash
set -euo pipefail
cd /workspace/mix_charging
mkdir -p /workspace/environment/mix_charging/mplconfig /workspace/environment/mix_charging/cache
export UV_CACHE_DIR=/workspace/environment/uv-cache
export MPLCONFIGDIR=/workspace/environment/mix_charging/mplconfig
export XDG_CACHE_HOME=/workspace/environment/mix_charging/cache
export MPLBACKEND=Agg
python3 -m venv /workspace/environment/mix_charging/.venv
uv pip install --python /workspace/environment/mix_charging/.venv/bin/python -r requirements-lock.txt
uv pip install --python /workspace/environment/mix_charging/.venv/bin/python --no-deps -e .
python3 tools/bootstrap_calculix.py
uv pip check --python /workspace/environment/mix_charging/.venv/bin/python
/workspace/environment/mix_charging/.venv/bin/python -m pytest -q
