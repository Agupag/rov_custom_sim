#!/usr/bin/env bash
# Run the ROV simulator using the conda environment with pybullet installed.
# This avoids ModuleNotFoundError when your project's .venv doesn't have pybullet.

ENV_NAME=rov_conda
PY=python

# If conda is not initialized in this shell, use `conda run` which works without activation.
# Prefer `conda run -n rov_conda python rov_sim.py` for one-off runs.

exec conda run -n "$ENV_NAME" $PY "$PWD/rov_sim.py" "$@"
