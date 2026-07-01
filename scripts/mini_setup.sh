#!/usr/bin/env bash
# Prepare the Mac mini Python environment.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE=${REMOTE:-mini}
REMOTE_DIR=${REMOTE_DIR:-"~/code/world-model-from-scratch"}
PYTHON=${PYTHON:-"/opt/homebrew/bin/python3.14"}
INSTALL_EVAL_DEPS=${INSTALL_EVAL_DEPS:-0}

ssh "$REMOTE" "cd $REMOTE_DIR && \
  test -x $PYTHON && \
  $PYTHON -m venv .venv && \
  ./.venv/bin/python -m pip install --upgrade pip wheel && \
  ./.venv/bin/python -m pip install torch numpy h5py tqdm plotly pytest imageio pillow && \
  if [ '$INSTALL_EVAL_DEPS' = '1' ]; then ./.venv/bin/python -m pip install gymnasium highway-env pygame || true; fi && \
  ./.venv/bin/python - <<'PY'
import torch, h5py, numpy
print('torch', torch.__version__)
print('mps', torch.backends.mps.is_available(), torch.backends.mps.is_built())
PY"
