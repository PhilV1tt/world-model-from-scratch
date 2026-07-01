#!/usr/bin/env bash
# Eval stable random/PD/model/model+PD-warm-start sur seeds fixes.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR=${1:-${RUN_DIR:-"runs/last"}}
EPISODES=${EPISODES:-20}
SEED=${SEED:-4242}
DEVICE=${DEVICE:-cpu}
OUT_DIR=${OUT_DIR:-"$RUN_DIR/eval_protocol"}
PLANNERS=${PLANNERS:-"random,pd,model,model_pd"}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-80}

cmd=(
  ./.venv/bin/python scripts/eval_protocol.py
  --run-dir "$RUN_DIR"
  --out-dir "$OUT_DIR"
  --planners "$PLANNERS"
  --episodes "$EPISODES"
  --seed "$SEED"
  --device "$DEVICE"
  --horizon "${HORIZON:-5}"
  --mpc-apply "${MPC_APPLY:-3}"
  --max-env-steps "$MAX_ENV_STEPS"
  --pop "${POP:-150}"
  --elites "${ELITES:-15}"
  --iters "${ITERS:-8}"
  --action-l2 "${ACTION_L2:-0.01}"
  --smooth-l2 "${SMOOTH_L2:-0.05}"
  --action-smoothing "${ACTION_SMOOTHING:-0.25}"
  --trajectory-weight "${TRAJECTORY_WEIGHT:-0.0}"
)

if [[ -n "${CKPT:-}" ]]; then
  cmd+=(--ckpt "$CKPT")
fi

if [[ "${GIFS:-1}" == "0" ]]; then
  cmd+=(--no-gifs)
fi

"${cmd[@]}"
