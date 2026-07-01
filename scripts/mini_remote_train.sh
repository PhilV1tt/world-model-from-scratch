#!/usr/bin/env bash
# Run LeWM on the Mac mini. Intended to be launched remotely by mini_launch.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR=${OUT_DIR:-"runs/experiments/mini_mps_${STAMP}"}
EPOCHS=${EPOCHS:-8}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-100}
DATA=${DATA:-"data/parking/train_v2.h5"}
BATCH=${BATCH:-128}
LR=${LR:-2e-4}
SEQ_LEN=${SEQ_LEN:-6}
FRAME_SKIP=${FRAME_SKIP:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
LOG_EVERY_STEPS=${LOG_EVERY_STEPS:-10}
VAL_EVERY_STEPS=${VAL_EVERY_STEPS:-200}
VAL_BATCHES=${VAL_BATCHES:-4}
ENC_DEPTH=${ENC_DEPTH:-4}
PRED_DEPTH=${PRED_DEPTH:-2}
AUTOREGRESSIVE_WEIGHT=${AUTOREGRESSIVE_WEIGHT:-0.15}
AUTOREGRESSIVE_RAMP_STEPS=${AUTOREGRESSIVE_RAMP_STEPS:-500}
STOP_AFTER_STEPS=${STOP_AFTER_STEPS:-0}
RESUME=${RESUME:-}
RESUME_SCHEDULE=${RESUME_SCHEDULE:-extend}

if [[ ! -f "$DATA" ]]; then
  echo "Missing dataset: $DATA"
  echo "Collect or sync dataset v2 first, or override DATA=..."
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export WM_TORCH_THREADS=${WM_TORCH_THREADS:-4}

ARGS=(
  scripts/train.py
  --data "$DATA"
  --out "$OUT_DIR"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --lr "$LR"
  --seq-len "$SEQ_LEN"
  --frame-skip "$FRAME_SKIP"
  --num-workers "$NUM_WORKERS"
  --device mps
  --bf16
  --enc-depth "$ENC_DEPTH"
  --pred-depth "$PRED_DEPTH"
  --save-every-steps "$SAVE_EVERY_STEPS"
  --log-every-steps "$LOG_EVERY_STEPS"
  --val-every-steps "$VAL_EVERY_STEPS"
  --val-batches "$VAL_BATCHES"
  --autoregressive-weight "$AUTOREGRESSIVE_WEIGHT"
  --autoregressive-ramp-steps "$AUTOREGRESSIVE_RAMP_STEPS"
)

if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume "$RESUME" --resume-schedule "$RESUME_SCHEDULE")
fi

if [[ "$STOP_AFTER_STEPS" != "0" ]]; then
  ARGS+=(--stop-after-steps "$STOP_AFTER_STEPS")
fi

./.venv/bin/python "${ARGS[@]}"
