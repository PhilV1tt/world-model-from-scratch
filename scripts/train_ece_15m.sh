#!/usr/bin/env bash
# Run lourd pour ia-lab-01: LeWM ~15M, seq_len plus long, sans BF16.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR=${OUT_DIR:-"runs/experiments/ece_15m_${STAMP}"}
EPOCHS=${EPOCHS:-20}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-100}
DATA=${DATA:-"data/parking/train_v2.h5"}
NPROC=${NPROC:-4}
BATCH_PER_GPU=${BATCH_PER_GPU:-96}
NUM_WORKERS=${NUM_WORKERS:-2}
SEQ_LEN=${SEQ_LEN:-6}
FRAME_SKIP=${FRAME_SKIP:-1}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
STOP_AFTER_STEPS=${STOP_AFTER_STEPS:-0}
DIST_BACKEND=${DIST_BACKEND:-gloo}
VAL_EVERY_STEPS=${VAL_EVERY_STEPS:-200}
VAL_BATCHES=${VAL_BATCHES:-8}
AUTOREGRESSIVE_WEIGHT=${AUTOREGRESSIVE_WEIGHT:-0.25}
AUTOREGRESSIVE_RAMP_STEPS=${AUTOREGRESSIVE_RAMP_STEPS:-1000}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export WM_TORCH_THREADS=${WM_TORCH_THREADS:-4}

if [[ ! -f "$DATA" ]]; then
  echo "Missing dataset: $DATA"
  echo "Collect or sync dataset v2 first, or override DATA=..."
  exit 1
fi

TRAIN_ARGS=(
  scripts/train.py
  --data "$DATA"
  --out "$OUT_DIR"
  --epochs "$EPOCHS"
  --batch "$BATCH_PER_GPU"
  --num-workers "$NUM_WORKERS"
  --device cuda
  --distributed
  --dist-backend "$DIST_BACKEND"
  --seq-len "$SEQ_LEN"
  --frame-skip "$FRAME_SKIP"
  --enc-depth 12
  --pred-depth 6
  --sigreg-n-proj 256
  --save-every-steps "$SAVE_EVERY_STEPS"
  --val-every-steps "$VAL_EVERY_STEPS"
  --val-batches "$VAL_BATCHES"
  --autoregressive-weight "$AUTOREGRESSIVE_WEIGHT"
  --autoregressive-ramp-steps "$AUTOREGRESSIVE_RAMP_STEPS"
)

if [[ "$STOP_AFTER_STEPS" != "0" ]]; then
  TRAIN_ARGS+=(--stop-after-steps "$STOP_AFTER_STEPS")
fi

./.venv/bin/python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  "${TRAIN_ARGS[@]}"
