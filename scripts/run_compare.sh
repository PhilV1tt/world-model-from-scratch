#!/usr/bin/env bash
# Compare checkpoint courant vs nouveau run clean avec budget local court.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +"%Y%m%d_%H%M%S")
EPOCHS=${EPOCHS:-1}
STEPS=${STEPS:-120}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-50}
EVAL_EPISODES=${EVAL_EPISODES:-10}
SEED=${SEED:-0}
BASE="runs/experiments/compare_${STAMP}"
CURRENT_OUT="${BASE}/current_fixed"
CLEAN_OUT="${BASE}/clean_v1"
CURRENT_CKPT=${CURRENT_CKPT:-"runs/last/ckpt_last.pt"}

mkdir -p "$BASE"

ENC_DEPTH=$(./.venv/bin/python -c "import torch; c=torch.load('$CURRENT_CKPT', map_location='cpu', weights_only=False); print(c['cfg']['enc_depth'])")
PRED_DEPTH=$(./.venv/bin/python -c "import torch; c=torch.load('$CURRENT_CKPT', map_location='cpu', weights_only=False); print(c['cfg']['pred_depth'])")

echo "A/current_fixed -> $CURRENT_OUT"
./.venv/bin/python scripts/train.py \
  --resume "$CURRENT_CKPT" \
  --resume-schedule extend \
  --epochs "$EPOCHS" \
  --stop-after-steps "$STEPS" \
  --save-every-steps "$SAVE_EVERY_STEPS" \
  --batch 128 --num-workers 4 --bf16 \
  --seed "$SEED" \
  --enc-depth "$ENC_DEPTH" --pred-depth "$PRED_DEPTH" \
  --val-every-steps 50 --val-batches 4 \
  --out "$CURRENT_OUT"

echo "B/clean_v1 -> $CLEAN_OUT"
./.venv/bin/python scripts/train.py \
  --epochs "$EPOCHS" \
  --stop-after-steps "$STEPS" \
  --save-every-steps "$SAVE_EVERY_STEPS" \
  --batch 128 --num-workers 4 --bf16 \
  --seed "$SEED" \
  --enc-depth "$ENC_DEPTH" --pred-depth "$PRED_DEPTH" \
  --val-every-steps 50 --val-batches 4 \
  --out "$CLEAN_OUT"

./.venv/bin/python scripts/eval_compare.py \
  "$CURRENT_OUT" "$CLEAN_OUT" \
  --episodes "$EVAL_EPISODES" \
  --seed 4242 \
  --device cpu \
  --warm-start-pd \
  --out "$BASE/eval_compare.csv" \
  --gif-dir "$BASE/compare_gifs"

echo "comparison saved in $BASE"
