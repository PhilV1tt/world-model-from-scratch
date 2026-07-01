#!/usr/bin/env bash
# Premier run de training - 5 epochs rapides, model leger (~7M params).
# Apres tu peux relancer: ./scripts/train_more.sh 5
set -e
cd "$(dirname "$0")/.."
STAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR=${OUT_DIR:-"runs/experiments/clean_v1_${STAMP}"}
./.venv/bin/python scripts/train.py \
  --epochs 5 \
  --batch 128 --num-workers 4 --bf16 \
  --enc-depth 6 --pred-depth 3 \
  --out "$OUT_DIR"
