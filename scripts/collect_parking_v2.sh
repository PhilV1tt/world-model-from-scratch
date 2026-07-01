#!/usr/bin/env bash
# Collecte dataset v2 plus utile pour planning: expert, reverse, near-goal, align.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-"data/parking/train_v2.h5"}
EPISODES=${EPISODES:-4000}
MAX_STEPS=${MAX_STEPS:-100}
WORKERS=${WORKERS:-6}
SEED=${SEED:-2026}
SPLIT_SEED=${SPLIT_SEED:-2026}
POLICY_MIX=${POLICY_MIX:-"random,pd,pd_noisy,expert,expert_noisy,reverse,reverse_noisy,near_goal_correction,near_goal_noisy,final_alignment,final_alignment_noisy"}

./.venv/bin/python scripts/collect_parking.py \
  --episodes "$EPISODES" \
  --max-steps "$MAX_STEPS" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --split-seed "$SPLIT_SEED" \
  --out "$OUT" \
  --policy-mix "$POLICY_MIX"
