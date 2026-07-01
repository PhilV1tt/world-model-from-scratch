#!/usr/bin/env bash
# Collecte dataset v3: plus de variantes initiales, env statiques/crowded, metadata enrichie.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-"data/parking/train_v3.h5"}
EPISODES=${EPISODES:-6000}
MAX_STEPS=${MAX_STEPS:-120}
WORKERS=${WORKERS:-6}
SEED=${SEED:-2027}
SPLIT_SEED=${SPLIT_SEED:-2027}
CHECK_READABLE=${CHECK_READABLE:-1}
STREAM_WRITE=${STREAM_WRITE:-1}
EPISODES_PER_CHUNK=${EPISODES_PER_CHUNK:-8}
POLICY_MIX=${POLICY_MIX:-"pd,pd_noisy,expert,expert_noisy,reverse,reverse_noisy,near_goal_correction,near_goal_noisy,final_alignment,final_alignment_noisy"}
ENV_VARIANTS=${ENV_VARIANTS:-"standard,wide_start,long_approach,near_goal,short_correction,reverse_entry,final_alignment,slot_offset,static_vehicles,crowded_static"}

cmd=(
  ./.venv/bin/python scripts/collect_parking.py
  --dataset-version v3
  --episodes "$EPISODES"
  --max-steps "$MAX_STEPS"
  --workers "$WORKERS"
  --seed "$SEED"
  --split-seed "$SPLIT_SEED"
  --out "$OUT"
  --policy-mix "$POLICY_MIX"
  --env-variants "$ENV_VARIANTS"
)

if [[ "$STREAM_WRITE" == "1" ]]; then
  cmd+=(--stream-write --episodes-per-chunk "$EPISODES_PER_CHUNK")
fi

if [[ "$CHECK_READABLE" == "1" ]]; then
  cmd+=(--check-readable --check-seq-len 3)
fi

"${cmd[@]}"
