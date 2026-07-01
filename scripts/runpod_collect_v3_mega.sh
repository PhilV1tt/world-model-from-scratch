#!/usr/bin/env bash
# Lance une collecte v3 massive sur RunPod/Linux, en arriere-plan et tail-able.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" == "Darwin" && "${ALLOW_LOCAL_MEGA:-0}" != "1" ]]; then
  echo "Refusing mega collection on macOS. Run this on RunPod, or set ALLOW_LOCAL_MEGA=1 for a deliberate local smoke." >&2
  exit 2
fi

EPISODES=${EPISODES:-50000}
MAX_STEPS=${MAX_STEPS:-120}
WORKERS=${WORKERS:-16}
EPISODES_PER_CHUNK=${EPISODES_PER_CHUNK:-8}
SEED=${SEED:-3027}
SPLIT_SEED=${SPLIT_SEED:-3027}
CHECK_READABLE=${CHECK_READABLE:-1}
BACKGROUND=${BACKGROUND:-1}
OUT=${OUT:-"data/parking/train_v3_mega_${EPISODES}.h5"}
LOG=${LOG:-"runs/collect_v3_mega_$(date +%Y%m%d_%H%M%S).log"}

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

cmd=(
  env
  OUT="$OUT"
  EPISODES="$EPISODES"
  MAX_STEPS="$MAX_STEPS"
  WORKERS="$WORKERS"
  EPISODES_PER_CHUNK="$EPISODES_PER_CHUNK"
  STREAM_WRITE=1
  SEED="$SEED"
  SPLIT_SEED="$SPLIT_SEED"
  CHECK_READABLE="$CHECK_READABLE"
  ./scripts/collect_parking_v3.sh
)

echo "dataset: $OUT"
echo "log: $LOG"
echo "episodes=$EPISODES max_steps=$MAX_STEPS workers=$WORKERS episodes_per_chunk=$EPISODES_PER_CHUNK"

if [[ "$BACKGROUND" == "1" ]]; then
  nohup "${cmd[@]}" > "$LOG" 2>&1 &
  echo "pid: $!"
  echo "tail: tail -f $LOG"
else
  "${cmd[@]}" 2>&1 | tee "$LOG"
fi
