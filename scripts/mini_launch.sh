#!/usr/bin/env bash
# Launch a detached Mac mini training run and write a tail-friendly log.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE=${REMOTE:-mini}
REMOTE_DIR=${REMOTE_DIR:-"~/code/world-model-from-scratch"}

REMOTE_CMD='
set -euo pipefail
cd '"$REMOTE_DIR"'
mkdir -p runs
if [ -f runs/mini_train.pid ] && kill -0 "$(cat runs/mini_train.pid)" 2>/dev/null; then
  echo "Mini training already running: pid=$(cat runs/mini_train.pid)"
  echo "log=$(cat runs/current_mini_log.txt 2>/dev/null || true)"
  exit 0
fi
STAMP=$(date +"%Y%m%d_%H%M%S")
LOG="runs/mini_mps_${STAMP}.log"
echo "$LOG" > runs/current_mini_log.txt
nohup env PYTHONUNBUFFERED=1 '"${EXTRA_ENV:-}"' ./scripts/mini_remote_train.sh > "$LOG" 2>&1 &
echo $! > runs/mini_train.pid
echo "pid=$(cat runs/mini_train.pid)"
echo "log=$LOG"
'

ssh "$REMOTE" "$REMOTE_CMD"
