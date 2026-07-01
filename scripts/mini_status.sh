#!/usr/bin/env bash
# Show Mac mini training status and latest metrics.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE=${REMOTE:-mini}
REMOTE_DIR=${REMOTE_DIR:-"~/code/world-model-from-scratch"}

ssh "$REMOTE" "cd $REMOTE_DIR && \
  echo log=\$(cat runs/current_mini_log.txt 2>/dev/null || true) && \
  if [ -f runs/mini_train.pid ] && kill -0 \$(cat runs/mini_train.pid) 2>/dev/null; then echo running pid=\$(cat runs/mini_train.pid); else echo not_running; fi && \
  echo ---process--- && ps -o pid,pcpu,pmem,etime,command -p \$(cat runs/mini_train.pid 2>/dev/null) 2>/dev/null || true && \
  echo ---log--- && tail -30 \$(cat runs/current_mini_log.txt 2>/dev/null) 2>/dev/null | sed -e 's/\r/\n/g' | tail -30 && \
  echo ---csv--- && find runs/experiments -maxdepth 2 -name train_log.csv -print | sort | tail -1 | xargs tail -5 2>/dev/null || true"
