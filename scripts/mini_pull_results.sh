#!/usr/bin/env bash
# Pull Mac mini run outputs back to this MacBook.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE=${REMOTE:-mini}
REMOTE_DIR=${REMOTE_DIR:-"~/code/world-model-from-scratch"}
LOCAL_DIR=${LOCAL_DIR:-"runs/mini_sync"}

mkdir -p "$LOCAL_DIR"

CURRENT_LOG=$(ssh "$REMOTE" "cd $REMOTE_DIR && cat runs/current_mini_log.txt 2>/dev/null || true")
LATEST_RUN=$(ssh "$REMOTE" "cd $REMOTE_DIR && find runs/experiments -maxdepth 2 -name train_log.csv -print 2>/dev/null | sort | tail -1 | xargs dirname 2>/dev/null || true")

rsync -azP "$REMOTE:$REMOTE_DIR/runs/current_mini_log.txt" "$LOCAL_DIR/" 2>/dev/null || true
if [[ -n "$CURRENT_LOG" ]]; then
  rsync -azP "$REMOTE:$REMOTE_DIR/$CURRENT_LOG" "$LOCAL_DIR/" 2>/dev/null || true
fi

if [[ -n "$LATEST_RUN" && "$LATEST_RUN" != "." ]]; then
  DEST="$LOCAL_DIR/$LATEST_RUN"
  mkdir -p "$DEST"
  rsync -azP \
    "$REMOTE:$REMOTE_DIR/$LATEST_RUN/train_log.csv" \
    "$REMOTE:$REMOTE_DIR/$LATEST_RUN/config.json" \
    "$DEST/" 2>/dev/null || true
  rsync -azP "$REMOTE:$REMOTE_DIR/$LATEST_RUN/ckpt_last.pt" "$DEST/" 2>/dev/null || true
fi

echo "Pulled mini results into $LOCAL_DIR"
