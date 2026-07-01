#!/usr/bin/env bash
# Sync code + dataset to the Mac mini, without local envs or run history.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE=${REMOTE:-mini}
REMOTE_DIR=${REMOTE_DIR:-"~/code/world-model-from-scratch"}

ssh "$REMOTE" "mkdir -p $REMOTE_DIR"
rsync -az --delete \
  --exclude '.venv/' \
  --exclude 'runs/' \
  --exclude '.git/' \
  --exclude '.DS_Store' \
  --exclude '.pytest_cache/' \
  --exclude '.understand-anything/' \
  --exclude '.vscode/' \
  --exclude '__pycache__/' \
  ./ "$REMOTE:$REMOTE_DIR/"

ssh "$REMOTE" "cd $REMOTE_DIR && chmod +x scripts/mini_remote_train.sh scripts/mini_launch.sh scripts/mini_status.sh scripts/mini_pull_results.sh 2>/dev/null || true"
