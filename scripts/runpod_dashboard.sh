#!/usr/bin/env bash
# Lightweight RunPod dashboard: collection progress now, training curves later.
set -euo pipefail
cd "$(dirname "$0")/.."

ACTION=${1:-start}
PORT=${PORT:-8788}
DASH_DIR=${DASH_DIR:-"runs/runpod_dashboard"}
RUN_DIR=${RUN_DIR:-""}
COLLECT_LOG=${COLLECT_LOG:-""}
COLLECT_DATASET=${COLLECT_DATASET:-"data/parking/train_v3_mega_50000.h5"}

if [[ -z "$COLLECT_LOG" ]]; then
  COLLECT_LOG="$(ls -t runs/collect_v3_mega_*.log 2>/dev/null | head -1 || true)"
fi

stop_dashboard() {
  if [[ -f "$DASH_DIR/watch.pid" ]]; then
    kill "$(cat "$DASH_DIR/watch.pid")" 2>/dev/null || true
  fi
  if [[ -f "$DASH_DIR/http.pid" ]]; then
    kill "$(cat "$DASH_DIR/http.pid")" 2>/dev/null || true
  fi
  pkill -f "scripts/watch_training.py.*$DASH_DIR" 2>/dev/null || true
  pkill -f "http.server $PORT .*--directory $DASH_DIR" 2>/dev/null || true
  rm -f "$DASH_DIR/watch.pid" "$DASH_DIR/http.pid"
}

if [[ "$ACTION" == "stop" ]]; then
  stop_dashboard
  echo "stopped RunPod dashboard"
  exit 0
fi

if [[ "$ACTION" != "start" ]]; then
  echo "usage: $0 [start|stop]" >&2
  exit 2
fi

mkdir -p "$DASH_DIR"
if [[ -n "$RUN_DIR" ]]; then
  mkdir -p "$RUN_DIR"
fi
stop_dashboard

watch_args=(
  ./.venv/bin/python scripts/watch_training.py
  --dashboard-dir "$DASH_DIR"
  --eval-episodes 0
)

if [[ -n "$RUN_DIR" ]]; then
  watch_args+=(--run-dir "$RUN_DIR")
else
  watch_args+=(--follow-latest)
fi

if [[ -n "$COLLECT_LOG" ]]; then
  watch_args+=(--collect-log "$COLLECT_LOG" --collect-dataset "$COLLECT_DATASET")
fi

nohup "${watch_args[@]}" > "$DASH_DIR/watch.log" 2> "$DASH_DIR/watch.err" &
echo $! > "$DASH_DIR/watch.pid"

nohup ./.venv/bin/python -m http.server "$PORT" --bind 0.0.0.0 --directory "$DASH_DIR" > "$DASH_DIR/http.log" 2> "$DASH_DIR/http.err" &
echo $! > "$DASH_DIR/http.pid"

cat <<EOF
RunPod dashboard started.
Port:        $PORT
Dashboard:   $DASH_DIR
Collect log: ${COLLECT_LOG:-none}
Dataset:     $COLLECT_DATASET

Tail:
  tail -f $DASH_DIR/watch.log
  tail -f ${COLLECT_LOG:-runs/collect_v3_mega_*.log}

Stop:
  ./scripts/runpod_dashboard.sh stop
EOF
