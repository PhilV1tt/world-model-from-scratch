#!/usr/bin/env bash
# Local dashboard for a Mac mini LeWM training run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$ROOT"

ACTION=${1:-start}
PORT=${PORT:-8777}
DASH_DIR=${DASH_DIR:-"runs/mini_dashboard"}
RUN_DIR=${RUN_DIR:-""}
SEED_GIF_DIR=${SEED_GIF_DIR:-"runs/experiments/compare_20260526_160358/compare_gifs/current_fixed"}
TRAJECTORY_DIR=${TRAJECTORY_DIR:-""}
EVAL_EPISODES=${EVAL_EPISODES:-0}
EVAL_PERIOD=${EVAL_PERIOD:-900}

UID_NUM="$(id -u)"
DOMAIN="gui/$UID_NUM"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
SYNC_LABEL="com.phil.worldmodel.mini-sync"
WATCH_LABEL="com.phil.worldmodel.mini-dashboard-watch"
HTTP_LABEL="com.phil.worldmodel.mini-dashboard-http"

xml_escape() {
  local value="$1"
  value="${value//&/&amp;}"
  value="${value//</&lt;}"
  value="${value//>/&gt;}"
  value="${value//\"/&quot;}"
  printf '%s' "$value"
}

write_plist() {
  local label="$1"
  local plist="$2"
  local stdout_path="$3"
  local stderr_path="$4"
  local keep_alive="$5"
  shift 5

  {
    printf '%s\n' '<?xml version="1.0" encoding="UTF-8"?>'
    printf '%s\n' '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
    printf '%s\n' '<plist version="1.0">'
    printf '%s\n' '<dict>'
    printf '  <key>Label</key><string>%s</string>\n' "$(xml_escape "$label")"
    printf '  <key>WorkingDirectory</key><string>%s</string>\n' "$(xml_escape "$ROOT")"
    printf '%s\n' '  <key>ProgramArguments</key>'
    printf '%s\n' '  <array>'
    for arg in "$@"; do
      printf '    <string>%s</string>\n' "$(xml_escape "$arg")"
    done
    printf '%s\n' '  </array>'
    printf '%s\n' '  <key>RunAtLoad</key><true/>'
    if [[ "$keep_alive" == "true" ]]; then
      printf '%s\n' '  <key>KeepAlive</key><true/>'
    fi
    printf '  <key>StandardOutPath</key><string>%s</string>\n' "$(xml_escape "$stdout_path")"
    printf '  <key>StandardErrorPath</key><string>%s</string>\n' "$(xml_escape "$stderr_path")"
    printf '%s\n' '</dict>'
    printf '%s\n' '</plist>'
  } > "$plist"
}

bootout_job() {
  local label="$1"
  launchctl bootout "$DOMAIN/$label" >/dev/null 2>&1 || true
}

wait_unloaded() {
  local label="$1"
  for _ in {1..50}; do
    if ! launchctl print "$DOMAIN/$label" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

start_job() {
  local label="$1"
  local plist="$2"
  bootout_job "$label"
  wait_unloaded "$label" || true
  if ! launchctl bootstrap "$DOMAIN" "$plist"; then
    sleep 0.5
    bootout_job "$label"
    wait_unloaded "$label" || true
    launchctl bootstrap "$DOMAIN" "$plist"
  fi
}

resolve_run_dir() {
  local found="$RUN_DIR"
  if [[ -z "$found" ]]; then
    local latest_csv
    latest_csv="$(find runs/mini_sync/runs/experiments -maxdepth 2 -name train_log.csv -print 2>/dev/null | sort | tail -1 || true)"
    if [[ -n "$latest_csv" ]]; then
      found="$(dirname "$latest_csv")"
    fi
  fi

  if [[ -z "$found" || ! -f "$found/train_log.csv" ]]; then
    echo "No synced mini train_log.csv found. Run ./scripts/mini_pull_results.sh first." >&2
    return 1
  fi

  (cd "$found" && pwd -P)
}

if [[ "$ACTION" == "stop" ]]; then
  bootout_job "$HTTP_LABEL"
  bootout_job "$WATCH_LABEL"
  bootout_job "$SYNC_LABEL"
  rm -f "$DASH_DIR/http.pid" "$DASH_DIR/watch.pid" runs/mini_sync_loop.pid
  echo "stopped mini dashboard jobs"
  exit 0
fi

if [[ "$ACTION" == "eval" ]]; then
  RUN_DIR="$(resolve_run_dir)"
  STABLE_EPISODES=${EPISODES:-5}
  STABLE_OUT_DIR=${OUT_DIR:-"$RUN_DIR/eval_protocol"}
  echo "Running one-off stable eval on $RUN_DIR"
  echo "Output: $STABLE_OUT_DIR"
  EPISODES="$STABLE_EPISODES" \
    OUT_DIR="$STABLE_OUT_DIR" \
    PLANNERS="${PLANNERS:-random,pd,model,model_pd}" \
    DEVICE="${DEVICE:-cpu}" \
    ./scripts/eval_stable.sh "$RUN_DIR"
  echo "Done. The dashboard watcher will display the new summary.json on its next poll."
  exit 0
fi

if [[ "$ACTION" != "start" ]]; then
  echo "usage: $0 [start|stop|eval]" >&2
  exit 2
fi

mkdir -p "$DASH_DIR" "$LAUNCH_DIR"
DASH_DIR="$(cd "$DASH_DIR" && pwd -P)"
rm -f "$DASH_DIR/http.pid" "$DASH_DIR/watch.pid" runs/mini_sync_loop.pid

RUN_DIR="$(resolve_run_dir)"
if [[ -d "$SEED_GIF_DIR" ]]; then
  SEED_GIF_DIR="$(cd "$SEED_GIF_DIR" && pwd -P)"
fi
if [[ -n "$TRAJECTORY_DIR" && -d "$TRAJECTORY_DIR" ]]; then
  TRAJECTORY_DIR="$(cd "$TRAJECTORY_DIR" && pwd -P)"
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  owner="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | tail -1 | awk '{print $2}')"
  http_pid="$(launchctl list | awk -v label="$HTTP_LABEL" '$3 == label {print $1}')"
  if [[ -z "$http_pid" || "$owner" != "$http_pid" ]]; then
    echo "Port $PORT is already in use by pid $owner. Set PORT=... and rerun." >&2
    exit 1
  fi
fi

SYNC_PLIST="$LAUNCH_DIR/$SYNC_LABEL.plist"
WATCH_PLIST="$LAUNCH_DIR/$WATCH_LABEL.plist"
HTTP_PLIST="$LAUNCH_DIR/$HTTP_LABEL.plist"

write_plist "$SYNC_LABEL" "$SYNC_PLIST" "$ROOT/runs/mini_sync_loop.log" "$ROOT/runs/mini_sync_loop.err" true \
  /bin/bash -lc "cd '$ROOT'; while true; do ./scripts/mini_pull_results.sh; sleep 120; done"

WATCH_ARGS=(
  "$ROOT/.venv/bin/python" "$ROOT/scripts/watch_training.py" \
  --run-dir "$RUN_DIR" \
  --dashboard-dir "$DASH_DIR" \
  --eval-episodes "$EVAL_EPISODES" \
  --eval-period "$EVAL_PERIOD" \
  --device cpu \
  --seed-gif-dir "$SEED_GIF_DIR"
)
if [[ -n "$TRAJECTORY_DIR" ]]; then
  WATCH_ARGS+=(--seed-gif-dir "$TRAJECTORY_DIR")
fi

write_plist "$WATCH_LABEL" "$WATCH_PLIST" "$DASH_DIR/watch.log" "$DASH_DIR/watch.err" true \
  "${WATCH_ARGS[@]}"

write_plist "$HTTP_LABEL" "$HTTP_PLIST" "$DASH_DIR/http.log" "$DASH_DIR/http.err" true \
  "$ROOT/.venv/bin/python" -m http.server "$PORT" --directory "$DASH_DIR"

start_job "$SYNC_LABEL" "$SYNC_PLIST"
start_job "$WATCH_LABEL" "$WATCH_PLIST"
start_job "$HTTP_LABEL" "$HTTP_PLIST"

cat <<EOF

Dashboard: http://127.0.0.1:$PORT
Run dir:   $RUN_DIR
Eval:      $EVAL_EPISODES local episodes per checkpoint
Jobs:
  launchctl list | rg 'com.phil.worldmodel.mini'
Logs:
  tail -f $DASH_DIR/watch.log
  tail -f $DASH_DIR/http.log
  tail -F $RUN_DIR/train_log.csv
Stop:
  ./scripts/mini_dashboard.sh stop
EOF
