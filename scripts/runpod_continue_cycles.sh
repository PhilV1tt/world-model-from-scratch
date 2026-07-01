#!/usr/bin/env bash
# Continue H100 training after the current run finishes.
set -euo pipefail

cd "$(dirname "$0")/.."

CURRENT_RUN=${CURRENT_RUN:-"runs/experiments/h100_v3_16m_b1536_e5_20260527_133609"}
CURRENT_PID=${CURRENT_PID:-"132957"}
DATA=${DATA:-"data/parking/train_v3_mega_50000.h5"}
STAMP=${STAMP:-"$(date +%Y%m%d_%H%M%S)"}
CYCLES=${CYCLES:-3}
EPOCHS_PER_CYCLE=${EPOCHS_PER_CYCLE:-2}
BATCH=${BATCH:-1536}
NUM_WORKERS=${NUM_WORKERS:-12}
SEQ_LEN=${SEQ_LEN:-6}
FRAME_SKIP=${FRAME_SKIP:-1}
ENC_DEPTH=${ENC_DEPTH:-12}
PRED_DEPTH=${PRED_DEPTH:-6}
SIGREG_N_PROJ=${SIGREG_N_PROJ:-256}
AUTOREGRESSIVE_WEIGHT=${AUTOREGRESSIVE_WEIGHT:-0.25}
AUTOREGRESSIVE_RAMP_STEPS=${AUTOREGRESSIVE_RAMP_STEPS:-1000}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-100}
VAL_EVERY_STEPS=${VAL_EVERY_STEPS:-200}
VAL_BATCHES=${VAL_BATCHES:-8}
SEED=${SEED:-42}
EVAL_EPISODES=${EVAL_EPISODES:-8}
EVAL_PLANNERS=${EVAL_PLANNERS:-"pd,model_pd"}
CYCLE_LRS=${CYCLE_LRS:-"8e-5,5e-5,3e-5"}

IFS=',' read -r -a LR_LIST <<< "$CYCLE_LRS"

log_status() {
  local run="$1"
  if [[ -f "$run/train_log.csv" ]]; then
    ./.venv/bin/python - "$run/train_log.csv" <<'PY' || true
import csv, sys
path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
if not rows:
    raise SystemExit
last = rows[-1]
keys = ["step", "epoch", "loss", "loss_ema", "val_loss", "lr", "steps_per_sec"]
print("status " + " ".join(f"{k}={last.get(k, '')}" for k in keys))
PY
  fi
}

echo "RunPod continuation supervisor"
echo "started_at=$(date -Is)"
echo "current_run=$CURRENT_RUN"
echo "current_pid=$CURRENT_PID"
echo "cycles=$CYCLES epochs_per_cycle=$EPOCHS_PER_CYCLE batch=$BATCH lrs=$CYCLE_LRS"

if [[ -n "$CURRENT_PID" ]] && kill -0 "$CURRENT_PID" 2>/dev/null; then
  echo "waiting for current PID $CURRENT_PID"
  while kill -0 "$CURRENT_PID" 2>/dev/null; do
    log_status "$CURRENT_RUN"
    sleep 60
  done
  echo "current PID finished at $(date -Is)"
else
  echo "current PID is not running; continuing from latest checkpoint if present"
fi

if [[ ! -f "$CURRENT_RUN/ckpt_last.pt" ]]; then
  echo "missing checkpoint: $CURRENT_RUN/ckpt_last.pt" >&2
  exit 1
fi

prev_run="$CURRENT_RUN"
for idx in $(seq 1 "$CYCLES"); do
  lr_index=$((idx - 1))
  if (( lr_index >= ${#LR_LIST[@]} )); then
    lr_index=$((${#LR_LIST[@]} - 1))
  fi
  lr="${LR_LIST[$lr_index]}"
  out="runs/experiments/h100_v3_cycle$(printf '%02d' "$idx")_16m_b${BATCH}_e${EPOCHS_PER_CYCLE}_${STAMP}"
  resume="$prev_run/ckpt_last.pt"

  echo
  echo "cycle=$idx started_at=$(date -Is)"
  echo "resume=$resume"
  echo "out=$out"
  echo "lr=$lr"

  ./.venv/bin/python scripts/train.py \
    --data "$DATA" \
    --out "$out" \
    --resume "$resume" \
    --resume-schedule extend \
    --allow-lr-jump \
    --device cuda \
    --bf16 \
    --epochs "$EPOCHS_PER_CYCLE" \
    --batch "$BATCH" \
    --lr "$lr" \
    --num-workers "$NUM_WORKERS" \
    --seq-len "$SEQ_LEN" \
    --frame-skip "$FRAME_SKIP" \
    --enc-depth "$ENC_DEPTH" \
    --pred-depth "$PRED_DEPTH" \
    --sigreg-n-proj "$SIGREG_N_PROJ" \
    --save-every-steps "$SAVE_EVERY_STEPS" \
    --val-every-steps "$VAL_EVERY_STEPS" \
    --val-batches "$VAL_BATCHES" \
    --val-bn-stats batch \
    --autoregressive-weight "$AUTOREGRESSIVE_WEIGHT" \
    --autoregressive-ramp-steps "$AUTOREGRESSIVE_RAMP_STEPS" \
    --seed "$SEED"

  echo "cycle=$idx train_done_at=$(date -Is)"
  log_status "$out"

  echo "cycle=$idx eval_started_at=$(date -Is)"
  ./.venv/bin/python scripts/eval_protocol.py \
    --ckpt "$out/ckpt_last.pt" \
    --out-dir "$out/eval_quick" \
    --planners "$EVAL_PLANNERS" \
    --episodes "$EVAL_EPISODES" \
    --seed 4242 \
    --device cuda \
    --horizon 5 \
    --mpc-apply 2 \
    --max-env-steps 80 \
    --pop 128 \
    --elites 16 \
    --iters 4 || echo "cycle=$idx eval_failed_but_training_keeps_going"

  prev_run="$out"
done

echo "all_cycles_done_at=$(date -Is)"
