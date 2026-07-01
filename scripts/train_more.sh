#!/usr/bin/env bash
# Continue training depuis runs/last/ckpt_last.pt pour N epochs supplementaires.
# Usage:
#   ./scripts/train_more.sh           # +5 epochs (default)
#   ./scripts/train_more.sh 10        # +10 epochs
set -e
cd "$(dirname "$0")/.."
N=${1:-5}
SAVE_EVERY=${SAVE_EVERY_STEPS:-50}
CKPT="runs/last/ckpt_last.pt"
STAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR=${OUT_DIR:-"runs/experiments/current_fixed_${STAMP}"}
if [ ! -f "$CKPT" ]; then
  echo "Pas de checkpoint a $CKPT - lance d'abord ./scripts/train_first.sh"
  exit 1
fi
# Lit l'archi du ckpt courant pour rester compatible (sinon mismatch state_dict)
ENC_DEPTH=$(./.venv/bin/python -c "import torch; c=torch.load('$CKPT', map_location='cpu', weights_only=False); print(c['cfg']['enc_depth'])")
PRED_DEPTH=$(./.venv/bin/python -c "import torch; c=torch.load('$CKPT', map_location='cpu', weights_only=False); print(c['cfg']['pred_depth'])")
echo "Resuming from $CKPT for +$N epochs (enc=$ENC_DEPTH pred=$PRED_DEPTH, save_every=${SAVE_EVERY} steps, out=$OUT_DIR)"

NCPU=$(sysctl -n hw.ncpu)
NPCORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 4)
export OMP_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU VECLIB_MAXIMUM_THREADS=$NCPU
export NUMEXPR_NUM_THREADS=$NCPU PYTORCH_ENABLE_MPS_FALLBACK=1 WM_TORCH_THREADS=$NCPU

./.venv/bin/python scripts/train.py \
  --resume "$CKPT" \
  --resume-schedule extend \
  --epochs "$N" \
  --batch 128 --num-workers $NPCORES --bf16 \
  --enc-depth $ENC_DEPTH --pred-depth $PRED_DEPTH \
  --save-every-steps "$SAVE_EVERY" \
  --out "$OUT_DIR"
