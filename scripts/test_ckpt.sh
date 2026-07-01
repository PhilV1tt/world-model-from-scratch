#!/usr/bin/env bash
# Test rapide d'un checkpoint (sans embêter le training en parallèle).
# Usage:
#   ./scripts/test_ckpt.sh                      # teste runs/last/ckpt_last.pt
#   ./scripts/test_ckpt.sh ckpt_epoch004.pt     # teste un epoch précis
#   ./scripts/test_ckpt.sh /chemin/absolu.pt    # teste n'importe quel chemin
set -e
cd "$(dirname "$0")/.."

CKPT_ARG="${1:-runs/last/ckpt_last.pt}"
# Si juste "ckpt_epochXXX.pt", préfixe avec runs/last/
case "$CKPT_ARG" in
  /*) CKPT="$CKPT_ARG" ;;
  ckpt_*) CKPT="runs/last/$CKPT_ARG" ;;
  *) CKPT="$CKPT_ARG" ;;
esac

if [ ! -f "$CKPT" ]; then
  echo "Pas trouvé: $CKPT"
  echo "Disponibles dans runs/last/:"
  ls runs/last/ckpt_*.pt 2>/dev/null
  exit 1
fi

# Copie pour éviter conflit avec training (qui écrase ckpt_last.pt)
TMP="/tmp/test_$(basename "$CKPT")"
cp "$CKPT" "$TMP"
echo "Testing $CKPT (copy at $TMP), CPU mode pour pas saturer le GPU"

./.venv/bin/python scripts/plan.py \
  --ckpt "$TMP" \
  --device cpu \
  --episodes 5 \
  --horizon 5 \
  --pop 150 \
  --elites 15 \
  --iters 8 \
  --mpc-apply 3 \
  --max-env-steps 80 \
  --seed 42
