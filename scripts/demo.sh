#!/usr/bin/env bash
# Pipeline visuel: preview anime -> baselines -> training (~40min) -> CEM planning -> GIFs.
set -e
cd "$(dirname "$0")/.."

C='\033[0;36m'; Y='\033[1;33m'; G='\033[0;32m'; M='\033[0;35m'; R='\033[0;31m'; B='\033[1m'; N='\033[0m'

banner() {
  echo
  echo -e "${C}${B}╔══════════════════════════════════════════════════════════════╗${N}"
  printf  "${C}${B}║  %-60s║${N}\n" "$1"
  echo -e "${C}${B}╚══════════════════════════════════════════════════════════════╝${N}"
  echo
}

# ── Perf maxee ─────────────────────────────────────────────────────────
NCPU=$(sysctl -n hw.ncpu)
NPCORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 4)
NECORES=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 0)
RAM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
CHIP=$(sysctl -n machdep.cpu.brand_string)

export OMP_NUM_THREADS=$NCPU
export MKL_NUM_THREADS=$NCPU
export VECLIB_MAXIMUM_THREADS=$NCPU
export NUMEXPR_NUM_THREADS=$NCPU
export TOKENIZERS_PARALLELISM=true
export PYTORCH_ENABLE_MPS_FALLBACK=1
export WM_TORCH_THREADS=$NCPU
NUM_WORKERS=$NPCORES
[ "$NUM_WORKERS" -lt 4 ] && NUM_WORKERS=4

T0=$(date +%s)

banner "LeWM — World Model from pixels (parking-v0)"
echo -e "${M}Hardware:${N}    $CHIP  |  ${NCPU} cores (${NPCORES}P + ${NECORES}E)  |  ${RAM_GB} GB RAM"
echo -e "${M}Threads:${N}     OMP/MKL/torch=$NCPU  |  DataLoader workers=$NUM_WORKERS (persistent)"
echo -e "${M}Device:${N}      $(./.venv/bin/python -c 'import torch; print("MPS — Apple Silicon GPU" if torch.backends.mps.is_available() else "CPU")')"
echo -e "${M}Model:${N}       ViT enc(4) + Transformer pred(2) + sigreg | bf16 autocast"
echo -e "${M}Dataset:${N}     data/parking/train.h5 ($(/bin/ls -lh data/parking/train.h5 | awk '{print $5}'))"

banner "[0/3] Live viewer — fenetre pygame en parallele"
echo -e "${Y}PD heuristic d'abord, puis bascule sur LeWM+CEM des le 1er ckpt${N}"
./.venv/bin/python scripts/live_viewer.py &
VIEWER_PID=$!
trap 'kill $VIEWER_PID 2>/dev/null || true' EXIT
sleep 2

banner "[1/3] Baselines sans modele — random + PD heuristic"
./.venv/bin/python scripts/eval.py --episodes 30 --seed 42 --max-steps 60 --out runs/last/baselines.npz

banner "[2/3] Training LeWM — 2 epochs, ETA en live (~40min)"
./.venv/bin/python scripts/train.py \
  --data data/parking/train.h5 \
  --epochs 2 \
  --batch 128 --num-workers $NUM_WORKERS --bf16 \
  --enc-depth 4 --pred-depth 2 \
  --out runs/last

banner "[3/3] Latent CEM planning + MPC sur 20 episodes"
./.venv/bin/python scripts/plan.py \
  --ckpt runs/last/ckpt_last.pt \
  --episodes 20 --horizon 5 --pop 200 --elites 20 --iters 8 \
  --mpc-apply 5 --max-env-steps 60 --seed 42

T1=$(date +%s); ELAPSED=$((T1 - T0)); H=$((ELAPSED/3600)); MIN=$(((ELAPSED%3600)/60)); SEC=$((ELAPSED%60))

banner "DONE en ${H}h${MIN}m${SEC}s"
echo -e "${Y}GIFs (modele entraine):${N}  runs/last/plan_results/gifs/"
echo -e "${Y}Results:${N}                runs/last/plan_results/results.npz"
echo -e "${Y}Logs:${N}                   runs/last/train_log.csv"
echo

# Ouvre les GIFs OK directement dans Preview (en plus du Finder)
for f in runs/last/plan_results/gifs/*_ok.gif; do
  [ -f "$f" ] && open "$f" 2>/dev/null || true
done
open runs/last/plan_results/gifs/ 2>/dev/null || true
echo -e "${G}${B}OK.${N}  (la fenetre live viewer continue de tourner — Cmd+W pour fermer)"
trap - EXIT
wait $VIEWER_PID 2>/dev/null || true
