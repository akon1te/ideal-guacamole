#!/usr/bin/env bash
# Low-data sweep on Rössler.
# Studies how T_valid / W1 / sym_KL / spec_L1 of each model degrades
# when the training prefix is shrunk from f=1.0 down to f=0.05.
#
# Models   : csode (P1B-curr), node, anode, lnode, csode_P1D_full  (5 rows)
# Fractions: 0.05  0.10  0.20  0.50  1.00                          (5 levels)
# => 25 runs.  Baseline 4 rows share the "P1B-curr" recipe (curriculum
# only); csode_P1D_full uses the full E-recipe (curriculum + TF=10 +
# cosine LR + grad-clip + spectral loss) — same recipe as the winning
# main run on Rössler.  This lets us see whether the rich recipe still
# pays off when data is scarce.  Each run: 1000 epochs, hidden=64,
# aug=4, seq_len=300.
#
# Layout: 2 GPUs, alternate fractions to GPU 0 / 1.  Launch all in
# parallel with a small stagger.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/timmiakov/internal_exps/dynsys-hub/.venv/bin/python
mkdir -p logs

EPOCHS=1000
SEED=0

COMMON=(
  --system rossler --seq_len 300
  --hidden 64 --aug 4
  --epochs "$EPOCHS" --batch 256 --lr 1e-3
  --seed "$SEED"
  --tvalid_every 50 --log_every 50
  --tvalid_threshold 1.0
  --curriculum
  --curriculum_schedule "20:1,50:1,85:1,150:2"
)

# Extra flags for the csode_P1D_full row only.
FULL_TRICKS=(
  --tf_chunk 10
  --grad_clip 1.0
  --cosine_lr --warmup_epochs 50
  --lambda_spec 3e-3
)

MODELS=(csode node anode)
FRACS=(0.05 0.10 0.20 0.50 1.00)

BG_PIDS=()
TAGS=()
g=0

# spawn <model_id> <frac> <recipe_tag>  [extra args...]
# model_id is the actual --model name; tag uses recipe_tag for naming.
spawn() {
  local model=$1 frac=$2 recipe=$3; shift 3
  local fname tag logf
  fname=$(printf 'd%03d' "$(echo "$frac*100/1" | bc)")
  tag="rossler_${recipe}_lowdata_${fname}"
  logf="logs/_lowdata_${tag}.out"
  echo "[GPU$g] $tag (model=$model frac=$frac)"
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" train.py \
    --model "$model" \
    --data_frac "$frac" \
    "${COMMON[@]}" \
    "$@" \
    --run_name "$tag" \
    > "$logf" 2>&1 &
  BG_PIDS+=($!)
  TAGS+=("$tag")
  g=$((1 - g))
  sleep 2
}

echo "===== Rössler low-data sweep — $(date) ====="
# 4 baseline rows: P1B-curr recipe
for m in "${MODELS[@]}"; do
  for f in "${FRACS[@]}"; do
    spawn "$m" "$f" "$m"
  done
done
# 5th row: csode with the full P1D recipe
for f in "${FRACS[@]}"; do
  spawn "csode" "$f" "csode_P1D_full" "${FULL_TRICKS[@]}"
done

echo "Launched ${#BG_PIDS[@]} jobs."
printf '%s\n' "${BG_PIDS[@]}" > /tmp/lowdata_rossler_pids.txt
printf '%s\n' "${TAGS[@]}"    > /tmp/lowdata_rossler_tags.txt
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

# Final validator: wait, then run validate.py on each.
(
  echo "=== waiting for ${#BG_PIDS[@]} jobs ($(date)) ==="
  for pid in "${BG_PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    echo "  PID $pid done at $(date)"
  done
  echo "=== validating ==="
  for tag in "${TAGS[@]}"; do
    recipe="${tag#rossler_}"; recipe="${recipe%%_lowdata_*}"
    case "$recipe" in
      csode_P1D_full) mdl="csode" ;;
      *)              mdl="$recipe" ;;
    esac
    ckpt="checkpoints/${tag}/rossler_${mdl}.pt"
    if [[ -f "$ckpt" ]]; then
      CUDA_VISIBLE_DEVICES=0 "$PY" validate.py --ckpt "$ckpt" \
        > "logs/_lowdata_validate_${tag}.out" 2>&1 \
        && echo "OK $tag" || echo "FAIL $tag"
    else
      echo "MISSING $ckpt"
    fi
  done
  echo "=== ALL DONE at $(date) ==="
) > logs/_lowdata_rossler_finalvalidate.out 2>&1 &
FV=$!
echo "Final-validator PID=$FV"
echo "$FV" > /tmp/lowdata_rossler_fv_pid.txt
