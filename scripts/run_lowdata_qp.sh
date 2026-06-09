#!/usr/bin/env bash
# Low-data sweep on Quasi-periodic.
# Mirror of run_lowdata_rossler.sh — same 4 models × 5 fractions.
# Settings match the QP P1D-full main run:
#   seq_len=200, hidden=64, aug=4, curriculum "20:1,50:1,85:1,150:2",
#   tvalid_threshold=1.0.  csode_P1D_full row adds the full recipe.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/timmiakov/internal_exps/dynsys-hub/.venv/bin/python
mkdir -p logs

EPOCHS=1000
SEED=0
SYS=quasi_periodic
PREFIX=qp

COMMON=(
  --system "$SYS" --seq_len 200
  --hidden 64 --aug 4
  --epochs "$EPOCHS" --batch 256 --lr 1e-3
  --seed "$SEED"
  --tvalid_every 50 --log_every 50
  --tvalid_threshold 1.0
  --curriculum
  --curriculum_schedule "20:1,50:1,85:1,150:2"
)

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

spawn() {
  local model=$1 frac=$2 recipe=$3; shift 3
  local fname tag logf
  fname=$(printf 'd%03d' "$(echo "$frac*100/1" | bc)")
  tag="${PREFIX}_${recipe}_lowdata_${fname}"
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

echo "===== QP low-data sweep — $(date) ====="
for m in "${MODELS[@]}"; do
  for f in "${FRACS[@]}"; do
    spawn "$m" "$f" "$m"
  done
done
for f in "${FRACS[@]}"; do
  spawn "csode" "$f" "csode_P1D_full" "${FULL_TRICKS[@]}"
done

echo "Launched ${#BG_PIDS[@]} jobs."
printf '%s\n' "${BG_PIDS[@]}" > /tmp/lowdata_qp_pids.txt
printf '%s\n' "${TAGS[@]}"    > /tmp/lowdata_qp_tags.txt
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

(
  echo "=== waiting for ${#BG_PIDS[@]} jobs ($(date)) ==="
  for pid in "${BG_PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    echo "  PID $pid done at $(date)"
  done
  echo "=== validating ==="
  for tag in "${TAGS[@]}"; do
    recipe="${tag#${PREFIX}_}"; recipe="${recipe%%_lowdata_*}"
    case "$recipe" in
      csode_P1D_full) mdl="csode" ;;
      *)              mdl="$recipe" ;;
    esac
    ckpt="checkpoints/${tag}/${SYS}_${mdl}.pt"
    if [[ -f "$ckpt" ]]; then
      CUDA_VISIBLE_DEVICES=0 "$PY" validate.py --ckpt "$ckpt" \
        > "logs/_lowdata_validate_${tag}.out" 2>&1 \
        && echo "OK $tag" || echo "FAIL $tag"
    else
      echo "MISSING $ckpt"
    fi
  done
  echo "=== ALL DONE at $(date) ==="
) > logs/_lowdata_qp_finalvalidate.out 2>&1 &
FV=$!
echo "Final-validator PID=$FV"
echo "$FV" > /tmp/lowdata_qp_fv_pid.txt
