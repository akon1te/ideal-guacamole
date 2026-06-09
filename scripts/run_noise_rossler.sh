#!/usr/bin/env bash
# Noise sweep on Rössler — counterpart to run_lowdata_rossler.sh.
# Studies how T_valid / W1 / sym_KL / spec_L1 of each model degrades
# when measurement noise σ is added to the training trajectory.
# Validation trajectory is ALWAYS clean (handled inside train.py).
#
# Models   : csode (P1B-curr), node, anode, csode_P1D_full          (4 rows)
# Sigmas   : 0.01  0.03  0.05  0.10  0.20                            (5 levels)
# => 20 runs.  Baseline 3 rows use the modest "P1B-curr" recipe
# (curriculum only); csode_P1D_full uses the full E-recipe — same
# split as in run_lowdata_rossler.sh, so the two studies are
# directly comparable.
#
# By default WAITS until the low-data sweep PIDs in
# /tmp/lowdata_rossler_pids.txt are done; override with WAIT=0.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/timmiakov/internal_exps/dynsys-hub/.venv/bin/python
mkdir -p logs

EPOCHS=1000
SEED=0
WAIT=${WAIT:-1}

if [[ "$WAIT" == "1" && -f /tmp/lowdata_rossler_pids.txt ]]; then
  echo "=== Waiting for low-data sweep PIDs to finish (WAIT=0 to skip) ==="
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    echo "  PID $pid done at $(date)"
  done < /tmp/lowdata_rossler_pids.txt
  # Also wait for the final-validator of the lowdata sweep
  if [[ -f /tmp/lowdata_rossler_fv_pid.txt ]]; then
    fv=$(cat /tmp/lowdata_rossler_fv_pid.txt)
    while kill -0 "$fv" 2>/dev/null; do sleep 30; done
    echo "  low-data FV $fv done at $(date)"
  fi
  echo "=== Low-data sweep finished, proceeding with noise sweep at $(date) ==="
fi

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

FULL_TRICKS=(
  --tf_chunk 10
  --grad_clip 1.0
  --cosine_lr --warmup_epochs 50
  --lambda_spec 3e-3
)

MODELS=(csode node anode)
SIGMAS=(0.01 0.03 0.05 0.10 0.20)

BG_PIDS=()
TAGS=()
g=0

# spawn <model_id> <sigma> <recipe_tag>  [extra args...]
spawn() {
  local model=$1 sigma=$2 recipe=$3; shift 3
  # sigma=0.05 -> tag "s005", 0.10 -> "s010" etc.  Use %02d on
  # sigma*100 to keep ordering stable.
  local sname tag logf
  sname=$(printf 's%03d' "$(echo "$sigma*100/1" | bc)")
  tag="rossler_${recipe}_noise_${sname}"
  logf="logs/_noise_${tag}.out"
  echo "[GPU$g] $tag (model=$model sigma=$sigma)"
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" train.py \
    --model "$model" \
    --data_noise "$sigma" \
    "${COMMON[@]}" \
    "$@" \
    --run_name "$tag" \
    > "$logf" 2>&1 &
  BG_PIDS+=($!)
  TAGS+=("$tag")
  g=$((1 - g))
  sleep 2
}

echo "===== Rössler noise sweep — $(date) ====="
# 3 baseline rows: P1B-curr recipe
for m in "${MODELS[@]}"; do
  for s in "${SIGMAS[@]}"; do
    spawn "$m" "$s" "$m"
  done
done
# 4th row: csode with the full P1D recipe
for s in "${SIGMAS[@]}"; do
  spawn "csode" "$s" "csode_P1D_full" "${FULL_TRICKS[@]}"
done

echo "Launched ${#BG_PIDS[@]} jobs."
printf '%s\n' "${BG_PIDS[@]}" > /tmp/noise_rossler_pids.txt
printf '%s\n' "${TAGS[@]}"    > /tmp/noise_rossler_tags.txt
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

# Final validator
(
  echo "=== waiting for ${#BG_PIDS[@]} jobs ($(date)) ==="
  for pid in "${BG_PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    echo "  PID $pid done at $(date)"
  done
  echo "=== validating ==="
  for tag in "${TAGS[@]}"; do
    recipe="${tag#rossler_}"; recipe="${recipe%%_noise_*}"
    case "$recipe" in
      csode_P1D_full) mdl="csode" ;;
      *)              mdl="$recipe" ;;
    esac
    ckpt="checkpoints/${tag}/rossler_${mdl}.pt"
    if [[ -f "$ckpt" ]]; then
      CUDA_VISIBLE_DEVICES=0 "$PY" validate.py --ckpt "$ckpt" \
        > "logs/_noise_validate_${tag}.out" 2>&1 \
        && echo "OK $tag" || echo "FAIL $tag"
    else
      echo "MISSING $ckpt"
    fi
  done
  echo "=== ALL DONE at $(date) ==="
) > logs/_noise_rossler_finalvalidate.out 2>&1 &
FV=$!
echo "Final-validator PID=$FV"
echo "$FV" > /tmp/noise_rossler_fv_pid.txt
