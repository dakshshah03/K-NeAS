#!/usr/bin/env bash
set -euo pipefail

# Run evaluation for ablations 1-6 for a given anatomy (uses checkpoint_epoch_1000)
#
# Usage: ./eval_ablations.sh [--gpu DEVICE] <anatomy>
# Example: ./eval_ablations.sh --gpu cuda:0 chest

GPU_DEVICE="cuda"

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu|-g)
      GPU_DEVICE="$2"
      shift 2
      ;;
    -*)
      echo "Unknown option: $1"
      echo "Usage: $0 [--gpu DEVICE] <anatomy>"
      exit 1
      ;;
    *)
      if [[ -z "${ANATOMY:-}" ]]; then
        ANATOMY="$1"
        shift
      else
        echo "Too many arguments. Usage: $0 [--gpu DEVICE] <anatomy>"
        exit 1
      fi
      ;;
  esac
done

if [[ -z "${ANATOMY:-}" ]]; then
  echo "Error: missing anatomy argument."
  echo "Usage: $0 [--gpu DEVICE] <anatomy>"
  echo "Valid anatomies: foot"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_CMD="python"
EPOCH=1000

if [[ "$ANATOMY" == "foot" ]]; then
  ABLATIONS=(
    "1_KSelector"
    "2_KSelector_Floater"
    "3_KSelector_Shared"
    "4_KNEAS"
  )
fi

OUTPUT_CSV="${OUTPUT_CSV:-$ROOT_DIR/eval_summary_${ANATOMY}_ablations.csv}"
CSV_MODE="overwrite"

if [[ "$CSV_MODE" == "overwrite" || ! -f "$OUTPUT_CSV" ]]; then
  printf "%s\n" "ablation,anatomy,epoch,avg_proj_psnr,avg_proj_ssim,vol_psnr,vol_ssim" > "$OUTPUT_CSV"
fi

for ablation in "${ABLATIONS[@]}"; do
  EXP_NAME="${ANATOMY}_${ablation}"
  CKPT_PATH="$ROOT_DIR/checkpoints/${ANATOMY}/${EXP_NAME}/checkpoint_epoch_${EPOCH}.pth"
  VAL_PICKLE="$ROOT_DIR/data/${ANATOMY}_50.pickle"

  echo "------------------------------------------------------------"
  echo "Running: anatomy=${ANATOMY}, ablation=${ablation}, epoch=${EPOCH}"
  echo "Checkpoint: ${CKPT_PATH}"
  echo "Val pickle: ${VAL_PICKLE}"

  if [[ ! -f "$CKPT_PATH" ]]; then
    echo "⚠️  Checkpoint not found — skipping: $CKPT_PATH" >&2
    continue
  fi
  if [[ ! -f "$VAL_PICKLE" ]]; then
    echo "⚠️  Validation pickle not found — skipping: $VAL_PICKLE" >&2
    continue
  fi

  TMP_CSV="$ROOT_DIR/.tmp_eval_${ANATOMY}_${ablation}_epoch${EPOCH}.csv"

  CKPT_DIR="$(dirname "$CKPT_PATH")"
  if [[ -n "${EVAL_ROOT:-}" ]]; then
      TARGET_EVAL_ROOT="$EVAL_ROOT"
  else
      TARGET_EVAL_ROOT="$CKPT_DIR/eval"
  fi
  mkdir -p "$TARGET_EVAL_ROOT"

  "$PY_CMD" src/eval_validation_metrics.py \
    --checkpoint "$CKPT_PATH" \
    --val_pickle "$VAL_PICKLE" \
    --device "$GPU_DEVICE" \
    --save_csv "$TMP_CSV" \
    --eval_id "${EXP_NAME}" \
    --eval_root "$TARGET_EVAL_ROOT"

  if [[ -f "$TMP_CSV" ]]; then
    read -r AVG_PROJ_PSNR AVG_PROJ_SSIM < <(awk -F, '$1=="avg"{gsub(/^[ \t]+|[ \t]+$/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $3); print $2, $3; exit}' "$TMP_CSV")
    read -r VOL_PSNR VOL_SSIM       < <(awk -F, '$1=="vol"{gsub(/^[ \t]+|[ \t]+$/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $3); print $2, $3; exit}' "$TMP_CSV")
    rm -f "$TMP_CSV"
  else
    echo "⚠️  Expected per-run CSV not found: $TMP_CSV" >&2
    continue
  fi

  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "$ablation" "$ANATOMY" "$EPOCH" "$AVG_PROJ_PSNR" "$AVG_PROJ_SSIM" "$VOL_PSNR" "$VOL_SSIM" >> "$OUTPUT_CSV"

  echo "✅ Metrics appended to $OUTPUT_CSV"
  echo
done

echo "✅ All done. (used checkpoint_epoch_${EPOCH} for all runs)"
echo "Summary CSV: $OUTPUT_CSV"
