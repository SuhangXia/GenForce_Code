#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_ROOT="${DATASET_ROOT:-/datasets/usa_static_v1_large_run/downstream_test_16_20_23}"
if [ ! -d "$DATASET_ROOT" ] && [ -d "/home/suhang/datasets/usa_static_v1_large_run/downstream_test_16_20_23" ]; then
  DATASET_ROOT="/home/suhang/datasets/usa_static_v1_large_run/downstream_test_16_20_23"
fi

REGRESSOR_CKPT="${REGRESSOR_CKPT:-/datasets/checkpoints/standalone_stea_experiments/canonical20_with_stea/best.pt}"
if [ ! -f "$REGRESSOR_CKPT" ] && [ -f "/home/suhang/datasets/checkpoints/standalone_stea_experiments/canonical20_with_stea/best.pt" ]; then
  REGRESSOR_CKPT="/home/suhang/datasets/checkpoints/standalone_stea_experiments/canonical20_with_stea/best.pt"
fi

STEA_CKPT="${STEA_CKPT:-/datasets/checkpoints/standalone_stea_adapter_fullpairs_bs320/best.pt}"
if [ ! -f "$STEA_CKPT" ] && [ -f "/home/suhang/datasets/checkpoints/standalone_stea_adapter_fullpairs_bs320/best.pt" ]; then
  STEA_CKPT="/home/suhang/datasets/checkpoints/standalone_stea_adapter_fullpairs_bs320/best.pt"
fi

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-4}"
RESULTS_DIR="${RESULTS_DIR:-/datasets/checkpoints/standalone_stea_experiments/results}"
if [ ! -d "$(dirname "$RESULTS_DIR")" ] && [ -d "/home/suhang/datasets/checkpoints/standalone_stea_experiments" ]; then
  RESULTS_DIR="/home/suhang/datasets/checkpoints/standalone_stea_experiments/results"
fi
mkdir -p "$RESULTS_DIR"
LOG_PATH="$RESULTS_DIR/canonical20_with_stea_matrix.txt"
: > "$LOG_PATH"

run_eval() {
  local scale_mm="$1"
  local split="$2"
  "$PYTHON_BIN" "$PROJECT_ROOT/standalone_stea_experiments/eval_matrix_pipeline.py" \
    --dataset-root "$DATASET_ROOT" \
    --regressor-ckpt "$REGRESSOR_CKPT" \
    --pipeline-kind with_stea \
    --stea-ckpt "$STEA_CKPT" \
    --scale-mm "$scale_mm" \
    --split "$split" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --device "$DEVICE" | tee -a "$LOG_PATH"
}

for scale_mm in 16 20 23; do
  for split in seen unseen; do
    run_eval "$scale_mm" "$split"
  done
done

echo "Saved matrix log to $LOG_PATH"

