#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_ROOT="${DATASET_ROOT:-/home/suhang/datasets/usa_static_v1_large_run/downstream_test_16_20_23}"
if [ ! -d "$DATASET_ROOT" ] && [ -d "/datasets/usa_static_v1_large_run/downstream_test_16_20_23" ]; then
  DATASET_ROOT="/datasets/usa_static_v1_large_run/downstream_test_16_20_23"
fi

REGRESSOR_CKPT="${REGRESSOR_CKPT:-/home/suhang/datasets/checkpoints/standalone_stea_regressor/best.pt}"
if [ ! -f "$REGRESSOR_CKPT" ] && [ -f "/datasets/checkpoints/standalone_stea_regressor/best.pt" ]; then
  REGRESSOR_CKPT="/datasets/checkpoints/standalone_stea_regressor/best.pt"
fi

STEA_CKPT="${STEA_CKPT:-/home/suhang/datasets/checkpoints/standalone_stea_adapter/best.pt}"
if [ ! -f "$STEA_CKPT" ] && [ -f "/datasets/checkpoints/standalone_stea_adapter/best.pt" ]; then
  STEA_CKPT="/datasets/checkpoints/standalone_stea_adapter/best.pt"
fi

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-4}"

run_eval() {
  local scale_mm="$1"
  local split="$2"
  local use_stea="$3"
  if [ "$use_stea" = "true" ]; then
    "$PYTHON_BIN" "$PROJECT_ROOT/standalone_stea/eval_regressor_with_stea.py"       --dataset-root "$DATASET_ROOT"       --regressor-ckpt "$REGRESSOR_CKPT"       --stea-ckpt "$STEA_CKPT"       --use-stea true       --scale-mm "$scale_mm"       --split "$split"       --batch-size "$BATCH_SIZE"       --workers "$WORKERS"       --device "$DEVICE"
  else
    "$PYTHON_BIN" "$PROJECT_ROOT/standalone_stea/eval_regressor_with_stea.py"       --dataset-root "$DATASET_ROOT"       --regressor-ckpt "$REGRESSOR_CKPT"       --use-stea false       --scale-mm "$scale_mm"       --split "$split"       --batch-size "$BATCH_SIZE"       --workers "$WORKERS"       --device "$DEVICE"
  fi
}

for scale_mm in 16 20 23; do
  for split in seen unseen; do
    run_eval "$scale_mm" "$split" false
    run_eval "$scale_mm" "$split" true
  done
done
