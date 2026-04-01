#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/datasets/usa_static_v1_large_run/downstream_test_16_20_23}"
REGRESSOR_CKPT="${REGRESSOR_CKPT:-/datasets/checkpoints/downstream_regressor_20mm_with_qformer/best.pt}"
QFORMER_CKPT="${QFORMER_CKPT:-/datasets/checkpoints/standalone_qformer_distill/best.pt}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"

scales=(16 20 23)
splits=(seen unseen)

for scale in "${scales[@]}"; do
  for split in "${splits[@]}"; do
    echo "============================================================"
    echo "Running eval WITHOUT Q-FORMER | scale=${scale}mm | split=${split}"
    echo "============================================================"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/standalone_qformer/eval_matrix_qformer.py"       --dataset-root "${DATASET_ROOT}"       --test-scale "${scale}"       --indenter-split "${split}"       --use-qformer false       --regressor-ckpt "${REGRESSOR_CKPT}"       --batch-size "${BATCH_SIZE}"       --workers "${WORKERS}"       --device "${DEVICE}"

    echo "============================================================"
    echo "Running eval WITH Q-FORMER | scale=${scale}mm | split=${split}"
    echo "============================================================"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/standalone_qformer/eval_matrix_qformer.py"       --dataset-root "${DATASET_ROOT}"       --test-scale "${scale}"       --indenter-split "${split}"       --use-qformer true       --qformer-ckpt "${QFORMER_CKPT}"       --regressor-ckpt "${REGRESSOR_CKPT}"       --batch-size "${BATCH_SIZE}"       --workers "${WORKERS}"       --device "${DEVICE}"
  done
done
