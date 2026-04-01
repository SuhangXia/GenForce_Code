#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET_ROOT="/datasets/usa_static_v1_large_run/downstream_test_16_20_23"
REGRESSOR_CKPT="../datasets/checkpoints/downstream_regressor_20mm_with_ddusa/best.pt"
USA_CKPT="../datasets/checkpoints/dd_usa_pretrain_v1_p24000_bs12/best.pt"
SCALES=(16 20 23)
SPLITS=("seen" "unseen")

for scale in "${SCALES[@]}"; do
  for split in "${SPLITS[@]}"; do
    echo "============================================================"
    echo "Running eval WITHOUT ADAPTER | scale=${scale}mm | split=${split}"
    echo "============================================================"
    python "${PROJECT_ROOT}/downstream_task/eval_matrix.py" \
      --dataset-root "${DATASET_ROOT}" \
      --test_scale "${scale}" \
      --indenter_split "${split}" \
      --use-adapter false \
      --regressor_ckpt "${PROJECT_ROOT}/${REGRESSOR_CKPT}"

    echo "============================================================"
    echo "Running eval WITH ADAPTER | scale=${scale}mm | split=${split}"
    echo "============================================================"
    python "${PROJECT_ROOT}/downstream_task/eval_matrix.py" \
      --dataset-root "${DATASET_ROOT}" \
      --test_scale "${scale}" \
      --indenter_split "${split}" \
      --use-adapter true \
      --adapter-ckpt "${PROJECT_ROOT}/${USA_CKPT}" \
      --regressor_ckpt "${PROJECT_ROOT}/${REGRESSOR_CKPT}"
  done
done
