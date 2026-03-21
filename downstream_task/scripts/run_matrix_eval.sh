#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET_ROOT="/datasets/usa_static_v1_large_run/downstream_test_16_20_23"
REGRESSOR_CKPT="logs/regressor_20mm_best.pt"
USA_CKPT="usa_train_runs/usa_adapter_full_5scales_ep100_boundarymix_zero_shot_laptop/best_loss.pt"
SCALES=(16 20 23)
SPLITS=("seen" "unseen")

for scale in "${SCALES[@]}"; do
  for split in "${SPLITS[@]}"; do
    echo "============================================================"
    echo "Running eval WITHOUT USA | scale=${scale}mm | split=${split}"
    echo "============================================================"
    python "${PROJECT_ROOT}/downstream_task/eval_matrix.py" \
      --dataset-root "${DATASET_ROOT}" \
      --test_scale "${scale}" \
      --indenter_split "${split}" \
      --use_usa False \
      --regressor_ckpt "${PROJECT_ROOT}/${REGRESSOR_CKPT}"

    echo "============================================================"
    echo "Running eval WITH USA | scale=${scale}mm | split=${split}"
    echo "============================================================"
    python "${PROJECT_ROOT}/downstream_task/eval_matrix.py" \
      --dataset-root "${DATASET_ROOT}" \
      --test_scale "${scale}" \
      --indenter_split "${split}" \
      --use_usa True \
      --usa_ckpt "${PROJECT_ROOT}/${USA_CKPT}" \
      --regressor_ckpt "${PROJECT_ROOT}/${REGRESSOR_CKPT}"
  done
done
