#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-/datasets/usa_static_v1_large_run/downstream_test_16_20_23}"
STEA_CKPT="${STEA_CKPT:-/datasets/checkpoints/standalone_stea_adapter_fullpairs_bs320/best.pt}"
DEVICE="${DEVICE:-cuda}"
RUN_TRAINING="${RUN_TRAINING:-0}"
RUN_MATRICES="${RUN_MATRICES:-0}"

TRAIN_A_CMD="cd /workspace && python standalone_stea_experiments/train_regressor_no_adapter.py --dataset-root ${DATASET_ROOT} --train-scales 16 20 23 --checkpoint-dir /datasets/checkpoints/standalone_stea_experiments/no_adapter_baseline --epochs 50 --batch-size 128 --lr 1e-4 --weight-decay 1e-2 --workers 4 --device ${DEVICE} --save-every 5"
TRAIN_B_CMD="cd /workspace && python standalone_stea_experiments/train_regressor_canonical20_no_adapter.py --dataset-root ${DATASET_ROOT} --train-scales 20 --checkpoint-dir /datasets/checkpoints/standalone_stea_experiments/canonical20_no_adapter --epochs 50 --batch-size 128 --lr 1e-4 --weight-decay 1e-2 --workers 4 --device ${DEVICE} --save-every 5"
TRAIN_C_CMD="cd /workspace && python standalone_stea_experiments/train_regressor_canonical20_with_frozen_stea.py --dataset-root ${DATASET_ROOT} --stea-ckpt ${STEA_CKPT} --train-scales 20 --canonical-scale-mm 20.0 --checkpoint-dir /datasets/checkpoints/standalone_stea_experiments/canonical20_with_stea --epochs 50 --batch-size 128 --lr 1e-4 --weight-decay 1e-2 --workers 4 --device ${DEVICE} --save-every 5"

echo "Recommended supplemental training commands:"
echo "$TRAIN_A_CMD"
echo "$TRAIN_B_CMD"
echo "$TRAIN_C_CMD"
echo

if [ "$RUN_TRAINING" = "1" ]; then
  bash -lc "$TRAIN_A_CMD"
  bash -lc "$TRAIN_B_CMD"
  bash -lc "$TRAIN_C_CMD"
fi

if [ "$RUN_MATRICES" = "1" ]; then
  DATASET_ROOT="$DATASET_ROOT" DEVICE="$DEVICE" \
    bash "$PROJECT_ROOT/standalone_stea_experiments/scripts/run_matrix_no_adapter_baseline.sh"
  DATASET_ROOT="$DATASET_ROOT" DEVICE="$DEVICE" \
    bash "$PROJECT_ROOT/standalone_stea_experiments/scripts/run_matrix_canonical20_no_adapter.sh"
  DATASET_ROOT="$DATASET_ROOT" DEVICE="$DEVICE" STEA_CKPT="$STEA_CKPT" \
    bash "$PROJECT_ROOT/standalone_stea_experiments/scripts/run_matrix_canonical20_with_stea.sh"
fi
