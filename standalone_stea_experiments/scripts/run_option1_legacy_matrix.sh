#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

DATASET_ROOT="${DATASET_ROOT:-/datasets/usa_static_v1_large_run/downstream_test_16_20_23}"
LEGACY_REGRESSOR_CKPT="${LEGACY_REGRESSOR_CKPT:-/datasets/checkpoints/downstream_regressor_20mm_no_adapter/best.pt}"
STEA_CKPT="${STEA_CKPT:-/datasets/checkpoints/standalone_stea_adapter_fullpairs_bs320/best.pt}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-4}"
RESULTS_DIR="${RESULTS_DIR:-/datasets/checkpoints/standalone_stea_experiments/results}"

mkdir -p "$RESULTS_DIR"
RAW_LOG="${RESULTS_DIR}/option1_legacy_matrix.txt"
: > "$RAW_LOG"

run_eval() {
  local scale_mm="$1"
  local split="$2"

  SCALE_MM="$scale_mm" \
  SPLIT="$split" \
  DATASET_ROOT="$DATASET_ROOT" \
  LEGACY_REGRESSOR_CKPT="$LEGACY_REGRESSOR_CKPT" \
  STEA_CKPT="$STEA_CKPT" \
  DEVICE="$DEVICE" \
  BATCH_SIZE="$BATCH_SIZE" \
  WORKERS="$WORKERS" \
  python - <<'PY' | tee -a "$RAW_LOG"
import os
import torch

from standalone_stea.data import build_downstream_split_dataset, create_loader, unpack_downstream_batch
from standalone_stea.utils import FrozenViTPatchExtractor, load_canonical_head_from_regressor_ckpt
from standalone_stea_experiments.utils import load_frozen_stea

scale_mm = float(os.environ['SCALE_MM'])
split = os.environ['SPLIT']
dataset_root = os.environ['DATASET_ROOT']
legacy_regressor_ckpt = os.environ['LEGACY_REGRESSOR_CKPT']
stea_ckpt = os.environ['STEA_CKPT']
device = torch.device(os.environ['DEVICE'])
batch_size = int(os.environ['BATCH_SIZE'])
workers = int(os.environ['WORKERS'])
canonical_scale_mm = 20.0

vit = FrozenViTPatchExtractor(model_name='vit_base_patch16_224', pretrained=True).to(device)
vit.eval()
head = load_canonical_head_from_regressor_ckpt(legacy_regressor_ckpt, device, freeze=True)
stea, _ = load_frozen_stea(stea_ckpt, device)

dataset = build_downstream_split_dataset(
    root_dir=dataset_root,
    scales_mm=[scale_mm],
    split=split,
    reference_scale_mm=canonical_scale_mm,
)
loader = create_loader(dataset, batch_size=batch_size, workers=workers, shuffle=False)

sum_sq_x = 0.0
sum_sq_y = 0.0
sum_sq_depth = 0.0
total_samples = 0

with torch.no_grad():
    for batch in loader:
        images, targets, _target_coords, _source_coords, source_scale_mm = unpack_downstream_batch(batch, device)
        features = vit(images)
        adapted_features, _ = stea(
            features,
            source_scale_mm=source_scale_mm,
            target_scale_mm=canonical_scale_mm,
        )
        preds = head(adapted_features)
        sq = (preds - targets).pow(2)
        sum_sq_x += sq[:, 0].sum().item()
        sum_sq_y += sq[:, 1].sum().item()
        sum_sq_depth += sq[:, 2].sum().item()
        total_samples += images.shape[0]

mse_x = sum_sq_x / max(total_samples, 1)
mse_y = sum_sq_y / max(total_samples, 1)
mse_depth = sum_sq_depth / max(total_samples, 1)
mse_total = (mse_x + mse_y + mse_depth) / 3.0
scale_text = str(int(scale_mm)) if float(scale_mm).is_integer() else f'{scale_mm:g}'
print(
    f'Eval scale={scale_text} split={split} use_stea=True adapter_kind=stea | '
    f'mse_x={mse_x:.6f} mse_y={mse_y:.6f} mse_depth={mse_depth:.6f} mse_total={mse_total:.6f}'
)
PY
}

for scale_mm in 16 20 23; do
  for split in seen unseen; do
    run_eval "$scale_mm" "$split"
  done
done

echo
echo "Summary table:"
python - "$RAW_LOG" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
line_re = re.compile(
    r'^Eval scale=(?P<scale>[\d.]+) split=(?P<split>\w+) use_stea=True adapter_kind=stea \| '
    r'mse_x=(?P<mse_x>[0-9.eE+-]+) mse_y=(?P<mse_y>[0-9.eE+-]+) '
    r'mse_depth=(?P<mse_depth>[0-9.eE+-]+) mse_total=(?P<mse_total>[0-9.eE+-]+)$'
)

rows = []
for line in log_path.read_text().splitlines():
    match = line_re.match(line.strip())
    if not match:
        continue
    rows.append(match.groupdict())

print('| scale | split | mse_x | mse_y | mse_depth | mse_total |')
print('| --- | --- | ---: | ---: | ---: | ---: |')
for row in rows:
    print(
        f'| {row["scale"]} | {row["split"]} | {float(row["mse_x"]):.6f} | '
        f'{float(row["mse_y"]):.6f} | {float(row["mse_depth"]):.6f} | {float(row["mse_total"]):.6f} |'
    )
PY

echo
echo "Saved raw matrix log to $RAW_LOG"

