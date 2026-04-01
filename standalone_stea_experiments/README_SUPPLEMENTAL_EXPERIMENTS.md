# Supplemental STEA Experiments

## Why These Experiments Exist
These supplemental experiments are designed to strengthen the paper story around STEA without changing any previous STEA, DD-USA, Q-Former, or downstream experiments.

We already have a trained STEA adapter and a frozen-STEA downstream pipeline. What is still missing are cleaner baselines that answer three specific questions:

1. Is STEA better than a fair no-adapter pipeline trained on the same downstream data?
2. How far does a strict legacy 20mm-only no-adapter head generalize by itself?
3. Does frozen STEA enable true canonical-interface reuse when the downstream head is trained only on canonical 20mm?

All supplemental code lives in `standalone_stea_experiments/` and writes only to new checkpoint/result directories.

## The Three New Experiments

### A. Fair No-Adapter Baseline
- train scales: `16 20 23`
- pipeline: `image -> frozen ViT -> trainable regressor head`
- no adapter anywhere

This is the fairest system-level baseline against the existing frozen-STEA pipeline.

### B. Strict Canonical20 No-Adapter
- train scales: `20` only
- pipeline: `image -> frozen ViT -> trainable regressor head`
- no adapter anywhere

This isolates raw legacy-canonical generalization without any scale correction.

### C. Strict Canonical20 With Frozen STEA
- train scales: `20` only
- pipeline: `image -> frozen ViT -> frozen STEA(target=20) -> trainable regressor head`

This is the canonical-interface reuse experiment: the head only sees canonical 20mm during training, while frozen STEA handles zero-shot cross-scale transfer during evaluation.

## How These Differ From Previous STEA Runs
- The original standalone STEA regressor experiment trains a head on scales `16 20 23` with frozen STEA.
- These supplemental experiments add cleaner no-adapter and canonical-only baselines.
- They are designed to produce paper-safe comparisons without touching prior checkpoints or scripts.

## New Safe Checkpoint Directories
- `/datasets/checkpoints/standalone_stea_experiments/no_adapter_baseline`
- `/datasets/checkpoints/standalone_stea_experiments/canonical20_no_adapter`
- `/datasets/checkpoints/standalone_stea_experiments/canonical20_with_stea`
- `/datasets/checkpoints/standalone_stea_experiments/results`

## Exact Training Commands

### A. Fair No-Adapter Baseline
```bash
cd /workspace && \
python standalone_stea_experiments/train_regressor_no_adapter.py \
  --dataset-root /datasets/usa_static_v1_large_run/downstream_test_16_20_23 \
  --train-scales 16 20 23 \
  --checkpoint-dir /datasets/checkpoints/standalone_stea_experiments/no_adapter_baseline \
  --epochs 50 \
  --batch-size 128 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --workers 4 \
  --device cuda \
  --save-every 5
```

### B. Strict Canonical20 No-Adapter
```bash
cd /workspace && \
python standalone_stea_experiments/train_regressor_canonical20_no_adapter.py \
  --dataset-root /datasets/usa_static_v1_large_run/downstream_test_16_20_23 \
  --train-scales 20 \
  --checkpoint-dir /datasets/checkpoints/standalone_stea_experiments/canonical20_no_adapter \
  --epochs 50 \
  --batch-size 128 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --workers 4 \
  --device cuda \
  --save-every 5
```

### C. Strict Canonical20 With Frozen STEA
```bash
cd /workspace && \
python standalone_stea_experiments/train_regressor_canonical20_with_frozen_stea.py \
  --dataset-root /datasets/usa_static_v1_large_run/downstream_test_16_20_23 \
  --stea-ckpt /datasets/checkpoints/standalone_stea_adapter_fullpairs_bs320/best.pt \
  --train-scales 20 \
  --canonical-scale-mm 20.0 \
  --checkpoint-dir /datasets/checkpoints/standalone_stea_experiments/canonical20_with_stea \
  --epochs 50 \
  --batch-size 128 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --workers 4 \
  --device cuda \
  --save-every 5
```

## Exact Matrix Evaluation Commands

### Run all three matrix baselines
```bash
cd /workspace && \
bash standalone_stea_experiments/scripts/run_matrix_no_adapter_baseline.sh && \
bash standalone_stea_experiments/scripts/run_matrix_canonical20_no_adapter.sh && \
bash standalone_stea_experiments/scripts/run_matrix_canonical20_with_stea.sh
```

### Or run the umbrella helper
```bash
cd /workspace && \
bash standalone_stea_experiments/scripts/run_all_supplemental_experiments.sh
```

## Result Interpretation
- Experiment A answers whether STEA beats a fair no-adapter system trained on the same downstream data.
- Experiment B answers how much cross-scale generalization a strict legacy canonical 20mm model gets for free.
- Experiment C answers whether frozen STEA really enables canonical-interface reuse when the head is trained only on 20mm.

The strongest paper story is usually:
- compare A vs the existing frozen-STEA fair system baseline
- compare B vs C to isolate whether STEA is necessary for canonical-only zero-shot reuse

## Matrix Log Summaries
After running the three matrix scripts, summarize them with:

```bash
cd /workspace && \
python standalone_stea_experiments/summarize_results.py \
  --results-dir /datasets/checkpoints/standalone_stea_experiments/results
```

