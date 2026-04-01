# STEA: Spatial-Topology & Energy-Aligned Adapter

## What STEA Is
STEA is a new standalone canonical-interface adapter for cross-scale tactile transfer.
It treats the frozen ViT patch grid as a strict 14x14 spatial interface and corrects scale mismatch conservatively rather than rewriting latent tokens from scratch.

The canonical interface is always **20mm**.

## Architecture Summary
STEA is built from four explicit steps on the fixed token grid:

1. **Token-space geometric resampling**
   - reshape `(B, 196, 768)` into `(B, 768, 14, 14)`
   - compute `ratio = source_scale / target_scale`
   - use `affine_grid + grid_sample` to canonicalize geometry on the same 14x14 output topology

2. **Canonical background latent imputation**
   - out-of-bounds or invalid regions are never zero-filled
   - a full canonical 20mm no-contact latent map `(1, 768, 14, 14)` fills unsupported regions

3. **Scale-conditioned AdaLN-Zero energy calibration**
   - conditioning uses `[r, 1/r, log(r), r-1]`
   - fixed 2D positional encodings preserve patch identity on the 14x14 grid
   - no global self-attention is used

4. **Conservative residual correction**
   - a small gated residual adjusts energy statistics while keeping `20 -> 20` near identity at initialization
   - optional depthwise boundary smoothing is available

## Why STEA Differs From DD-USA And Q-Former
### Compared with DD-USA
- DD-USA uses a more expressive latent-interface correction pipeline with attention-style feature transfer.
- STEA is intentionally more conservative and explicitly topology preserving.
- STEA performs a direct token-grid geometric correction first, then a lightweight energy correction.

### Compared with Q-Former
- Q-Former learns query-token interactions and can rewrite latents more aggressively.
- STEA is **not** a token generator.
- STEA never discards the fixed 14x14 token lattice.
- STEA is designed to preserve downstream X/Y localization topology.

## Datasets
There are two different dataset roots and they must not be confused.

### A. STEA adapter training dataset
- `/home/suhang/datasets/usa_static_v1_large_run/full_5scales_ep100_boundarymix`
- scales: `15 18 20 22 25`
- canonical target scale: `20mm`
- paired source->target training cases:
  - `15 -> 20`
  - `18 -> 20`
  - `20 -> 20`
  - `22 -> 20`
  - `25 -> 20`

### B. Downstream dataset
- `/home/suhang/datasets/usa_static_v1_large_run/downstream_test_16_20_23`
- scales: `16 20 23`
- used for downstream regressor training and matrix evaluation

## Build The Background Latent Map
You need one or more canonical 20mm no-contact images.

```bash
cd /home/suhang/projects/test_code/GenForce_Code && python standalone_stea/build_background_latent.py   --image-paths     /home/suhang/datasets/usa_static_v1_large_run/full_5scales_ep100_boundarymix/<canonical_20mm_no_contact_image_1>.png     /home/suhang/datasets/usa_static_v1_large_run/full_5scales_ep100_boundarymix/<canonical_20mm_no_contact_image_2>.png   --output-path /home/suhang/datasets/checkpoints/stea_background/canonical_20mm_background.pt   --also-save-npy   --device cuda
```

## Train The STEA Adapter
This stage uses:
- frozen ViT
- frozen canonical downstream regressor head
- trainable STEA only

Recommended canonical regressor checkpoint:
- `/home/suhang/datasets/checkpoints/downstream_regressor_20mm_no_adapter/best.pt`

```bash
cd /home/suhang/projects/test_code/GenForce_Code && python standalone_stea/train_stea_adapter.py   --dataset-root /home/suhang/datasets/usa_static_v1_large_run/full_5scales_ep100_boundarymix   --canonical-scale-mm 20.0   --source-scales 15 18 20 22 25   --target-scale-mm 20.0   --regressor-ckpt /home/suhang/datasets/checkpoints/downstream_regressor_20mm_no_adapter/best.pt   --background-latent-path /home/suhang/datasets/checkpoints/stea_background/canonical_20mm_background.pt   --checkpoint-dir /home/suhang/datasets/checkpoints/standalone_stea_adapter   --epochs 50   --batch-size 12   --lr 1e-4   --weight-decay 1e-2   --workers 4   --device cuda   --save-every 5   --use-boundary-smoothing
```

## Train A Downstream Regressor With Frozen STEA
This stage uses:
- frozen ViT
- frozen pretrained STEA
- trainable downstream regressor head

```bash
cd /home/suhang/projects/test_code/GenForce_Code && python standalone_stea/train_regressor_with_frozen_stea.py   --dataset-root /home/suhang/datasets/usa_static_v1_large_run/downstream_test_16_20_23   --stea-ckpt /home/suhang/datasets/checkpoints/standalone_stea_adapter/best.pt   --canonical-scale-mm 20.0   --train-scales 16 20 23   --checkpoint-dir /home/suhang/datasets/checkpoints/standalone_stea_regressor   --epochs 50   --batch-size 32   --lr 1e-4   --weight-decay 1e-2   --workers 4   --device cuda   --save-every 5
```

## Run A Single Evaluation
```bash
cd /home/suhang/projects/test_code/GenForce_Code && python standalone_stea/eval_regressor_with_stea.py   --dataset-root /home/suhang/datasets/usa_static_v1_large_run/downstream_test_16_20_23   --regressor-ckpt /home/suhang/datasets/checkpoints/standalone_stea_regressor/best.pt   --stea-ckpt /home/suhang/datasets/checkpoints/standalone_stea_adapter/best.pt   --use-stea true   --scale-mm 16   --split seen   --device cuda
```

## Run The Full Matrix
```bash
cd /home/suhang/projects/test_code/GenForce_Code && REGRESSOR_CKPT=/home/suhang/datasets/checkpoints/standalone_stea_regressor/best.pt STEA_CKPT=/home/suhang/datasets/checkpoints/standalone_stea_adapter/best.pt DATASET_ROOT=/home/suhang/datasets/usa_static_v1_large_run/downstream_test_16_20_23 DEVICE=cuda bash standalone_stea/scripts/run_matrix_eval_stea.sh
```

This evaluates:
- without STEA
- with STEA
for:
- `scale=16, split=seen`
- `scale=16, split=unseen`
- `scale=20, split=seen`
- `scale=20, split=unseen`
- `scale=23, split=seen`
- `scale=23, split=unseen`

## Notes
- All STEA code lives under `standalone_stea/` only.
- Old DD-USA, USA, Q-Former, and downstream files are left untouched.
- The scripts can map `/home/suhang/datasets/...` paths to `/datasets/...` automatically when running inside the Docker container.
