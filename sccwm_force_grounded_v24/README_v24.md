# Force-Grounded SCCWM v24

This package implements **v24-FGOF**: force-guided geometry/operator factorization.

It is a conservative extension on top of the v21a-clean / v23 lessons:

- the model remains a **world model**
- this is **not** an explicit force-supervised model
- the backbone / projector / decoder are not redesigned
- the main change is geometry/operator factorization plus force-guided canonicalization

## Main idea

`v23-GOF` showed that an explicit operator code can absorb nuisance information, but canonical geometry still stayed entangled with scale / marker. `v24-FGOF` keeps the operator branch and uses the empirically cleaner force-side latent as a teacher to shape canonical geometry.

Conceptually:

- `geometry_latent_canonical`: canonical state / contact content
- `operator_code`: nuisance / rendering style / scale-marker-branch carrier
- `z_force_global`: cleaner force-side teacher

## Main toggles

Use one config plus two mechanism toggles:

- full v24-FGOF:
  - `loss.force_guidance_weight > 0`
  - `model.enable_swap_decode = true`
- no-force-guidance ablation:
  - `loss.force_guidance_weight = 0.0`
  - `model.enable_swap_decode = true`
- no-swap ablation:
  - `loss.force_guidance_weight > 0`
  - `model.enable_swap_decode = false`

## Exposed embedding views

- `geometry_raw_only`
- `geometry_canonical_only`
- `operator_only`
- `force_only`
- `force_map_pooled`
- `geom_canonical_plus_force`
- `full_state`
- `full_state_factorized`

## Minimal reduced-budget run

```bash
cd /workspace
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python sccwm_force_grounded_v24/train/train_sccwm_stage2_force_grounded_v24.py \
  --config sccwm_force_grounded_v24/configs/sccwm_stage2_force_grounded_v24fgof.yaml \
  --override train.output_dir=/datasets/checkpoints/sccwm_force_grounded_v24/stage2_force_grounded_v24fgof \
  --override train.epochs=1 \
  --override train.sample_episodes_per_indenter_per_epoch=8 \
  --override train.max_train_samples_per_epoch=4096 \
  --override train.max_val_samples=2048 \
  --override train.save_every=1 \
  --override system.device=cuda
```
