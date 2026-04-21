# Force-Grounded SCCWM v2.1a

This package is a **small corrective fork** of force-grounded SCCWM v2.1.

It is still:

- a **world model** built on the SCCWM paired world-lattice backbone,
- **not** an explicit 6D force/torque supervised model,
- **not** a UniForce reproduction,
- **not** a same-force-event equilibrium model.

## What v2.1a fixes

### 1. Anchor semantics are now explicit

v2.1a does **not** rely on `seq_valid_mask.argmax()` as a generic anchor rule.

Instead, the overlay dataset now returns:

- `anchor_index_in_window`
- `anchor_position_in_sequence`
- `anchor_selection_mode`
- `anchor_phase_name`
- `anchor_phase_progress`

Losses and sanity checks use this explicit anchor first, then fall back to:

1. window center,
2. last valid frame,

only if the explicit field is unavailable.

### 2. Full training recipe is intentionally more conservative

Default full recipe:

- `epochs = 6`
- `sample_episodes_per_indenter_per_epoch = 20`

This is meant to be a safer first full run than the more aggressive v2.1 recipe.

### 3. Preflight sanity logging is built in

Before full training starts, v2.1a writes `preflight_sanity.json` and prints:

- anchor selection mode,
- anchor index statistics,
- example batch rows with:
  - phase name,
  - phase progress,
  - selected anchor frame,
  - resulting load-progress target,
- penetration/contact/load proxy target stats,
- first-batch force losses and adversarial accuracies.

There is also a `--sanity-only` mode for running only these checks.

## Data handling

Training uses the black-border image roots through the overlay dataset:

- images:
  - `/datasets/usa_static_v1_large_run_blackborder_3px/...`
- metadata / coord maps / pair indices:
  - `/datasets/usa_static_v1_large_run/...`

## First-pass training guidance

For the first full run, do **not** optimize for `mae_mean` alone.

Prioritize:

- `val/state_mae_depth`
- strict `16↔23` cross-band `mae_depth`
- `ccauc`
- `force_scale_adv_acc`
- `force_branch_adv_acc`
- `force_latent_consistency_global`
- `force_latent_consistency_map`

If these move in the right direction while `mae_mean` is flat, that is still useful signal for this force-grounded line.
