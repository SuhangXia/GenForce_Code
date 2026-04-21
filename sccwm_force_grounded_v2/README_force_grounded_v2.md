# Force-Grounded SCCWM v2

This package is an isolated, incremental extension of SCCWM.

It is **not**:

- an explicit force-supervised model,
- a 6D force/torque regressor,
- or a UniForce2-style equilibrium model.

It is:

- a SCCWM stage-2 style paired world model,
- with a small force-like latent subspace `z_force`,
- grounded by depth/contact proxy supervision and paired latent consistency.

The intended purpose is to move SCCWM closer to future 6D force supervision without
rewriting the existing backbone or contaminating the original SCCWM codepath.

## Key ideas

- Keep the original SCCWM backbone: patch encoder, sensor encoder, projector, ConvGRU,
  paired decode, and direct state prediction.
- Add a dedicated force-like latent `z_force`.
- Train `z_force` with:
  - depth proxy supervision,
  - contact-intensity proxy supervision,
  - matched-pair latent consistency,
  - variance regularization to avoid collapse.

## Data handling

Training/eval on the black-border dataset uses an overlay dataset:

- images come from the black-border root,
- metadata, coord maps, and pair indices come from the original root.

This is necessary because the black-border dataset currently only contains rendered
marker images, not the full paired-sequence metadata assets.
