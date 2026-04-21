# Force-Grounded SCCWM v2.1

This package is an incremental extension of force-grounded SCCWM v2.

It is **not**:

- an explicit 6D force/torque supervised model,
- a UniForce reproduction,
- or a same-force-event equilibrium model.

It **is**:

- a world model built on the SCCWM paired world-lattice backbone,
- with a spatial force-like latent `z_force_map`,
- a pooled global force summary `z_force_global`,
- explicit force-conditioned decoding,
- and anti-shortcut regularization against scale/branch/sensor leakage.

## What changed relative to v2

1. `z_force` is no longer only a global pooled vector.
   - `z_force_map` carries spatial contact/load structure in world coordinates.
   - `z_force_global` summarizes that map.

2. The decoder now explicitly uses force latents.
   - spatial force map injection enters the decode feature path,
   - global force summary modulates decode activations through FiLM-style conditioning.

3. Anti-shortcut regularization is stronger.
   - covariance-based orthogonality against sensor and visibility latents,
   - adversarial confusion on scale bucket and branch identity via gradient reversal.

4. Force proxies are less degenerate.
   - normalized penetration proxy,
   - contact-intensity proxy,
   - load-progress proxy,
   - and a same-episode penetration ranking loss.

## Data handling

Training still uses the black-border image roots through an overlay dataset:

- images come from `/datasets/usa_static_v1_large_run_blackborder_3px/...`
- metadata, coord maps, and pair indices come from `/datasets/usa_static_v1_large_run/...`

This keeps the force-grounded experiments isolated from the original SCCWM codepath.
