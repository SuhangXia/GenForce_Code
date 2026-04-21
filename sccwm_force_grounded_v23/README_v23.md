# Force-Grounded SCCWM v23

This package contains the next geometry-focused research pass on top of the
cleaned `v21a` baseline. These models remain world models. They are not
explicit force-supervised models and they do not replace the backbone,
projector, or decoder wholesale.

The working hypothesis is that the geometry-side latent is the main remaining
source of operator leakage. v23 tests two conservative interventions:

- `v23gd`: geometry-side deconfounding only
- `v23gof`: explicit geometry/operator factorization

Both variants:

- warm-start from the same `v21a-clean` checkpoint strategy
- keep the v21a force branch intact
- expose auditable embedding views for direct eval, latent eval, and frozen
  probes

Key embedding views:

- `geometry_raw_only`
- `geometry_deconf_only` for `v23gd`
- `geometry_canonical_only` for `v23gof`
- `operator_only` for `v23gof`
- `force_only`
- `force_map_pooled`
- `geom_deconf_plus_force` / `geom_canonical_plus_force`
- `full_state`

The scientific goal is not only lower MAE. The main question is whether
geometry-side scale/marker leakage can be reduced while keeping strict `16↔23`
cross-band regression and CCAUC competitive.
