# Force-Grounded SCCWM v25

`v25-SCFGOF` is a conservative follow-on to `v24-FGOF`.

This package remains a world model. It is not explicitly force supervised.

The guiding hypothesis is narrow:

- `v24` showed that force guidance helps clean canonical geometry.
- But a single canonical branch still appears to trade away localization detail.
- `v25` therefore splits canonical geometry into:
  - `z_canon_load`
  - `z_canon_pose`
- Force guidance is applied mainly to `z_canon_load`, not to the whole canonical representation.

Key design points:

- explicit operator factorization is retained
- operator code remains the nuisance carrier
- `x/y` residual refinement is pose-dominant
- `depth` residual refinement is load-plus-force dominant
- decode stays conservative and continues to use canonical content plus operator code

Main reduced-budget control:

- `model.enable_split_canonical=false`

This collapses `v25` back toward a monolithic canonical branch while keeping the
same operator branch and force-guidance machinery.
