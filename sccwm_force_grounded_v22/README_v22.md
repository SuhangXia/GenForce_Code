# Force-Grounded SCCWM v22 Ablations

This directory contains conservative, auditable ablations on top of the cleaned
`v21a` baseline. These remain world models. They are not explicit force
supervised models and they do not replace the backbone, projector, or decoder.

Variants:

- `v22t`: trajectory/load-order constraints only
- `v22s`: spatial contact-support head only
- `v22ts`: both together

The goal is to test whether sequence-level ordering and/or a canonical spatial
contact-support representation improve strict cross-band invariance with minimal
architecture drift.
