from __future__ import annotations

import torch

from sccwm.losses.sccwm_losses import build_negative_labels, compute_counterfactual_ranking_loss
from sccwm.metrics.ccauc_metric import compute_ccauc
from sccwm.metrics.sass_metric import compute_sass
from sccwm.models.sccwm import SCCWM
from sccwm.models.world_lattice import WorldLatticeProjector


def test_world_lattice_shapes() -> None:
    projector = WorldLatticeProjector()
    features = torch.randn(2, 3, 8, 16, 16)
    coord_map = torch.randn(2, 16, 16, 2)
    world, occ = projector.splat_to_world_lattice(features, coord_map)
    gathered, gathered_occ = projector.gather_from_world_lattice(world, coord_map)
    assert world.shape == (2, 3, 8, 32, 32)
    assert occ.shape == (2, 3, 1, 32, 32)
    assert gathered.shape == (2, 3, 8, 16, 16)
    assert gathered_occ.shape == (2, 3, 1, 16, 16)


def test_sccwm_forward_pair() -> None:
    model = SCCWM(feature_dim=32, sensor_dim=16, world_hidden_dim=32, geometry_dim=16, visibility_dim=8, lattice_size=16)
    source_obs = torch.randn(2, 3, 3, 256, 256)
    target_obs = torch.randn(2, 3, 3, 256, 256)
    coord_map = torch.randn(2, 16, 16, 2)
    out = model.forward_pair(
        source_obs=source_obs,
        target_obs=target_obs,
        source_coord_map=coord_map,
        target_coord_map=coord_map,
        source_scale_mm=torch.tensor([20.0, 22.0]),
        target_scale_mm=torch.tensor([20.0, 18.0]),
    )
    assert out["source"]["pred_x_norm"].shape == (2,)
    assert out["source"]["pred_depth_mm_seq"].shape == (2, 3)
    assert out["source_to_target"]["decoded_target_features"].shape[:3] == (2, 3, 32)


def test_counterfactual_helpers() -> None:
    occupancy = torch.tensor([0.1, 0.12, 0.9, 0.95], dtype=torch.float32)
    depth = torch.tensor([0.2, 0.8, 0.21, 0.81], dtype=torch.float32)
    idx = build_negative_labels(occupancy, depth)
    assert idx.shape == (4,)
    emb = torch.randn(4, 8)
    loss = compute_counterfactual_ranking_loss(emb, occupancy, depth)
    assert torch.isfinite(loss)


def test_metrics_smoke() -> None:
    sass = compute_sass(
        [
            {"event_key": "a", "pred": [0.0, 0.0, 0.0]},
            {"event_key": "a", "pred": [0.1, 0.0, 0.2]},
            {"event_key": "b", "pred": [0.0, 0.0, 0.0]},
            {"event_key": "b", "pred": [0.0, 0.0, 0.0]},
        ]
    )
    ccauc = compute_ccauc([0.9, 0.8], [0.2, 0.1])
    assert "sass_mean" in sass
    assert ccauc["ccauc"] >= 0.9
