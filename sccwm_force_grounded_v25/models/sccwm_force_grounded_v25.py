from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn

from sccwm.models.common import MLP
from sccwm_force_grounded_v21a.models.sccwm_force_grounded_v21a import SCCWMForceGroundedV21A, grad_reverse
from sccwm_force_grounded_v23.models.sccwm_force_grounded_v23 import (
    DEFAULT_MARKER_VOCAB,
    _StateResidualReadout,
    _zero_init_last_linear,
)
from sccwm_force_grounded_v24.models.sccwm_force_grounded_v24 import SCCWMForceGroundedV24FGOF


EMBEDDING_VIEW_TO_KEY = {
    "full_state": "state_embedding_full",
    "latent_only": "state_embedding_canonical_pose_plus_load_plus_force",
    "geometry_only": "state_embedding_geometry_raw_only",
    "geometry_raw_only": "state_embedding_geometry_raw_only",
    "geometry_canonical_only": "state_embedding_geometry_canonical_only",
    "canonical_load_only": "state_embedding_canonical_load_only",
    "canonical_pose_only": "state_embedding_canonical_pose_only",
    "operator_only": "state_embedding_operator_only",
    "force_only": "state_embedding_force_only",
    "force_map_pooled": "state_embedding_force_map_pooled",
    "canonical_load_plus_force": "state_embedding_canonical_load_plus_force",
    "canonical_pose_plus_load": "state_embedding_canonical_pose_plus_load",
    "canonical_pose_plus_load_plus_force": "state_embedding_canonical_pose_plus_load_plus_force",
    "geom_canonical_plus_force": "state_embedding_canonical_pose_plus_load_plus_force",
    "full_state_factorized": "state_embedding_full",
}


def select_embedding_view_v25(encoding: dict[str, torch.Tensor], embedding_view: str) -> torch.Tensor:
    key = EMBEDDING_VIEW_TO_KEY.get(str(embedding_view))
    if key is None:
        raise ValueError(f"Unsupported embedding view: {embedding_view}")
    if key not in encoding:
        raise KeyError(f"Encoding does not contain embedding view {embedding_view!r} ({key!r})")
    return encoding[key]


class SCCWMForceGroundedV25SCFGOF(SCCWMForceGroundedV24FGOF):
    """Split-canonical, force-guided geometry/operator factorization.

    This remains a world model. v25 reuses the v24 factorized world-model path,
    but splits canonical geometry into a load-guided subspace and a
    localization-preserving subspace.
    """

    variant_name = "v25scfgof"

    def __init__(
        self,
        *,
        geometry_hidden_dim: int = 64,
        state_refine_hidden_dim: int = 64,
        operator_dim: int = 32,
        geometry_adv_grl_lambda: float = 0.25,
        pose_adv_grl_lambda: float = 0.10,
        marker_vocab: Iterable[str] = DEFAULT_MARKER_VOCAB,
        enable_swap_decode: bool = True,
        canonical_force_align_dim: int | None = None,
        enable_split_canonical: bool = True,
        canon_shared_dim: int | None = None,
        canon_load_dim: int | None = None,
        canon_pose_dim: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            geometry_hidden_dim=geometry_hidden_dim,
            state_refine_hidden_dim=state_refine_hidden_dim,
            operator_dim=operator_dim,
            geometry_adv_grl_lambda=geometry_adv_grl_lambda,
            marker_vocab=marker_vocab,
            enable_swap_decode=enable_swap_decode,
            canonical_force_align_dim=canonical_force_align_dim,
            **kwargs,
        )
        self.enable_split_canonical = bool(enable_split_canonical)
        self.pose_adv_grl_lambda = float(pose_adv_grl_lambda)
        self.canon_shared_dim = int(canon_shared_dim or self.geometry_dim)
        self.canon_load_dim = int(canon_load_dim or self.geometry_dim)
        self.canon_pose_dim = int(canon_pose_dim or self.geometry_dim)
        if self.canon_shared_dim != self.geometry_dim:
            raise ValueError("v25 currently expects canon_shared_dim == geometry_dim for conservative warm-start compatibility")
        if self.canon_load_dim != self.geometry_dim or self.canon_pose_dim != self.geometry_dim:
            raise ValueError("v25 currently expects canon_load_dim == canon_pose_dim == geometry_dim")

        self.canonical_load_delta = nn.Sequential(
            nn.LayerNorm(self.geometry_dim),
            MLP([self.geometry_dim, self.geometry_hidden_dim, self.geometry_dim]),
        )
        self.canonical_pose_delta = nn.Sequential(
            nn.LayerNorm(self.geometry_dim),
            MLP([self.geometry_dim, self.geometry_hidden_dim, self.geometry_dim]),
        )
        _zero_init_last_linear(self.canonical_load_delta)
        _zero_init_last_linear(self.canonical_pose_delta)

        self.pose_scale_adv = MLP([self.geometry_dim, self.geometry_hidden_dim, self.scale_class_count])
        self.pose_branch_adv = MLP([self.geometry_dim, self.geometry_hidden_dim, 2])
        self.pose_marker_adv = (
            MLP([self.geometry_dim, self.geometry_hidden_dim, self.marker_class_count]) if self.marker_class_count > 0 else None
        )

        self.xy_pose_head = MLP([self.geometry_dim, self.geometry_hidden_dim, 2])
        self.depth_load_head = MLP([self.geometry_dim + self.z_force_dim, self.geometry_hidden_dim, 1])
        self.split_state_refine_head = _StateResidualReadout(
            self.geometry_dim + self.geometry_dim + self.z_force_dim,
            state_refine_hidden_dim,
        )
        _zero_init_last_linear(self.xy_pose_head)
        _zero_init_last_linear(self.depth_load_head)

    def _split_canonical(self, shared: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enable_split_canonical:
            return shared, shared
        load = shared + self.canonical_load_delta(shared)
        pose = shared + self.canonical_pose_delta(shared)
        return load, pose

    def encode_sequence(
        self,
        obs_seq: torch.Tensor,
        coord_map: torch.Tensor,
        scale_mm: torch.Tensor | float,
        *,
        absolute_contact_xy_mm: torch.Tensor | None = None,
        world_origin_xy_mm: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoding = SCCWMForceGroundedV21A.encode_sequence(
            self,
            obs_seq,
            coord_map,
            scale_mm,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
            valid_mask=valid_mask,
        )
        geometry_raw = encoding["geometry_latent"]
        geometry_shared = self.geometry_canonical_proj(geometry_raw)
        z_canon_load, z_canon_pose = self._split_canonical(geometry_shared)
        scale_feature = encoding["scale_mm"].unsqueeze(1)
        operator_in = torch.cat(
            [encoding["sensor_embedding"], encoding["anchor_visibility_latent"], encoding["appearance_pooled"], scale_feature],
            dim=1,
        )
        operator_code = self.operator_encoder(operator_in)

        load_adv = grad_reverse(z_canon_load, self.geometry_adv_grl_lambda)
        pose_adv = grad_reverse(z_canon_pose, self.pose_adv_grl_lambda)
        canonical_force_projected = self.canonical_force_proj(z_canon_load)
        force_teacher = encoding["z_force_global"]

        xy_delta = self.xy_pose_head(z_canon_pose)
        depth_delta = self.depth_load_head(torch.cat([z_canon_load, force_teacher], dim=1)).squeeze(1)
        refine_delta = self.split_state_refine_head(torch.cat([z_canon_pose, z_canon_load, force_teacher], dim=1))
        pred_x_norm = encoding["pred_x_norm"] + xy_delta[:, 0] + refine_delta[:, 0]
        pred_y_norm = encoding["pred_y_norm"] + xy_delta[:, 1] + refine_delta[:, 1]
        pred_depth_mm = encoding["pred_depth_mm"] + depth_delta + refine_delta[:, 2]
        pred_x_mm, pred_y_mm = self._metric_from_sensor_norm(pred_x_norm, pred_y_norm, encoding["scale_mm"])

        canonical_pose_plus_load = torch.cat([z_canon_pose, z_canon_load], dim=1)
        canonical_load_plus_force = torch.cat([z_canon_load, force_teacher], dim=1)
        canonical_pose_plus_load_plus_force = torch.cat([z_canon_pose, z_canon_load, force_teacher], dim=1)
        state_embedding_full = torch.cat(
            [
                z_canon_pose,
                z_canon_load,
                force_teacher,
                pred_depth_mm.unsqueeze(1),
                pred_x_mm.unsqueeze(1),
                pred_y_mm.unsqueeze(1),
            ],
            dim=1,
        )

        encoding.update(
            {
                "pred_x_norm": pred_x_norm,
                "pred_y_norm": pred_y_norm,
                "pred_depth_mm": pred_depth_mm,
                "pred_x_mm": pred_x_mm,
                "pred_y_mm": pred_y_mm,
                "geometry_latent_raw": geometry_raw,
                "geometry_latent_canonical_shared": geometry_shared,
                "geometry_latent_canonical": canonical_pose_plus_load,
                "z_canon_shared": geometry_shared,
                "z_canon_load": z_canon_load,
                "z_canon_pose": z_canon_pose,
                "operator_code": operator_code,
                "split_canonical_enabled": torch.tensor(
                    1.0 if self.enable_split_canonical else 0.0,
                    device=geometry_raw.device,
                    dtype=geometry_raw.dtype,
                ),
                "canonical_load_scale_adv_logits": self.canonical_scale_adv(load_adv),
                "canonical_load_branch_adv_logits": self.canonical_branch_adv(load_adv),
                "canonical_pose_scale_adv_logits": self.pose_scale_adv(pose_adv),
                "canonical_pose_branch_adv_logits": self.pose_branch_adv(pose_adv),
                "operator_scale_logits": self.operator_scale_classifier(operator_code),
                "operator_branch_logits": self.operator_branch_classifier(operator_code),
                "canonical_force_projected": canonical_force_projected,
                "force_teacher_latent": force_teacher,
                "state_embedding": state_embedding_full,
                "state_embedding_full": state_embedding_full,
                "state_embedding_full_state_factorized": state_embedding_full,
                "state_embedding_latent_only": canonical_pose_plus_load_plus_force,
                "state_embedding_geometry_only": geometry_raw,
                "state_embedding_geometry_raw_only": geometry_raw,
                "state_embedding_geometry_canonical_only": canonical_pose_plus_load,
                "state_embedding_canonical_load_only": z_canon_load,
                "state_embedding_canonical_pose_only": z_canon_pose,
                "state_embedding_operator_only": operator_code,
                "state_embedding_force_only": force_teacher,
                "state_embedding_force_map_pooled": encoding["z_force_map_pooled"],
                "state_embedding_canonical_load_plus_force": canonical_load_plus_force,
                "state_embedding_canonical_pose_plus_load": canonical_pose_plus_load,
                "state_embedding_canonical_pose_plus_load_plus_force": canonical_pose_plus_load_plus_force,
                "state_embedding_geom_canonical_plus_force": canonical_pose_plus_load_plus_force,
            }
        )
        if self.canonical_marker_adv is not None:
            encoding["canonical_load_marker_adv_logits"] = self.canonical_marker_adv(load_adv)
        if self.pose_marker_adv is not None:
            encoding["canonical_pose_marker_adv_logits"] = self.pose_marker_adv(pose_adv)
        if self.operator_marker_classifier is not None:
            encoding["operator_marker_logits"] = self.operator_marker_classifier(operator_code)
        return encoding

    def decode_to_target(
        self,
        source_encoding: dict[str, torch.Tensor],
        target_coord_map: torch.Tensor,
        target_scale_mm: torch.Tensor | float,
        *,
        absolute_contact_xy_mm: torch.Tensor | None = None,
        world_origin_xy_mm: torch.Tensor | None = None,
        target_operator_code: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        return self._decode_to_target_common(
            source_encoding,
            target_coord_map,
            target_scale_mm,
            geometry_latent=source_encoding["z_canon_shared"],
            target_operator_code=target_operator_code,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )

    def forward_pair(
        self,
        *,
        source_obs: torch.Tensor,
        target_obs: torch.Tensor,
        source_coord_map: torch.Tensor,
        target_coord_map: torch.Tensor,
        source_scale_mm: torch.Tensor | float,
        target_scale_mm: torch.Tensor | float,
        source_valid_mask: torch.Tensor | None = None,
        target_valid_mask: torch.Tensor | None = None,
        absolute_contact_xy_mm: torch.Tensor | None = None,
        world_origin_xy_mm: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        source_encoding = self.encode_sequence(
            source_obs,
            source_coord_map,
            source_scale_mm,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
            valid_mask=source_valid_mask,
        )
        target_encoding = self.encode_sequence(
            target_obs,
            target_coord_map,
            target_scale_mm,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
            valid_mask=target_valid_mask,
        )
        if self.enable_swap_decode:
            source_to_target_operator = target_encoding["operator_code"]
            target_to_source_operator = source_encoding["operator_code"]
        else:
            source_to_target_operator = source_encoding["operator_code"]
            target_to_source_operator = target_encoding["operator_code"]
        source_to_target = self.decode_to_target(
            source_encoding,
            target_coord_map,
            target_scale_mm,
            target_operator_code=source_to_target_operator,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
        target_to_source = self.decode_to_target(
            target_encoding,
            source_coord_map,
            source_scale_mm,
            target_operator_code=target_to_source_operator,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
        return {
            "source": source_encoding,
            "target": target_encoding,
            "source_to_target": source_to_target,
            "target_to_source": target_to_source,
        }


def build_force_grounded_v25_model(model_cfg: dict[str, Any]) -> SCCWMForceGroundedV25SCFGOF:
    variant = str(model_cfg.get("variant", "v25scfgof")).strip().lower()
    if variant != "v25scfgof":
        raise ValueError(f"Unsupported v25 variant: {variant}")
    return SCCWMForceGroundedV25SCFGOF(
        input_channels=int(model_cfg.get("input_channels", 3)),
        feature_dim=int(model_cfg.get("feature_dim", 128)),
        sensor_dim=int(model_cfg.get("sensor_dim", 64)),
        world_hidden_dim=int(model_cfg.get("world_hidden_dim", 128)),
        geometry_dim=int(model_cfg.get("geometry_dim", 64)),
        visibility_dim=int(model_cfg.get("visibility_dim", 32)),
        lattice_size=int(model_cfg.get("lattice_size", 32)),
        reconstruct_observation=bool(model_cfg.get("reconstruct_observation", False)),
        enable_contact_mask=bool(model_cfg.get("enable_contact_mask", False)),
        z_force_dim=int(model_cfg.get("z_force_dim", 6)),
        adv_grl_lambda=float(model_cfg.get("adv_grl_lambda", 0.25)),
        scale_bucket_edges_mm=[float(v) for v in model_cfg.get("scale_bucket_edges_mm", [14.0, 18.0, 22.0])],
        geometry_adv_grl_lambda=float(model_cfg.get("geometry_adv_grl_lambda", model_cfg.get("adv_grl_lambda", 0.25))),
        pose_adv_grl_lambda=float(model_cfg.get("pose_adv_grl_lambda", 0.10)),
        geometry_hidden_dim=int(model_cfg.get("geometry_hidden_dim", 64)),
        state_refine_hidden_dim=int(model_cfg.get("state_refine_hidden_dim", 64)),
        operator_dim=int(model_cfg.get("operator_dim", 32)),
        marker_vocab=[str(v) for v in model_cfg.get("marker_vocab", list(DEFAULT_MARKER_VOCAB))],
        enable_swap_decode=bool(model_cfg.get("enable_swap_decode", True)),
        canonical_force_align_dim=int(model_cfg.get("canonical_force_align_dim", model_cfg.get("z_force_dim", 6))),
        enable_split_canonical=bool(model_cfg.get("enable_split_canonical", True)),
        canon_shared_dim=int(model_cfg.get("canon_shared_dim", model_cfg.get("geometry_dim", 64))),
        canon_load_dim=int(model_cfg.get("canon_load_dim", model_cfg.get("geometry_dim", 64))),
        canon_pose_dim=int(model_cfg.get("canon_pose_dim", model_cfg.get("geometry_dim", 64))),
    )
