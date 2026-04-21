from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn

from sccwm.models.common import MLP
from sccwm_force_grounded_v21a.models.sccwm_force_grounded_v21a import grad_reverse
from sccwm_force_grounded_v23.models.sccwm_force_grounded_v23 import (
    DEFAULT_MARKER_VOCAB,
    EMBEDDING_VIEW_TO_KEY as V23_EMBEDDING_VIEW_TO_KEY,
    SCCWMForceGroundedV23Base,
    _StateResidualReadout,
    _zero_init_last_linear,
)


EMBEDDING_VIEW_TO_KEY = dict(V23_EMBEDDING_VIEW_TO_KEY)


def select_embedding_view_v24(encoding: dict[str, torch.Tensor], embedding_view: str) -> torch.Tensor:
    key = EMBEDDING_VIEW_TO_KEY.get(str(embedding_view))
    if key is None:
        raise ValueError(f"Unsupported embedding view: {embedding_view}")
    if key not in encoding:
        raise KeyError(f"Encoding does not contain embedding view {embedding_view!r} ({key!r})")
    return encoding[key]


class SCCWMForceGroundedV24FGOF(SCCWMForceGroundedV23Base):
    """Force-guided geometry/operator factorization on top of v21a-clean.

    This remains a world model. The backbone/projector/decoder stay intact.
    v24 adds a conservative geometry/operator factorization plus force-guided
    canonicalization. It is not an explicit force-supervised model.
    """

    variant_name = "v24fgof"

    def __init__(
        self,
        *,
        geometry_hidden_dim: int = 64,
        state_refine_hidden_dim: int = 64,
        operator_dim: int = 32,
        geometry_adv_grl_lambda: float = 0.25,
        marker_vocab: Iterable[str] = DEFAULT_MARKER_VOCAB,
        enable_swap_decode: bool = True,
        canonical_force_align_dim: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            geometry_adv_grl_lambda=geometry_adv_grl_lambda,
            geometry_hidden_dim=geometry_hidden_dim,
            state_refine_hidden_dim=state_refine_hidden_dim,
            operator_dim=operator_dim,
            marker_vocab=marker_vocab,
            **kwargs,
        )
        self.enable_swap_decode = bool(enable_swap_decode)
        self.canonical_force_align_dim = int(canonical_force_align_dim or self.z_force_dim)

        self.geometry_canonical_proj = nn.Sequential(
            nn.LayerNorm(self.geometry_dim),
            MLP([self.geometry_dim, self.geometry_hidden_dim, self.geometry_dim]),
        )
        operator_in_dim = self.sensor_dim + self.visibility_dim + self.feature_dim + 1
        self.operator_encoder = nn.Sequential(
            nn.LayerNorm(operator_in_dim),
            MLP([operator_in_dim, self.geometry_hidden_dim, self.operator_dim]),
        )

        self.canonical_scale_adv = MLP([self.geometry_dim, self.geometry_hidden_dim, self.scale_class_count])
        self.canonical_branch_adv = MLP([self.geometry_dim, self.geometry_hidden_dim, 2])
        self.canonical_marker_adv = (
            MLP([self.geometry_dim, self.geometry_hidden_dim, self.marker_class_count]) if self.marker_class_count > 0 else None
        )

        self.operator_scale_classifier = MLP([self.operator_dim, self.geometry_hidden_dim, self.scale_class_count])
        self.operator_branch_classifier = MLP([self.operator_dim, self.geometry_hidden_dim, 2])
        self.operator_marker_classifier = (
            MLP([self.operator_dim, self.geometry_hidden_dim, self.marker_class_count]) if self.marker_class_count > 0 else None
        )
        self.operator_to_sensor_delta = MLP([self.operator_dim, self.geometry_hidden_dim, self.sensor_dim])
        _zero_init_last_linear(self.operator_to_sensor_delta)

        # Force-guided canonicalization uses a small projection from canonical geometry
        # into a force-alignment space. The force branch acts as the cleaner teacher.
        self.canonical_force_proj = nn.Sequential(
            nn.LayerNorm(self.geometry_dim),
            MLP([self.geometry_dim, self.geometry_hidden_dim, self.canonical_force_align_dim]),
        )
        self.geometry_state_refine_head = _StateResidualReadout(
            self.geometry_dim + self.z_force_dim,
            state_refine_hidden_dim,
        )

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
        encoding = super().encode_sequence(
            obs_seq,
            coord_map,
            scale_mm,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
            valid_mask=valid_mask,
        )
        geometry_raw = encoding["geometry_latent"]
        geometry_canonical = self.geometry_canonical_proj(geometry_raw)
        scale_feature = encoding["scale_mm"].unsqueeze(1)
        operator_in = torch.cat(
            [encoding["sensor_embedding"], encoding["anchor_visibility_latent"], encoding["appearance_pooled"], scale_feature],
            dim=1,
        )
        operator_code = self.operator_encoder(operator_in)
        geometry_adv = grad_reverse(geometry_canonical, self.geometry_adv_grl_lambda)
        canonical_force_projected = self.canonical_force_proj(geometry_canonical)
        force_teacher = encoding["z_force_global"]
        encoding.update(
            {
                "geometry_latent_raw": geometry_raw,
                "geometry_latent_canonical": geometry_canonical,
                "operator_code": operator_code,
                "canonical_scale_adv_logits": self.canonical_scale_adv(geometry_adv),
                "canonical_branch_adv_logits": self.canonical_branch_adv(geometry_adv),
                "operator_scale_logits": self.operator_scale_classifier(operator_code),
                "operator_branch_logits": self.operator_branch_classifier(operator_code),
                "canonical_force_projected": canonical_force_projected,
                "force_teacher_latent": force_teacher,
            }
        )
        if self.canonical_marker_adv is not None:
            encoding["canonical_marker_adv_logits"] = self.canonical_marker_adv(geometry_adv)
        if self.operator_marker_classifier is not None:
            encoding["operator_marker_logits"] = self.operator_marker_classifier(operator_code)

        # Main state refinement path relies on canonical geometry + force, not operator.
        self._apply_state_refinement(encoding, geometry_state_code=geometry_canonical, head=self.geometry_state_refine_head)
        self._update_shared_views(
            encoding,
            geometry_raw=geometry_raw,
            geometry_main=geometry_canonical,
            operator_code=operator_code,
            main_pair_key="geom_canonical_plus_force",
        )
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
            geometry_latent=source_encoding["geometry_latent_canonical"],
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


def build_force_grounded_v24_model(model_cfg: dict[str, Any]) -> SCCWMForceGroundedV24FGOF:
    variant = str(model_cfg.get("variant", "v24fgof")).strip().lower()
    if variant != "v24fgof":
        raise ValueError(f"Unsupported v24 variant: {variant}")
    return SCCWMForceGroundedV24FGOF(
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
        geometry_hidden_dim=int(model_cfg.get("geometry_hidden_dim", 64)),
        state_refine_hidden_dim=int(model_cfg.get("state_refine_hidden_dim", 64)),
        operator_dim=int(model_cfg.get("operator_dim", 32)),
        marker_vocab=[str(v) for v in model_cfg.get("marker_vocab", list(DEFAULT_MARKER_VOCAB))],
        enable_swap_decode=bool(model_cfg.get("enable_swap_decode", True)),
        canonical_force_align_dim=int(model_cfg.get("canonical_force_align_dim", model_cfg.get("z_force_dim", 6))),
    )
