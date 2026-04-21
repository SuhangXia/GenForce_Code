from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn

from sccwm.models.common import MLP
from sccwm_force_grounded_v21a.models.sccwm_force_grounded_v21a import SCCWMForceGroundedV21A, grad_reverse


DEFAULT_MARKER_VOCAB = (
    "marker_Array1.jpg",
    "marker_Array2.jpg",
    "marker_Array_Gelsight.jpg",
    "marker_Diamond2.jpg",
)

EMBEDDING_VIEW_TO_KEY = {
    "full_state": "state_embedding_full",
    "latent_only": "state_embedding_latent_only",
    "force_only": "state_embedding_force_only",
    "force_map_pooled": "state_embedding_force_map_pooled",
    "geometry_only": "state_embedding_geometry_only",
    "geometry_raw_only": "state_embedding_geometry_raw_only",
    "geometry_deconf_only": "state_embedding_geometry_deconf_only",
    "geometry_canonical_only": "state_embedding_geometry_canonical_only",
    "operator_only": "state_embedding_operator_only",
    "geom_deconf_plus_force": "state_embedding_geom_deconf_plus_force",
    "geom_canonical_plus_force": "state_embedding_geom_canonical_plus_force",
    "full_state_factorized": "state_embedding_full_state_factorized",
}


def select_embedding_view_v23(encoding: dict[str, torch.Tensor], embedding_view: str) -> torch.Tensor:
    key = EMBEDDING_VIEW_TO_KEY.get(str(embedding_view))
    if key is None:
        raise ValueError(f"Unsupported embedding view: {embedding_view}")
    if key not in encoding:
        raise KeyError(f"Encoding does not contain embedding view {embedding_view!r} ({key!r})")
    return encoding[key]


def _last_linear(module: nn.Module) -> nn.Linear | None:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            return child
    return None


def _zero_init_last_linear(module: nn.Module) -> None:
    layer = _last_linear(module)
    if layer is None:
        return
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class _StateResidualReadout(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = MLP([in_dim, hidden_dim, 3])
        _zero_init_last_linear(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SCCWMForceGroundedV23Base(SCCWMForceGroundedV21A):
    """Conservative geometry-focused v23 base on top of v21a-clean.

    This remains a world model. The backbone/projector/decoder are preserved.
    v23 only adds geometry-side deconfounding / factorization heads and small
    state/decode adapters needed to test the geometry-side bottleneck.
    """

    variant_name = "v23_base"

    def __init__(
        self,
        *,
        geometry_adv_grl_lambda: float = 0.25,
        geometry_hidden_dim: int = 64,
        state_refine_hidden_dim: int = 64,
        operator_dim: int = 32,
        marker_vocab: Iterable[str] = DEFAULT_MARKER_VOCAB,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.geometry_adv_grl_lambda = float(geometry_adv_grl_lambda)
        self.geometry_hidden_dim = int(geometry_hidden_dim)
        self.operator_dim = int(operator_dim)
        self.marker_vocab = tuple(str(v) for v in marker_vocab)
        self.scale_class_count = len(self.scale_bucket_edges_mm) + 1
        self.marker_class_count = len(self.marker_vocab)

    def _metric_from_sensor_norm(
        self,
        x_norm: torch.Tensor,
        y_norm: torch.Tensor,
        scale_mm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        view_shape = [scale_mm.shape[0]] + [1] * max(x_norm.dim() - 1, 0)
        half_scale = (scale_mm.reshape(view_shape) / 2.0).clamp_min(1e-6)
        return x_norm * half_scale, y_norm * half_scale

    def _apply_state_refinement(
        self,
        encoding: dict[str, torch.Tensor],
        *,
        geometry_state_code: torch.Tensor,
        head: nn.Module,
    ) -> None:
        refine_in = torch.cat([geometry_state_code, encoding["z_force_global"]], dim=1)
        delta = head(refine_in)
        pred_x_norm = encoding["pred_x_norm"] + delta[:, 0]
        pred_y_norm = encoding["pred_y_norm"] + delta[:, 1]
        pred_depth_mm = encoding["pred_depth_mm"] + delta[:, 2]
        pred_x_mm, pred_y_mm = self._metric_from_sensor_norm(pred_x_norm, pred_y_norm, encoding["scale_mm"])
        encoding.update(
            {
                "pred_x_norm": pred_x_norm,
                "pred_y_norm": pred_y_norm,
                "pred_depth_mm": pred_depth_mm,
                "pred_x_mm": pred_x_mm,
                "pred_y_mm": pred_y_mm,
            }
        )

    def _update_shared_views(
        self,
        encoding: dict[str, torch.Tensor],
        *,
        geometry_raw: torch.Tensor,
        geometry_main: torch.Tensor,
        operator_code: torch.Tensor | None = None,
        main_pair_key: str,
    ) -> None:
        state_embedding_full = torch.cat(
            [
                geometry_main,
                encoding["z_force_global"],
                encoding["pred_depth_mm"].unsqueeze(1),
                encoding["pred_x_mm"].unsqueeze(1),
                encoding["pred_y_mm"].unsqueeze(1),
            ],
            dim=1,
        )
        latent_only = torch.cat([geometry_main, encoding["z_force_global"]], dim=1)
        encoding.update(
            {
                "state_embedding": state_embedding_full,
                "state_embedding_full": state_embedding_full,
                "state_embedding_full_state_factorized": state_embedding_full,
                "state_embedding_latent_only": latent_only,
                "state_embedding_force_only": encoding["z_force_global"],
                "state_embedding_force_map_pooled": encoding["z_force_map_pooled"],
                "state_embedding_geometry_only": geometry_raw,
                "state_embedding_geometry_raw_only": geometry_raw,
            }
        )
        if main_pair_key == "geom_deconf_plus_force":
            encoding["state_embedding_geometry_deconf_only"] = geometry_main
            encoding["state_embedding_geom_deconf_plus_force"] = latent_only
        elif main_pair_key == "geom_canonical_plus_force":
            encoding["state_embedding_geometry_canonical_only"] = geometry_main
            encoding["state_embedding_geom_canonical_plus_force"] = latent_only
        if operator_code is not None:
            encoding["state_embedding_operator_only"] = operator_code

    def _decode_to_target_common(
        self,
        source_encoding: dict[str, torch.Tensor],
        target_coord_map: torch.Tensor,
        target_scale_mm: torch.Tensor | float,
        *,
        geometry_latent: torch.Tensor,
        target_operator_code: torch.Tensor | None = None,
        absolute_contact_xy_mm: torch.Tensor | None = None,
        world_origin_xy_mm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        b = source_encoding["world_latent_seq"].shape[0]
        device = source_encoding["world_latent_seq"].device
        dtype = source_encoding["world_latent_seq"].dtype
        target_scale = self._prepare_scale(target_scale_mm, b, device, dtype)
        target_coord_map_t = target_coord_map.to(device=device, dtype=dtype)
        target_sensor_embedding = self.sensor_encoder(target_coord_map_t, target_scale)
        if target_operator_code is not None and hasattr(self, "operator_to_sensor_delta"):
            target_sensor_embedding = target_sensor_embedding + self.operator_to_sensor_delta(target_operator_code)
        gathered_world_base_features, gathered_world_base_occ = self.projector.gather_from_world_lattice(
            source_encoding["world_features"],
            target_coord_map_t,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
        world_feature_seq = self.world_to_feature(
            source_encoding["world_latent_seq"].reshape(
                -1,
                self.world_hidden_dim,
                source_encoding["world_latent_seq"].shape[-2],
                source_encoding["world_latent_seq"].shape[-1],
            )
        ).reshape(
            b,
            source_encoding["world_latent_seq"].shape[1],
            self.feature_dim,
            source_encoding["world_latent_seq"].shape[-2],
            source_encoding["world_latent_seq"].shape[-1],
        )
        gathered_world_features, gathered_world_occ = self.projector.gather_from_world_lattice(
            world_feature_seq,
            target_coord_map_t,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
        force_map_world = source_encoding["z_force_map"].unsqueeze(1).expand(-1, source_encoding["world_latent_seq"].shape[1], -1, -1, -1)
        gathered_force_map, _ = self.projector.gather_from_world_lattice(
            force_map_world,
            target_coord_map_t,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
        decoded_features, decoded_obs = self.decoder(
            gathered_world_features=gathered_world_features,
            gathered_world_base_features=gathered_world_base_features,
            gathered_force_map=gathered_force_map,
            target_sensor_embedding=target_sensor_embedding,
            geometry_latent=geometry_latent,
            visibility_latent=source_encoding["visibility_latent"],
            z_force_global=source_encoding["z_force_global"],
        )
        return {
            "target_sensor_embedding": target_sensor_embedding,
            "gathered_world_features": gathered_world_features,
            "gathered_world_base_features": gathered_world_base_features,
            "gathered_force_map": gathered_force_map,
            "decoded_target_features": decoded_features,
            "decoded_target_observation": decoded_obs,
            "decoded_target_images": decoded_obs,
            "decoded_target_occupancy": gathered_world_occ,
            "gathered_world_base_occupancy": gathered_world_base_occ,
            "decoded_world_occupancy": gathered_world_occ,
        }


class SCCWMForceGroundedV23GD(SCCWMForceGroundedV23Base):
    variant_name = "v23gd"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.geometry_deconf_proj = nn.Sequential(
            nn.LayerNorm(self.geometry_dim),
            MLP([self.geometry_dim, self.geometry_hidden_dim, self.geometry_dim]),
        )
        self.geometry_deconf_scale_adv = MLP([self.geometry_dim, self.geometry_hidden_dim, self.scale_class_count])
        self.geometry_deconf_branch_adv = MLP([self.geometry_dim, self.geometry_hidden_dim, 2])
        self.geometry_deconf_marker_adv = (
            MLP([self.geometry_dim, self.geometry_hidden_dim, self.marker_class_count]) if self.marker_class_count > 0 else None
        )
        self.geometry_state_refine_head = _StateResidualReadout(self.geometry_dim + self.z_force_dim, self.geometry_hidden_dim)

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
        geometry_deconf = self.geometry_deconf_proj(geometry_raw)
        geometry_adv = grad_reverse(geometry_deconf, self.geometry_adv_grl_lambda)
        encoding.update(
            {
                "geometry_latent_raw": geometry_raw,
                "geometry_latent_deconf": geometry_deconf,
                "geometry_scale_adv_logits": self.geometry_deconf_scale_adv(geometry_adv),
                "geometry_branch_adv_logits": self.geometry_deconf_branch_adv(geometry_adv),
            }
        )
        if self.geometry_deconf_marker_adv is not None:
            encoding["geometry_marker_adv_logits"] = self.geometry_deconf_marker_adv(geometry_adv)
        self._apply_state_refinement(encoding, geometry_state_code=geometry_deconf, head=self.geometry_state_refine_head)
        self._update_shared_views(
            encoding,
            geometry_raw=geometry_raw,
            geometry_main=geometry_deconf,
            operator_code=None,
            main_pair_key="geom_deconf_plus_force",
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
            geometry_latent=source_encoding["geometry_latent_deconf"],
            target_operator_code=target_operator_code,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )


class SCCWMForceGroundedV23GOF(SCCWMForceGroundedV23Base):
    variant_name = "v23gof"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
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
        self.canonical_marker_adv = MLP([self.geometry_dim, self.geometry_hidden_dim, self.marker_class_count]) if self.marker_class_count > 0 else None
        self.operator_scale_classifier = MLP([self.operator_dim, self.geometry_hidden_dim, self.scale_class_count])
        self.operator_branch_classifier = MLP([self.operator_dim, self.geometry_hidden_dim, 2])
        self.operator_marker_classifier = MLP([self.operator_dim, self.geometry_hidden_dim, self.marker_class_count]) if self.marker_class_count > 0 else None
        self.operator_to_sensor_delta = MLP([self.operator_dim, self.geometry_hidden_dim, self.sensor_dim])
        _zero_init_last_linear(self.operator_to_sensor_delta)
        self.geometry_state_refine_head = _StateResidualReadout(self.geometry_dim + self.z_force_dim, self.geometry_hidden_dim)

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
        encoding.update(
            {
                "geometry_latent_raw": geometry_raw,
                "geometry_latent_canonical": geometry_canonical,
                "operator_code": operator_code,
                "canonical_scale_adv_logits": self.canonical_scale_adv(geometry_adv),
                "canonical_branch_adv_logits": self.canonical_branch_adv(geometry_adv),
                "operator_scale_logits": self.operator_scale_classifier(operator_code),
                "operator_branch_logits": self.operator_branch_classifier(operator_code),
            }
        )
        if self.canonical_marker_adv is not None:
            encoding["canonical_marker_adv_logits"] = self.canonical_marker_adv(geometry_adv)
        if self.operator_marker_classifier is not None:
            encoding["operator_marker_logits"] = self.operator_marker_classifier(operator_code)
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
        source_to_target = self.decode_to_target(
            source_encoding,
            target_coord_map,
            target_scale_mm,
            target_operator_code=target_encoding["operator_code"],
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
        target_to_source = self.decode_to_target(
            target_encoding,
            source_coord_map,
            source_scale_mm,
            target_operator_code=source_encoding["operator_code"],
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
        return {
            "source": source_encoding,
            "target": target_encoding,
            "source_to_target": source_to_target,
            "target_to_source": target_to_source,
        }


def build_force_grounded_v23_model(model_cfg: dict[str, Any]) -> SCCWMForceGroundedV23Base:
    variant = str(model_cfg.get("variant", "v23gof")).strip().lower()
    common_kwargs = dict(
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
    )
    if variant == "v23gd":
        return SCCWMForceGroundedV23GD(**common_kwargs)
    if variant == "v23gof":
        return SCCWMForceGroundedV23GOF(**common_kwargs)
    raise ValueError(f"Unsupported v23 variant: {variant}")
