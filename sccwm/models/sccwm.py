from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .common import MLP, repeat_time, sequence_last_valid, spatial_softargmax2d
from .counterfactual_decoder import CounterfactualDecoder
from .patch_feature_encoder import PatchFeatureEncoder
from .sensor_context_encoder import SensorContextEncoder
from .temporal_state_encoder import ConvGRU
from .world_lattice import WorldLatticeConfig, WorldLatticeProjector


class SCCWM(nn.Module):
    """Sensor-Conditioned Counterfactual Contact World Model."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        feature_dim: int = 128,
        sensor_dim: int = 64,
        world_hidden_dim: int = 128,
        geometry_dim: int = 64,
        visibility_dim: int = 32,
        lattice_size: int = 32,
        reconstruct_observation: bool = True,
        enable_contact_mask: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.sensor_dim = int(sensor_dim)
        self.world_hidden_dim = int(world_hidden_dim)
        self.geometry_dim = int(geometry_dim)
        self.visibility_dim = int(visibility_dim)
        self.reconstruct_observation = bool(reconstruct_observation)
        self.enable_contact_mask = bool(enable_contact_mask)

        self.patch_encoder = PatchFeatureEncoder(in_channels=input_channels, feature_dim=feature_dim)
        self.sensor_encoder = SensorContextEncoder(embedding_dim=sensor_dim)
        self.projector = WorldLatticeProjector(WorldLatticeConfig(lattice_size=lattice_size))
        self.temporal = ConvGRU(feature_dim, hidden_dim=world_hidden_dim, num_layers=2)

        self.geometry_head = MLP([world_hidden_dim + sensor_dim, world_hidden_dim, geometry_dim])
        self.visibility_head = MLP([world_hidden_dim + sensor_dim, world_hidden_dim, visibility_dim])
        self.position_head = nn.Conv2d(world_hidden_dim, 1, kernel_size=3, padding=1)
        self.depth_head = MLP([world_hidden_dim + geometry_dim + visibility_dim, world_hidden_dim, 1])
        self.contact_mask_head = nn.Conv2d(world_hidden_dim, 1, kernel_size=3, padding=1) if self.enable_contact_mask else None

        self.world_to_feature = nn.Conv2d(world_hidden_dim, feature_dim, kernel_size=1)
        self.decoder = CounterfactualDecoder(
            world_hidden_dim=world_hidden_dim,
            feature_dim=feature_dim,
            sensor_dim=sensor_dim,
            geometry_dim=geometry_dim,
            visibility_dim=visibility_dim,
            reconstruct_observation=reconstruct_observation,
        )

    def _prepare_scale(self, scale_mm: torch.Tensor | float, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(scale_mm, torch.Tensor):
            if scale_mm.dim() == 0:
                return scale_mm.to(device=device, dtype=dtype).expand(batch_size)
            return scale_mm.to(device=device, dtype=dtype).reshape(batch_size)
        return torch.full((batch_size,), float(scale_mm), device=device, dtype=dtype)

    def _metric_to_sensor_norm(self, x_mm: torch.Tensor, y_mm: torch.Tensor, scale_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        view_shape = [scale_mm.shape[0]] + [1] * max(x_mm.dim() - 1, 0)
        half_scale = (scale_mm.reshape(view_shape) / 2.0).clamp_min(1e-6)
        return x_mm / half_scale, y_mm / half_scale

    def _predict_state(
        self,
        hidden_seq: torch.Tensor,
        sensor_embedding: torch.Tensor,
        geometry_latent: torch.Tensor,
        visibility_latent: torch.Tensor,
        scale_mm: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, t, c, h, w = hidden_seq.shape
        logits = self.position_head(hidden_seq.reshape(b * t, c, h, w))
        lattice_x_norm, lattice_y_norm, heatmap = spatial_softargmax2d(logits)
        lattice_x_norm = lattice_x_norm.reshape(b, t)
        lattice_y_norm = lattice_y_norm.reshape(b, t)
        x_mm, y_mm = self.projector.lattice_normalized_to_mm(lattice_x_norm, lattice_y_norm)
        x_norm, y_norm = self._metric_to_sensor_norm(x_mm, y_mm, scale_mm)
        heatmap = heatmap.reshape(b, t, 1, h, w)
        pooled = hidden_seq.mean(dim=(3, 4))
        depth_in = torch.cat([pooled, repeat_time(geometry_latent, t), visibility_latent], dim=-1)
        depth = self.depth_head(depth_in.reshape(b * t, -1)).reshape(b, t)
        out = {
            "pred_x_mm_seq": x_mm,
            "pred_y_mm_seq": y_mm,
            "pred_x_norm_seq": x_norm,
            "pred_y_norm_seq": y_norm,
            "pred_depth_mm_seq": depth,
            "pred_x_mm": sequence_last_valid(x_mm.unsqueeze(-1), valid_mask).squeeze(-1),
            "pred_y_mm": sequence_last_valid(y_mm.unsqueeze(-1), valid_mask).squeeze(-1),
            "pred_x_norm": sequence_last_valid(x_norm.unsqueeze(-1), valid_mask).squeeze(-1),
            "pred_y_norm": sequence_last_valid(y_norm.unsqueeze(-1), valid_mask).squeeze(-1),
            "pred_depth_mm": sequence_last_valid(depth.unsqueeze(-1), valid_mask).squeeze(-1),
            "world_position_heatmap": heatmap,
            "position_heatmap": heatmap,
        }
        if self.contact_mask_head is not None:
            contact_logits = self.contact_mask_head(hidden_seq.reshape(b * t, c, h, w)).reshape(b, t, 1, h, w)
            out["contact_mask_logits"] = contact_logits
            out["contact_mask"] = torch.sigmoid(contact_logits)
        return out

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
        b = obs_seq.shape[0]
        device = obs_seq.device
        dtype = obs_seq.dtype
        scale = self._prepare_scale(scale_mm, b, device, dtype)
        patch_features = self.patch_encoder(obs_seq)
        sensor_embedding = self.sensor_encoder(coord_map.to(device=device, dtype=dtype), scale)
        world_features, occupancy = self.projector.splat_to_world_lattice(
            patch_features,
            coord_map.to(device=device, dtype=dtype),
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
        hidden_seq, hidden_states = self.temporal(world_features)
        pooled_world = hidden_seq.mean(dim=(1, 3, 4))
        geometry_latent = self.geometry_head(torch.cat([pooled_world, sensor_embedding], dim=1))
        visibility_input = torch.cat([hidden_seq.mean(dim=(3, 4)), repeat_time(sensor_embedding, hidden_seq.shape[1])], dim=-1)
        visibility_latent = self.visibility_head(visibility_input.reshape(b * hidden_seq.shape[1], -1)).reshape(b, hidden_seq.shape[1], -1)
        state = self._predict_state(hidden_seq, sensor_embedding, geometry_latent, visibility_latent, scale, valid_mask=valid_mask)
        return {
            "patch_features": patch_features,
            "scale_mm": scale,
            "sensor_embedding": sensor_embedding,
            "world_features": world_features,
            "world_occupancy": occupancy,
            "world_latent_seq": hidden_seq,
            "hidden_states": hidden_states,
            "geometry_latent": geometry_latent,
            "visibility_latent": visibility_latent,
            "last_hidden": sequence_last_valid(hidden_seq, valid_mask),
            **state,
        }

    def decode_to_target(
        self,
        source_encoding: dict[str, torch.Tensor],
        target_coord_map: torch.Tensor,
        target_scale_mm: torch.Tensor | float,
        *,
        absolute_contact_xy_mm: torch.Tensor | None = None,
        world_origin_xy_mm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        b = source_encoding["world_latent_seq"].shape[0]
        device = source_encoding["world_latent_seq"].device
        dtype = source_encoding["world_latent_seq"].dtype
        target_scale = self._prepare_scale(target_scale_mm, b, device, dtype)
        target_sensor_embedding = self.sensor_encoder(target_coord_map.to(device=device, dtype=dtype), target_scale)
        gathered_world_base_features, gathered_world_base_occ = self.projector.gather_from_world_lattice(
            source_encoding["world_features"],
            target_coord_map.to(device=device, dtype=dtype),
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
            target_coord_map.to(device=device, dtype=dtype),
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
        decoded_features, decoded_obs = self.decoder(
            gathered_world_features=gathered_world_features,
            gathered_world_base_features=gathered_world_base_features,
            target_sensor_embedding=target_sensor_embedding,
            geometry_latent=source_encoding["geometry_latent"],
            visibility_latent=source_encoding["visibility_latent"],
        )
        return {
            "target_sensor_embedding": target_sensor_embedding,
            "gathered_world_features": gathered_world_features,
            "gathered_world_base_features": gathered_world_base_features,
            "decoded_target_features": decoded_features,
            "decoded_target_observation": decoded_obs,
            "decoded_target_images": decoded_obs,
            "decoded_target_occupancy": gathered_world_occ,
            "gathered_world_base_occupancy": gathered_world_base_occ,
            "decoded_world_occupancy": gathered_world_occ,
        }

    def forward_single(
        self,
        obs_seq: torch.Tensor,
        coord_map: torch.Tensor,
        scale_mm: torch.Tensor | float,
        *,
        absolute_contact_xy_mm: torch.Tensor | None = None,
        world_origin_xy_mm: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.encode_sequence(
            obs_seq,
            coord_map,
            scale_mm,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
            valid_mask=valid_mask,
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
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
        target_to_source = self.decode_to_target(
            target_encoding,
            source_coord_map,
            source_scale_mm,
            absolute_contact_xy_mm=absolute_contact_xy_mm,
            world_origin_xy_mm=world_origin_xy_mm,
        )
        return {
            "source": source_encoding,
            "target": target_encoding,
            "source_to_target": source_to_target,
            "target_to_source": target_to_source,
        }

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        return self.forward_pair(**kwargs)
