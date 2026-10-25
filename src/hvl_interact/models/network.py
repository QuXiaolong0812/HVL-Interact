"""Observation-only queries and granularity-matched semantic projections."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hvl_interact.data.layout import LEVELS
from hvl_interact.models.skeleton import HierarchicalSkeletonEncoder


def projection(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim), nn.GELU(), nn.Linear(output_dim, output_dim)
    )


def unit_embedding(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1, eps=1e-6).to(value.dtype)


class HVLInteract(nn.Module):
    def __init__(
        self,
        num_classes: int,
        width: int = 512,
        embed_dim: int = 512,
        clip_dim: int = 512,
        channels: tuple[int, ...] = (128, 256, 512),
        heads: int = 8,
        temporal_layers: int = 1,
        spatial_layers: int = 2,
        dropout: float = 0.1,
        max_frames: int = 120,
        groups: dict | None = None,
        clip_encoder: nn.Module | None = None,
        use_visual: bool = True,
    ):
        super().__init__()
        if min(num_classes, width, embed_dim, clip_dim) < 1:
            raise ValueError("Model dimensions must be positive")
        self.skeleton = HierarchicalSkeletonEncoder(
            width, channels, heads, temporal_layers, spatial_layers, dropout, max_frames, groups
        )
        self.query_fine = projection(width, embed_dim)
        self.query_coarse = projection(width, embed_dim)
        self.stream_fine = projection(width, embed_dim)
        self.stream_coarse = projection(width, embed_dim)
        self.visual_projection = projection(clip_dim, embed_dim) if use_visual else None
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1)
        )
        self.global_projection = projection(embed_dim, embed_dim)
        self.text_projections = nn.ModuleDict({h: projection(clip_dim, embed_dim) for h in LEVELS})
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.clip_encoder = clip_encoder
        self.use_visual = use_visual

    def project_text(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if set(features) != set(LEVELS):
            raise ValueError("Text features must contain fine, coarse and global levels")
        output = {}
        for level in LEVELS:
            if features[level].ndim != 3 or features[level].shape[1] < 1:
                raise ValueError("Text features must have shape [classes,paraphrases,CLIP width]")
            projected = self.text_projections[level](features[level])
            output[level] = unit_embedding(projected.mean(dim=1))
        return output

    def forward(
        self,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None,
        text_features: dict[str, torch.Tensor] | None = None,
    ) -> dict:
        tokens = self.skeleton(coordinates, valid)
        fine, coarse = tokens["fine_pool"], tokens["coarse_pool"]
        queries = {
            "fine": unit_embedding(self.query_fine(fine)),
            "coarse": unit_embedding(self.query_coarse(coarse)),
        }
        streams = [self.stream_fine(fine), self.stream_coarse(coarse)]
        if self.use_visual:
            if visual_features is not None and pixel_values is not None:
                raise ValueError("Supply pixels or visual features, not both")
            if visual_features is None:
                if self.clip_encoder is None or pixel_values is None:
                    raise ValueError("Visual encoding requires local CLIP and rendered pixels")
                visual_features = self.clip_encoder.encode_images(pixel_values)
            if visual_features.ndim != 3:
                raise ValueError("Visual features must have shape [batch,frames,CLIP width]")
            streams.append(unit_embedding(self.visual_projection(visual_features.mean(dim=1))))
        stacked = torch.stack(streams, dim=1)
        weights = self.gate(stacked).softmax(dim=1)
        queries["global"] = unit_embedding(self.global_projection((stacked * weights).sum(dim=1)))
        result = {
            "queries": queries,
            "logits": self.classifier(queries["global"]),
            "gate_weights": weights.squeeze(-1),
            "tokens": tokens,
        }
        if text_features is not None:
            result["prototypes"] = self.project_text(text_features)
        return result
