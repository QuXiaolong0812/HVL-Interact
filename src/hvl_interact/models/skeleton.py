"""Masked spatiotemporal features and dual-resolution skeleton tokens."""

from __future__ import annotations

import math

import torch
from torch import nn

from hvl_interact.data.layout import BODY_GROUPS, EDGES


def masked_mean(x: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    weights = valid.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(dim) / weights.sum(dim).clamp_min(1)


class SpatialTemporalBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        adjacency = torch.eye(25)
        for a, b in EDGES:
            adjacency[a, b] = adjacency[b, a] = 1
        self.register_buffer("adjacency", adjacency)
        self.graph = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.temporal = nn.Conv1d(output_dim, output_dim, kernel_size=9, padding=4)
        self.residual = (
            nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        clean = torch.where(valid.unsqueeze(-1), x, 0)
        degree = torch.einsum("vw,bmtw->bmtv", self.adjacency, valid.to(x.dtype))
        neighbors = torch.einsum("vw,bmtwd->bmtvd", self.adjacency, clean)
        h = self.norm(self.graph(neighbors / degree.clamp_min(1).unsqueeze(-1)))
        h = torch.nn.functional.gelu(h)
        h = torch.where(valid.unsqueeze(-1), h, 0)
        b, m, t, v, d = h.shape
        temporal = self.temporal(h.permute(0, 1, 3, 4, 2).reshape(b * m * v, d, t))
        temporal = temporal.reshape(b, m, v, d, t).permute(0, 1, 4, 2, 3)
        h = torch.nn.functional.gelu(self.residual(clean) + self.dropout(temporal))
        return torch.where(valid.unsqueeze(-1), h, 0)


def make_attention(width: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    block = nn.TransformerEncoderLayer(
        width,
        heads,
        dim_feedforward=width * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    encoder = nn.TransformerEncoder(
        block, num_layers=layers, norm=nn.LayerNorm(width), enable_nested_tensor=False
    )
    for module in encoder.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
    return encoder


class TemporalPool(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        layers: int,
        dropout: float,
        max_frames: int,
        chunk_size: int = 128,
    ):
        super().__init__()
        self.attention = make_attention(width, heads, layers, dropout)
        position = torch.arange(max_frames, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, width, 2) * (-math.log(10000) / width))
        encoding = torch.zeros(max_frames, width)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[: width // 2])
        self.register_buffer("position", encoding, persistent=True)
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, m, t, v, d = x.shape
        if t > self.position.shape[0]:
            raise ValueError("Sequence exceeds configured temporal position capacity")
        tokens = x.permute(0, 1, 3, 2, 4).reshape(b * m * v, t, d)
        mask = valid.permute(0, 1, 3, 2).reshape(b * m * v, t)
        tokens = torch.where(mask.unsqueeze(-1), tokens + self.position[:t], 0)
        pieces = []
        for start in range(0, len(tokens), self.chunk_size):
            chunk, observed = (
                tokens[start : start + self.chunk_size],
                mask[start : start + self.chunk_size],
            )
            padding = ~observed.clone()
            padding[~observed.any(dim=1), 0] = False
            attended = self.attention(chunk, src_key_padding_mask=padding)
            attended = torch.where(observed.unsqueeze(-1), attended, 0)
            pieces.append(masked_mean(attended, observed, 1))
        return torch.cat(pieces).reshape(b, m * v, d), mask.any(dim=1).reshape(b, m * v)


class HierarchicalSkeletonEncoder(nn.Module):
    def __init__(
        self,
        width: int = 512,
        channels: tuple[int, ...] = (128, 256, 512),
        heads: int = 8,
        temporal_layers: int = 1,
        spatial_layers: int = 2,
        dropout: float = 0.1,
        max_frames: int = 120,
        groups: dict | None = None,
    ):
        super().__init__()
        if not channels or channels[-1] != width or width % heads:
            raise ValueError("Final channel width must equal width and be divisible by heads")
        if min(temporal_layers, spatial_layers, max_frames, heads, *channels) < 1:
            raise ValueError("Encoder dimensions and layer counts must be positive")
        groups = BODY_GROUPS if groups is None else groups
        if set(groups) != set(BODY_GROUPS):
            raise ValueError("Body groups must be head, torso, left_arm, right_arm and legs")
        groups = {name: groups[name] for name in BODY_GROUPS}
        if len(groups) != 5 or any(not group for group in groups.values()):
            raise ValueError("Exactly five nonempty body groups are required")
        membership = torch.zeros(25, 5)
        for i, joints in enumerate(groups.values()):
            if len(set(joints)) != len(joints) or any(j < 0 or j >= 25 for j in joints):
                raise ValueError("Invalid or repeated joint indices within a body group")
            membership[list(joints), i] = 1 / len(joints)
        if not (membership.sum(dim=1) > 0).all():
            raise ValueError("Body groups must cover every joint")
        self.register_buffer("pooling", membership)
        dims = (3, *channels)
        self.backbone = nn.ModuleList(
            SpatialTemporalBlock(a, b, dropout) for a, b in zip(dims[:-1], dims[1:], strict=True)
        )
        self.fine_temporal = TemporalPool(width, heads, temporal_layers, dropout, max_frames)
        self.coarse_temporal = TemporalPool(width, heads, temporal_layers, dropout, max_frames)
        self.token_identity = nn.Parameter(torch.empty(1, 60, width))
        nn.init.trunc_normal_(self.token_identity, std=0.02)
        self.spatial = make_attention(width, heads, spatial_layers, dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> dict:
        if x.ndim != 5 or x.shape[1] != 2 or x.shape[3:] != (25, 3):
            raise ValueError("Skeleton input must have shape [B,2,T,25,3]")
        if valid.shape != x.shape[:-1] or valid.dtype != torch.bool:
            raise ValueError("Skeleton mask must be boolean [B,2,T,25]")
        h = torch.where(valid.unsqueeze(-1), x, 0)
        for block in self.backbone:
            h = block(h, valid)
        coarse_weight = torch.einsum("bmtv,vg->bmtg", valid.to(h.dtype), self.pooling)
        coarse = torch.einsum("bmtvd,vg->bmtgd", h, self.pooling)
        coarse = coarse / coarse_weight.clamp_min(1e-6).unsqueeze(-1)
        fine, fine_mask = self.fine_temporal(h, valid)
        grouped, coarse_mask = self.coarse_temporal(coarse, coarse_weight > 0)
        observed = torch.cat((fine_mask, coarse_mask), dim=1)
        tokens = torch.cat((fine, grouped), dim=1) + self.token_identity
        tokens = torch.where(observed.unsqueeze(-1), tokens, 0)
        padding = ~observed.clone()
        padding[~observed.any(dim=1), 0] = False
        tokens = self.spatial(tokens, src_key_padding_mask=padding)
        tokens = torch.where(observed.unsqueeze(-1), tokens, 0)
        return {
            "fine": tokens[:, :50],
            "coarse": tokens[:, 50:],
            "fine_mask": fine_mask,
            "coarse_mask": coarse_mask,
            "fine_pool": masked_mean(tokens[:, :50], fine_mask, 1),
            "coarse_pool": masked_mean(tokens[:, 50:], coarse_mask, 1),
        }
