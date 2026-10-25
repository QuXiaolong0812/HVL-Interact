"""Class-prototype alignment, supervised classification and candidate scoring."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hvl_interact.data.layout import LEVELS


class HierarchicalObjective(nn.Module):
    def __init__(
        self,
        class_ids: list[int],
        beta: float = 0.5,
        temperature: float = 0.07,
        level_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        if (
            not class_ids
            or len(class_ids) != len(set(class_ids))
            or min(class_ids) < 1
            or temperature <= 0
            or beta < 0
            or len(level_weights) != 3
            or min(level_weights) < 0
            or not 0 <= label_smoothing < 1
        ):
            raise ValueError("Invalid objective class IDs or hyperparameters")
        mapping = torch.full((max(class_ids) + 1,), -1, dtype=torch.long)
        mapping[class_ids] = torch.arange(len(class_ids))
        self.register_buffer("label_map", mapping)
        self.beta, self.temperature = beta, temperature
        self.level_weights, self.label_smoothing = level_weights, label_smoothing

    def forward(self, output: dict, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        if (
            labels.ndim != 1
            or labels.dtype != torch.long
            or (labels < 0).any()
            or (labels >= len(self.label_map)).any()
        ):
            raise ValueError("Training label is outside the permitted class set")
        target = self.label_map[labels]
        if (target < 0).any():
            raise ValueError("Training label belongs to an excluded class")
        cls = F.cross_entropy(
            output["logits"].float(), target, label_smoothing=self.label_smoothing
        )
        result = {"classification": cls}
        align = cls.new_zeros(())
        if self.beta:
            if "prototypes" not in output:
                raise ValueError("Alignment requires class prototypes")
            for level, weight in zip(LEVELS, self.level_weights, strict=True):
                scores = output["queries"][level].float() @ output["prototypes"][level].float().T
                value = F.cross_entropy(scores / self.temperature, target)
                result[f"alignment_{level}"] = value
                align = align + weight * value
        result["alignment"] = align
        result["loss"] = cls + self.beta * align
        return result


def semantic_scores(
    queries: dict[str, torch.Tensor],
    prototypes: dict[str, torch.Tensor],
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    if len(weights) != 3 or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("Semantic weights must be three nonnegative values with positive sum")
    total = sum(weights)
    return sum(
        (weight / total) * (queries[h].float() @ prototypes[h].float().T)
        for h, weight in zip(LEVELS, weights, strict=True)
    )
