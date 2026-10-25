"""Indexed coordinate sequences with deterministic epoch-specific augmentation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from hvl_interact.config import Configuration
from hvl_interact.data.preprocess import (
    augment,
    normalize_shared,
    resample,
    trim_empty,
    validate_arrays,
)
from hvl_interact.io import contained_path
from hvl_interact.rendering import SkeletonRenderer


def load_coordinates(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        x = np.asarray(archive["coordinates"], dtype=np.float32)
        valid = np.asarray(archive["valid"])
    validate_arrays(x, valid)
    return x, valid


def transform_coordinates(
    x: np.ndarray, valid: np.ndarray, config: Configuration, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    x, valid = trim_empty(x, valid)
    positions = None
    if rng is not None:
        duration = x.shape[1] - 1
        length = duration * rng.uniform(config.data.crop_min, 1)
        start = rng.uniform(0, max(0.0, duration - length))
        positions = np.linspace(start, start + length, config.data.frames)
        jitter = config.data.temporal_jitter * length / max(config.data.frames - 1, 1)
        positions = np.sort(
            np.clip(positions + rng.uniform(-jitter, jitter, len(positions)), 0, duration)
        )
    x, valid = resample(x, valid, config.data.frames, positions)
    x = normalize_shared(x, valid, config.data.root_joint, config.data.scale_joint)
    if rng is not None:
        x, valid = augment(x, valid, rng, **asdict(config.augmentation))
    return x, valid


class InteractionDataset(Dataset):
    def __init__(self, index: dict, sample_ids: list[str], config: Configuration, training: bool):
        self.root = Path(config.paths.data)
        by_id = {r["sample_id"]: r for r in index["samples"]}
        self.records = [by_id[s] for s in sample_ids]
        self.config, self.training, self.epoch = config, training, 0
        self.renderer = (
            SkeletonRenderer(**asdict(config.render)) if config.model.use_visual else None
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        x, valid = load_coordinates(contained_path(self.root, record["path"]))
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, self.epoch, index]))
        x, valid = transform_coordinates(x, valid, self.config, rng if self.training else None)
        output = {
            "coordinates": torch.from_numpy(x),
            "valid": torch.from_numpy(valid),
            "label": record["class_id"],
            "sample_id": record["sample_id"],
        }
        if self.renderer is not None:
            output["pixel_values"] = torch.from_numpy(self.renderer(x, valid))
        return output
