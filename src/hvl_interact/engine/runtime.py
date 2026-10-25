"""Validated experiment inputs, model construction and run identity."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from hvl_interact import __version__
from hvl_interact.config import (
    AugmentationConfig,
    Configuration,
    DataConfig,
    LossConfig,
    ModelConfig,
    Paths,
    RenderConfig,
    TrainConfig,
    construct,
    validate_config,
)
from hvl_interact.data.protocols import validate_split
from hvl_interact.io import clip_fingerprint, contained_path, file_sha256, object_sha256, read_json
from hvl_interact.models.clip import FrozenCLIP, load_prompt_bank
from hvl_interact.models.network import HVLInteract


def config_from_snapshot(raw: dict) -> Configuration:
    sections = {
        key: construct(cls, raw[key], key)
        for key, cls in (
            ("paths", Paths),
            ("data", DataConfig),
            ("augmentation", AugmentationConfig),
            ("render", RenderConfig),
            ("model", ModelConfig),
            ("loss", LossConfig),
            ("train", TrainConfig),
        )
    }
    config = Configuration(raw["root"], raw["mode"], raw["protocol"], raw["seed"], **sections)
    validate_config(config)
    return config


def validate_inputs(config: Configuration) -> tuple[dict, dict, dict, dict]:
    if config.paths.split is None:
        raise ValueError("An explicit split manifest path is required")
    root = Path(config.paths.data)
    index_path, split_path = root / "index.json", Path(config.paths.split)
    index, split = read_json(index_path), read_json(split_path)
    if index.get("schema_version") != 1:
        raise ValueError("Unsupported dataset index schema")
    validate_split(split, index)
    if (split["mode"], split["protocol"]) != (config.mode, config.protocol):
        raise ValueError("Configuration mode/protocol disagrees with split manifest")
    if split.get("dataset_index_sha256") != file_sha256(index_path):
        raise ValueError("Split manifest does not match the dataset index fingerprint")
    for record in index["samples"]:
        path = contained_path(root, record["path"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Dataset content changed: {record['sample_id']}")
    if config.mode == "zeroshot" and config.loss.beta <= 0:
        raise ValueError("Zero-shot training requires a positive semantic alignment weight")
    bank = load_prompt_bank(config.paths.prompts)
    fingerprints = {
        "dataset": file_sha256(index_path),
        "split": file_sha256(split_path),
        "prompts": file_sha256(config.paths.prompts),
        "clip": clip_fingerprint(config.paths.clip),
    }
    return index, split, bank, fingerprints


def create_model(config: Configuration, class_ids: list[int], device: torch.device) -> HVLInteract:
    clip = FrozenCLIP(config.paths.clip)
    if config.model.use_visual and config.render.size != clip.image_size:
        raise ValueError("Rendering size must equal the local CLIP image size")
    options = asdict(config.model)
    return HVLInteract(
        len(class_ids),
        clip_dim=clip.output_dim,
        max_frames=config.data.frames,
        clip_encoder=clip,
        **options,
    ).to(device)


def signature(config: Configuration, fingerprints: dict, world_size: int) -> str:
    values = config.as_dict()
    values.pop("root")
    values.pop("paths")
    return object_sha256(
        {"configuration": values, "fingerprints": fingerprints, "world_size": world_size}
    )


def parameter_counts(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "resident": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "checkpoint": sum(
            v.numel() for k, v in model.state_dict().items() if not k.startswith("clip_encoder.")
        ),
    }


def run_metadata(config: Configuration, split: dict, fingerprints: dict, model, context) -> dict:
    try:
        revision = subprocess.run(
            ["git", "-C", config.root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": __version__,
        "configuration": config.as_dict(),
        "fingerprints": fingerprints,
        "signature": signature(config, fingerprints, context.world_size),
        "class_ids": split["seen_classes"],
        "unseen_class_ids": split["unseen_classes"],
        "parameters": parameter_counts(model),
        "git_revision": revision,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "world_size": context.world_size,
            "device": str(context.device),
        },
    }
