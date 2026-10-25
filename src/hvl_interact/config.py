"""Strict YAML configuration with paths relative to an explicit project root."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Paths:
    data: str = "data/processed/ntu120"
    split: str | None = None
    clip: str = "weights/clip-vit-base-patch16"
    prompts: str = "resources/prompts/ntu120_interactions.json"
    output: str = "runs/experiment"


@dataclass
class DataConfig:
    frames: int = 120
    root_joint: int = 0
    scale_joint: int = 20
    temporal_jitter: float = 0.0
    crop_min: float = 1.0


@dataclass
class AugmentationConfig:
    rotation_degrees: float = 0.0
    noise_std: float = 0.0
    mask_probability: float = 0.0
    swap_probability: float = 0.0


@dataclass
class RenderConfig:
    size: int = 224
    frames: int = 8
    margin: float = 0.08
    line_width: int = 2
    joint_radius: int = 2


@dataclass
class ModelConfig:
    width: int = 512
    channels: list[int] = field(default_factory=lambda: [128, 256, 512])
    embed_dim: int = 512
    heads: int = 8
    temporal_layers: int = 1
    spatial_layers: int = 2
    dropout: float = 0.1
    groups: dict | None = None
    use_visual: bool = True


@dataclass
class LossConfig:
    beta: float = 0.5
    temperature: float = 0.07
    level_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    semantic_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    label_smoothing: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 110
    warmup_epochs: int = 10
    batch_size: int = 16
    accumulation_steps: int = 1
    num_workers: int = 4
    learning_rate: float = 0.001
    min_learning_rate: float = 0.000001
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    precision: str = "fp16"
    device: str = "auto"
    deterministic: bool = True


@dataclass
class Configuration:
    root: str
    mode: str
    protocol: str
    seed: int
    paths: Paths
    data: DataConfig
    augmentation: AugmentationConfig
    render: RenderConfig
    model: ModelConfig
    loss: LossConfig
    train: TrainConfig

    def as_dict(self) -> dict:
        return asdict(self)


def construct(cls, values: dict, section: str):
    if not isinstance(values, dict):
        raise ValueError(f"{section} must be a mapping")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {section} configuration keys: {sorted(unknown)}")
    return cls(**values)


def validate_config(config: Configuration) -> None:
    if config.mode not in ("supervised", "zeroshot"):
        raise ValueError("mode must be supervised or zeroshot")
    if config.protocol not in ("xsub", "xset"):
        raise ValueError("An explicit xsub or xset protocol is required")
    if not isinstance(config.seed, int) or not 0 <= config.seed < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    integers = (
        config.data.frames,
        config.render.frames,
        config.render.size,
        config.model.width,
        config.model.embed_dim,
        config.model.heads,
        config.model.temporal_layers,
        config.model.spatial_layers,
        config.train.epochs,
        config.train.batch_size,
        config.train.accumulation_steps,
    )
    if any(type(v) is not int or v < 1 for v in integers):
        raise ValueError("Dimensions, epochs and batch sizes must be positive integers")
    if not config.model.channels or config.model.channels[-1] != config.model.width:
        raise ValueError("Final backbone channel must equal model.width")
    if config.model.width % config.model.heads:
        raise ValueError("model.width must be divisible by model.heads")
    if not 0 <= config.model.dropout < 1 or not 0 <= config.loss.label_smoothing < 1:
        raise ValueError("dropout and label_smoothing must be in [0,1)")
    if not 0 < config.data.crop_min <= 1 or not 0 <= config.data.temporal_jitter <= 0.5:
        raise ValueError("Invalid temporal crop or jitter")
    if not (
        0 <= config.augmentation.mask_probability < 1
        and 0 <= config.augmentation.swap_probability <= 1
    ):
        raise ValueError("Invalid augmentation probabilities")
    for values in (config.loss.level_weights, config.loss.semantic_weights):
        if (
            len(values) != 3
            or any(not math.isfinite(v) or v < 0 for v in values)
            or sum(values) <= 0
        ):
            raise ValueError(
                "Three finite nonnegative level weights with positive sum are required"
            )
    nonnegative = (
        config.loss.beta,
        config.train.weight_decay,
        config.train.grad_clip,
        config.augmentation.rotation_degrees,
        config.augmentation.noise_std,
    )
    if any(not math.isfinite(v) or v < 0 for v in nonnegative):
        raise ValueError(
            "Loss, regularization and augmentation strengths must be finite and nonnegative"
        )
    if (
        not math.isfinite(config.loss.temperature)
        or config.loss.temperature <= 0
        or not 0 < config.train.min_learning_rate <= config.train.learning_rate
    ):
        raise ValueError("Invalid temperature or learning-rate range")
    if not 0 <= config.train.warmup_epochs < config.train.epochs or config.train.num_workers < 0:
        raise ValueError("Invalid warmup epochs or worker count")
    if config.train.precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")


def load_config(
    path: str | Path,
    protocol: str | None = None,
    split: str | None = None,
    output: str | None = None,
) -> Configuration:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    unknown = set(raw) - {f.name for f in fields(Configuration)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    root = (path.parent / raw.get("root", "..")).resolve()
    sections = {
        name: construct(cls, raw.get(name, {}), name)
        for name, cls in (
            ("paths", Paths),
            ("data", DataConfig),
            ("augmentation", AugmentationConfig),
            ("render", RenderConfig),
            ("model", ModelConfig),
            ("loss", LossConfig),
            ("train", TrainConfig),
        )
    }
    if split is not None:
        sections["paths"].split = str(Path(split).resolve())
    if output is not None:
        sections["paths"].output = str(Path(output).resolve())
    for f in fields(Paths):
        value = getattr(sections["paths"], f.name)
        if value is not None:
            setattr(sections["paths"], f.name, str((root / value).resolve()))
    config = Configuration(
        str(root),
        raw.get("mode", "supervised"),
        protocol or raw.get("protocol"),
        raw.get("seed", 3407),
        **sections,
    )
    validate_config(config)
    return config
