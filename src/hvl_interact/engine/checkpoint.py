"""Epoch-boundary checkpoints with external encoder identity and random states."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def capture_rng() -> dict:
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), *state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    name, keys, position, has_gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def network_state(model) -> dict:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("clip_encoder.")
    }


def restore_network(model, state: dict) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    required_missing = [key for key in missing if not key.startswith("clip_encoder.")]
    if required_missing or unexpected:
        raise ValueError(
            f"Checkpoint architecture mismatch: missing={required_missing}, unexpected={unexpected}"
        )


def save_checkpoint(path: str | Path, checkpoint: dict) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    return checkpoint


def validate_resume_history(directory: str | Path, checkpoint_epoch: int) -> None:
    lines = (Path(directory) / "history.jsonl").read_text(encoding="utf-8").splitlines()
    epochs = [json.loads(line)["epoch"] for line in lines]
    if epochs != list(range(1, checkpoint_epoch + 1)):
        raise ValueError("Resume requires the latest completed epoch and a contiguous run history")
