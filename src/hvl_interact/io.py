"""Atomic artifacts and deterministic content fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def clip_fingerprint(directory: str | Path) -> str:
    directory = Path(directory)
    files = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix in (".json", ".txt", ".safetensors", ".bin")
    )
    if not any(p.suffix in (".safetensors", ".bin") for p in files):
        raise FileNotFoundError(f"No CLIP model weights found in {directory}")
    return object_sha256({p.relative_to(directory).as_posix(): file_sha256(p) for p in files})


def contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes its root: {relative}")
    return path
