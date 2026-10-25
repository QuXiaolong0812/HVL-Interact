"""Validity-aware temporal resampling and shared interaction coordinates."""

from __future__ import annotations

import numpy as np


def validate_arrays(x: np.ndarray, valid: np.ndarray) -> None:
    if x.ndim != 4 or x.shape[0] != 2 or x.shape[2:] != (25, 3) or x.shape[1] < 1:
        raise ValueError(f"Expected coordinates [2,T,25,3], received {x.shape}")
    if valid.shape != x.shape[:-1] or valid.dtype != np.bool_:
        raise ValueError("Validity must be boolean with shape [2,T,25]")
    if not np.isfinite(x[valid]).all() or not valid.any():
        raise ValueError("Coordinates must contain finite valid observations")


def trim_empty(x: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    validate_arrays(x, valid)
    indices = np.flatnonzero(valid.any(axis=(0, 2)))
    sl = slice(indices[0], indices[-1] + 1)
    return x[:, sl].copy(), valid[:, sl].copy()


def resample(
    x: np.ndarray, valid: np.ndarray, frames: int, positions: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    validate_arrays(x, valid)
    if frames < 1:
        raise ValueError("frames must be positive")
    if positions is None:
        positions = np.linspace(0, x.shape[1] - 1, frames)
    positions = np.asarray(positions)
    if (
        positions.shape != (frames,)
        or not np.isfinite(positions).all()
        or (np.diff(positions) < 0).any()
        or positions.min() < 0
        or positions.max() > x.shape[1] - 1
    ):
        raise ValueError("Resampling positions must be finite, monotonic and within the sequence")
    low = np.floor(positions).astype(int)
    high = np.ceil(positions).astype(int)
    alpha = (positions - low)[None, :, None, None]
    mask = valid[:, low] & valid[:, high]
    clean = np.where(valid[..., None], x, 0)
    out = clean[:, low] * (1 - alpha) + clean[:, high] * alpha
    return np.where(mask[..., None], out, 0).astype(np.float32), mask


def normalize_shared(
    x: np.ndarray, valid: np.ndarray, root: int = 0, scale_joint: int = 20, eps: float = 1e-6
) -> np.ndarray:
    validate_arrays(x, valid)
    if not (0 <= root < 25 and 0 <= scale_joint < 25 and root != scale_joint):
        raise ValueError("Distinct valid root and scale joint indices are required")
    clean = np.where(valid[..., None], x, 0).astype(np.float32)
    root_valid = valid[:, :, root]
    count = root_valid.sum(axis=0)
    origin = clean[:, :, root].sum(axis=0) / np.maximum(count[:, None], 1)
    known = np.flatnonzero(count)
    if not known.size:
        raise ValueError("No observed root joints for shared centering")
    for axis in range(3):
        origin[:, axis] = np.interp(np.arange(x.shape[1]), known, origin[known, axis])
    length_mask = valid[:, :, root] & valid[:, :, scale_joint]
    lengths = np.linalg.norm(clean[:, :, scale_joint] - clean[:, :, root], axis=-1)
    lengths = lengths[length_mask & (lengths > eps)]
    if not lengths.size:
        raise ValueError("No positive observed body scale")
    scale = float(np.median(lengths))
    out = (clean - origin[None, :, None]) / (scale + eps)
    return np.where(valid[..., None], out, 0).astype(np.float32)


def augment(
    x: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator,
    rotation_degrees: float,
    noise_std: float,
    mask_probability: float,
    swap_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(rng.uniform(-rotation_degrees, rotation_degrees))
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    x = x @ rotation.T
    if noise_std:
        x += rng.normal(0, noise_std, x.shape).astype(np.float32)
    valid = valid.copy()
    if mask_probability:
        proposed = valid & (rng.random(valid.shape) >= mask_probability)
        if proposed.any():
            valid = proposed
    if rng.random() < swap_probability:
        x, valid = x[::-1].copy(), valid[::-1].copy()
    return np.where(valid[..., None], x, 0).astype(np.float32), valid
