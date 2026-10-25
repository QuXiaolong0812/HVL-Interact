"""Sequence-consistent front projection of the two skeletons."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from hvl_interact.data.layout import EDGES
from hvl_interact.data.preprocess import validate_arrays

CLIP_MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), np.float32)
CLIP_STD = np.asarray((0.26862954, 0.26130258, 0.27577711), np.float32)


class SkeletonRenderer:
    def __init__(
        self,
        size: int = 224,
        frames: int = 8,
        margin: float = 0.08,
        line_width: int = 2,
        joint_radius: int = 2,
    ):
        if size < 16 or frames < 1 or not 0 <= margin < 0.5:
            raise ValueError("Invalid rendering size, frame count or margin")
        if line_width < 1 or joint_radius < 0:
            raise ValueError("Invalid skeleton marker sizes")
        self.size, self.frames, self.margin = size, frames, margin
        self.line_width, self.joint_radius = line_width, joint_radius
        self.colors = ((35, 115, 210), (235, 125, 35))

    def project(self, x: np.ndarray, valid: np.ndarray) -> np.ndarray:
        validate_arrays(x, valid)
        points = x[..., :2].copy()
        observed = points[valid]
        lo, hi = observed.min(axis=0), observed.max(axis=0)
        center = (lo + hi) / 2
        extent = max(float((hi - lo).max()), 1e-6)
        points = (points - center) * ((self.size - 1) * (1 - 2 * self.margin) / extent)
        points[..., 1] *= -1
        return points + (self.size - 1) / 2

    def __call__(self, x: np.ndarray, valid: np.ndarray) -> np.ndarray:
        points = self.project(x, valid)
        indices = np.rint(np.linspace(0, x.shape[1] - 1, self.frames)).astype(int)
        output = []
        for t in indices:
            image = Image.new("RGB", (self.size, self.size), "white")
            draw = ImageDraw.Draw(image)
            for m, color in enumerate(self.colors):
                for a, b in EDGES:
                    if valid[m, t, a] and valid[m, t, b]:
                        draw.line(
                            [tuple(points[m, t, a]), tuple(points[m, t, b])],
                            fill=color,
                            width=self.line_width,
                        )
                r = self.joint_radius
                for v in np.flatnonzero(valid[m, t]):
                    px, py = points[m, t, v]
                    draw.ellipse((px - r, py - r, px + r, py + r), fill=color)
            pixels = np.asarray(image, dtype=np.float32) / 255
            output.append(((pixels - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1))
        return np.stack(output)
