"""Parse the official text skeleton format without changing tracking identities."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SAMPLE_PATTERN = re.compile(r"^S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})$")


def parse_sample_id(sample_id: str) -> dict[str, int]:
    match = SAMPLE_PATTERN.fullmatch(Path(sample_id).stem)
    if match is None:
        raise ValueError(f"Invalid NTU sample ID: {sample_id}")
    setup, camera, subject, replication, action = map(int, match.groups())
    if not (
        1 <= setup <= 32
        and 1 <= camera <= 3
        and 1 <= subject <= 106
        and 1 <= replication <= 2
        and 1 <= action <= 120
    ):
        raise ValueError(f"Out-of-range NTU sample ID: {sample_id}")
    return dict(
        setup=setup, camera=camera, subject=subject, replication=replication, class_id=action
    )


def read_skeleton(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return [2,T,25,3] coordinates, validity and persistent track IDs.

    Ranking uses valid joint count, motion variance and tracking ID, in that order.
    Inferred joints (tracking state 1) are retained; state 0 is missing.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    cursor = 0

    def take() -> list[str]:
        nonlocal cursor
        if cursor >= len(lines):
            raise ValueError(f"Truncated skeleton at line {cursor + 1}: {path.name}")
        fields = lines[cursor].split()
        cursor += 1
        if not fields:
            raise ValueError(f"Empty skeleton line {cursor}: {path.name}")
        return fields

    try:
        frames = int(take()[0])
        if not 1 <= frames <= 10000:
            raise ValueError("Invalid skeleton frame count")
        tracks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for t in range(frames):
            count = int(take()[0])
            if not 0 <= count <= 20:
                raise ValueError("Invalid skeleton body count")
            frame_ids = set()
            for _ in range(count):
                info = take()
                if len(info) != 10:
                    raise ValueError("A skeleton body header must contain 10 fields")
                body_id = info[0]
                if body_id in frame_ids:
                    raise ValueError("Duplicate skeleton body ID within a frame")
                frame_ids.add(body_id)
                if body_id not in tracks:
                    tracks[body_id] = (
                        np.zeros((frames, 25, 3), np.float32),
                        np.zeros((frames, 25), bool),
                    )
                coordinates, valid = tracks[body_id]
                if int(take()[0]) != 25:
                    raise ValueError("A skeleton body must contain 25 joints")
                for v in range(25):
                    fields = take()
                    if len(fields) != 12:
                        raise ValueError("A skeleton joint line must contain 12 fields")
                    xyz = np.asarray(fields[:3], dtype=np.float32)
                    state = int(fields[11])
                    if state not in (0, 1, 2):
                        raise ValueError("Invalid skeleton joint tracking state")
                    if state > 0 and np.isfinite(xyz).all():
                        coordinates[t, v] = xyz
                        valid[t, v] = True
        if any(line.strip() for line in lines[cursor:]):
            raise ValueError("Unexpected trailing skeleton lines")
        active = [k for k, (_, mask) in tracks.items() if mask.any()]

        def ranking(key: str) -> tuple:
            x, mask = tracks[key]
            return (-int(mask.sum()), -float(x[mask].var(axis=0).sum()), key)

        ids = sorted(active, key=ranking)[:2]
        if len(ids) != 2:
            raise ValueError("Dual-person skeleton requires two nonempty body tracks")
        return (np.stack([tracks[k][0] for k in ids]), np.stack([tracks[k][1] for k in ids]), ids)
    except (ValueError, IndexError, OverflowError) as exc:
        raise ValueError(f"Invalid skeleton {path.name}, line {cursor}: {exc}") from exc
