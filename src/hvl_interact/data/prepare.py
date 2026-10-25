"""Convert local NTU skeleton files into checksummed sequence archives."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from hvl_interact.data.layout import INTERACTION_IDS
from hvl_interact.data.ntu import parse_sample_id, read_skeleton
from hvl_interact.data.preprocess import normalize_shared, trim_empty
from hvl_interact.io import file_sha256, write_json


def prepare(
    skeleton_root: str | Path, output: str | Path, excluded: str | Path, overwrite: bool = False
) -> dict:
    skeleton_root, output, excluded = Path(skeleton_root), Path(output), Path(excluded)
    if not skeleton_root.is_dir() or not excluded.is_file():
        raise FileNotFoundError("Skeleton root and official missing-skeleton list must exist")
    if (output / "index.json").exists() and not overwrite:
        raise FileExistsError(f"Dataset index already exists: {output / 'index.json'}")
    ignored = set(
        re.findall(r"S\d{3}C\d{3}P\d{3}R\d{3}A\d{3}", excluded.read_text(encoding="utf-8-sig"))
    )
    if not ignored:
        raise ValueError("Missing-skeleton list contains no NTU sample identifiers")
    paths = sorted(skeleton_root.rglob("*.skeleton"))
    if not paths:
        raise ValueError(f"No .skeleton files found under {skeleton_root}")
    records, rejected, visited = [], [], set()
    (output / "samples").mkdir(parents=True, exist_ok=True)
    for path in paths:
        sid = path.stem
        metadata = parse_sample_id(sid)
        if metadata["class_id"] not in INTERACTION_IDS:
            continue
        if sid in visited:
            raise ValueError(f"Duplicate skeleton sample: {sid}")
        visited.add(sid)
        if sid in ignored:
            rejected.append({"sample_id": sid, "reason": "official_missing_skeleton_list"})
            continue
        try:
            x, valid, tracks = read_skeleton(path)
            x, valid = trim_empty(x, valid)
            normalize_shared(x, valid)
        except ValueError as exc:
            rejected.append({"sample_id": sid, "reason": str(exc)})
            continue
        target = output / "samples" / f"{sid}.npz"
        temporary = target.with_name(f".{sid}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, coordinates=x, valid=valid)
        temporary.replace(target)
        records.append(
            {
                "sample_id": sid,
                **metadata,
                "path": target.relative_to(output).as_posix(),
                "sha256": file_sha256(target),
                "source_sha256": file_sha256(path),
                "body_ids": tracks,
                "frames": x.shape[1],
                "valid_joint_fraction": float(valid.mean()),
            }
        )
    if not records:
        raise ValueError("No valid dual-person interaction sequences remain after preparation")
    index = dict(
        schema_version=1,
        dataset="NTU RGB+D 120 interaction subset",
        exclusion_sha256=file_sha256(excluded),
        samples=records,
        rejected=rejected,
        preparation={
            "track_ranking": "valid_count,motion_variance,body_id",
            "tracking_states": [1, 2],
            "minimum_tracks": 2,
        },
    )
    write_json(output / "index.json", index)
    return {
        "accepted": len(records),
        "rejected": len(rejected),
        "index": str(output / "index.json"),
    }
