"""Official evaluation partitions and explicit seen/unseen class boundaries."""

from __future__ import annotations

import hashlib
import math

from hvl_interact.data.layout import INTERACTION_IDS, TRAIN_SUBJECTS
from hvl_interact.data.ntu import parse_sample_id


def official_training(sample_id: str, protocol: str) -> bool:
    item = parse_sample_id(sample_id)
    if protocol == "xsub":
        return item["subject"] in TRAIN_SUBJECTS
    if protocol == "xset":
        return item["setup"] % 2 == 0
    raise ValueError("protocol must be xsub or xset")


def class_partition(mode: str, unseen: list[int] | None = None) -> tuple[list[int], list[int]]:
    if mode == "supervised":
        if unseen:
            raise ValueError("Supervised experiments cannot declare unseen classes")
        return list(INTERACTION_IDS), []
    if mode != "zeroshot":
        raise ValueError("mode must be supervised or zeroshot")
    if (
        unseen is None
        or len(unseen) != 6
        or len(set(unseen)) != 6
        or not set(unseen) <= set(INTERACTION_IDS)
    ):
        raise ValueError("Zero-shot experiments require six distinct NTU interaction class IDs")
    return [c for c in INTERACTION_IDS if c not in unseen], sorted(unseen)


def validation_group(sample_id: str) -> tuple[int, int, int]:
    item = parse_sample_id(sample_id)
    return item["setup"], item["subject"], item["replication"]


def build_split(
    index: dict,
    mode: str,
    protocol: str,
    unseen: list[int] | None = None,
    validation_fraction: float = 0.15,
    seed: int = 3407,
) -> dict:
    seen, novel = class_partition(mode, unseen)
    if protocol not in ("xsub", "xset") or not 0 < validation_fraction < 1:
        raise ValueError("Explicit protocol and validation_fraction in (0,1) are required")
    records = index["samples"]
    ids = [r["sample_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs in dataset index")
    for r in records:
        if parse_sample_id(r["sample_id"])["class_id"] != r["class_id"]:
            raise ValueError("Dataset index class does not match sample ID")
    training = [
        r for r in records if r["class_id"] in seen and official_training(r["sample_id"], protocol)
    ]
    groups = sorted(
        {validation_group(r["sample_id"]) for r in training},
        key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(),
    )
    if len(groups) < 2:
        raise ValueError("At least two training capture groups are required for validation")
    nval = min(len(groups) - 1, max(1, math.ceil(len(groups) * validation_fraction)))
    heldout = set(groups[:nval])
    train = [r["sample_id"] for r in training if validation_group(r["sample_id"]) not in heldout]
    val = [r["sample_id"] for r in training if validation_group(r["sample_id"]) in heldout]
    candidates = novel if mode == "zeroshot" else seen
    test = [
        r["sample_id"]
        for r in records
        if r["class_id"] in candidates and not official_training(r["sample_id"], protocol)
    ]
    split = dict(
        schema_version=1,
        mode=mode,
        protocol=protocol,
        seed=seed,
        validation_fraction=validation_fraction,
        seen_classes=seen,
        unseen_classes=novel,
        train=sorted(train),
        validation=sorted(val),
        test=sorted(test),
    )
    validate_split(split, index)
    return split


def validate_split(split: dict, index: dict) -> None:
    if split.get("schema_version") != 1:
        raise ValueError("Unsupported split schema")
    seen, unseen = class_partition(split["mode"], split["unseen_classes"])
    if split["seen_classes"] != seen or split["unseen_classes"] != unseen:
        raise ValueError("Inconsistent seen/unseen class partition")
    records = {r["sample_id"]: r for r in index["samples"]}
    if len(records) != len(index["samples"]):
        raise ValueError("Duplicate indexed sample IDs")
    visited: set[str] = set()
    for part in ("train", "validation", "test"):
        ids = split[part]
        if not ids or len(ids) != len(set(ids)) or set(ids) & visited:
            raise ValueError(f"Empty, overlapping or duplicated {part} partition")
        visited.update(ids)
        expected = set(unseen if part == "test" and split["mode"] == "zeroshot" else seen)
        observed = set()
        for sample_id in ids:
            if sample_id not in records:
                raise ValueError(f"Unknown sample in split: {sample_id}")
            action = parse_sample_id(sample_id)["class_id"]
            if records[sample_id]["class_id"] != action or action not in expected:
                raise ValueError(f"Class leakage or inconsistent class ID: {sample_id}")
            if official_training(sample_id, split["protocol"]) != (part != "test"):
                raise ValueError(f"Official protocol boundary violated: {sample_id}")
            observed.add(action)
        if observed != expected:
            raise ValueError(
                f"{part} does not cover all required classes: {sorted(expected - observed)}"
            )
    train_groups = {validation_group(s) for s in split["train"]}
    if train_groups & {validation_group(s) for s in split["validation"]}:
        raise ValueError("Training and validation share a capture group")
    expected_training = {
        sid
        for sid, row in records.items()
        if row["class_id"] in seen and official_training(sid, split["protocol"])
    }
    expected_test = {
        sid
        for sid, row in records.items()
        if row["class_id"] in (unseen if split["mode"] == "zeroshot" else seen)
        and not official_training(sid, split["protocol"])
    }
    if (
        set(split["train"]) | set(split["validation"]) != expected_training
        or set(split["test"]) != expected_test
    ):
        raise ValueError("Incomplete official partition: eligible samples were omitted")
