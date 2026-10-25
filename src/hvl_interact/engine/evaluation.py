"""Exact sample-level evaluation with official class IDs and raw class scores."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.distributed as dist

from hvl_interact.io import write_json
from hvl_interact.objectives import semantic_scores


def batch_inputs(batch: dict, device: torch.device) -> dict:
    output = {
        "coordinates": batch["coordinates"].to(device, non_blocking=True),
        "valid": batch["valid"].to(device, non_blocking=True),
    }
    if "pixel_values" in batch:
        output["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True)
    return output


@torch.no_grad()
def evaluate_batches(
    model,
    loader,
    class_ids: list[int],
    text_features: dict,
    head: str,
    weights: list[float],
    context,
) -> tuple[dict, list[dict]]:
    model.eval()
    prototypes = model.project_text(text_features) if head == "semantic" else None
    rows = []
    for batch in loader:
        output = model(**batch_inputs(batch, context.device))
        scores = (
            semantic_scores(output["queries"], prototypes, weights)
            if head == "semantic"
            else output["logits"].float()
        )
        if scores.shape[1] != len(class_ids) or not torch.isfinite(scores).all():
            raise ValueError("Evaluation produced invalid class scores")
        predictions = scores.argmax(dim=1).tolist()
        for sid, true, pred, vector in zip(
            batch["sample_id"],
            batch["label"].tolist(),
            predictions,
            scores.cpu().tolist(),
            strict=True,
        ):
            rows.append(
                {
                    "sample_id": sid,
                    "true_class_id": true,
                    "predicted_class_id": class_ids[pred],
                    "candidate_class_ids": class_ids,
                    "scores": vector,
                }
            )
    if context.world_size > 1:
        shards = [None] * context.world_size
        dist.all_gather_object(shards, rows)
        rows = [row for shard in shards for row in shard]
    rows.sort(key=lambda row: row["sample_id"])
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Evaluation contains duplicated samples")
    if not rows:
        raise ValueError("Cannot evaluate an empty partition")
    confusion = [[0] * len(class_ids) for _ in class_ids]
    mapping = {c: i for i, c in enumerate(class_ids)}
    for row in rows:
        confusion[mapping[row["true_class_id"]]][mapping[row["predicted_class_id"]]] += 1
    per_class = []
    for i, cid in enumerate(class_ids):
        support = sum(confusion[i])
        per_class.append(
            {
                "class_id": cid,
                "support": support,
                "accuracy_percent": 100 * confusion[i][i] / support if support else None,
            }
        )
    available = [r["accuracy_percent"] for r in per_class if r["support"]]
    correct = sum(confusion[i][i] for i in range(len(class_ids)))
    metrics = {
        "count": len(rows),
        "correct": correct,
        "top1_percent": 100 * correct / len(rows),
        "macro_accuracy_percent": sum(available) / len(available),
        "head": head,
        "candidate_class_ids": class_ids,
        "per_class": per_class,
        "confusion_counts": confusion,
    }
    return metrics, rows


def write_evaluation(
    directory: str | Path, metrics: dict, rows: list[dict], provenance: dict
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "metrics.json", {**metrics, "provenance": provenance})
    with (directory / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    with (directory / "confusion.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *metrics["candidate_class_ids"]])
        for cid, row in zip(
            metrics["candidate_class_ids"], metrics["confusion_counts"], strict=True
        ):
            writer.writerow([cid, *row])
