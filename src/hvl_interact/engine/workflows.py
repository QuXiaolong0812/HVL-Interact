"""Checkpoint-backed dataset evaluation and label-free single-sequence inference."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from hvl_interact.config import Configuration
from hvl_interact.data.dataset import InteractionDataset, load_coordinates, transform_coordinates
from hvl_interact.engine.checkpoint import load_checkpoint, restore_network
from hvl_interact.engine.distributed import initialize
from hvl_interact.engine.evaluation import evaluate_batches, write_evaluation
from hvl_interact.engine.runtime import config_from_snapshot, create_model, validate_inputs
from hvl_interact.engine.training import make_loader
from hvl_interact.io import clip_fingerprint, file_sha256, write_json
from hvl_interact.models.clip import load_prompt_bank
from hvl_interact.objectives import semantic_scores
from hvl_interact.rendering import SkeletonRenderer


def evaluate(
    config: Configuration, checkpoint_path: str, partition: str = "test", head: str | None = None
) -> dict:
    context = initialize(config.train.device)
    try:
        index, split, bank, fingerprints = validate_inputs(config)
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint["fingerprints"] != fingerprints:
            raise ValueError("Evaluation inputs differ from the checkpoint fingerprints")
        for key in ("mode", "protocol", "data", "render", "model", "loss"):
            if config.as_dict()[key] != checkpoint["config"][key]:
                raise ValueError(f"Evaluation configuration differs from checkpoint: {key}")
        model = create_model(config, checkpoint["class_ids"], context.device)
        restore_network(model, checkpoint["model"])
        zero_test = config.mode == "zeroshot" and partition == "test"
        candidates = split["unseen_classes"] if zero_test else split["seen_classes"]
        head = head or ("semantic" if config.mode == "zeroshot" else "classifier")
        if zero_test and head != "semantic":
            raise ValueError("A seen-class classifier cannot produce unseen labels")
        if head == "semantic" and config.loss.beta <= 0:
            raise ValueError("Semantic evaluation requires a model trained with alignment")
        text = model.clip_encoder.encode_descriptions(bank, candidates)
        dataset = InteractionDataset(index, split[partition], config, False)
        loader = make_loader(dataset, config, context, False)
        metrics, rows = evaluate_batches(
            model, loader, candidates, text, head, config.loss.semantic_weights, context
        )
        if context.primary:
            directory = Path(config.paths.output) / "evaluation" / partition
            write_evaluation(
                directory,
                metrics,
                rows,
                {
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "fingerprints": fingerprints,
                    "protocol": config.protocol,
                    "mode": config.mode,
                    "partition": partition,
                },
            )
        return metrics
    finally:
        context.close()


@torch.no_grad()
def predict(
    checkpoint_path: str,
    clip_path: str,
    prompt_path: str,
    input_path: str,
    output_path: str,
    device: str = "auto",
) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    config = config_from_snapshot(checkpoint["config"])
    if (
        clip_fingerprint(clip_path) != checkpoint["fingerprints"]["clip"]
        or file_sha256(prompt_path) != checkpoint["fingerprints"]["prompts"]
    ):
        raise ValueError("Inference CLIP or prompt bank differs from the checkpoint")
    config.paths.clip, config.paths.prompts = (
        str(Path(clip_path).resolve()),
        str(Path(prompt_path).resolve()),
    )
    context = initialize(device)
    try:
        if context.world_size != 1:
            raise ValueError("Single-sequence prediction runs in one process")
        model = create_model(config, checkpoint["class_ids"], context.device).eval()
        restore_network(model, checkpoint["model"])
        bank = load_prompt_bank(prompt_path)
        x, valid = load_coordinates(input_path)
        x, valid = transform_coordinates(x, valid, config)
        inputs = {
            "coordinates": torch.from_numpy(x).unsqueeze(0).to(context.device),
            "valid": torch.from_numpy(valid).unsqueeze(0).to(context.device),
        }
        if config.model.use_visual:
            pixels = SkeletonRenderer(**asdict(config.render))(x, valid)
            inputs["pixel_values"] = torch.from_numpy(pixels).unsqueeze(0).to(context.device)
        output = model(**inputs)
        if config.mode == "zeroshot":
            candidates = checkpoint["unseen_class_ids"]
            text = model.clip_encoder.encode_descriptions(bank, candidates)
            scores = semantic_scores(
                output["queries"], model.project_text(text), config.loss.semantic_weights
            )
            score_type = "weighted_cosine_similarity"
        else:
            candidates, scores, score_type = (
                checkpoint["class_ids"],
                output["logits"].float(),
                "classifier_logit",
            )
        predicted = candidates[scores.argmax(dim=1).item()]
        result = {
            "predicted_class_id": predicted,
            "predicted_class_name": bank[predicted]["name"],
            "candidate_class_ids": candidates,
            "scores": scores[0].cpu().tolist(),
            "score_type": score_type,
            "gate_weights": output["gate_weights"][0].cpu().tolist(),
            "checkpoint_sha256": file_sha256(checkpoint_path),
        }
        write_json(output_path, result)
        return result
    finally:
        context.close()
