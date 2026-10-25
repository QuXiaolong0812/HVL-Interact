"""AdamW training with seen-only selection, AMP, accumulation and torchrun."""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from hvl_interact.config import Configuration
from hvl_interact.data.dataset import InteractionDataset
from hvl_interact.engine.checkpoint import (
    capture_rng,
    load_checkpoint,
    network_state,
    restore_network,
    restore_rng,
    save_checkpoint,
    seed_everything,
    validate_resume_history,
)
from hvl_interact.engine.distributed import EvaluationSampler, initialize
from hvl_interact.engine.evaluation import batch_inputs, evaluate_batches
from hvl_interact.engine.runtime import create_model, run_metadata, validate_inputs
from hvl_interact.io import read_json, write_json
from hvl_interact.objectives import HierarchicalObjective


def make_loader(dataset, config: Configuration, context, training: bool):
    if training and context.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            context.world_size,
            context.rank,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
        )
    elif training:
        sampler = None
    else:
        sampler = EvaluationSampler(dataset, context.rank, context.world_size)
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        sampler=sampler,
        shuffle=training and sampler is None,
        num_workers=config.train.num_workers,
        pin_memory=context.device.type == "cuda",
        persistent_workers=False,
        drop_last=False,
    )


def learning_rate(step: int, total: int, warmup: int, initial: float, minimum: float) -> float:
    if step < warmup:
        return initial * (step + 1) / max(1, warmup)
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup - 1)))
    return minimum + 0.5 * (initial - minimum) * (1 + math.cos(math.pi * progress))


def accumulation_weight(
    batch_index: int,
    accumulation: int,
    loader_batches: int,
    batch_size: int,
    sampler_length: int,
    actual_batch_size: int,
) -> float:
    window_start = (batch_index // accumulation) * accumulation
    microbatches = min(accumulation, loader_batches - window_start)
    window_samples = min(microbatches * batch_size, sampler_length - window_start * batch_size)
    return actual_batch_size / window_samples


def initialize_output(output: Path, resume: str | None, metadata: dict, context) -> None:
    status = [None]
    if context.primary:
        try:
            if resume is None:
                if output.exists() and any(output.iterdir()):
                    raise FileExistsError(f"Run directory is not empty: {output}")
                output.mkdir(parents=True, exist_ok=True)
                write_json(output / "run.json", metadata)
            else:
                if Path(resume).resolve().parent != output.resolve():
                    raise ValueError(
                        "Resume checkpoint must be inside the configured run directory"
                    )
                previous = read_json(output / "run.json")
                if previous["signature"] != metadata["signature"]:
                    raise ValueError("Run metadata does not match this resume experiment")
        except (ValueError, OSError) as exc:
            status[0] = f"{type(exc).__name__}: {exc}"
    if context.world_size > 1:
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise ValueError(status[0])


def train(
    config: Configuration, resume: str | None = None, stop_after_epoch: int | None = None
) -> dict:
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= config.train.epochs:
        raise ValueError("stop_after_epoch must lie between 1 and the configured epoch count")
    context = initialize(config.train.device)
    try:
        return train_experiment(config, context, resume, stop_after_epoch)
    finally:
        context.close()


def train_experiment(
    config: Configuration, context, resume: str | None, stop_after_epoch: int | None
) -> dict:
    seed_everything(config.seed, config.train.deterministic)
    index, split, bank, fingerprints = validate_inputs(config)
    model = create_model(config, split["seen_classes"], context.device)
    metadata = run_metadata(config, split, fingerprints, model, context)
    output = Path(config.paths.output)
    initialize_output(output, resume, metadata, context)
    text = model.clip_encoder.encode_descriptions(bank, split["seen_classes"])
    training_data = InteractionDataset(index, split["train"], config, True)
    validation_data = InteractionDataset(index, split["validation"], config, False)
    loader = make_loader(training_data, config, context, True)
    validation = make_loader(validation_data, config, context, False)
    objective = HierarchicalObjective(
        split["seen_classes"],
        config.loss.beta,
        config.loss.temperature,
        tuple(config.loss.level_weights),
        config.loss.label_smoothing,
    ).to(context.device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    amp = context.device.type == "cuda" and config.train.precision != "fp32"
    dtype = torch.bfloat16 if config.train.precision == "bf16" else torch.float16
    if amp and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16")
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
    start_epoch, global_step, best = 0, 0, -1.0
    if context.world_size > 1:
        distributed = DistributedDataParallel(
            model,
            device_ids=[context.device.index] if context.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=config.loss.beta == 0,
        )
    else:
        distributed = model
    seed_everything(config.seed + context.rank, config.train.deterministic)
    if resume is not None:
        checkpoint = load_checkpoint(resume)
        if checkpoint["signature"] != metadata["signature"]:
            raise ValueError(
                "Resume configuration, inputs or world size differ from the checkpoint"
            )
        restore_network(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, global_step, best = (
            checkpoint["epoch"],
            checkpoint["global_step"],
            checkpoint["best_metric"],
        )
        validate_resume_history(output, start_epoch)
        restore_rng(checkpoint["rng_states"][context.rank])
    steps_per_epoch = math.ceil(len(loader) / config.train.accumulation_steps)
    total_steps = config.train.epochs * steps_per_epoch
    warmup_steps = config.train.warmup_epochs * steps_per_epoch
    end_epoch = min(config.train.epochs, stop_after_epoch or config.train.epochs)
    if end_epoch <= start_epoch:
        raise ValueError("No remaining epochs; stop_after_epoch must exceed the checkpoint epoch")
    for epoch in range(start_epoch, end_epoch):
        training_data.set_epoch(epoch)
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        distributed.train()
        optimizer.zero_grad(set_to_none=True)
        sums = torch.zeros(2, dtype=torch.float64, device=context.device)
        for batch_index, batch in enumerate(loader):
            boundary = (
                batch_index + 1
            ) % config.train.accumulation_steps == 0 or batch_index + 1 == len(loader)
            sync = (
                distributed.no_sync()
                if context.world_size > 1 and not boundary
                else contextlib.nullcontext()
            )
            with sync:
                with torch.autocast(context.device.type, dtype=dtype, enabled=amp):
                    prediction = distributed(
                        **batch_inputs(batch, context.device),
                        text_features=text if config.loss.beta else None,
                    )
                    losses = objective(prediction, batch["label"].to(context.device))
                    weight = accumulation_weight(
                        batch_index,
                        config.train.accumulation_steps,
                        len(loader),
                        config.train.batch_size,
                        len(loader.sampler),
                        len(batch["label"]),
                    )
                    scaled = losses["loss"] * weight
                if not torch.isfinite(scaled):
                    raise FloatingPointError(f"Nonfinite training loss at epoch {epoch + 1}")
                scaler.scale(scaled).backward()
            count = len(batch["label"])
            sums[0] += losses["loss"].detach().double() * count
            sums[1] += count
            if boundary:
                lr = learning_rate(
                    global_step,
                    total_steps,
                    warmup_steps,
                    config.train.learning_rate,
                    config.train.min_learning_rate,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                scaler.unscale_(optimizer)
                if config.train.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.train.grad_clip,
                        error_if_nonfinite=not scaler.is_enabled(),
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        if context.world_size > 1:
            dist.all_reduce(sums)
        head = "semantic" if config.mode == "zeroshot" else "classifier"
        metrics, _ = evaluate_batches(
            model,
            validation,
            split["seen_classes"],
            text,
            head,
            config.loss.semantic_weights,
            context,
        )
        improved = metrics["top1_percent"] > best
        best = max(best, metrics["top1_percent"])
        random_states = [capture_rng()]
        if context.world_size > 1:
            gathered = [None] * context.world_size
            dist.all_gather_object(gathered, random_states[0])
            random_states = gathered
        if context.primary:
            checkpoint = {
                "schema_version": 1,
                "model": network_state(model),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_metric": best,
                "rng_states": random_states,
                "signature": metadata["signature"],
                "class_ids": split["seen_classes"],
                "unseen_class_ids": split["unseen_classes"],
                "config": config.as_dict(),
                "fingerprints": fingerprints,
            }
            save_checkpoint(output / "last.pt", checkpoint)
            if improved:
                save_checkpoint(output / "best.pt", checkpoint)
            log = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "learning_rate": lr,
                "training_loss": (sums[0] / sums[1]).item(),
                "validation_top1_percent": metrics["top1_percent"],
                "validation_head": head,
                "validation_count": metrics["count"],
                "best_validation_top1_percent": best,
            }
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(log, allow_nan=False) + "\n")
            print(json.dumps(log), flush=True)
        if context.world_size > 1:
            dist.barrier()
    return {
        "epochs_completed": end_epoch,
        "best_validation_top1_percent": best,
        "output": str(output),
    }
