"""Command-line entry points for dataset preparation and experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hvl_interact import __version__


def action_id(value: str) -> int:
    try:
        return int(value.upper().removeprefix("A"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid NTU action ID: {value}") from exc


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="hvl-interact", description="HVL-Interact interaction recognition"
    )
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Convert local NTU skeleton files")
    preparation.add_argument("--skeleton-root", required=True)
    preparation.add_argument("--excluded", required=True, help="Official missing-skeleton list")
    preparation.add_argument("--output", required=True)
    preparation.add_argument("--overwrite", action="store_true")
    split = commands.add_parser("split", help="Create an explicit experiment manifest")
    split.add_argument("--index", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--mode", choices=("supervised", "zeroshot"), required=True)
    split.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    split.add_argument("--unseen", nargs=6, type=action_id)
    split.add_argument("--validation-fraction", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=3407)
    split.add_argument("--overwrite", action="store_true")
    for name in ("train", "evaluate", "doctor"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--protocol", choices=("xsub", "xset"))
        sub.add_argument("--split")
        sub.add_argument("--output", help="Run output directory")
        if name == "train":
            sub.add_argument("--resume")
            sub.add_argument(
                "--stop-after-epoch",
                type=int,
                help="End at this epoch while retaining the configured LR schedule",
            )
        elif name == "evaluate":
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--partition", choices=("validation", "test"), default="test")
            sub.add_argument("--head", choices=("classifier", "semantic"))
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--clip", required=True)
    prediction.add_argument("--prompts", required=True)
    prediction.add_argument("--input", required=True, help="NPZ with coordinates and valid arrays")
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--device", default="auto")
    return result


def main(argv: list[str] | None = None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        from hvl_interact.data.prepare import prepare

        value = prepare(args.skeleton_root, args.output, args.excluded, args.overwrite)
    elif args.command == "split":
        from hvl_interact.data.protocols import build_split
        from hvl_interact.io import file_sha256, read_json, write_json

        if Path(args.output).exists() and not args.overwrite:
            raise FileExistsError(f"Split manifest already exists: {args.output}")
        value = build_split(
            read_json(args.index),
            args.mode,
            args.protocol,
            args.unseen,
            args.validation_fraction,
            args.seed,
        )
        value["dataset_index_sha256"] = file_sha256(args.index)
        write_json(args.output, value)
        value = {key: len(value[key]) for key in ("train", "validation", "test")}
    elif args.command == "predict":
        from hvl_interact.engine.workflows import predict

        value = predict(
            args.checkpoint, args.clip, args.prompts, args.input, args.output, args.device
        )
    else:
        from hvl_interact.config import load_config

        config = load_config(args.config, args.protocol, args.split, args.output)
        if args.command == "train":
            from hvl_interact.engine.training import train

            value = train(config, args.resume, args.stop_after_epoch)
        elif args.command == "evaluate":
            from hvl_interact.engine.workflows import evaluate

            value = evaluate(config, args.checkpoint, args.partition, args.head)
        else:
            import torch

            from hvl_interact.engine.runtime import create_model, parameter_counts, validate_inputs
            from hvl_interact.models.clip import load_prompt_bank

            checks = {
                "dataset_index": Path(config.paths.data, "index.json").is_file(),
                "split_manifest": config.paths.split is not None
                and Path(config.paths.split).is_file(),
                "clip_configuration": Path(config.paths.clip, "config.json").is_file(),
                "prompt_bank": Path(config.paths.prompts).is_file(),
            }
            value = {"ready": all(checks.values()), "checks": checks}
            if value["ready"]:
                _, split, _, fingerprints = validate_inputs(config)
                model = create_model(config, split["seen_classes"], torch.device("cpu"))
                bank = load_prompt_bank(config.paths.prompts)
                model.clip_encoder.encode_descriptions(bank, split["seen_classes"])
                value.update(parameters=parameter_counts(model), fingerprints=fingerprints)
    print(json.dumps(value, indent=2, allow_nan=False), flush=True)
    return value


def entrypoint() -> None:
    try:
        value = main()
        if isinstance(value, dict) and value.get("ready") is False:
            raise SystemExit(1)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"hvl-interact: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
