import json

import pytest
import torch
import yaml


def test_local_clip_is_frozen_in_train_mode(experiment):
    from hvl_interact.models.clip import FrozenCLIP

    root, _, _ = experiment
    encoder = FrozenCLIP(root / "weights")
    encoder.train()
    assert not encoder.training and not encoder.model.training
    assert not any(p.requires_grad for p in encoder.parameters())
    images = encoder.encode_images(torch.randn(2, 2, 3, 32, 32))
    assert images.shape == (2, 2, 16) and not images.requires_grad
    with pytest.raises(FileNotFoundError):
        FrozenCLIP(root / "nonexistent")


def test_training_resume_evaluation_and_prediction(experiment):
    from hvl_interact.cli import main

    root, config, config_path = experiment
    main(
        [
            "split",
            "--index",
            str(root / "data/index.json"),
            "--output",
            str(root / "split.json"),
            "--mode",
            "supervised",
            "--protocol",
            "xsub",
        ]
    )
    main(["train", "--config", str(config_path), "--stop-after-epoch", "1"])
    first = torch.load(root / "run/last.pt", weights_only=True)
    assert first["epoch"] == 1
    main(["train", "--config", str(config_path), "--resume", str(root / "run/last.pt")])
    checkpoint = torch.load(root / "run/last.pt", weights_only=True)
    assert checkpoint["epoch"] == 2
    assert not any(k.startswith("clip_encoder.") for k in checkpoint["model"])
    main(["evaluate", "--config", str(config_path), "--checkpoint", str(root / "run/best.pt")])
    metrics = json.loads((root / "run/evaluation/test/metrics.json").read_text())
    records = [
        json.loads(s)
        for s in (root / "run/evaluation/test/predictions.jsonl").read_text().splitlines()
    ]
    assert metrics["count"] == 26 == len(records)
    assert all(len(r["scores"]) == 26 for r in records)
    assert len({r["sample_id"] for r in records}) == 26
    main(
        [
            "predict",
            "--checkpoint",
            str(root / "run/best.pt"),
            "--clip",
            str(root / "weights"),
            "--prompts",
            str(root / "prompts.json"),
            "--input",
            str(next((root / "data").glob("*.npz"))),
            "--output",
            str(root / "prediction.json"),
            "--device",
            "cpu",
        ]
    )
    prediction = json.loads((root / "prediction.json").read_text())
    assert prediction["predicted_class_id"] in prediction["candidate_class_ids"]
    config["paths"]["output"] = "continuous"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    main(["train", "--config", str(config_path)])
    continuous = torch.load(root / "continuous/last.pt", weights_only=True)
    assert continuous["global_step"] == checkpoint["global_step"]
    for key in checkpoint["model"]:
        torch.testing.assert_close(
            checkpoint["model"][key], continuous["model"][key], rtol=0, atol=0
        )


def test_zero_shot_boundaries_and_required_protocol(experiment):
    from hvl_interact.cli import main
    from hvl_interact.config import load_config
    from hvl_interact.data.protocols import validate_split

    root, config, config_path = experiment
    main(
        [
            "split",
            "--index",
            str(root / "data/index.json"),
            "--output",
            str(root / "split.json"),
            "--mode",
            "zeroshot",
            "--protocol",
            "xset",
            "--unseen",
            "50",
            "51",
            "52",
            "53",
            "54",
            "55",
        ]
    )
    config.update(mode="zeroshot", protocol=None)
    config["train"]["epochs"] = 1
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        load_config(config_path)
    main(["train", "--config", str(config_path), "--protocol", "xset"])
    main(
        [
            "evaluate",
            "--config",
            str(config_path),
            "--protocol",
            "xset",
            "--checkpoint",
            str(root / "run/best.pt"),
        ]
    )
    checkpoint = torch.load(root / "run/best.pt", weights_only=True)
    assert len(checkpoint["class_ids"]) == 20
    records = [
        json.loads(s)
        for s in (root / "run/evaluation/test/predictions.jsonl").read_text().splitlines()
    ]
    assert len(records) == 6
    assert all(set(r["candidate_class_ids"]) == set(range(50, 56)) for r in records)
    split = json.loads((root / "split.json").read_text())
    index = json.loads((root / "data/index.json").read_text())
    split["train"][0] = split["test"][0]
    with pytest.raises(ValueError, match="overlap|leakage|protocol"):
        validate_split(split, index)


def test_config_rejects_unknown_keys(experiment):
    from hvl_interact.config import load_config

    _, config, config_path = experiment
    config["train"]["lerning_rate"] = 0.1
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="lerning_rate"):
        load_config(config_path)
