import hashlib
import json

import numpy as np
import pytest
import torch
import yaml


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def experiment(tmp_path):
    from transformers import (
        CLIPConfig,
        CLIPImageProcessor,
        CLIPModel,
        CLIPProcessor,
        CLIPTextConfig,
        CLIPTokenizer,
        CLIPVisionConfig,
    )
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    root = tmp_path
    clip_dir = root / "weights"
    clip_dir.mkdir()
    vocab = {c: i for i, c in enumerate(bytes_to_unicode().values())}
    vocab.update({"<|startoftext|>": 256, "<|endoftext|>": 257})
    (clip_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (clip_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = CLIPTokenizer(str(clip_dir / "vocab.json"), str(clip_dir / "merges.txt"))
    image_processor = CLIPImageProcessor(
        size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
    )
    CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer).save_pretrained(clip_dir)
    text = CLIPTextConfig(
        vocab_size=258,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=77,
        bos_token_id=256,
        eos_token_id=257,
        pad_token_id=257,
    )
    vision = CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=32,
        patch_size=16,
    )
    torch.manual_seed(19)
    model = CLIPModel(CLIPConfig.from_text_vision_configs(text, vision, projection_dim=16))
    model.save_pretrained(clip_dir, safe_serialization=True)
    data = root / "data"
    data.mkdir()
    classes = list(range(50, 61)) + list(range(106, 121))
    samples = []
    for c in classes:
        for setup, subject, rep in ((2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 4, 1), (1, 3, 1)):
            sid = f"S{setup:03}C001P{subject:03}R{rep:03}A{c:03}"
            path = data / f"{sid}.npz"
            x = np.random.default_rng(c).normal(0, 0.1, (2, 4, 25, 3)).astype(np.float32)
            x[1, :, :, 0] += 1
            x[:, :, 20, 1] += 1
            np.savez_compressed(path, coordinates=x, valid=np.ones(x.shape[:-1], bool))
            samples.append(
                {
                    "sample_id": sid,
                    "class_id": c,
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    (data / "index.json").write_text(
        json.dumps({"schema_version": 1, "samples": samples}), encoding="utf-8"
    )
    prompts = {
        "schema_version": 1,
        "classes": [
            dict(
                id=c,
                name=f"action {c}",
                fine=["arm motion"],
                coarse=["body motion"],
                **{"global": ["two people interact"]},
            )
            for c in classes
        ],
    }
    (root / "prompts.json").write_text(json.dumps(prompts), encoding="utf-8")
    config = {
        "root": ".",
        "mode": "supervised",
        "protocol": "xsub",
        "seed": 19,
        "paths": {
            "data": "data",
            "split": "split.json",
            "clip": "weights",
            "prompts": "prompts.json",
            "output": "run",
        },
        "data": {"frames": 4},
        "render": {"size": 32, "frames": 2},
        "model": {
            "width": 8,
            "channels": [8],
            "embed_dim": 8,
            "heads": 2,
            "temporal_layers": 1,
            "spatial_layers": 1,
            "dropout": 0.1,
        },
        "train": {
            "epochs": 2,
            "warmup_epochs": 0,
            "batch_size": 32,
            "num_workers": 0,
            "device": "cpu",
            "precision": "fp32",
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return root, config, config_path
