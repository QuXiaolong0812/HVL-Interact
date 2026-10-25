"""Frozen CLIP loaded exclusively from a local Transformers checkpoint directory."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor

from hvl_interact.data.layout import INTERACTION_IDS, LEVELS
from hvl_interact.io import read_json


def load_prompt_bank(path: str | Path) -> dict[int, dict]:
    document = read_json(path)
    if document.get("schema_version") != 1 or not isinstance(document.get("classes"), list):
        raise ValueError("Unsupported prompt-bank schema")
    bank = {entry["id"]: entry for entry in document["classes"]}
    if len(bank) != len(document["classes"]) or set(bank) != set(INTERACTION_IDS):
        raise ValueError("Prompt bank must contain each of the 26 interaction classes exactly once")
    for cid, entry in bank.items():
        if not isinstance(entry.get("name"), str) or not entry["name"].strip():
            raise ValueError(f"Missing class name: A{cid:03}")
        for h in LEVELS:
            prompts = entry.get(h)
            if (
                not isinstance(prompts, list)
                or not prompts
                or any(not isinstance(p, str) or not p.strip() for p in prompts)
            ):
                raise ValueError(f"Invalid {h} descriptions for A{cid:03}")
    for level in LEVELS:
        if len({len(row[level]) for row in bank.values()}) != 1:
            raise ValueError(f"All classes need the same paraphrase count at level {level}")
    return bank


class FrozenCLIP(nn.Module):
    def __init__(self, directory: str | Path, image_batch_size: int = 64):
        super().__init__()
        directory = Path(directory)
        if not directory.is_dir() or not (directory / "config.json").is_file():
            raise FileNotFoundError(f"Local CLIP checkpoint directory is incomplete: {directory}")
        self.model = CLIPModel.from_pretrained(directory, local_files_only=True)
        self.processor = CLIPProcessor.from_pretrained(directory, local_files_only=True)
        self.model.requires_grad_(False)
        self.image_batch_size = image_batch_size
        self.output_dim = self.model.config.projection_dim
        self.image_size = self.model.config.vision_config.image_size
        self.train(False)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode_images(self, pixels: torch.Tensor) -> torch.Tensor:
        if (
            pixels.ndim != 5
            or pixels.shape[2:] != (3, self.image_size, self.image_size)
            or pixels.shape[1] < 1
        ):
            raise ValueError(f"CLIP pixels must be [B,N,3,{self.image_size},{self.image_size}]")
        b, n = pixels.shape[:2]
        flattened = pixels.reshape(b * n, *pixels.shape[2:])
        features = [
            self.model.get_image_features(pixel_values=part).float()
            for part in flattened.split(self.image_batch_size)
        ]
        return torch.cat(features).reshape(b, n, self.output_dim)

    @torch.no_grad()
    def encode_descriptions(
        self, bank: dict[int, dict], class_ids: list[int]
    ) -> dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        output = {}
        for level in LEVELS:
            descriptions = [text for cid in class_ids for text in bank[cid][level]]
            encoded = []
            for start in range(0, len(descriptions), 32):
                tokens = self.processor.tokenizer(
                    descriptions[start : start + 32],
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                if (
                    tokens["input_ids"].shape[1]
                    > self.model.config.text_config.max_position_embeddings
                ):
                    raise ValueError(f"A {level} description exceeds the CLIP context length")
                tokens = {k: v.to(device) for k, v in tokens.items()}
                encoded.append(self.model.get_text_features(**tokens).float())
            count = len(bank[class_ids[0]][level])
            output[level] = torch.cat(encoded).reshape(len(class_ids), count, self.output_dim)
        return output
