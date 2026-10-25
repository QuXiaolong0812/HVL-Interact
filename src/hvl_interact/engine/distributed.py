"""Torchrun process context and evaluation sampling without padding."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class EvaluationSampler(Sampler[int]):
    def __init__(self, dataset, rank: int = 0, world_size: int = 1):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed sampler rank")
        self.size, self.rank, self.world_size = len(dataset), rank, world_size

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self):
        return len(range(self.rank, self.size, self.world_size))


@dataclass
class ProcessContext:
    device: torch.device
    rank: int = 0
    world_size: int = 1

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


def initialize(device: str = "auto") -> ProcessContext:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        index = local_rank if world > 1 else (selected.index or 0)
        torch.cuda.set_device(index)
        selected = torch.device("cuda", index)
    elif selected.type != "cpu":
        raise ValueError("Supported devices: auto, cpu, cuda, cuda:N")
    if world > 1:
        dist.init_process_group("nccl" if selected.type == "cuda" else "gloo")
    return ProcessContext(selected, rank, world)
