import pytest
import torch
from torch import nn


def test_accumulation_uses_sample_counts_in_short_microbatch():
    from hvl_interact.engine.training import accumulation_weight

    x = torch.tensor([[1.0], [2.0], [9.0]])
    full = nn.Linear(1, 1, bias=False)
    full.weight.data.fill_(0.5)
    (full(x).square().mean()).backward()
    micro = nn.Linear(1, 1, bias=False)
    micro.weight.data.copy_(full.weight.data)
    for i, part in enumerate(x.split(2)):
        weight = accumulation_weight(i, 2, 2, 2, 3, len(part))
        (micro(part).square().mean() * weight).backward()
    torch.testing.assert_close(full.weight.grad, micro.weight.grad)


def test_warmup_cosine_schedule_boundaries():
    from hvl_interact.engine.training import learning_rate

    assert learning_rate(0, 12, 2, 0.1, 0.001) == 0.05
    assert learning_rate(1, 12, 2, 0.1, 0.001) == 0.1
    assert learning_rate(2, 12, 2, 0.1, 0.001) == 0.1
    assert learning_rate(11, 12, 2, 0.1, 0.001) == 0.001


def test_stop_epoch_rejects_zero_before_loading_assets():
    from pathlib import Path

    from hvl_interact.config import load_config
    from hvl_interact.engine.training import train

    config = load_config(Path(__file__).resolve().parents[1] / "configs/supervised_xsub.yaml")
    with pytest.raises(ValueError, match="stop_after_epoch"):
        train(config, stop_after_epoch=0)
