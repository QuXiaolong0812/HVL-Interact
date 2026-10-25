from dataclasses import asdict
from pathlib import Path

import torch


def test_full_width_and_sequence_length_forward():
    from hvl_interact.config import load_config
    from hvl_interact.models.clip import load_prompt_bank
    from hvl_interact.models.network import HVLInteract

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/supervised_xsub.yaml")
    bank = load_prompt_bank(root / "resources/prompts/ntu120_interactions.json")
    model = HVLInteract(
        26, clip_dim=512, max_frames=config.data.frames, **asdict(config.model)
    ).eval()
    x = torch.randn(1, 2, 120, 25, 3)
    valid = torch.ones(x.shape[:-1], dtype=torch.bool)
    with torch.no_grad():
        out = model(
            x,
            valid,
            visual_features=torch.randn(1, 8, 512),
            text_features={h: torch.randn(26, 1, 512) for h in ("fine", "coarse", "global")},
        )
    assert out["tokens"]["fine"].shape == (1, 50, 512)
    assert out["tokens"]["coarse"].shape == (1, 10, 512)
    assert out["logits"].shape == (1, 26)
    assert len(bank) == 26
    assert all(torch.isfinite(x).all() for x in out["prototypes"].values())
