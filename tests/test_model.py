import inspect

import torch


def test_model_queries_and_missing_tracks_are_finite():
    from hvl_interact.models.network import HVLInteract

    torch.manual_seed(7)
    model = HVLInteract(
        num_classes=4,
        width=16,
        embed_dim=8,
        clip_dim=12,
        channels=(8, 16),
        heads=4,
        temporal_layers=1,
        spatial_layers=1,
        dropout=0,
        max_frames=8,
    )
    x = torch.randn(2, 2, 6, 25, 3)
    valid = torch.ones(2, 2, 6, 25, dtype=torch.bool)
    valid[0, 1] = False
    x[0, 1] = 999
    features = torch.randn(2, 3, 12)
    model.eval()
    with torch.no_grad():
        out = model(x, valid, visual_features=features)
    for level in ("fine", "coarse", "global"):
        assert out["queries"][level].shape == (2, 8)
        assert torch.isfinite(out["queries"][level]).all()
        torch.testing.assert_close(out["queries"][level].norm(dim=-1), torch.ones(2))
    assert out["logits"].shape == (2, 4)
    assert "labels" not in inspect.signature(model.forward).parameters
    assert out["tokens"]["fine"].shape == (2, 50, 16)
    assert out["tokens"]["coarse"].shape == (2, 10, 16)


def test_text_does_not_change_observation_queries():
    from hvl_interact.models.network import HVLInteract

    model = HVLInteract(
        num_classes=3,
        width=8,
        embed_dim=8,
        clip_dim=8,
        channels=(8,),
        heads=2,
        temporal_layers=1,
        spatial_layers=1,
        dropout=0,
        max_frames=4,
    ).eval()
    x = torch.randn(1, 2, 4, 25, 3)
    mask = torch.ones(x.shape[:-1], dtype=torch.bool)
    visual = torch.randn(1, 2, 8)
    a = {k: torch.randn(3, 1, 8) for k in ("fine", "coarse", "global")}
    b = {k: torch.randn(6, 2, 8) for k in a}
    first = model(x, mask, visual_features=visual, text_features=a)
    second = model(x, mask, visual_features=visual, text_features=b)
    torch.testing.assert_close(first["queries"]["global"], second["queries"]["global"])
    loss = sum(v.square().sum() for v in first["prototypes"].values())
    loss.backward()
    assert all(p.grad is not None for p in model.text_projections.parameters())


def test_zero_embeddings_remain_finite_in_half_precision():
    from hvl_interact.models.network import HVLInteract

    model = (
        HVLInteract(
            num_classes=2,
            width=8,
            embed_dim=8,
            clip_dim=8,
            channels=(8,),
            heads=2,
            temporal_layers=1,
            spatial_layers=1,
            dropout=0,
            max_frames=2,
        )
        .half()
        .eval()
    )
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    x = torch.zeros(1, 2, 2, 25, 3, dtype=torch.float16)
    mask = torch.zeros(x.shape[:-1], dtype=torch.bool)
    out = model(
        x,
        mask,
        visual_features=torch.zeros(1, 2, 8, dtype=torch.float16),
        text_features={
            k: torch.zeros(2, 1, 8, dtype=torch.float16) for k in ("fine", "coarse", "global")
        },
    )
    assert all(torch.isfinite(t).all() for t in out["queries"].values())
    assert all(torch.isfinite(t).all() for t in out["prototypes"].values())


def test_body_group_order_is_canonical():
    from hvl_interact.data.layout import BODY_GROUPS
    from hvl_interact.models.skeleton import HierarchicalSkeletonEncoder

    first = HierarchicalSkeletonEncoder(width=8, channels=(8,), heads=2, groups=BODY_GROUPS)
    reverse = dict(reversed(list(BODY_GROUPS.items())))
    second = HierarchicalSkeletonEncoder(width=8, channels=(8,), heads=2, groups=reverse)
    torch.testing.assert_close(first.pooling, second.pooling)
