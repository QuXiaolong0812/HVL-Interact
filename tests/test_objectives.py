import pytest
import torch
from torch.nn import functional as F


def test_class_prototype_loss_matches_cross_entropy():
    from hvl_interact.objectives import HierarchicalObjective

    query = F.normalize(torch.tensor([[1.0, 2.0], [1.0, 2.0], [-1.0, 0.0]]), dim=-1)
    proto = F.normalize(torch.tensor([[1.0, 1.0], [-1.0, 1.0]]), dim=-1)
    labels = torch.tensor([50, 50, 58])
    logits = torch.tensor([[2.0, 1.0], [1.0, 2.0], [0.0, 1.0]])
    outputs = {
        "queries": {k: query for k in ("fine", "coarse", "global")},
        "prototypes": {k: proto for k in ("fine", "coarse", "global")},
        "logits": logits,
    }
    objective = HierarchicalObjective(
        [50, 58], beta=0.5, temperature=0.07, level_weights=(1.0, 2.0, 3.0), label_smoothing=0.1
    )
    got = objective(outputs, labels)
    expected = F.cross_entropy(logits, torch.tensor([0, 0, 1]), label_smoothing=0.1)
    expected += 0.5 * 6 * F.cross_entropy(query @ proto.T / 0.07, torch.tensor([0, 0, 1]))
    torch.testing.assert_close(got["loss"], expected)
    with pytest.raises(ValueError, match="class"):
        objective(outputs, torch.tensor([1, 50, 58]))


def test_candidate_order_only_permutates_semantic_scores():
    from hvl_interact.objectives import semantic_scores

    queries = {k: F.normalize(torch.randn(3, 8), dim=-1) for k in ("fine", "coarse", "global")}
    protos = {k: F.normalize(torch.randn(6, 8), dim=-1) for k in queries}
    perm = torch.tensor([4, 0, 3, 1, 5, 2])
    original = semantic_scores(queries, protos, (1.0, 1.0, 1.0))
    reordered = semantic_scores(queries, {k: p[perm] for k, p in protos.items()}, (1.0, 1.0, 1.0))
    torch.testing.assert_close(original[:, perm], reordered)
