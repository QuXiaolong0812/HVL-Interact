import pytest


def test_official_partitions():
    from hvl_interact.data.protocols import official_training

    assert official_training("S001C001P001R001A050", "xsub")
    assert not official_training("S002C002P003R001A050", "xsub")
    assert official_training("S002C001P003R001A050", "xset")
    assert not official_training("S001C001P001R001A050", "xset")


def test_zero_shot_requires_six_explicit_classes():
    from hvl_interact.data.protocols import class_partition

    for invalid in ([], [50], [50, 50, 51, 52, 53, 54], [1, 2, 3, 4, 5, 6]):
        with pytest.raises(ValueError):
            class_partition("zeroshot", invalid)
    seen, unseen = class_partition("zeroshot", [50, 51, 52, 53, 54, 55])
    assert len(seen) == 20 and len(unseen) == 6
    assert not set(seen) & set(unseen)


def test_unpadded_evaluation_sampler():
    from hvl_interact.engine.distributed import EvaluationSampler

    shards = [list(EvaluationSampler(list(range(7)), rank=r, world_size=3)) for r in range(3)]
    assert sorted(i for shard in shards for i in shard) == list(range(7))
    assert sum(map(len, shards)) == 7


def test_split_cannot_silently_drop_official_samples(experiment):
    import json

    from hvl_interact.data.protocols import build_split, validate_split

    root, _, _ = experiment
    index = json.loads((root / "data/index.json").read_text())
    split = build_split(index, "supervised", "xsub")
    split["train"].pop()
    with pytest.raises(ValueError, match="complete|omitted"):
        validate_split(split, index)
