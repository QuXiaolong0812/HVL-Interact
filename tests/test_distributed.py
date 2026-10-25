import contextlib
import json
import os
import socket
import time

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml


def distributed_worker(rank, port, config_path, root, command):
    from pathlib import Path

    from hvl_interact.cli import main
    from hvl_interact.engine import training

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_RANK=str(rank),
        USE_LIBUV="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
    )
    original = training.create_model

    def delayed_model(*args, **kwargs):
        value = original(*args, **kwargs)
        if rank == 1:
            time.sleep(2)
        return value

    training.create_model = delayed_model
    with (Path(root) / f"{command}-rank-{rank}.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            if command == "train":
                main(["train", "--config", config_path])
            else:
                main(
                    [
                        "evaluate",
                        "--config",
                        config_path,
                        "--checkpoint",
                        str(Path(root) / "run/best.pt"),
                    ]
                )


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)
@pytest.mark.parametrize("skeleton_only", [False, True])
def test_two_process_training_and_uneven_evaluation(experiment, skeleton_only):
    import shutil

    root, config, config_path = experiment
    config["train"].update(epochs=1, accumulation_steps=2, batch_size=13)
    if skeleton_only:
        config["model"]["use_visual"] = False
        config["loss"] = {"beta": 0}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    index_path = root / "data/index.json"
    index = json.loads(index_path.read_text())
    extra = next(r.copy() for r in index["samples"] if r["sample_id"] == "S001C001P003R001A050")
    extra.update(sample_id="S001C002P003R001A050", path="S001C002P003R001A050.npz")
    shutil.copyfile(root / "data/S001C001P003R001A050.npz", root / "data" / extra["path"])
    index["samples"].append(extra)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    from hvl_interact.cli import main

    main(
        [
            "split",
            "--index",
            str(index_path),
            "--output",
            str(root / "split.json"),
            "--mode",
            "supervised",
            "--protocol",
            "xsub",
        ]
    )
    for command in ("train", "evaluate"):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(
            distributed_worker,
            args=(port, str(config_path), str(root), command),
            nprocs=2,
            join=True,
        )
    metrics = json.loads((root / "run/evaluation/test/metrics.json").read_text())
    assert metrics["count"] == 27
    rows = (root / "run/evaluation/test/predictions.jsonl").read_text().splitlines()
    assert len(rows) == len({json.loads(r)["sample_id"] for r in rows}) == 27
