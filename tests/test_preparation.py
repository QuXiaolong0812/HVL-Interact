import json

import numpy as np


def write_raw(path):
    lines = ["2"]
    for _ in range(2):
        lines.append("2")
        for body in (10, 20):
            lines.extend([f"{body} 0 0 0 0 0 0 0 0 2", "25"])
            for j in range(25):
                lines.append(f"{body / 10} {j / 25} 0 0 0 0 0 1 0 0 0 2")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_prepare_records_quality_exclusions(tmp_path):
    from hvl_interact.cli import main

    raw = tmp_path / "raw"
    raw.mkdir()
    good = "S002C001P001R001A050"
    ignored = "S002C001P001R001A051"
    broken = "S002C001P001R001A052"
    write_raw(raw / f"{good}.skeleton")
    write_raw(raw / f"{ignored}.skeleton")
    (raw / f"{broken}.skeleton").write_text("2\n", encoding="utf-8")
    excluded = tmp_path / "excluded.txt"
    excluded.write_text(ignored, encoding="utf-8")
    out = tmp_path / "processed"
    result = main(
        ["prepare", "--skeleton-root", str(raw), "--excluded", str(excluded), "--output", str(out)]
    )
    assert result["accepted"] == 1 and result["rejected"] == 2
    index = json.loads((out / "index.json").read_text())
    with np.load(out / index["samples"][0]["path"], allow_pickle=False) as data:
        assert data["coordinates"].shape == (2, 2, 25, 3)
        assert data["valid"].all()
