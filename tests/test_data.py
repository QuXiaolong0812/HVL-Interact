import numpy as np
import pytest


def test_shared_normalization_preserves_relative_displacement():
    from hvl_interact.data.preprocess import normalize_shared

    x = np.zeros((2, 4, 25, 3), dtype=np.float32)
    x[1, :, :, 0] = 2
    x[:, :, 20, 1] = 2
    valid = np.ones(x.shape[:-1], dtype=bool)
    out = normalize_shared(x, valid)
    np.testing.assert_allclose(
        out[1, :, 0] - out[0, :, 0], np.tile([2 / (2 + 1e-6), 0, 0], (4, 1)), rtol=1e-7
    )
    shifted = normalize_shared(x + [10, -3, 7], valid)
    np.testing.assert_allclose(out, shifted, atol=1e-6)


def test_interpolation_does_not_bridge_missing_observations():
    from hvl_interact.data.preprocess import resample

    x = np.ones((2, 3, 25, 3), dtype=np.float32)
    mask = np.ones((2, 3, 25), dtype=bool)
    mask[0, 1, 2] = False
    y, valid = resample(x, mask, 5)
    assert not valid[0, 1:4, 2].any()
    assert (y[~valid] == 0).all()
    assert valid[1].all()


def test_skeleton_reader_tracks_ids_and_retains_valid_zero(tmp_path):
    from hvl_interact.data.ntu import read_skeleton

    path = tmp_path / "S001C001P001R001A050.skeleton"
    lines = ["2"]
    for order in ([91, 73], [73, 91]):
        lines.append("2")
        for body in order:
            lines.extend([f"{body} 0 0 0 0 0 0 0 0 2", "25"])
            for j in range(25):
                lines.append(f"{0 if j == 0 else body} {j} 0 0 0 0 0 1 0 0 0 2")
    path.write_text("\n".join(lines), encoding="utf-8")
    x, mask, ids = read_skeleton(path)
    assert x.shape == (2, 2, 25, 3)
    assert mask.all()
    assert len(set(ids)) == 2
    np.testing.assert_array_equal(x[:, 0], x[:, 1])
    assert (x[:, :, 0] == 0).all()


def test_malformed_skeleton_is_rejected(tmp_path):
    from hvl_interact.data.ntu import read_skeleton

    p = tmp_path / "broken.skeleton"
    p.write_text("2\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="skeleton|line|body"):
        read_skeleton(p)


def test_fixed_viewport_keeps_relative_distance():
    from hvl_interact.rendering import SkeletonRenderer

    x = np.zeros((2, 2, 25, 3), dtype=np.float32)
    x[1, 0, :, 0] = 1
    x[1, 1, :, 0] = 2
    x[:, :, :, 1] = np.linspace(0, 1, 25)
    mask = np.ones(x.shape[:-1], dtype=bool)
    renderer = SkeletonRenderer(size=64, frames=2)
    points = renderer.project(x, mask)
    assert points[1, 1, 0, 0] - points[0, 1, 0, 0] > points[1, 0, 0, 0] - points[0, 0, 0, 0]
    images = renderer(x, mask)
    assert images.shape == (2, 3, 64, 64)
    assert np.isfinite(images).all()
