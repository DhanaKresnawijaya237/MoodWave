"""Test z-normalization utilities."""
import sys
sys.path.insert(0, "..")

import numpy as np
from core.utils import compute_z_stats, normalize_z, denormalize_z


def test_compute_z_stats():
    yv = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    ya = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    stats = compute_z_stats(yv, ya)
    assert "v_mean" in stats and "v_std" in stats
    assert "a_mean" in stats and "a_std" in stats
    assert stats["v_mean"] == 3.0
    assert stats["a_mean"] == 3.0
    print("PASS: test_compute_z_stats")


def test_normalize_denormalize_roundtrip():
    yv = np.random.randn(100).astype(np.float32)
    ya = np.random.randn(100).astype(np.float32)
    stats = compute_z_stats(yv, ya)
    yv_z, ya_z = normalize_z(yv, ya, stats)
    yv_back, ya_back = denormalize_z(yv_z, ya_z, stats)
    assert np.allclose(yv, yv_back)
    assert np.allclose(ya, ya_back)
    print("PASS: test_normalize_denormalize_roundtrip")


def test_zero_mean_after_normalize():
    yv = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    ya = np.array([0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    stats = compute_z_stats(yv, ya)
    yv_z, ya_z = normalize_z(yv, ya, stats)
    assert abs(yv_z.mean()) < 1e-5
    assert abs(ya_z.mean()) < 1e-5
    print("PASS: test_zero_mean_after_normalize")


if __name__ == "__main__":
    test_compute_z_stats()
    test_normalize_denormalize_roundtrip()
    test_zero_mean_after_normalize()
