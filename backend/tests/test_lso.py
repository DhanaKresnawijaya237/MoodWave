"""Test LSO encoding/decoding and search loop."""
import sys
sys.path.insert(0, "..")

import numpy as np
import torch
from training.lso import _encode, _decode, LSOSearch


def test_encode_decode_roundtrip():
    params = {
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "conv1_filters": 64,
        "conv2_filters": 128,
        "kernel_size": 5,
        "hidden_size": 256,
        "dropout": 0.35,
    }
    vec = _encode(params)
    decoded = _decode(vec)
    assert decoded["lr"] == params["lr"]
    assert decoded["conv1_filters"] == params["conv1_filters"]
    assert decoded["conv2_filters"] == params["conv2_filters"]
    assert decoded["kernel_size"] == params["kernel_size"]
    assert decoded["hidden_size"] == params["hidden_size"]
    assert abs(decoded["dropout"] - params["dropout"]) < 0.02
    assert abs(decoded["weight_decay"] - params["weight_decay"]) < 1e-4
    print("PASS: test_encode_decode_roundtrip")


def test_decode_clipping():
    vec = np.array([-0.5, 1.2, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    decoded = _decode(vec)
    assert decoded["lr"] in [5e-5, 1e-4, 1e-3]
    assert 0.0 <= _encode(decoded)[0] <= 1.0
    print("PASS: test_decode_clipping")


def test_lso_search_one_iteration():
    """Run LSO for 1 iteration on tiny synthetic data to verify the loop executes."""
    device = torch.device("cpu")
    n = 20
    X_train = np.random.randn(n, 1, 64, 64).astype(np.float16)
    yv_train = np.random.randn(n).astype(np.float32)
    ya_train = np.random.randn(n).astype(np.float32)
    X_val = np.random.randn(n, 1, 64, 64).astype(np.float16)
    yv_val = np.random.randn(n).astype(np.float32)
    ya_val = np.random.randn(n).astype(np.float32)

    lso = LSOSearch(
        X_train, yv_train, ya_train,
        X_val, yv_val, ya_val,
        device,
        pop_size=4,
        iterations=1,
        eval_epochs=2,
        eval_patience=1,
        batch_size=8,
        seed=42,
    )
    best = lso.optimize()
    assert isinstance(best, dict)
    assert "lr" in best
    assert "hidden_size" in best
    print("PASS: test_lso_search_one_iteration")


if __name__ == "__main__":
    test_encode_decode_roundtrip()
    test_decode_clipping()
    test_lso_search_one_iteration()
