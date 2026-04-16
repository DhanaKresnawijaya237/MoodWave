"""Test MoodCNNBiGRU dynamic architecture."""
import sys
sys.path.insert(0, "..")

import torch
from core.model import MoodCNNBiGRU


def test_default_architecture():
    model = MoodCNNBiGRU(n_mels=64)
    x = torch.randn(2, 1, 64, 64)
    v, a = model(x)
    assert v.shape == (2,)
    assert a.shape == (2,)
    assert v.min() >= -1.0 and v.max() <= 1.0
    assert a.min() >= -1.0 and a.max() <= 1.0
    print("PASS: test_default_architecture")


def test_lso_hyperparameter_variants():
    configs = [
        {"conv1_filters": 32, "conv2_filters": 64, "kernel_size": 3, "hidden_size": 128, "dropout": 0.3},
        {"conv1_filters": 64, "conv2_filters": 128, "kernel_size": 5, "hidden_size": 256, "dropout": 0.5},
        {"conv1_filters": 32, "conv2_filters": 128, "kernel_size": 3, "hidden_size": 64, "dropout": 0.2},
    ]
    for cfg in configs:
        model = MoodCNNBiGRU(n_mels=64, **cfg)
        x = torch.randn(2, 1, 64, 64)
        v, a = model(x)
        assert v.shape == (2,)
        assert a.shape == (2,)
    print("PASS: test_lso_hyperparameter_variants")


def test_different_pooling():
    model = MoodCNNBiGRU(n_mels=64, pool1=(2, 2), pool2=(2, 2))
    x = torch.randn(2, 1, 64, 64)
    v, a = model(x)
    assert v.shape == (2,)
    assert a.shape == (2,)
    print("PASS: test_different_pooling")


def test_gru_input_dim_computed_correctly():
    """Ensure the model computes GRU input dim automatically for various CNN configs."""
    model = MoodCNNBiGRU(n_mels=64, conv1_filters=64, conv2_filters=128, kernel_size=5)
    x = torch.randn(1, 1, 64, 64)
    feat = model.cnn(x)
    _, C, F, T = feat.shape
    expected_input = C * F
    assert model.bigru.input_size == expected_input
    print("PASS: test_gru_input_dim_computed_correctly")


if __name__ == "__main__":
    test_default_architecture()
    test_lso_hyperparameter_variants()
    test_different_pooling()
    test_gru_input_dim_computed_correctly()
