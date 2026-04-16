"""Test training engine with loss-mode and model_kwargs."""
import sys
sys.path.insert(0, "..")

import numpy as np
import torch
from features.dataset import MelRegressionDataset
from torch.utils.data import DataLoader
from training.engine import train_model, evaluate_loader


def test_train_model_with_loss_mode():
    device = torch.device("cpu")
    n = 16
    X_train = np.random.randn(n, 1, 64, 64).astype(np.float16)
    yv_train = np.random.randn(n).astype(np.float32)
    ya_train = np.random.randn(n).astype(np.float32)
    X_val = np.random.randn(n, 1, 64, 64).astype(np.float16)
    yv_val = np.random.randn(n).astype(np.float32)
    ya_val = np.random.randn(n).astype(np.float32)

    model = train_model(
        X_train, yv_train, ya_train,
        X_val, yv_val, ya_val,
        device,
        epochs=2,
        patience=5,
        lr=1e-3,
        batch_size=8,
        model_kwargs={"conv1_filters": 32, "conv2_filters": 64, "kernel_size": 3, "hidden_size": 64, "dropout": 0.2},
        best_metric_mode="loss",
        quiet=True,
    )
    assert model is not None
    print("PASS: test_train_model_with_loss_mode")


def test_evaluate_loader_returns_metrics():
    from core.model import MoodCNNBiGRU
    device = torch.device("cpu")
    n = 8
    X = np.random.randn(n, 1, 64, 64).astype(np.float16)
    yv = np.random.randn(n).astype(np.float32)
    ya = np.random.randn(n).astype(np.float32)
    loader = DataLoader(MelRegressionDataset(X, yv, ya, augment=False), batch_size=4)
    model = MoodCNNBiGRU(n_mels=64).to(device)
    metrics = evaluate_loader(model, loader, device)
    required = ["valence_mae", "valence_r2", "arousal_mae", "arousal_r2", "quadrant_acc"]
    for k in required:
        assert k in metrics
    print("PASS: test_evaluate_loader_returns_metrics")


if __name__ == "__main__":
    test_train_model_with_loss_mode()
    test_evaluate_loader_returns_metrics()
