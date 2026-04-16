"""Test inference denormalization with z_stats."""
import sys
import os
import tempfile
sys.path.insert(0, "..")

import torch
import joblib
import numpy as np
from core.model import MoodCNNBiGRU


def test_inference_with_z_stats():
    model = MoodCNNBiGRU(n_mels=64, hidden_size=64)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_kwargs": {"hidden_size": 64},
    }
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        model_path = f.name
    torch.save(checkpoint, model_path)

    mel_stats = {"mean": -50.0, "std": 20.0}
    z_stats = {"v_mean": 0.5, "v_std": 0.2, "a_mean": -0.3, "a_std": 0.4}

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        mel_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        z_path = f.name
    joblib.dump(mel_stats, mel_path)
    joblib.dump(z_stats, z_path)

    # Simulate _predict_cnnbigru logic
    from core.utils import denormalize_z
    x = torch.randn(2, 1, 64, 64)
    model.eval()
    with torch.no_grad():
        pred_v, pred_a = model(x)
    valence_z = float(pred_v.mean().cpu().numpy())
    arousal_z = float(pred_a.mean().cpu().numpy())

    valence, arousal = denormalize_z(
        np.array([valence_z]), np.array([arousal_z]), z_stats
    )
    expected_v = valence_z * z_stats["v_std"] + z_stats["v_mean"]
    expected_a = arousal_z * z_stats["a_std"] + z_stats["a_mean"]
    assert abs(float(valence[0]) - expected_v) < 1e-5
    assert abs(float(arousal[0]) - expected_a) < 1e-5

    os.unlink(model_path)
    os.unlink(mel_path)
    os.unlink(z_path)
    print("PASS: test_inference_with_z_stats")


if __name__ == "__main__":
    test_inference_with_z_stats()
