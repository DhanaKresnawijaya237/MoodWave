"""Test checkpoint saving/loading and backward compatibility."""
import sys
import os
import tempfile
sys.path.insert(0, "..")

import torch
from core.model import MoodCNNBiGRU


def test_new_checkpoint_roundtrip():
    model = MoodCNNBiGRU(n_mels=64, conv1_filters=64, conv2_filters=128, kernel_size=5, hidden_size=256, dropout=0.4)
    state = model.state_dict()
    checkpoint = {
        "state_dict": state,
        "model_kwargs": {
            "conv1_filters": 64,
            "conv2_filters": 128,
            "kernel_size": 5,
            "hidden_size": 256,
            "dropout": 0.4,
        },
    }
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        path = f.name
    torch.save(checkpoint, path)

    loaded = torch.load(path, map_location="cpu")
    kwargs = loaded.get("model_kwargs", {})
    restored = MoodCNNBiGRU(n_mels=64, **kwargs)
    restored.load_state_dict(loaded["state_dict"])

    model.eval()
    restored.eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        v1, a1 = model(x)
        v2, a2 = restored(x)
    assert torch.allclose(v1, v2)
    assert torch.allclose(a1, a2)
    os.unlink(path)
    print("PASS: test_new_checkpoint_roundtrip")


def test_old_checkpoint_backward_compatible():
    """Simulate loading an old-format checkpoint (just state_dict, no model_kwargs)."""
    model = MoodCNNBiGRU(n_mels=64)
    state = model.state_dict()
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        path = f.name
    torch.save(state, path)

    loaded = torch.load(path, map_location="cpu")
    kwargs = loaded.get("model_kwargs", {}) if isinstance(loaded, dict) else {}
    if "state_dict" in loaded:
        kwargs = loaded.get("model_kwargs", {})
        state_to_load = loaded["state_dict"]
    else:
        state_to_load = loaded

    restored = MoodCNNBiGRU(n_mels=64, **kwargs)
    restored.load_state_dict(state_to_load)
    model.eval()
    restored.eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        v1, a1 = model(x)
        v2, a2 = restored(x)
    assert torch.allclose(v1, v2)
    os.unlink(path)
    print("PASS: test_old_checkpoint_backward_compatible")


if __name__ == "__main__":
    test_new_checkpoint_roundtrip()
    test_old_checkpoint_backward_compatible()
