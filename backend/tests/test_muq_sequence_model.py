"""Smoke tests for MuQ song-window sequence modeling."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.model import MoodMuQBiGRU
from features.muq_extractor import _build_song_window_sequences
from training.helpers import SequenceRegressionDataset, evaluate_sequence_model


def test_muq_windows_group_into_ordered_song_sequences():
    X = torch.tensor([
        [2.0, 2.0],
        [1.0, 1.0],
        [4.0, 4.0],
        [3.0, 3.0],
    ]).numpy()
    yv = torch.tensor([0.2, 0.1, 0.4, 0.3]).numpy()
    ya = torch.tensor([-0.2, -0.1, -0.4, -0.3]).numpy()
    track_ids = torch.tensor([10, 10, 11, 11], dtype=torch.int32).numpy()
    starts = torch.tensor([0.5, 0.0, 0.5, 0.0]).numpy()

    X_seq, lengths, yv_seq, ya_seq, song_ids = _build_song_window_sequences(X, yv, ya, track_ids, starts)
    assert song_ids.tolist() == [10, 11]
    assert lengths.tolist() == [2, 2]
    assert X_seq[0, :, 0].tolist() == [1.0, 2.0]
    assert np.allclose(yv_seq[1], [0.3, 0.4])
    assert np.allclose(ya_seq[1], [-0.3, -0.4])
    print("PASS: test_muq_windows_group_into_ordered_song_sequences")


def test_muq_bigru_outputs_per_window():
    model = MoodMuQBiGRU(input_dim=4, hidden_size=8, num_layers=1, dropout=0.0)
    x = torch.randn(2, 5, 4)
    lengths = torch.tensor([5, 3], dtype=torch.long)
    pred_v, pred_a = model(x, lengths)
    assert pred_v.shape == (2, 5)
    assert pred_a.shape == (2, 5)
    print("PASS: test_muq_bigru_outputs_per_window")


def test_sequence_eval_masks_padded_windows():
    class EchoModel(nn.Module):
        def forward(self, x, lengths):
            return x[:, :, 0], x[:, :, 1]

    X = torch.zeros(2, 4, 2).numpy()
    lengths = torch.tensor([4, 2], dtype=torch.int32).numpy()
    X[0, :, 0] = [0.1, 0.2, 0.3, 0.4]
    X[0, :, 1] = [-0.1, -0.2, -0.3, -0.4]
    X[1, :2, 0] = [0.5, 0.6]
    X[1, :2, 1] = [-0.5, -0.6]

    yv = X[:, :, 0].copy()
    ya = X[:, :, 1].copy()
    yv[1, 2:] = 99.0
    ya[1, 2:] = -99.0

    ds = SequenceRegressionDataset(X, lengths, yv, ya)
    loader = DataLoader(ds, batch_size=2)
    metrics = evaluate_sequence_model(EchoModel(), loader, torch.device("cpu"), loss_fn=nn.MSELoss())
    assert metrics["loss"] == 0.0
    assert metrics["valence_mae"] == 0.0
    assert metrics["arousal_mae"] == 0.0
    print("PASS: test_sequence_eval_masks_padded_windows")


if __name__ == "__main__":
    test_muq_windows_group_into_ordered_song_sequences()
    test_muq_bigru_outputs_per_window()
    test_sequence_eval_masks_padded_windows()
