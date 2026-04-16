import random

import numpy as np
import torch


MOOD_CENTERS = {
    "energetic": (0.7, 0.7),
    "happy": (0.5, 0.3),
    "calm": (0.4, -0.6),
    "romantic": (0.5, -0.5),
    "sad": (-0.6, -0.5),
    "angry": (-0.6, 0.7),
}


def valence_arousal_to_mood_distribution(valence: float, arousal: float, temperature: int = 3) -> dict:
    distances = {}
    for mood, (v_center, a_center) in MOOD_CENTERS.items():
        dist = np.sqrt((valence - v_center) ** 2 + (arousal - a_center) ** 2)
        distances[mood] = dist

    inv_distances = {mood: 1.0 / (dist + 1e-6) ** temperature for mood, dist in distances.items()}
    total = sum(inv_distances.values())
    return {mood: round(weight / total, 4) for mood, weight in inv_distances.items()}


def _quadrant_label(valence, arousal):
    if valence >= 0 and arousal >= 0:
        return 0
    if valence < 0 <= arousal:
        return 1
    if valence < 0 and arousal < 0:
        return 2
    return 3


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_mel_stats(X_train):
    train32 = X_train.astype(np.float32)
    return {
        "mean": float(train32.mean()),
        "std": float(train32.std()),
    }


def normalize_segments(X, mel_stats):
    mean = mel_stats["mean"]
    std = mel_stats["std"]
    return ((X.astype(np.float32) - mean) / (std + 1e-6)).astype(np.float16)


def compute_z_stats(yv_train, ya_train):
    return {
        "v_mean": float(yv_train.mean()),
        "v_std": float(yv_train.std() + 1e-6),
        "a_mean": float(ya_train.mean()),
        "a_std": float(ya_train.std() + 1e-6),
    }


def normalize_z(yv, ya, z_stats):
    yv_z = (yv - z_stats["v_mean"]) / z_stats["v_std"]
    ya_z = (ya - z_stats["a_mean"]) / z_stats["a_std"]
    return yv_z, ya_z


def denormalize_z(yv_z, ya_z, z_stats):
    yv = yv_z * z_stats["v_std"] + z_stats["v_mean"]
    ya = ya_z * z_stats["a_std"] + z_stats["a_mean"]
    return yv, ya


def write_report(title, metrics, report_path=None):
    lines = [
        title,
        f"Valence MAE: {metrics['valence_mae']:.6f}",
        f"Valence R2: {metrics['valence_r2']:.6f}",
        f"Arousal MAE: {metrics['arousal_mae']:.6f}",
        f"Arousal R2: {metrics['arousal_r2']:.6f}",
        f"Quadrant Accuracy: {metrics['quadrant_acc']:.6f}",
        f"Quadrant Weighted F1: {metrics['quadrant_weighted_f1']:.6f}",
        f"Quadrant Macro F1: {metrics['quadrant_macro_f1']:.6f}",
    ]

    print(f"\n{title}")
    for line in lines[1:]:
        print(f"  {line}")

    if report_path is not None:
        import os

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
