"""Dataset splitting utilities for DEAM."""

import numpy as np
from sklearn.model_selection import train_test_split

from features.processor import load_dynamic_annotations
from core.utils import _quadrant_label

SEED = 42


def split_song_ids(dynamic_annotations):
    all_ids = list(dynamic_annotations.keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_ids)

    trainval_ids, test_ids = train_test_split(all_ids, test_size=0.15, random_state=SEED)
    val_ratio = 0.15 / 0.85
    train_ids, val_ids = train_test_split(trainval_ids, test_size=val_ratio, random_state=SEED)
    return set(train_ids), set(val_ids), set(test_ids)


def split_indices(yv, ya, track_ids, split_mode):
    indices = np.arange(len(yv))
    quadrant_labels = np.array([_quadrant_label(v, a) for v, a in zip(yv, ya)], dtype=np.int64)
    if split_mode == "paper":
        trainval_idx, test_idx = train_test_split(
            indices,
            test_size=0.15,
            random_state=SEED,
            stratify=quadrant_labels,
        )
        val_ratio = 0.15 / 0.85
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=val_ratio,
            random_state=SEED,
            stratify=quadrant_labels[trainval_idx],
        )
        return train_idx, val_idx, test_idx

    dynamic = load_dynamic_annotations()
    train_ids, val_ids, test_ids = split_song_ids(dynamic)
    train_idx = indices[np.isin(track_ids, list(train_ids))]
    val_idx = indices[np.isin(track_ids, list(val_ids))]
    test_idx = indices[np.isin(track_ids, list(test_ids))]
    return train_idx, val_idx, test_idx
