"""Test audio augmentation during segment extraction."""
import sys
sys.path.insert(0, "..")

import numpy as np
from features.processor import _extract_song_segments, MEL_SAMPLE_RATE, MEL_SEGMENT_FRAMES, MEL_N_MELS


def _make_task(n_segments=3):
    sid = 999
    sr = MEL_SAMPLE_RATE
    duration_s = 60
    audio = np.sin(2 * np.pi * 440.0 * np.arange(int(duration_s * sr)) / sr).astype(np.float32)
    samples = []
    for i in range(n_segments):
        t = 25.0 + i * 5.0  # well after AUDIO_OFFSET (15s) and within bounds
        samples.append((t, float(0.1 * i), float(-0.1 * i)))
    return sid, audio, samples


def test_extract_without_augment():
    sid, audio, samples = _make_task()
    task = (sid, None, samples)  # audio_path=None, we'll patch loading

    # Save temp wav
    import tempfile, os
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    sf.write(path, audio, MEL_SAMPLE_RATE)

    task = (sid, path, samples)
    result = _extract_song_segments(task, augment=False)
    _, X, yv, ya = result
    assert X.shape[0] == len(samples)
    assert X.shape[1] == 1
    assert X.shape[2] == MEL_N_MELS
    assert X.shape[3] == MEL_SEGMENT_FRAMES
    os.unlink(path)
    print("PASS: test_extract_without_augment")


def test_extract_with_augment():
    sid, audio, samples = _make_task(n_segments=2)
    import tempfile, os
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    sf.write(path, audio, MEL_SAMPLE_RATE)

    task = (sid, path, samples)
    _, X, yv, ya = _extract_song_segments(task, augment=False)
    n_original = X.shape[0]

    _, X_aug, yv_aug, ya_aug = _extract_song_segments(task, augment=True)
    n_aug = X_aug.shape[0]

    # Augmentations added: original + pitch_shift x2 + time_stretch x2 + noise x1 = 6x total
    assert n_aug == n_original * 6
    assert np.allclose(yv_aug[:n_original], yv)
    assert np.allclose(ya_aug[:n_original], ya)
    os.unlink(path)
    print("PASS: test_extract_with_augment")


if __name__ == "__main__":
    test_extract_without_augment()
    test_extract_with_augment()
