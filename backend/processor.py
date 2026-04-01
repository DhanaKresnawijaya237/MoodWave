import scipy.signal
# scipy >= 1.8 removed hann from scipy.signal — patch it back before librosa imports it
if not hasattr(scipy.signal, 'hann'):
    scipy.signal.hann = scipy.signal.windows.hann
    
import librosa
import numpy as np
import pandas as pd
import os


def features_to_vector(features: dict) -> list:
    base = [
        features["tempo"],
        features["rms_energy"],
        features["spectral_centroid"],
        features["zero_crossing_rate"],
        features["spectral_rolloff"],
        features["valence"],
        features["onset_strength"],
    ]
    return base + features["mfcc"] + features["chroma"] + features["spectral_contrast"] + features["tonnetz"]

CHUNK_DURATION = 10  # seconds, must match main.py
AUDIO_OFFSET = 15.0  # DEAM excerpts start at 15s of original song


def extract_features_from_array(y: np.ndarray, sr: int) -> dict:
    """Extract audio features from a numpy array chunk."""

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    harmonic, _ = librosa.effects.hpss(y)
    valence = np.mean(np.abs(harmonic)) / (np.mean(np.abs(y)) + 1e-6)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr).mean(axis=1)
    onset_strength = np.mean(librosa.onset.onset_strength(y=y, sr=sr))

    return {
        "tempo": float(tempo),
        "rms_energy": float(rms),
        "spectral_centroid": float(centroid),
        "zero_crossing_rate": float(zcr),
        "spectral_rolloff": float(rolloff),
        "valence": float(valence),
        "onset_strength": float(onset_strength),
        "mfcc": [float(x) for x in mfcc],
        "chroma": [float(x) for x in chroma],
        "spectral_contrast": [float(x) for x in spectral_contrast],
        "tonnetz": [float(x) for x in tonnetz],
    }


def load_dynamic_annotations():
    """Load dynamic per-second annotations and return lookup dicts."""
    base = "data/deam/annotations/annotations averaged per song/dynamic (per second annotations)"
    valence_df = pd.read_csv(f"{base}/valence.csv")
    arousal_df = pd.read_csv(f"{base}/arousal.csv")

    # Parse timestamp columns (e.g., "sample_15000ms" → 15.0 seconds)
    time_cols = [c for c in valence_df.columns if c.startswith("sample_")]
    timestamps = [int(c.replace("sample_", "").replace("ms", "")) / 1000.0 for c in time_cols]

    # Build lookup: song_id → list of (timestamp_seconds, valence, arousal)
    # Keep raw -1 to 1 scale — no rescaling
    lookup = {}
    for _, v_row in valence_df.iterrows():
        sid = int(v_row["song_id"])
        a_row = arousal_df[arousal_df["song_id"] == sid]
        if a_row.empty:
            continue
        a_row = a_row.iloc[0]

        samples = []
        for t, col in zip(timestamps, time_cols):
            v = v_row[col]
            a = a_row[col]
            if pd.notna(v) and pd.notna(a):
                samples.append((t, float(v), float(a)))
        lookup[sid] = samples

    return lookup


def get_chunk_labels(samples, chunk_start_s, chunk_end_s):
    """Average the dynamic annotations within a time window.
    chunk_start_s/chunk_end_s are in seconds relative to audio file start (0-based).
    Annotations use original song timestamps (offset by AUDIO_OFFSET)."""
    annot_start = chunk_start_s + AUDIO_OFFSET
    annot_end = chunk_end_s + AUDIO_OFFSET

    chunk_vals = [(v, a) for t, v, a in samples if annot_start <= t < annot_end]
    if not chunk_vals:
        return None, None

    valence = np.mean([v for v, a in chunk_vals])
    arousal = np.mean([a for v, a in chunk_vals])
    return float(valence), float(arousal)


def process_song(args):
    """Process a single song: load audio, chunk it, extract features.
    Returns list of (vector, valence, arousal) tuples."""
    sid, samples = args
    path = f"data/deam/MEMD_audio/{sid}.mp3"
    if not os.path.exists(path):
        return sid, []

    try:
        audio, sr = librosa.load(path)
        chunk_samples = int(CHUNK_DURATION * sr)
        results = []

        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start : start + chunk_samples]
            # Skip chunks shorter than 5 seconds
            if len(chunk) < sr * 5:
                continue

            chunk_start_s = start / sr
            chunk_end_s = (start + len(chunk)) / sr

            v, a = get_chunk_labels(samples, chunk_start_s, chunk_end_s)
            if v is None:
                continue

            f = extract_features_from_array(chunk, sr)
            vector = features_to_vector(f)
            results.append((vector, v, a))

        return sid, results
    except Exception as e:
        print(f"  Skipped {sid}: {e}")
        return sid, []
