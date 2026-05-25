"""OpenL3 feature extraction for DEAM frame-level VA regression.

Matches the paper setup:
- 512-dim music-domain embeddings
- hop_size = 0.5 s (aligned to DEAM 2 Hz annotations)
- audio sampled at 44.1 kHz

Optimizations:
- Uses julian sampler (GPU-native, ~1.6× faster than resampy)
- Chunked incremental caching: saves partial results every 200 songs
  so extraction can resume after interruption.
"""

import glob
import os

import librosa
import numpy as np
import pandas as pd
import torchopenl3
from tqdm import tqdm

OPENL3_SR = 44100
OPENL3_HOP = 0.5
OPENL3_EMBED_DIM = 512
_AUDIO_OFFSET = 15.0
_CHUNK_SIZE = 200  # save partial cache every N songs


def load_dynamic_annotations():
    """Load dynamic per-second annotations and return lookup dicts."""
    base = "data/deam/annotations/annotations averaged per song/dynamic (per second annotations)"
    valence_df = pd.read_csv(f"{base}/valence.csv")
    arousal_df = pd.read_csv(f"{base}/arousal.csv")

    time_cols = [c for c in valence_df.columns if c.startswith("sample_")]
    timestamps = [int(c.replace("sample_", "").replace("ms", "")) / 1000.0 for c in time_cols]

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


def _partial_cache_path(cache_file, chunk_idx):
    base, ext = os.path.splitext(cache_file)
    return f"{base}_partial_{chunk_idx:04d}{ext}"


def _load_partial_caches(cache_file):
    """Load any existing partial caches and return processed song IDs + data."""
    base, ext = os.path.splitext(cache_file)
    pattern = f"{base}_partial_*{ext}"
    files = sorted(glob.glob(pattern))
    if not files:
        return set(), [], [], [], []

    done_sids = set()
    all_emb, all_yv, all_ya, all_tids = [], [], [], []
    for pf in files:
        try:
            data = np.load(pf)
            sids = set(np.unique(data["track_ids"]).tolist())
            done_sids.update(sids)
            all_emb.append(data["X"])
            all_yv.append(data["yv"])
            all_ya.append(data["ya"])
            all_tids.append(data["track_ids"])
        except Exception:
            continue

    print(f"  Found {len(done_sids)} songs in partial caches — resuming...")
    return done_sids, all_emb, all_yv, all_ya, all_tids


def _save_partial_cache(cache_file, chunk_idx, embeddings, valences, arousals, track_ids):
    path = _partial_cache_path(cache_file, chunk_idx)
    np.savez_compressed(
        path,
        X=np.concatenate(embeddings, axis=0),
        yv=np.concatenate(valences, axis=0),
        ya=np.concatenate(arousals, axis=0),
        track_ids=np.concatenate(track_ids, axis=0),
    )


def _extract_song_openl3(task):
    sid, audio_path, samples = task
    try:
        if not os.path.exists(audio_path):
            return sid, None, None, None

        y, sr_loaded = librosa.load(audio_path, sr=OPENL3_SR)
        if len(y) == 0:
            return sid, None, None, None

        emb, ts = torchopenl3.get_audio_embedding(
            y,
            sr_loaded,
            embedding_size=OPENL3_EMBED_DIM,
            hop_size=OPENL3_HOP,
            content_type="music",
            batch_size=32,
            verbose=False,
            sampler="julian",  # GPU-native resampler, ~1.6× faster than resampy
        )
        # emb: (1, n_frames, 512), ts: (1, n_frames)
        emb = emb.squeeze(0).cpu().numpy().astype(np.float32)
        ts = ts.squeeze(0).cpu().numpy().astype(np.float32)

        n_annotated = len(samples)
        if n_annotated == 0:
            return sid, None, None, None

        # Keep only frames that have annotations.
        # DEAM annotations cover excerpt_time 0.0 to 29.5s (60 frames at 0.5s)
        # OpenL3 timestamps are relative to audio start, so frame i is at i*0.5s.
        if emb.shape[0] < n_annotated:
            # Pad with last frame if audio is shorter than expected
            pad = n_annotated - emb.shape[0]
            emb = np.concatenate([emb, np.repeat(emb[-1:], pad, axis=0)], axis=0)
        else:
            emb = emb[:n_annotated]

        valence = np.array([v for _, v, _ in samples], dtype=np.float32)
        arousal = np.array([a for _, _, a in samples], dtype=np.float32)

        return sid, emb, valence, arousal
    except Exception as exc:
        tqdm.write(f"  Skipped {sid}: {exc}")
        return sid, None, None, None


def _cleanup_partial_caches(cache_file):
    base, ext = os.path.splitext(cache_file)
    pattern = f"{base}_partial_*{ext}"
    for pf in glob.glob(pattern):
        try:
            os.remove(pf)
        except OSError:
            pass


def load_or_extract_openl3_frames(
    rebuild_cache=False,
    cache_file=None,
    worker_count=1,
):
    """Extract or load cached OpenL3 frame-level embeddings for DEAM.

    Supports resume via chunked partial caches.

    Returns
    -------
    X : np.ndarray, shape (N, 512)
    yv : np.ndarray, shape (N,)
    ya : np.ndarray, shape (N,)
    track_ids : np.ndarray, shape (N,)
    """
    if cache_file is None:
        cache_file = "data/deam/openl3_embeddings_cache.npz"

    if os.path.exists(cache_file) and not rebuild_cache:
        print(f"Loading cached OpenL3 embeddings from {cache_file}...")
        cache = np.load(cache_file)
        return cache["X"], cache["yv"], cache["ya"], cache["track_ids"]

    if rebuild_cache:
        _cleanup_partial_caches(cache_file)

    print("Loading dynamic annotations...")
    dynamic = load_dynamic_annotations()
    print(f"  DEAM: {len(dynamic)} songs")

    audio_dir = "data/deam/MEMD_audio"
    tasks = []
    for sid, samples in dynamic.items():
        path = os.path.join(audio_dir, f"{sid}.mp3")
        if os.path.exists(path):
            tasks.append((sid, path, samples))

    if not tasks:
        raise RuntimeError("No DEAM audio files found for OpenL3 extraction.")

    # Resume support: skip songs already in partial caches
    done_sids, all_embeddings, all_valence, all_arousal, all_track_ids = _load_partial_caches(cache_file)
    tasks = [t for t in tasks if t[0] not in done_sids]

    if not tasks:
        # All done from partial caches — just merge and return
        X = np.concatenate(all_embeddings, axis=0)
        yv = np.concatenate(all_valence, axis=0)
        ya = np.concatenate(all_arousal, axis=0)
        track_ids = np.concatenate(all_track_ids, axis=0)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez_compressed(cache_file, X=X, yv=yv, ya=ya, track_ids=track_ids)
        _cleanup_partial_caches(cache_file)
        print(f"  Merged partial caches → {len(X)} OpenL3 frames saved to {cache_file}")
        return X, yv, ya, track_ids

    print(f"  Extracting OpenL3 embeddings for {len(tasks)} songs...")
    chunk_emb, chunk_yv, chunk_ya, chunk_tids = [], [], [], []
    chunk_counter = 0

    # OpenL3 uses GPU; sequential processing avoids CUDA OOM
    for task in tqdm(tasks, desc="OpenL3 extract", unit="song"):
        sid, emb, yv, ya = _extract_song_openl3(task)
        if emb is None:
            continue
        all_embeddings.append(emb)
        all_valence.append(yv)
        all_arousal.append(ya)
        all_track_ids.append(np.full(len(yv), sid, dtype=np.int32))

        chunk_emb.append(emb)
        chunk_yv.append(yv)
        chunk_ya.append(ya)
        chunk_tids.append(np.full(len(yv), sid, dtype=np.int32))

        # Save partial cache every _CHUNK_SIZE songs
        if len(chunk_emb) >= _CHUNK_SIZE:
            _save_partial_cache(cache_file, chunk_counter, chunk_emb, chunk_yv, chunk_ya, chunk_tids)
            chunk_counter += 1
            chunk_emb, chunk_yv, chunk_ya, chunk_tids = [], [], [], []

    # Flush remaining chunk
    if chunk_emb:
        _save_partial_cache(cache_file, chunk_counter, chunk_emb, chunk_yv, chunk_ya, chunk_tids)

    if not all_embeddings:
        raise RuntimeError("No OpenL3 embeddings were extracted.")

    X = np.concatenate(all_embeddings, axis=0)
    yv = np.concatenate(all_valence, axis=0)
    ya = np.concatenate(all_arousal, axis=0)
    track_ids = np.concatenate(all_track_ids, axis=0)

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.savez_compressed(cache_file, X=X, yv=yv, ya=ya, track_ids=track_ids)
    _cleanup_partial_caches(cache_file)
    print(f"  Cached {len(X)} OpenL3 frames to {cache_file}")
    return X, yv, ya, track_ids
