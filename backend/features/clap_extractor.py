"""CLAP feature extraction for DEAM dynamic window-level VA regression.

Supports two backends (tries msclap first, falls back to transformers):
    pip install msclap transformers

CLAP normally produces one embedding for an audio clip. Here each clip is the
0.5s audio window aligned with a DEAM dynamic valence/arousal annotation.
"""

import glob
import os

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

CLAP_SR = 48000
_AUDIO_OFFSET = 15.0
_WINDOW_SECONDS = 0.5
_CHUNK_SIZE = 200


def load_dynamic_annotations():
    """Load dynamic per-0.5s annotations and return lookup dicts."""
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
        for timestamp_s, col in zip(timestamps, time_cols):
            v = v_row[col]
            a = a_row[col]
            if pd.notna(v) and pd.notna(a):
                samples.append((timestamp_s, float(v), float(a)))
        lookup[sid] = samples

    return lookup


def _partial_cache_path(cache_file, chunk_idx):
    base, ext = os.path.splitext(cache_file)
    return f"{base}_partial_{chunk_idx:04d}{ext}"


def _next_partial_index(cache_file):
    base, ext = os.path.splitext(cache_file)
    return len(sorted(glob.glob(f"{base}_partial_*{ext}")))


def _cleanup_partial_caches(cache_file):
    base, ext = os.path.splitext(cache_file)
    for path in glob.glob(f"{base}_partial_*{ext}"):
        try:
            os.remove(path)
        except OSError:
            pass


def _annotation_arrays(samples):
    starts = []
    valence = []
    arousal = []
    for timestamp_s, v, a in samples:
        excerpt_time_s = float(timestamp_s) - _AUDIO_OFFSET
        if excerpt_time_s < 0:
            continue
        starts.append(excerpt_time_s)
        valence.append(float(v))
        arousal.append(float(a))
    return (
        np.array(starts, dtype=np.float32),
        np.array(valence, dtype=np.float32),
        np.array(arousal, dtype=np.float32),
    )


def _build_tasks(dynamic):
    audio_dir = "data/deam/MEMD_audio"
    tasks = []
    for sid, samples in dynamic.items():
        path = os.path.join(audio_dir, f"{sid}.mp3")
        if os.path.exists(path):
            tasks.append((sid, path, samples))
    return tasks


def _get_clap_embed_fn():
    """Return a callable (audio_np, sr) -> embedding_np and the embedding dim."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from msclap import CLAP

        clap_model = CLAP(version="2023", use_cuda=torch.cuda.is_available())
        print("  Using msclap backend")

        def _embed(audio, sr):
            import tempfile

            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio, sr)
                tmp_path = tmp.name
            try:
                emb = clap_model.get_audio_embeddings([tmp_path])
                if hasattr(emb, "cpu"):
                    emb = emb.cpu().numpy()
                else:
                    emb = np.array(emb)
                return emb.squeeze().astype(np.float32)
            finally:
                os.unlink(tmp_path)

        dummy = np.zeros(int(CLAP_SR * _WINDOW_SECONDS), dtype=np.float32)
        emb_dummy = _embed(dummy, CLAP_SR)
        return _embed, int(emb_dummy.shape[-1])
    except Exception as exc_msclap:
        print(f"  msclap not available ({exc_msclap}), trying transformers...")

    from transformers import AutoFeatureExtractor, AutoModel

    model_id = "laion/larger_clap_music"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    print(f"  Using transformers backend ({model_id})")

    embed_dim = model.config.projection_dim

    def _embed(audio, sr):
        if sr != feature_extractor.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=feature_extractor.sampling_rate)
        inputs = feature_extractor(audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.get_audio_features(**inputs)
        return outputs.squeeze(0).cpu().numpy().astype(np.float32)

    return _embed, embed_dim


def _slice_window(audio, start_s):
    start = int(round(start_s * CLAP_SR))
    end = start + int(round(_WINDOW_SECONDS * CLAP_SR))
    if start >= len(audio):
        return None
    segment = audio[start:min(end, len(audio))]
    if len(segment) == 0:
        return None
    if len(segment) < end - start:
        segment = np.pad(segment, (0, end - start - len(segment)), mode="constant")
    return segment.astype(np.float32, copy=False)


def _extract_song_clap(task, embed_fn):
    sid, audio_path, samples = task
    try:
        if not os.path.exists(audio_path):
            return sid, None, None, None

        y, sr_loaded = librosa.load(audio_path, sr=CLAP_SR, mono=True)
        if len(y) == 0:
            return sid, None, None, None

        starts, valence, arousal = _annotation_arrays(samples)
        if len(starts) == 0:
            return sid, None, None, None

        embeddings = []
        kept_valence = []
        kept_arousal = []
        for start_s, v, a in zip(starts, valence, arousal):
            segment = _slice_window(y, start_s)
            if segment is None:
                continue
            embeddings.append(embed_fn(segment, sr_loaded))
            kept_valence.append(v)
            kept_arousal.append(a)

        if not embeddings:
            return sid, None, None, None

        return (
            sid,
            np.stack(embeddings, axis=0).astype(np.float32),
            np.array(kept_valence, dtype=np.float32),
            np.array(kept_arousal, dtype=np.float32),
        )
    except Exception as exc:
        tqdm.write(f"  Skipped {sid}: {exc}")
        return sid, None, None, None


def _save_partial_cache(cache_file, chunk_idx, embeddings, valences, arousals, track_ids):
    np.savez_compressed(
        _partial_cache_path(cache_file, chunk_idx),
        X=np.concatenate(embeddings, axis=0),
        yv=np.concatenate(valences, axis=0).astype(np.float32),
        ya=np.concatenate(arousals, axis=0).astype(np.float32),
        track_ids=np.concatenate(track_ids, axis=0).astype(np.int32),
    )


def _load_partial_caches(cache_file):
    base, ext = os.path.splitext(cache_file)
    files = sorted(glob.glob(f"{base}_partial_*{ext}"))
    if not files:
        return set(), [], [], [], []

    done_sids = set()
    all_emb, all_yv, all_ya, all_tids = [], [], [], []
    for path in files:
        try:
            data = np.load(path)
            done_sids.update(set(np.unique(data["track_ids"]).tolist()))
            all_emb.append(data["X"])
            all_yv.append(data["yv"])
            all_ya.append(data["ya"])
            all_tids.append(data["track_ids"])
        except Exception:
            continue

    print(f"  Found {len(done_sids)} songs in partial caches; resuming...")
    return done_sids, all_emb, all_yv, all_ya, all_tids


def load_or_extract_clap_embeddings(
    rebuild_cache=False,
    cache_file=None,
):
    """Extract or load cached CLAP window-level embeddings for DEAM.

    Returns
    -------
    X : np.ndarray, shape (N_windows, embed_dim)
    yv : np.ndarray, shape (N_windows,)
    ya : np.ndarray, shape (N_windows,)
    track_ids : np.ndarray, shape (N_windows,)
    embed_dim : int
    """
    if cache_file is None:
        cache_file = "data/deam/clap_window_embeddings_cache.npz"

    if os.path.exists(cache_file) and not rebuild_cache:
        print(f"Loading cached CLAP window embeddings from {cache_file}...")
        cache = np.load(cache_file)
        return cache["X"], cache["yv"], cache["ya"], cache["track_ids"], int(cache["embed_dim"])

    if rebuild_cache:
        _cleanup_partial_caches(cache_file)

    print("Initializing CLAP model...")
    embed_fn, embed_dim = _get_clap_embed_fn()
    print(f"  CLAP embedding dimension: {embed_dim}")

    print("Loading dynamic annotations...")
    dynamic = load_dynamic_annotations()
    print(f"  DEAM: {len(dynamic)} songs")
    tasks = _build_tasks(dynamic)
    if not tasks:
        raise RuntimeError("No DEAM audio files found for CLAP extraction.")

    done_sids, all_emb, all_yv, all_ya, all_tids = _load_partial_caches(cache_file)
    tasks = [task for task in tasks if task[0] not in done_sids]

    if not tasks:
        X = np.concatenate(all_emb, axis=0)
        yv = np.concatenate(all_yv, axis=0)
        ya = np.concatenate(all_ya, axis=0)
        track_ids = np.concatenate(all_tids, axis=0)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez_compressed(cache_file, X=X, yv=yv, ya=ya, track_ids=track_ids, embed_dim=embed_dim)
        _cleanup_partial_caches(cache_file)
        print(f"  Merged partial caches into {len(X)} CLAP window embeddings at {cache_file}")
        return X, yv, ya, track_ids, embed_dim

    print(f"  Extracting CLAP window embeddings for {len(tasks)} songs...")
    chunk_emb, chunk_yv, chunk_ya, chunk_tids = [], [], [], []
    chunk_counter = _next_partial_index(cache_file)

    for task in tqdm(tasks, desc="CLAP window extract", unit="song"):
        sid, emb, yv, ya = _extract_song_clap(task, embed_fn)
        if emb is None:
            continue
        tids = np.full(len(yv), sid, dtype=np.int32)
        all_emb.append(emb)
        all_yv.append(yv)
        all_ya.append(ya)
        all_tids.append(tids)
        chunk_emb.append(emb)
        chunk_yv.append(yv)
        chunk_ya.append(ya)
        chunk_tids.append(tids)

        if len(chunk_emb) >= _CHUNK_SIZE:
            _save_partial_cache(cache_file, chunk_counter, chunk_emb, chunk_yv, chunk_ya, chunk_tids)
            chunk_counter += 1
            chunk_emb, chunk_yv, chunk_ya, chunk_tids = [], [], [], []

    if chunk_emb:
        _save_partial_cache(cache_file, chunk_counter, chunk_emb, chunk_yv, chunk_ya, chunk_tids)

    if not all_emb:
        raise RuntimeError("No CLAP window embeddings were extracted.")

    X = np.concatenate(all_emb, axis=0)
    yv = np.concatenate(all_yv, axis=0).astype(np.float32)
    ya = np.concatenate(all_ya, axis=0).astype(np.float32)
    track_ids = np.concatenate(all_tids, axis=0).astype(np.int32)

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.savez_compressed(cache_file, X=X, yv=yv, ya=ya, track_ids=track_ids, embed_dim=embed_dim)
    _cleanup_partial_caches(cache_file)
    print(f"  Cached {len(X)} CLAP window embeddings to {cache_file}")
    return X, yv, ya, track_ids, embed_dim
