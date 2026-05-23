"""MuQ feature extraction for DEAM dynamic window-level VA regression.

Requires:
    pip install muq transformers

MuQ expects 24 kHz mono audio and outputs frame-level Conformer features.
DEAM dynamic annotations are sampled every 0.5 seconds, starting at 15s in
the original song. The audio clips used here are treated as excerpts, so
annotation timestamps are shifted by 15s and aligned to excerpt time.
"""

import gc
import glob
import os

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

MUQ_SR = 24000
_AUDIO_OFFSET = 15.0
_WINDOW_SECONDS = 0.5
_CHUNK_SIZE = 200


def _is_cuda_oom(exc):
    message = str(exc).lower()
    return isinstance(exc, RuntimeError) and "cuda" in message and "out of memory" in message


def _clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


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


def _window_frame_indices(frame_times, start_s, end_s):
    mask = (frame_times >= start_s) & (frame_times < end_s)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        idx = np.array([int(np.argmin(np.abs(frame_times - start_s)))], dtype=np.int64)
    return idx


def _infer_starts_from_track_ids(track_ids):
    starts = np.zeros(len(track_ids), dtype=np.float32)
    counters = {}
    for index, tid in enumerate(track_ids):
        count = counters.get(int(tid), 0)
        starts[index] = count * _WINDOW_SECONDS
        counters[int(tid)] = count + 1
    return starts


def _pad_song_sequences(sequences):
    lengths = np.array([len(seq) for seq in sequences], dtype=np.int32)
    max_len = int(lengths.max())
    return np.stack(
        [np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode="constant") for seq in sequences],
        axis=0,
    ).astype(np.float32), lengths


def _pad_song_targets(targets):
    max_len = max(len(target) for target in targets)
    return np.stack(
        [np.pad(target, (0, max_len - len(target)), mode="constant") for target in targets],
        axis=0,
    ).astype(np.float32)


def _build_song_window_sequences(X, yv, ya, track_ids, starts):
    song_ids = np.array(sorted(np.unique(track_ids)), dtype=np.int32)
    sequences = []
    valences = []
    arousals = []
    for sid in song_ids:
        idx = np.where(track_ids == sid)[0]
        order = np.argsort(starts[idx], kind="stable")
        idx = idx[order]
        sequences.append(X[idx].astype(np.float32))
        valences.append(yv[idx].astype(np.float32))
        arousals.append(ya[idx].astype(np.float32))

    X_seq, lengths = _pad_song_sequences(sequences)
    yv_seq = _pad_song_targets(valences)
    ya_seq = _pad_song_targets(arousals)
    return X_seq, lengths, yv_seq, ya_seq, song_ids


def _build_tasks(dynamic):
    audio_dir = "data/deam/MEMD_audio"
    tasks = []
    for sid, samples in dynamic.items():
        path = os.path.join(audio_dir, f"{sid}.mp3")
        if os.path.exists(path):
            tasks.append((sid, path, samples))
    return tasks


def _load_muq_model(model_name, device=None, infer_embed_dim=True):
    from muq import MuQ

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading MuQ model ({model_name}) on {device}...")
    model = MuQ.from_pretrained(model_name).to(device).eval()
    embed_dim = None
    if infer_embed_dim:
        with torch.inference_mode():
            dummy_wav = torch.zeros(1, MUQ_SR * 2, device=device)
            dummy_out = model(dummy_wav, output_hidden_states=True)
            embed_dim = dummy_out.last_hidden_state.shape[-1]
        del dummy_wav, dummy_out
        _clear_memory()
        print(f"  MuQ embedding dimension: {embed_dim}")
    return model, embed_dim, device


def _run_muq_hidden(y, muq_model, device):
    wav = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        output = muq_model(wav, output_hidden_states=True)
        hidden = output.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)
    del wav, output
    return hidden


def _reload_muq_on_cpu(model_name):
    tqdm.write("  Reloading MuQ on CPU and continuing extraction there.")
    _clear_memory()
    return _load_muq_model(model_name, device=torch.device("cpu"), infer_embed_dim=False)


def _extract_hidden(task, muq_model, device):
    sid, audio_path, samples = task
    if not os.path.exists(audio_path):
        return sid, None, None, None, None, None, None

    y, _ = librosa.load(audio_path, sr=MUQ_SR, mono=True)
    if len(y) == 0:
        return sid, None, None, None, None, None, None

    starts, valence, arousal = _annotation_arrays(samples)
    if len(starts) == 0:
        return sid, None, None, None, None, None, None

    hidden = _run_muq_hidden(y, muq_model, device)

    duration_s = len(y) / MUQ_SR
    frame_times = np.linspace(0.0, duration_s, hidden.shape[0], endpoint=False, dtype=np.float32)
    return sid, hidden, frame_times, valence, arousal, starts, duration_s


def _extract_song_muq(task, muq_model, device):
    """Extract one pooled MuQ embedding per 0.5s dynamic annotation window."""
    try:
        sid, hidden, frame_times, valence, arousal, starts, duration_s = _extract_hidden(task, muq_model, device)
        if hidden is None:
            return sid, None, None, None, None

        embeddings = []
        kept_valence = []
        kept_arousal = []
        kept_starts = []
        for start_s, v, a in zip(starts, valence, arousal):
            if start_s >= duration_s:
                continue
            idx = _window_frame_indices(frame_times, start_s, start_s + _WINDOW_SECONDS)
            embeddings.append(hidden[idx].mean(axis=0).astype(np.float32))
            kept_valence.append(v)
            kept_arousal.append(a)
            kept_starts.append(start_s)

        if not embeddings:
            return sid, None, None, None, None

        return (
            sid,
            np.stack(embeddings, axis=0).astype(np.float32),
            np.array(kept_valence, dtype=np.float32),
            np.array(kept_arousal, dtype=np.float32),
            np.array(kept_starts, dtype=np.float32),
        )
    except Exception as exc:
        if _is_cuda_oom(exc):
            raise
        tqdm.write(f"  Skipped {task[0]}: {exc}")
        return task[0], None, None, None, None


def _save_vector_partial_cache(cache_file, chunk_idx, embeddings, valences, arousals, track_ids, starts):
    np.savez_compressed(
        _partial_cache_path(cache_file, chunk_idx),
        X=np.concatenate(embeddings, axis=0),
        yv=np.concatenate(valences, axis=0).astype(np.float32),
        ya=np.concatenate(arousals, axis=0).astype(np.float32),
        track_ids=np.concatenate(track_ids, axis=0).astype(np.int32),
        starts=np.concatenate(starts, axis=0).astype(np.float32),
    )


def _load_vector_partial_caches(cache_file):
    base, ext = os.path.splitext(cache_file)
    files = sorted(glob.glob(f"{base}_partial_*{ext}"))
    if not files:
        return set(), [], [], [], [], []

    done_sids = set()
    all_emb, all_yv, all_ya, all_tids, all_starts = [], [], [], [], []
    for path in files:
        try:
            data = np.load(path)
            track_ids = data["track_ids"]
            done_sids.update(set(np.unique(track_ids).tolist()))
            all_emb.append(data["X"])
            all_yv.append(data["yv"])
            all_ya.append(data["ya"])
            all_tids.append(track_ids)
            all_starts.append(data["starts"] if "starts" in data else _infer_starts_from_track_ids(track_ids))
        except Exception:
            continue

    print(f"  Found {len(done_sids)} songs in partial caches; resuming...")
    return done_sids, all_emb, all_yv, all_ya, all_tids, all_starts


def load_or_extract_muq_embeddings(
    rebuild_cache=False,
    cache_file=None,
    model_name="OpenMuQ/MuQ-large-msd-iter",
    return_starts=False,
):
    """Extract or load cached MuQ window-level embeddings for DEAM.

    Returns
    -------
    X : np.ndarray, shape (N_windows, embed_dim)
    yv : np.ndarray, shape (N_windows,)
    ya : np.ndarray, shape (N_windows,)
    track_ids : np.ndarray, shape (N_windows,)
    starts : np.ndarray, shape (N_windows,), returned when return_starts=True
    embed_dim : int
    """
    if cache_file is None:
        cache_file = "data/deam/muq_window_embeddings_cache.npz"

    if os.path.exists(cache_file) and not rebuild_cache:
        print(f"Loading cached MuQ window embeddings from {cache_file}...")
        cache = np.load(cache_file)
        starts = cache["starts"] if "starts" in cache else _infer_starts_from_track_ids(cache["track_ids"])
        if return_starts:
            return cache["X"], cache["yv"], cache["ya"], cache["track_ids"], starts, int(cache["embed_dim"])
        return cache["X"], cache["yv"], cache["ya"], cache["track_ids"], int(cache["embed_dim"])

    if rebuild_cache:
        _cleanup_partial_caches(cache_file)

    muq_model, embed_dim, device = _load_muq_model(model_name)

    print("Loading dynamic annotations...")
    dynamic = load_dynamic_annotations()
    print(f"  DEAM: {len(dynamic)} songs")
    tasks = _build_tasks(dynamic)
    if not tasks:
        raise RuntimeError("No DEAM audio files found for MuQ extraction.")

    done_sids, all_emb, all_yv, all_ya, all_tids, all_starts = _load_vector_partial_caches(cache_file)
    tasks = [task for task in tasks if task[0] not in done_sids]

    if not tasks:
        X = np.concatenate(all_emb, axis=0)
        yv = np.concatenate(all_yv, axis=0)
        ya = np.concatenate(all_ya, axis=0)
        track_ids = np.concatenate(all_tids, axis=0)
        starts = np.concatenate(all_starts, axis=0)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez_compressed(cache_file, X=X, yv=yv, ya=ya, track_ids=track_ids, starts=starts, embed_dim=embed_dim)
        _cleanup_partial_caches(cache_file)
        print(f"  Merged partial caches into {len(X)} MuQ window embeddings at {cache_file}")
        if return_starts:
            return X, yv, ya, track_ids, starts, embed_dim
        return X, yv, ya, track_ids, embed_dim

    print(f"  Extracting MuQ window embeddings for {len(tasks)} songs...")
    chunk_emb, chunk_yv, chunk_ya, chunk_tids, chunk_starts = [], [], [], [], []
    chunk_counter = _next_partial_index(cache_file)

    for task in tqdm(tasks, desc="MuQ window extract", unit="song"):
        try:
            sid, emb, yv, ya, starts = _extract_song_muq(task, muq_model, device)
        except RuntimeError as exc:
            if device.type != "cuda" or not _is_cuda_oom(exc):
                raise
            tqdm.write(f"  Song {task[0]}: CUDA OOM during MuQ extraction.")
            del muq_model
            muq_model, _, device = _reload_muq_on_cpu(model_name)
            sid, emb, yv, ya, starts = _extract_song_muq(task, muq_model, device)
        _clear_memory()
        if emb is None:
            continue
        tids = np.full(len(yv), sid, dtype=np.int32)
        all_emb.append(emb)
        all_yv.append(yv)
        all_ya.append(ya)
        all_tids.append(tids)
        all_starts.append(starts)
        chunk_emb.append(emb)
        chunk_yv.append(yv)
        chunk_ya.append(ya)
        chunk_tids.append(tids)
        chunk_starts.append(starts)

        if len(chunk_emb) >= _CHUNK_SIZE:
            _save_vector_partial_cache(cache_file, chunk_counter, chunk_emb, chunk_yv, chunk_ya, chunk_tids, chunk_starts)
            chunk_counter += 1
            chunk_emb, chunk_yv, chunk_ya, chunk_tids, chunk_starts = [], [], [], [], []

    if chunk_emb:
        _save_vector_partial_cache(cache_file, chunk_counter, chunk_emb, chunk_yv, chunk_ya, chunk_tids, chunk_starts)

    if not all_emb:
        raise RuntimeError("No MuQ window embeddings were extracted.")

    X = np.concatenate(all_emb, axis=0)
    yv = np.concatenate(all_yv, axis=0).astype(np.float32)
    ya = np.concatenate(all_ya, axis=0).astype(np.float32)
    track_ids = np.concatenate(all_tids, axis=0).astype(np.int32)
    starts = np.concatenate(all_starts, axis=0).astype(np.float32)

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.savez_compressed(cache_file, X=X, yv=yv, ya=ya, track_ids=track_ids, starts=starts, embed_dim=embed_dim)
    _cleanup_partial_caches(cache_file)
    print(f"  Cached {len(X)} MuQ window embeddings to {cache_file}")
    if return_starts:
        return X, yv, ya, track_ids, starts, embed_dim
    return X, yv, ya, track_ids, embed_dim


def _extract_song_muq_seq(task, muq_model, device):
    """Extract local MuQ frame sequences for each dynamic annotation window."""
    try:
        sid, hidden, frame_times, valence, arousal, starts, duration_s = _extract_hidden(task, muq_model, device)
        if hidden is None:
            return sid, None, None, None

        sequences = []
        kept_valence = []
        kept_arousal = []
        for start_s, v, a in zip(starts, valence, arousal):
            if start_s >= duration_s:
                continue
            idx = _window_frame_indices(frame_times, start_s, start_s + _WINDOW_SECONDS)
            sequences.append(hidden[idx].astype(np.float32))
            kept_valence.append(v)
            kept_arousal.append(a)

        if not sequences:
            return sid, None, None, None

        return sid, sequences, np.array(kept_valence, dtype=np.float32), np.array(kept_arousal, dtype=np.float32)
    except Exception as exc:
        if _is_cuda_oom(exc):
            raise
        tqdm.write(f"  Skipped {task[0]}: {exc}")
        return task[0], None, None, None


def _save_seq_partial_cache(cache_file, chunk_idx, sequences, lengths, valences, arousals, track_ids):
    np.savez_compressed(
        _partial_cache_path(cache_file, chunk_idx),
        X=np.array(sequences, dtype=object),
        lengths=np.array(lengths, dtype=np.int32),
        yv=np.array(valences, dtype=np.float32),
        ya=np.array(arousals, dtype=np.float32),
        track_ids=np.array(track_ids, dtype=np.int32),
    )


def _load_seq_partial_caches(cache_file):
    base, ext = os.path.splitext(cache_file)
    files = sorted(glob.glob(f"{base}_partial_*{ext}"))
    if not files:
        return set(), [], [], [], [], []

    done_sids = set()
    all_seq, all_len, all_yv, all_ya, all_tids = [], [], [], [], []
    for path in files:
        try:
            data = np.load(path, allow_pickle=True)
            done_sids.update(set(np.unique(data["track_ids"]).tolist()))
            all_seq.extend(list(data["X"]))
            all_len.extend(list(data["lengths"]))
            all_yv.extend(list(data["yv"]))
            all_ya.extend(list(data["ya"]))
            all_tids.extend(list(data["track_ids"]))
        except Exception:
            continue

    print(f"  Found {len(done_sids)} songs in partial caches; resuming...")
    return done_sids, all_seq, all_len, all_yv, all_ya, all_tids


def _pad_sequences(sequences, lengths):
    max_len = max(lengths)
    return np.stack(
        [np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode="constant") for seq in sequences],
        axis=0,
    ).astype(np.float32)


def load_or_extract_muq_sequences(
    rebuild_cache=False,
    cache_file=None,
    model_name="OpenMuQ/MuQ-large-msd-iter",
):
    """Load MuQ embeddings and group them into per-song window sequences.

    Returns
    -------
    X : np.ndarray, shape (N_songs, T_max, embed_dim)
    lengths : np.ndarray, shape (N_songs,)
    yv : np.ndarray, shape (N_songs, T_max)
    ya : np.ndarray, shape (N_songs, T_max)
    track_ids : np.ndarray, shape (N_songs,)
    embed_dim : int
    """
    X, yv, ya, track_ids, starts, embed_dim = load_or_extract_muq_embeddings(
        rebuild_cache=rebuild_cache,
        cache_file=cache_file,
        model_name=model_name,
        return_starts=True,
    )
    X_seq, lengths, yv_seq, ya_seq, song_ids = _build_song_window_sequences(X, yv, ya, track_ids, starts)
    return X_seq, lengths, yv_seq, ya_seq, song_ids, embed_dim


def load_or_extract_muq_frame_sequences(
    rebuild_cache=False,
    cache_file=None,
    model_name="OpenMuQ/MuQ-large-msd-iter",
):
    """Extract or load cached local MuQ frame sequences for each DEAM window."""
    if cache_file is None:
        cache_file = "data/deam/muq_window_frame_sequences_cache.npz"

    if os.path.exists(cache_file) and not rebuild_cache:
        print(f"Loading cached MuQ window frame sequences from {cache_file}...")
        cache = np.load(cache_file)
        return (
            cache["X"],
            cache["lengths"],
            cache["yv"],
            cache["ya"],
            cache["track_ids"],
            int(cache["embed_dim"]),
        )

    if rebuild_cache:
        _cleanup_partial_caches(cache_file)

    muq_model, embed_dim, device = _load_muq_model(model_name)

    print("Loading dynamic annotations...")
    dynamic = load_dynamic_annotations()
    print(f"  DEAM: {len(dynamic)} songs")
    tasks = _build_tasks(dynamic)
    if not tasks:
        raise RuntimeError("No DEAM audio files found for MuQ sequence extraction.")

    done_sids, all_seq, all_len, all_yv, all_ya, all_tids = _load_seq_partial_caches(cache_file)
    tasks = [task for task in tasks if task[0] not in done_sids]

    if not tasks:
        X = _pad_sequences(all_seq, all_len)
        lengths = np.array(all_len, dtype=np.int32)
        yv = np.array(all_yv, dtype=np.float32)
        ya = np.array(all_ya, dtype=np.float32)
        track_ids = np.array(all_tids, dtype=np.int32)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez_compressed(cache_file, X=X, lengths=lengths, yv=yv, ya=ya, track_ids=track_ids, embed_dim=embed_dim)
        _cleanup_partial_caches(cache_file)
        print(f"  Merged partial caches into {len(X)} MuQ window sequences at {cache_file}")
        return X, lengths, yv, ya, track_ids, embed_dim

    print(f"  Extracting MuQ window sequences for {len(tasks)} songs...")
    chunk_seq, chunk_len, chunk_yv, chunk_ya, chunk_tids = [], [], [], [], []
    chunk_counter = _next_partial_index(cache_file)

    for task in tqdm(tasks, desc="MuQ window seq extract", unit="song"):
        try:
            sid, seqs, yv, ya = _extract_song_muq_seq(task, muq_model, device)
        except RuntimeError as exc:
            if device.type != "cuda" or not _is_cuda_oom(exc):
                raise
            tqdm.write(f"  Song {task[0]}: CUDA OOM during MuQ sequence extraction.")
            del muq_model
            muq_model, _, device = _reload_muq_on_cpu(model_name)
            sid, seqs, yv, ya = _extract_song_muq_seq(task, muq_model, device)
        _clear_memory()
        if seqs is None:
            continue
        lengths = [seq.shape[0] for seq in seqs]
        tids = np.full(len(seqs), sid, dtype=np.int32)
        all_seq.extend(seqs)
        all_len.extend(lengths)
        all_yv.extend(list(yv))
        all_ya.extend(list(ya))
        all_tids.extend(list(tids))
        chunk_seq.extend(seqs)
        chunk_len.extend(lengths)
        chunk_yv.extend(list(yv))
        chunk_ya.extend(list(ya))
        chunk_tids.extend(list(tids))

        if len(chunk_seq) >= _CHUNK_SIZE:
            _save_seq_partial_cache(cache_file, chunk_counter, chunk_seq, chunk_len, chunk_yv, chunk_ya, chunk_tids)
            chunk_counter += 1
            chunk_seq, chunk_len, chunk_yv, chunk_ya, chunk_tids = [], [], [], [], []

    if chunk_seq:
        _save_seq_partial_cache(cache_file, chunk_counter, chunk_seq, chunk_len, chunk_yv, chunk_ya, chunk_tids)

    if not all_seq:
        raise RuntimeError("No MuQ window sequences were extracted.")

    X = _pad_sequences(all_seq, all_len)
    lengths = np.array(all_len, dtype=np.int32)
    yv = np.array(all_yv, dtype=np.float32)
    ya = np.array(all_ya, dtype=np.float32)
    track_ids = np.array(all_tids, dtype=np.int32)

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.savez_compressed(cache_file, X=X, lengths=lengths, yv=yv, ya=ya, track_ids=track_ids, embed_dim=embed_dim)
    _cleanup_partial_caches(cache_file)
    print(f"  Cached {len(X)} MuQ window sequences to {cache_file}")
    return X, lengths, yv, ya, track_ids, embed_dim
