import librosa
import numpy as np
import pandas as pd
import os


CHUNK_DURATION = 10  # seconds, must match main.py
AUDIO_OFFSET = 15.0  # DEAM excerpts start at 15s of original song


def extract_features_from_array(y: np.ndarray, sr: int) -> dict:
    """Extract lightweight audio features for UI metadata."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    return {
        "tempo": float(tempo),
        "rms_energy": float(rms),
        "spectral_centroid": float(centroid),
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




# ---------------------------------------------------------------------------
# CNN-BiGRU mel-segment extraction
# ---------------------------------------------------------------------------

MEL_SAMPLE_RATE = 22_050
MEL_N_FFT = 2_048
MEL_HOP_LENGTH = 512
MEL_N_MELS = 64
MEL_SEGMENT_FRAMES = 64


def _slice_segment(log_mel, center_time_s):
    center_frame = int(round(center_time_s * MEL_SAMPLE_RATE / MEL_HOP_LENGTH))
    start = center_frame - MEL_SEGMENT_FRAMES // 2
    end = start + MEL_SEGMENT_FRAMES
    pad_left = max(0, -start)
    pad_right = max(0, end - log_mel.shape[1])

    if pad_left or pad_right:
        log_mel = np.pad(log_mel, ((0, 0), (pad_left, pad_right)), mode="edge")
        start += pad_left
        end += pad_left

    return log_mel[:, start:end]


def _audio_to_segments(audio, samples, time_scale=1.0):
    """Extract log-mel segments from audio with optional time scaling."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=MEL_SAMPLE_RATE,
        n_fft=MEL_N_FFT,
        hop_length=MEL_HOP_LENGTH,
        n_mels=MEL_N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel + 1e-10, ref=np.max).astype(np.float32)

    segments = []
    valence_targets = []
    arousal_targets = []
    for timestamp_s, valence, arousal in samples:
        excerpt_time_s = float(timestamp_s) - AUDIO_OFFSET
        if excerpt_time_s < 0:
            continue
        scaled_time = excerpt_time_s * time_scale

        segment = _slice_segment(log_mel, scaled_time)
        if segment.shape[1] != MEL_SEGMENT_FRAMES:
            continue

        segments.append(segment[None, :, :].astype(np.float16))
        valence_targets.append(float(valence))
        arousal_targets.append(float(arousal))

    return segments, valence_targets, arousal_targets


def _extract_song_segments(task, augment=False):
    sid, audio_path, samples = task

    try:
        audio, _ = librosa.load(audio_path, sr=MEL_SAMPLE_RATE, duration=60)

        all_segments = []
        all_valence = []
        all_arousal = []

        # Original audio
        segs, yvs, yas = _audio_to_segments(audio, samples, time_scale=1.0)
        all_segments.extend(segs)
        all_valence.extend(yvs)
        all_arousal.extend(yas)

        if augment:
            # Pitch shifting (Eq. 9: Δf ∈ [-2, 2])
            for n_steps in (-2, 2):
                y_shifted = librosa.effects.pitch_shift(audio, sr=MEL_SAMPLE_RATE, n_steps=n_steps)
                segs, yvs, yas = _audio_to_segments(y_shifted, samples, time_scale=1.0)
                all_segments.extend(segs)
                all_valence.extend(yvs)
                all_arousal.extend(yas)

            # Time stretching (Eq. 8: α ∈ (0.8, 1.2))
            for rate in (0.9, 1.1):
                y_stretched = librosa.effects.time_stretch(audio, rate=rate)
                segs, yvs, yas = _audio_to_segments(y_stretched, samples, time_scale=1.0 / rate)
                all_segments.extend(segs)
                all_valence.extend(yvs)
                all_arousal.extend(yas)

            # Noise injection (Eq. 10: η ~ N(0, σ²))
            noise_factor = 0.005
            sigma = noise_factor * np.max(np.abs(audio))
            y_noisy = audio + sigma * np.random.randn(len(audio)).astype(audio.dtype)
            segs, yvs, yas = _audio_to_segments(y_noisy, samples, time_scale=1.0)
            all_segments.extend(segs)
            all_valence.extend(yvs)
            all_arousal.extend(yas)

        if not all_segments:
            empty_x = np.empty((0, 1, MEL_N_MELS, MEL_SEGMENT_FRAMES), dtype=np.float16)
            empty_y = np.empty((0,), dtype=np.float32)
            return sid, empty_x, empty_y, empty_y

        return (
            sid,
            np.stack(all_segments, axis=0),
            np.array(all_valence, dtype=np.float32),
            np.array(all_arousal, dtype=np.float32),
        )
    except Exception as exc:
        from tqdm import tqdm

        tqdm.write(f"  Skipped {sid}: {exc}")
        empty_x = np.empty((0, 1, MEL_N_MELS, MEL_SEGMENT_FRAMES), dtype=np.float16)
        empty_y = np.empty((0,), dtype=np.float32)
        return sid, empty_x, empty_y, empty_y


def _build_song_tasks(dynamic_annotations, audio_dir="data/deam/MEMD_audio"):
    tasks = []
    for sid, samples in dynamic_annotations.items():
        path = os.path.join(audio_dir, f"{sid}.mp3")
        if os.path.exists(path):
            tasks.append((sid, path, samples))
    return tasks


def _iter_song_results(tasks, worker_count, augment=False):
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from multiprocessing import get_context
    import functools

    extract_fn = functools.partial(_extract_song_segments, augment=augment)

    if worker_count <= 1:
        from tqdm import tqdm

        with tqdm(total=len(tasks), desc="Segment prep", unit="song") as prep_bar:
            for task in tasks:
                yield extract_fn(task)
                prep_bar.update(1)
        return

    import sys
    ctx = get_context("spawn") if sys.platform == "win32" else get_context("fork")
    

    def submit_next(executor, pending, task_iter):
        next_task = next(task_iter, None)
        if next_task is None:
            return
        pending[executor.submit(extract_fn, next_task)] = next_task[0]

    with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
        task_iter = iter(tasks)
        pending = {}
        for _ in range(worker_count):
            submit_next(executor, pending, task_iter)

        from tqdm import tqdm

        with tqdm(total=len(tasks), desc="Segment prep", unit="song") as prep_bar:
            while pending:
                done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    sid = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        tqdm.write(f"  Skipped {sid}: worker failed ({exc})")
                        empty_x = np.empty((0, 1, MEL_N_MELS, MEL_SEGMENT_FRAMES), dtype=np.float16)
                        empty_y = np.empty((0,), dtype=np.float32)
                        result = (sid, empty_x, empty_y, empty_y)
                    yield result
                    prep_bar.update(1)
                    submit_next(executor, pending, task_iter)


def load_or_extract_base_segments(
    rebuild_cache=False,
    worker_count=4,
    augment_cache=False,
    cache_file=None,
):
    if cache_file is None:
        suffix = "aug" if augment_cache else "base"
        cache_file = f"data/deam/cnn_bigru_multitask_segments_cache_{suffix}.npz"

    if os.path.exists(cache_file) and not rebuild_cache:
        print(f"Loading cached segments from {cache_file}...")
        cache = np.load(cache_file)
        return cache["X"], cache["yv"], cache["ya"], cache["track_ids"]

    print("Loading dynamic annotations...")
    dynamic = load_dynamic_annotations()
    print(f"  DEAM: {len(dynamic)} songs")

    tasks = _build_song_tasks(dynamic)
    if not tasks:
        raise RuntimeError("No DEAM audio files were found for CNN-BiGRU multitask training.")

    worker_count = min(worker_count, len(tasks))
    print(f"  CPU prep workers: {worker_count}")
    if augment_cache:
        print("  Audio augmentation enabled (pitch shift, time stretch, noise injection)")

    segments = []
    y_valence = []
    y_arousal = []
    track_ids = []
    for sid, song_segments, song_yv, song_ya in _iter_song_results(tasks, worker_count, augment=augment_cache):
        if len(song_yv) == 0:
            continue
        segments.append(song_segments)
        y_valence.append(song_yv)
        y_arousal.append(song_ya)
        track_ids.append(np.full(len(song_yv), sid, dtype=np.int32))

    if not segments:
        raise RuntimeError("No labeled mel segments were extracted.")

    X = np.concatenate(segments, axis=0)
    yv = np.concatenate(y_valence, axis=0)
    ya = np.concatenate(y_arousal, axis=0)
    track_ids = np.concatenate(track_ids, axis=0)

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.savez_compressed(cache_file, X=X, yv=yv, ya=ya, track_ids=track_ids)
    print(f"  Cached {len(X)} segments to {cache_file}")
    return X, yv, ya, track_ids


def extract_mel_segments(y, sr, segment_hop_s=1.0):
    """Extract log-Mel segments from an audio signal for CNN-BiGRU inference.

    Returns a numpy array of shape (N, 1, MEL_N_MELS, MEL_SEGMENT_FRAMES)
    or None if the audio is too short.
    """
    if sr != MEL_SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=MEL_SAMPLE_RATE)

    min_samples = MEL_SEGMENT_FRAMES * MEL_HOP_LENGTH
    if len(y) < min_samples:
        return None

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=MEL_SAMPLE_RATE,
        n_fft=MEL_N_FFT,
        hop_length=MEL_HOP_LENGTH,
        n_mels=MEL_N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel + 1e-10, ref=np.max).astype(np.float32)

    duration_s = len(y) / MEL_SAMPLE_RATE
    segments = []
    for center in np.arange(segment_hop_s, duration_s, segment_hop_s):
        seg = _slice_segment(log_mel, center)
        if seg.shape[1] == MEL_SEGMENT_FRAMES:
            segments.append(seg[None, :, :])

    if not segments:
        center = duration_s / 2
        seg = _slice_segment(log_mel, center)
        if seg.shape[1] == MEL_SEGMENT_FRAMES:
            segments.append(seg[None, :, :])

    if not segments:
        return None

    return np.stack(segments, axis=0).astype(np.float32)
