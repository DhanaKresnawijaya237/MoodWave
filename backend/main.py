import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime

import joblib
import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from core.model import MoodCNNBiGRU
from features.processor import MEL_N_MELS, MEL_WINDOW_SECONDS, extract_features_from_array, extract_mel_segments
from core.utils import valence_arousal_to_mood_distribution
from core.separator import separate_stems

app = FastAPI(title="MoodWave API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHUNK_DURATION = 10  # seconds
MOODS = ["energetic", "happy", "calm", "romantic", "sad", "angry"]
STEM_NAMES = ["vocals", "drums", "bass", "guitar", "piano", "other"]

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
FRONTEND_PATH = os.path.join(PROJECT_ROOT, "frontend", "moodwave.html")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
CURRENT_AUDIO_PATH = os.path.join(UPLOAD_DIR, "current.wav")
_latest_audio_path = CURRENT_AUDIO_PATH

SAVED_ANALYSES_DIR = os.path.join(BASE_DIR, "saved_analyses")
os.makedirs(SAVED_ANALYSES_DIR, exist_ok=True)

# Lazy-loaded CNN-BiGRU cache
_cnnbigru_cache = None


def _stats_to_numpy(stats):
    return {
        key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
        for key, value in stats.items()
    }


def _load_cnnbigru():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/weights/cnn_bigru_multitask_paper.pth"
    legacy_model_path = "models/cnn_bigru_multitask_paper.pth"
    legacy_mel_stats_path = "models/cnn_bigru_multitask_paper_mel_stats.pkl"

    if not os.path.exists(model_path) and os.path.exists(legacy_model_path):
        model_path = legacy_model_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_kwargs = dict(checkpoint.get("model_kwargs", {}))
        in_ch = checkpoint.get("in_channels", model_kwargs.pop("in_channels", 2))
        model_kwargs.pop("in_channels", None)
        use_qh = checkpoint.get("use_quadrant_head", False)
    else:
        state_dict = checkpoint
        model_kwargs = {}
        in_ch = 2
        use_qh = False

    model = MoodCNNBiGRU(
        n_mels=MEL_N_MELS,
        in_channels=in_ch,
        num_quadrant_classes=4 if use_qh else 0,
        **model_kwargs,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    mel_stats = checkpoint.get("feature_stats") if isinstance(checkpoint, dict) else None
    if mel_stats is None:
        if not os.path.exists(legacy_mel_stats_path):
            raise FileNotFoundError(
                "Feature normalization stats are missing from the checkpoint. "
                f"Retrain with train.py or keep the legacy stats file at {legacy_mel_stats_path}."
            )
        mel_stats = joblib.load(legacy_mel_stats_path)
    else:
        mel_stats = _stats_to_numpy(mel_stats)
    return model, mel_stats, device


def _predict_cnnbigru(y, sr):
    global _cnnbigru_cache
    if _cnnbigru_cache is None:
        _cnnbigru_cache = _load_cnnbigru()
    model, mel_stats, device = _cnnbigru_cache

    extracted = extract_mel_segments(y, sr, return_times=True)
    if extracted is None:
        return None
    segments, start_times = extracted

    mean = mel_stats["mean"]
    std = mel_stats["std"]
    segments = (segments.astype(np.float32) - mean) / (std + 1e-6)

    with torch.no_grad():
        x = torch.from_numpy(segments).to(device)
        model_out = model(x)
        pred_v, pred_a = model_out[0], model_out[1]
        pred_v_np = pred_v.cpu().numpy()
        pred_a_np = pred_a.cpu().numpy()
        valence = float(pred_v_np.mean())
        arousal = float(pred_a_np.mean())

    timeline = []
    for start_s, window_v, window_a in zip(start_times, pred_v_np, pred_a_np):
        v = float(window_v)
        a = float(window_a)
        timeline.append({
            "start": round(float(start_s), 2),
            "end": round(float(start_s + MEL_WINDOW_SECONDS), 2),
            "valence": round(v, 3),
            "arousal": round(a, 3),
            "distribution": valence_arousal_to_mood_distribution(v, a),
        })

    return {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "distribution": valence_arousal_to_mood_distribution(valence, arousal),
        "timeline": timeline,
    }


def _write_touchdesigner_wav(input_path, output_path, target_sr=44_100):
    """Decode uploaded audio and write a conservative WAV for TouchDesigner."""
    try:
        audio_out, sr = sf.read(input_path, always_2d=True, dtype="float32")
    except Exception:
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        audio_out = np.column_stack([audio, audio]) if audio.ndim == 1 else audio.T

    if sr != target_sr:
        from math import gcd
        from scipy.signal import resample_poly

        common = gcd(int(sr), int(target_sr))
        audio_out = resample_poly(audio_out, target_sr // common, sr // common, axis=0).astype(np.float32)

    if audio_out.shape[1] == 1:
        audio_out = np.repeat(audio_out, 2, axis=1)
    elif audio_out.shape[1] > 2:
        audio_out = audio_out[:, :2]
    sf.write(output_path, audio_out, target_sr, subtype="PCM_16", format="WAV")


def _td_path(path):
    return os.path.abspath(path).replace("\\", "/")


def _slugify(name):
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-").lower()
    return (slug or "analysis")[:60]


def _saved_dir(saved_id):
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", saved_id):
        raise HTTPException(status_code=404, detail="Saved analysis not found")
    path = os.path.join(SAVED_ANALYSES_DIR, saved_id)
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail="Saved analysis not found")
    return path


def _ensure_saved_child_path(path):
    root = os.path.abspath(SAVED_ANALYSES_DIR)
    target = os.path.abspath(path)
    if os.path.commonpath([root, target]) != root or target == root:
        raise HTTPException(status_code=400, detail="Invalid saved analysis path")
    return target


def _resolve_audio_source(payload):
    candidates = []
    payload_path = payload.get("filepath")
    if payload_path:
        candidates.append(os.path.normpath(str(payload_path)))
    candidates.append(_latest_audio_path)
    candidates.append(CURRENT_AUDIO_PATH)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise HTTPException(status_code=404, detail="No analyzed audio file is available to save")


def _summarize_chunks(chunks):
    summary = {
        "average_valence": 0.0,
        "average_arousal": 0.0,
        "average_mood": {mood: 0.0 for mood in MOODS},
        "dominant_mood": None,
        "dominant_counts": {mood: 0 for mood in MOODS},
        "average_features": {"tempo": 0.0, "energy": 0.0, "brightness": 0.0},
    }
    if not chunks:
        return summary

    n = len(chunks)
    summary["average_valence"] = round(sum(float(c.get("valence", 0)) for c in chunks) / n, 4)
    summary["average_arousal"] = round(sum(float(c.get("arousal", 0)) for c in chunks) / n, 4)

    for mood in MOODS:
        summary["average_mood"][mood] = round(
            sum(float((c.get("mood") or {}).get(mood, 0)) for c in chunks) / n,
            4,
        )

    for chunk in chunks:
        dominant = chunk.get("dominant")
        if dominant in summary["dominant_counts"]:
            summary["dominant_counts"][dominant] += 1
    summary["dominant_mood"] = max(summary["dominant_counts"], key=summary["dominant_counts"].get)

    for key in summary["average_features"]:
        summary["average_features"][key] = round(
            sum(float((c.get("features") or {}).get(key, 0)) for c in chunks) / n,
            4,
        )
    return summary


def _metadata_response(saved_id, include_chunks=False):
    path = _saved_dir(saved_id)
    metadata_path = os.path.join(path, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Saved analysis metadata not found")

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    audio_file = metadata.get("audio_file")
    if audio_file:
        metadata["filepath"] = _td_path(os.path.join(path, audio_file))

    stems = {}
    for name, rel_path in (metadata.get("stems") or {}).items():
        stem_path = os.path.join(path, rel_path)
        if os.path.exists(stem_path):
            stems[name] = _td_path(stem_path)
    metadata["stems"] = stems

    if not include_chunks:
        metadata.pop("chunks", None)
    return metadata


@app.get("/")
def frontend():
    """Serve the MoodWave frontend for launcher-based startup."""
    if not os.path.exists(FRONTEND_PATH):
        raise HTTPException(status_code=404, detail="frontend/moodwave.html not found")
    return FileResponse(FRONTEND_PATH, media_type="text/html")


@app.post("/analyze-stream")
async def analyze_stream(
    file: UploadFile = File(...),
    encoder: str = Form("librosa"),
):
    """
    Split audio into chunks, extract features and predict mood
    for each chunk, streaming results back via Server-Sent Events.

    The uploaded audio is persisted to backend/uploads/current.wav so
    that TouchDesigner can load it directly by file path.
    """
    global _latest_audio_path

    run_id = uuid.uuid4().hex[:12]
    run_dir = os.path.join(UPLOAD_DIR, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    audio_path = os.path.join(run_dir, "current.wav")

    # Persist to a stable per-run path for TD to read from. The browser may upload MP3,
    # FLAC, etc.; always decode and rewrite as real PCM WAV because TD relies
    # more heavily on the file extension/RIFF header matching.
    raw_upload_path = os.path.join(run_dir, "upload_source")
    with open(raw_upload_path, "wb") as out:
        out.write(await file.read())
    try:
        _write_touchdesigner_wav(raw_upload_path, audio_path)
    finally:
        if os.path.exists(raw_upload_path):
            os.remove(raw_upload_path)
    _latest_audio_path = audio_path

    # Forward-slash path for Windows compatibility inside TD
    td_filepath = _td_path(audio_path)

    def generate():
        y, sr = librosa.load(audio_path)
        total_duration = librosa.get_duration(y=y, sr=sr)
        samples_per_chunk = int(CHUNK_DURATION * sr)

        # Announce the saved filepath up front so the frontend can forward it
        yield f"data: {json.dumps({'filepath': td_filepath, 'total_duration': round(total_duration, 2)})}\n\n"

        # Run stem separation. Demucs does not expose fine-grained progress
        # here, so the frontend animates between these coarse milestones.
        yield f"data: {json.dumps({'status': 'separating', 'progress': 5, 'message': 'Preparing stem separation'})}\n\n"
        stems = None
        try:
            yield f"data: {json.dumps({'status': 'separating', 'progress': 18, 'message': 'Separating vocals, drums, bass, guitar, piano, and other stems'})}\n\n"
            stems = separate_stems(audio_path, run_dir)
            yield f"data: {json.dumps({'status': 'separating', 'progress': 96, 'message': 'Saving separated stems'})}\n\n"
        except Exception as exc:
            print(f"[Separator] Stem separation failed: {exc}")
            yield f"data: {json.dumps({'status': 'separating', 'progress': 96, 'message': 'Stem separation skipped'})}\n\n"

        # Transition back to analyzing as chunk inference begins
        yield f"data: {json.dumps({'status': 'analyzing', 'progress': 100, 'message': 'Analyzing mood timeline'})}\n\n"

        for i, start in enumerate(range(0, len(y), samples_per_chunk)):
            chunk = y[start : start + samples_per_chunk]

            # Skip chunks shorter than 5 seconds
            if len(chunk) < sr * 5:
                continue

            prediction = _predict_cnnbigru(chunk, sr)
            if prediction is None:
                continue

            features = extract_features_from_array(chunk, sr)
            features_info = {
                "tempo": round(float(features["tempo"]), 1),
                "energy": round(float(features["rms_energy"]), 3),
                "brightness": round(float(features["spectral_centroid"]), 3),
            }

            result = {
                "chunk": i,
                "time_start": round(i * CHUNK_DURATION, 2),
                "time_end": round(min((i + 1) * CHUNK_DURATION, total_duration), 2),
                "valence": prediction["valence"],
                "arousal": prediction["arousal"],
                "mood": prediction["distribution"],
                "timeline": prediction["timeline"],
                "encoder": "cnnbigru",
                "features": features_info,
                "total_duration": round(total_duration, 2),
            }

            yield f"data: {json.dumps(result)}\n\n"

        # Signal completion (include stem paths if separation succeeded)
        done_payload = {"done": True}
        if stems:
            done_payload["stems"] = {k: v.replace("\\", "/") for k, v in stems.items()}
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/saved-analyses")
async def save_analysis(request: Request):
    payload = await request.json()
    name = str(payload.get("name") or "").strip()
    chunks = payload.get("chunks")
    if not name:
        raise HTTPException(status_code=400, detail="A save name is required")
    if not isinstance(chunks, list) or not chunks:
        raise HTTPException(status_code=400, detail="No analyzed chunks were provided")

    saved_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{_slugify(name)}_{uuid.uuid4().hex[:8]}"
    save_dir = os.path.join(SAVED_ANALYSES_DIR, saved_id)
    stems_dir = os.path.join(save_dir, "stems")
    os.makedirs(stems_dir, exist_ok=True)

    audio_src = _resolve_audio_source(payload)
    audio_dst_name = "audio" + (os.path.splitext(audio_src)[1] or ".wav")
    shutil.copy2(audio_src, os.path.join(save_dir, audio_dst_name))

    saved_stems = {}
    input_stems = payload.get("stems") or {}
    if isinstance(input_stems, dict):
        for stem_name in STEM_NAMES:
            src = input_stems.get(stem_name)
            if not src:
                continue
            src_path = os.path.normpath(str(src))
            if not os.path.exists(src_path):
                continue
            ext = os.path.splitext(src_path)[1] or ".wav"
            stem_file = f"{stem_name}{ext}"
            shutil.copy2(src_path, os.path.join(stems_dir, stem_file))
            saved_stems[stem_name] = os.path.join("stems", stem_file).replace("\\", "/")

    duration = float(payload.get("duration") or chunks[-1].get("timeEnd") or 0)
    metadata = {
        "id": saved_id,
        "name": name,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_filename": payload.get("source_filename") or "",
        "duration": round(duration, 3),
        "chunk_duration": float(payload.get("chunk_duration") or CHUNK_DURATION),
        "num_chunks": len(chunks),
        "audio_file": audio_dst_name,
        "stems": saved_stems,
        "summary": _summarize_chunks(chunks),
        "chunks": chunks,
    }

    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return _metadata_response(saved_id, include_chunks=True)


@app.get("/saved-analyses")
def list_saved_analyses():
    items = []
    for saved_id in sorted(os.listdir(SAVED_ANALYSES_DIR), reverse=True):
        path = os.path.join(SAVED_ANALYSES_DIR, saved_id)
        if not os.path.isdir(path):
            continue
        try:
            items.append(_metadata_response(saved_id, include_chunks=False))
        except HTTPException:
            continue
    return {"items": items}


@app.get("/saved-analyses/{saved_id}")
def get_saved_analysis(saved_id: str):
    return _metadata_response(saved_id, include_chunks=True)


@app.delete("/saved-analyses/{saved_id}")
def delete_saved_analysis(saved_id: str):
    path = _ensure_saved_child_path(_saved_dir(saved_id))
    shutil.rmtree(path)
    return {"deleted": True, "id": saved_id}


@app.get("/audio")
def current_audio():
    """Serve the most recently uploaded audio file."""
    if not os.path.exists(_latest_audio_path):
        return {"error": "no audio uploaded yet"}
    return FileResponse(_latest_audio_path, media_type="audio/wav")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
