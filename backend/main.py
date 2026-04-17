import json
import os
import tempfile

import joblib
import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from core.model import MoodCNNBiGRU
from features.processor import MEL_N_MELS, extract_features_from_array, extract_mel_segments
from core.utils import valence_arousal_to_mood_distribution

app = FastAPI(title="MoodWave API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHUNK_DURATION = 10  # seconds

# Lazy-loaded CNN-BiGRU cache
_cnnbigru_cache = None


def _load_cnnbigru():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/cnn_bigru_multitask_paper.pth"
    mel_stats_path = "models/cnn_bigru_multitask_paper_mel_stats.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    if not os.path.exists(mel_stats_path):
        raise FileNotFoundError(f"Mel stats not found at {mel_stats_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model_kwargs = checkpoint.get("model_kwargs", {})
    model = MoodCNNBiGRU(n_mels=MEL_N_MELS, **model_kwargs).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    mel_stats = joblib.load(mel_stats_path)
    return model, mel_stats, device


def _predict_cnnbigru(y, sr):
    global _cnnbigru_cache
    if _cnnbigru_cache is None:
        _cnnbigru_cache = _load_cnnbigru()
    model, mel_stats, device = _cnnbigru_cache

    segments = extract_mel_segments(y, sr)
    if segments is None:
        return None

    mean = mel_stats["mean"]
    std = mel_stats["std"]
    segments = (segments.astype(np.float32) - mean) / (std + 1e-6)

    with torch.no_grad():
        x = torch.from_numpy(segments).to(device)
        pred_v, pred_a = model(x)
        valence = float(pred_v.mean().cpu().numpy())
        arousal = float(pred_a.mean().cpu().numpy())



    return {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "distribution": valence_arousal_to_mood_distribution(valence, arousal),
    }


@app.post("/analyze-stream")
async def analyze_stream(
    file: UploadFile = File(...),
    encoder: str = Form("librosa"),
):
    """
    Split audio into chunks, extract features and predict mood
    for each chunk, streaming results back via Server-Sent Events.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    def generate():
        try:
            y, sr = librosa.load(tmp_path)
            total_duration = librosa.get_duration(y=y, sr=sr)
            samples_per_chunk = int(CHUNK_DURATION * sr)

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
                    "encoder": "cnnbigru",
                    "features": features_info,
                    "total_duration": round(total_duration, 2),
                }

                yield f"data: {json.dumps(result)}\n\n"

            # Signal completion
            yield f"data: {json.dumps({'done': True})}\n\n"

        finally:
            os.unlink(tmp_path)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
