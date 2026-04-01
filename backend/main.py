from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from processor import extract_features_from_array
from classifier import predict_mood
import librosa
import numpy as np
import json
import os
import scipy.signal
from pythonosc import udp_client
# scipy >= 1.8 removed hann — patch before librosa loads
if not hasattr(scipy.signal, 'hann'):
    scipy.signal.hann = scipy.signal.windows.hann

# OSC client — notifies TouchDesigner of new file + playback trigger
TD_HOST = "127.0.0.1"
TD_PORT = 7000
osc = udp_client.SimpleUDPClient(TD_HOST, TD_PORT)

# Fixed upload path so TouchDesigner always knows where to look
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
CURRENT_AUDIO_PATH = os.path.join(UPLOAD_DIR, "current.wav")

app = FastAPI(title="MoodWave API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHUNK_DURATION = 10  # seconds

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the MoodWave frontend. Place moodwave.html next to main.py."""
    html_path = os.path.join(os.path.dirname(__file__), "moodwave.html")
    if not os.path.exists(html_path):
        return HTMLResponse(
            "<h2>moodwave.html not found</h2>"
            "<p>Place <code>moodwave.html</code> in the same folder as <code>main.py</code> and restart.</p>",
            status_code=404,
        )
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/analyze-stream")
async def analyze_stream(file: UploadFile = File(...)):
    """
    Split audio into chunks, extract features and predict mood
    for each chunk, streaming results back via Server-Sent Events.
    """
    audio_bytes = await file.read()
    with open(CURRENT_AUDIO_PATH, "wb") as f:
        f.write(audio_bytes)

    # Tell TouchDesigner where the file is (forward slashes for TD compatibility)
    td_path = CURRENT_AUDIO_PATH.replace("\\", "/")
    osc.send_message("/moodwave/filepath", td_path)
    osc.send_message("/moodwave/play", 1)

    def generate():
        try:
            y, sr = librosa.load(CURRENT_AUDIO_PATH)
            total_duration = librosa.get_duration(y=y, sr=sr)
            samples_per_chunk = int(CHUNK_DURATION * sr)

            for i, start in enumerate(range(0, len(y), samples_per_chunk)):
                chunk = y[start : start + samples_per_chunk]

                # Skip chunks shorter than 5 seconds
                if len(chunk) < sr * 5:
                    continue

                features = extract_features_from_array(chunk, sr)
                prediction = predict_mood(features)

                osc.send_message("/moodwave/filepath", td_path)

                result = {
                    "chunk": i,
                    "time_start": round(i * CHUNK_DURATION, 2),
                    "time_end": round(min((i + 1) * CHUNK_DURATION, total_duration), 2),
                    "valence": prediction["valence"],
                    "arousal": prediction["arousal"],
                    "mood": prediction["distribution"],
                    "features": {
                        "tempo":      round(float(features["tempo"]), 1),
                        "energy":     round(float(features["rms_energy"]), 3),
                        "brightness": round(float(features["spectral_centroid"]), 3),
                    },
                    "total_duration": round(total_duration, 2),
                }

                yield f"data: {json.dumps(result)}\n\n"

            # Signal completion
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

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
