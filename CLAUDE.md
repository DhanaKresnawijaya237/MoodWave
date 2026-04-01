# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MoodWave** — a music mood analysis system for CS330 (Multimedia Information Processing). Uploads audio, extracts features, predicts valence/arousal via pre-trained MLP models, and streams results to a web UI. Optionally bridges mood data to TouchDesigner via OSC.

## Running the Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The frontend (`moodwave.html`) is served at `http://localhost:8000/`.

## OSC Bridge (TouchDesigner integration)

Run one of these alongside the backend — they are independent scripts:

```bash
python osc_bridge.py               # minimal bridge
python moodwave_osc_bridge.py      # extended bridge with more OSC channels
python osc_debug.py                # diagnose OSC connectivity to TouchDesigner
```

## Model Training

Training requires the **DEAM dataset** audio files. Edit paths in `train.py` then:

```bash
cd backend
python train.py      # trains valence & arousal MLPs, saves to models/
python generator.py  # regenerates model artifacts (input_dim.pkl, scaler.pkl)
```

## Running Inference on a Single File

```bash
cd backend
python infer.py <audio_file.mp3>   # prints per-chunk mood + writes JSON output
```

## Architecture

### Data Flow

```
Audio Upload → main.py (FastAPI/SSE)
    → processor.py  (45-D feature extraction per 10s chunk)
    → classifier.py (valence + arousal MLP inference → 6-mood distribution)
    → SSE stream to moodwave.html (Canvas visualizations)
    → [optional] OSC bridge → TouchDesigner (UDP port 7000)
```

### Key Modules

| File | Role |
|---|---|
| `backend/main.py` | FastAPI server; `/analyze-stream` SSE endpoint; serves HTML |
| `backend/processor.py` | librosa feature extraction (tempo, MFCCs, chroma, spectral contrast, tonnetz → 45-D vector) |
| `backend/model.py` | `MoodMLP`: 4-layer MLP (Input → 512 → 256 → 64 → 1) with BatchNorm + Dropout + Tanh |
| `backend/classifier.py` | Loads valence/arousal models; maps (v, a) to 6 moods via Euclidean distance |
| `backend/train.py` | Training pipeline: 70/15/15 split, Adam + warmup + ReduceLROnPlateau, early stopping |
| `backend/infer.py` | Standalone CLI inference |
| `backend/generator.py` | Regenerates model artifacts in `models/` |
| `backend/moodwave.html` | Single-page frontend (vanilla JS + Canvas API) |
| `osc_bridge.py` / `moodwave_osc_bridge.py` | WebSocket → OSC forwarding to TouchDesigner |

### Pre-trained Models (`backend/models/`)

- `valence_mlp.pth`, `arousal_mlp.pth` — 690 KB each (Input→512→256→64→1)
- `mood_mlp.pth` — auxiliary model
- `input_dim.pkl` — feature dimension = 45
- `scaler.pkl` — `StandardScaler` for feature normalization

### Mood Mapping

Six moods are defined by (valence, arousal) centroids: **energetic, happy, calm, romantic, sad, angry**. `classifier.py` assigns probabilities via softmax over negative Euclidean distances.

### OSC Protocol

The bridge sends to `127.0.0.1:7000` (TouchDesigner default). Channels include `/mood/valence`, `/mood/arousal`, `/mood/tempo`, `/mood/energy`, `/mood/brightness`, and `/mood/<name>` per mood.
