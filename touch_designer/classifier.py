import torch
import joblib
import numpy as np
from model import MoodMLP
from processor import features_to_vector

# Mood centers on the valence-arousal 2D space (scale -1 to 1)
MOOD_CENTERS = {
    'energetic': ( 0.7,  0.7),
    'happy':     ( 0.5,  0.3),
    'calm':      ( 0.4, -0.6),
    'romantic':  ( 0.5, -0.5),
    'sad':       (-0.6, -0.5),
    'angry':     (-0.6,  0.7),
}

_valence_model, _arousal_model, _scaler, _device = None, None, None, None


def _load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = joblib.load("models/input_dim.pkl")
    scaler = joblib.load("models/scaler.pkl")

    valence_model = MoodMLP(input_dim=input_dim).to(device)
    valence_model.load_state_dict(torch.load("models/valence_mlp.pth", map_location=device))
    valence_model.eval()

    arousal_model = MoodMLP(input_dim=input_dim).to(device)
    arousal_model.load_state_dict(torch.load("models/arousal_mlp.pth", map_location=device))
    arousal_model.eval()

    return valence_model, arousal_model, scaler, device


def valence_arousal_to_mood_distribution(valence: float, arousal: float, temperature: int = 3) -> dict:
    distances = {}
    for mood, (v_center, a_center) in MOOD_CENTERS.items():
        dist = np.sqrt((valence - v_center) ** 2 + (arousal - a_center) ** 2)
        distances[mood] = dist

    inv_distances = {mood: 1.0 / (dist + 1e-6) ** temperature for mood, dist in distances.items()}
    total = sum(inv_distances.values())
    distribution = {mood: round(weight / total, 4) for mood, weight in inv_distances.items()}
    return distribution


def predict_mood(features: dict) -> dict:
    global _valence_model, _arousal_model, _scaler, _device
    if _valence_model is None:
        _valence_model, _arousal_model, _scaler, _device = _load_models()

    vector = features_to_vector(features)
    x = np.array([vector], dtype=np.float32)
    x = _scaler.transform(x)

    with torch.no_grad():
        t = torch.from_numpy(x).to(_device)
        valence = float(_valence_model(t).cpu().numpy()[0])
        arousal = float(_arousal_model(t).cpu().numpy()[0])

    distribution = valence_arousal_to_mood_distribution(valence, arousal)

    return {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "distribution": distribution,
    }
