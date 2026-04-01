"""Run inference on a single audio file and save results to JSON."""
import sys
import json
import librosa
import numpy as np
from processor import extract_features_from_array, CHUNK_DURATION
from classifier import predict_mood


def infer(audio_path, output_path="inference_result.json"):
    print(f"Loading: {audio_path}")
    y, sr = librosa.load(audio_path)
    duration = len(y) / sr
    print(f"Duration: {duration:.1f}s  Sample rate: {sr}")

    chunk_samples = int(CHUNK_DURATION * sr)
    results = []

    for i, start in enumerate(range(0, len(y), chunk_samples)):
        chunk = y[start : start + chunk_samples]
        if len(chunk) < sr * 5:
            continue

        time_start = start / sr
        time_end = (start + len(chunk)) / sr
        print(f"  Chunk {i}: {time_start:.1f}s - {time_end:.1f}s", end="", flush=True)

        features = extract_features_from_array(chunk, sr)
        prediction = predict_mood(features)

        dominant = max(prediction["distribution"], key=prediction["distribution"].get)
        print(f"  -> V={prediction['valence']:.3f} A={prediction['arousal']:.3f} [{dominant}]")

        results.append({
            "chunk": i,
            "time_start": round(time_start, 1),
            "time_end": round(time_end, 1),
            "valence": prediction["valence"],
            "arousal": prediction["arousal"],
            "mood_distribution": prediction["distribution"],
            "dominant_mood": dominant,
            "features": {
                "tempo": features["tempo"],
                "energy": features["rms_energy"],
                "brightness": features["spectral_centroid"],
            },
        })

    # Overall averages
    if results:
        avg_val = round(np.mean([r["valence"] for r in results]), 3)
        avg_aro = round(np.mean([r["arousal"] for r in results]), 3)
        moods = list(results[0]["mood_distribution"].keys())
        avg_mood = {m: round(np.mean([r["mood_distribution"][m] for r in results]), 4) for m in moods}
        dominant = max(avg_mood, key=avg_mood.get)
    else:
        avg_val = avg_aro = 0
        avg_mood = {}
        dominant = "unknown"

    output = {
        "file": audio_path,
        "total_duration": round(duration, 1),
        "num_chunks": len(results),
        "overall": {
            "valence": avg_val,
            "arousal": avg_aro,
            "mood_distribution": avg_mood,
            "dominant_mood": dominant,
        },
        "chunks": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nOverall: valence={avg_val}, arousal={avg_aro}, dominant_mood={dominant}")
    print(f"Mood distribution: {avg_mood}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../song2.mp3"
    infer(path)
