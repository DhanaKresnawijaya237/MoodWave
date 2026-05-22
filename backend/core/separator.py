import os
import shutil
import subprocess
import sys

import librosa
import numpy as np
import soundfile as sf

# Stems exposed to TouchDesigner.
# Demucs htdemucs_6s produces: vocals, drums, bass, guitar, piano, other.
TARGET_STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]
DEMUCS_MODEL = "htdemucs_6s"
TD_STEM_SAMPLE_RATE = 44_100


def _write_td_wav(output_path: str, audio: np.ndarray, sr: int, subtype: str = "PCM_16"):
    """Write TouchDesigner-friendly stereo PCM WAV."""
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=0 if y.shape[0] <= y.shape[-1] else 1)
    if sr != TD_STEM_SAMPLE_RATE:
        from math import gcd
        from scipy.signal import resample_poly

        common = gcd(int(sr), int(TD_STEM_SAMPLE_RATE))
        y = resample_poly(y, TD_STEM_SAMPLE_RATE // common, sr // common).astype(np.float32)
        sr = TD_STEM_SAMPLE_RATE
    stereo = np.column_stack([y, y])
    sf.write(output_path, stereo, sr, subtype=subtype, format="WAV")


def _find_stem_file(stem_dir: str, name: str):
    """Find a stem file by name, preferring .wav over .mp3."""
    for ext in (".wav", ".mp3", ".flac"):
        path = os.path.join(stem_dir, name + ext)
        if os.path.exists(path):
            return path
    return None


def cleanup_old_stems(upload_dir: str) -> str:
    """Remove any previously separated stems and return a clean stems directory."""
    stems_dir = os.path.join(upload_dir, "stems")
    if os.path.exists(stems_dir):
        shutil.rmtree(stems_dir)
    os.makedirs(stems_dir, exist_ok=True)
    return stems_dir


def _separate_with_api(input_path: str, output_dir: str, target_sr: int = 22050):
    """
    Use Demucs's native Python API with librosa/soundfile for I/O.
    This bypasses torchaudio entirely, avoiding torchcodec/ffmpeg issues.
    """
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    print("[Separator] Loading model...")
    model = get_model(DEMUCS_MODEL)
    model.cpu()

    print(f"[Separator] Loading audio (target sr={target_sr})...")
    wav, sr = librosa.load(input_path, sr=target_sr, mono=True)
    # Demucs models are trained on stereo — duplicate mono to 2 channels
    wav_stereo = np.stack([wav, wav], axis=0)  # (2, samples)
    wav_tensor = torch.from_numpy(wav_stereo).float()

    # Normalize using demucs's standard normalization
    ref = wav_tensor.mean(0)
    mean = ref.mean()
    std = ref.std() + 1e-8
    wav_tensor = (wav_tensor - mean) / std

    print("[Separator] Running separation (this may take a while)...")
    with torch.no_grad():
        sources = apply_model(
            model,
            wav_tensor[None],   # add batch dim: (1, 2, samples)
            device="cpu",
            split=True,
            overlap=0.25,
            progress=False,
        )[0]                    # -> (num_sources, 2, samples)

    # Denormalize
    sources = sources * std + mean

    # Save each stem as standard stereo PCM WAV for TouchDesigner.
    source_names = model.sources
    print(f"[Separator] Saving {len(source_names)} stems...")
    source_paths = {}
    for i, name in enumerate(source_names):
        src_np = sources[i].mean(0).cpu().numpy()  # (samples,)
        out_path = os.path.join(output_dir, f"{name}.wav")
        _write_td_wav(out_path, src_np, target_sr)
        source_paths[name] = out_path

    return source_paths


def _separate_with_cli(input_path: str, output_dir: str, target_sr: int = 22050):
    """
    Fallback: use Demucs CLI via subprocess.
    Requires torchaudio (and optionally torchcodec) to be working.
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    resampled_path = os.path.join(output_dir, f"{base_name}_22k.wav")
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)
    sf.write(resampled_path, y, target_sr, subtype="PCM_16", format="WAV")

    demucs_out = os.path.join(output_dir, "demucs_out")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "demucs.separate",
                "-n",
                DEMUCS_MODEL,
                "-o",
                demucs_out,
                resampled_path,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        os.remove(resampled_path)
        raise RuntimeError(
            "Demucs separation timed out after 10 minutes."
        ) from exc
    except subprocess.CalledProcessError as exc:
        os.remove(resampled_path)
        out = (exc.stdout or "").strip()
        err = (exc.stderr or "").strip()
        combined = "\n".join([line for line in [out, err] if line])
        print("[Demucs STDOUT]", out)
        print("[Demucs STDERR]", err)
        if "torchcodec" in combined.lower():
            raise RuntimeError(
                "Demucs CLI failed because torchaudio requires torchcodec. "
                "The native API should have been tried first — this is unexpected."
            ) from exc
        raise RuntimeError(
            f"Demucs CLI failed (exit code {exc.returncode})."
        ) from exc

    stem_source_dir = os.path.join(demucs_out, DEMUCS_MODEL, base_name + "_22k")
    if not os.path.isdir(stem_source_dir):
        raise RuntimeError(f"Demucs output directory not found: {stem_source_dir}")

    source_paths = {}
    for name in TARGET_STEMS:
        p = _find_stem_file(stem_source_dir, name)
        if p:
            y, sr = librosa.load(p, sr=None, mono=True)
            out_path = os.path.join(output_dir, f"{name}.wav")
            _write_td_wav(out_path, y, sr)
            source_paths[name] = out_path

    # Clean up
    shutil.rmtree(demucs_out)
    os.remove(resampled_path)
    return source_paths


def separate_stems(input_path: str, upload_dir: str, target_sr: int = 22050):
    """
    Separate audio into stems using Demucs htdemucs_6s.

    Uses the native Python API first (avoids torchaudio/torchcodec issues).
    Falls back to CLI subprocess if the API fails.

    Returns
    -------
    dict
        Mapping ``{"vocals": path, "drums": path, "bass": path,
        "guitar": path, "piano": path, "other": path}``.
        All paths are absolute stereo WAV files at 44.1 kHz, 16-bit PCM.
        The "other" stem is Demucs' original residual stem.
    """
    stems_dir = cleanup_old_stems(upload_dir)

    # Try native Python API first (no torchaudio, no subprocess)
    try:
        source_paths = _separate_with_api(input_path, stems_dir, target_sr)
    except Exception as api_exc:
        print(f"[Separator] Native API failed: {api_exc}")
        print("[Separator] Falling back to Demucs CLI...")
        source_paths = _separate_with_cli(input_path, stems_dir, target_sr)

    # Validate
    missing = [s for s in TARGET_STEMS if s not in source_paths]
    if missing:
        raise RuntimeError(f"Missing demucs output stems: {missing}")

    # Finalize target stems (skip copy if already in place)
    def _final_path(name: str) -> str:
        return os.path.join(stems_dir, f"{name}.wav")

    def _copy_if_different(src: str, dst: str):
        if os.path.normpath(src) == os.path.normpath(dst):
            return
        shutil.copy2(src, dst)

    for name in TARGET_STEMS:
        _copy_if_different(source_paths[name], _final_path(name))

    return {name: _final_path(name) for name in TARGET_STEMS}
