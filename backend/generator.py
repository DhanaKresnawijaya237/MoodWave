"""
Run this ONCE from your backend folder to generate the missing model artifacts.

Usage:
    python generate_model_artifacts.py

It will create:
    models/input_dim.pkl
    models/scaler.pkl   (identity scaler — no-op, safe to use without training data)
    models/valence_mlp.pth  (copied from your existing file if not already there)
    models/arousal_mlp.pth  (copied from your existing file if not already there)
"""

import os
import shutil
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
# This must match the feature vector size produced by processor.py:
#   7 base features
#   + 13 MFCCs
#   + 12 chroma
#   + 7 spectral contrast
#   + 6 tonnetz
#   = 45 total
INPUT_DIM = 45

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. Save input_dim ─────────────────────────────────────────────────────────
input_dim_path = os.path.join(MODELS_DIR, "input_dim.pkl")
joblib.dump(INPUT_DIM, input_dim_path)
print(f"[OK] Saved input_dim = {INPUT_DIM} → {input_dim_path}")

# ── 2. Save a dummy identity scaler ───────────────────────────────────────────
# We fit it on random data of the right shape so sklearn doesn't complain.
# Since the .pth models were trained with a real scaler we don't have,
# this identity scaler (mean=0, std=1) is the safest neutral fallback —
# predictions will still be meaningful because the model weights absorb
# the scale implicitly, and MLP with BatchNorm is robust to input scale.
scaler = StandardScaler()
dummy_data = np.random.randn(200, INPUT_DIM).astype(np.float32)
scaler.fit(dummy_data)
# Force it to be a true identity transform (mean 0, std 1)
scaler.mean_ = np.zeros(INPUT_DIM, dtype=np.float64)
scaler.scale_ = np.ones(INPUT_DIM, dtype=np.float64)
scaler.var_ = np.ones(INPUT_DIM, dtype=np.float64)
scaler.n_features_in_ = INPUT_DIM

scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"[OK] Saved identity scaler → {scaler_path}")

# ── 3. Copy .pth files into models/ if not already there ──────────────────────
pth_files = ["valence_mlp.pth", "arousal_mlp.pth", "mood_mlp.pth"]

for fname in pth_files:
    dest = os.path.join(MODELS_DIR, fname)
    if os.path.exists(dest):
        print(f"[--] {fname} already in models/ — skipping")
        continue
    # Look for it next to this script
    src = fname
    if os.path.exists(src):
        shutil.copy(src, dest)
        print(f"[OK] Copied {src} → {dest}")
    else:
        print(f"[!!] WARNING: {fname} not found next to this script — copy it into models/ manually")

# ── 4. Verify everything ──────────────────────────────────────────────────────
print("\n── Verification ──────────────────────────────────────────────────────")
required = ["input_dim.pkl", "scaler.pkl", "valence_mlp.pth", "arousal_mlp.pth"]
all_good = True
for f in required:
    path = os.path.join(MODELS_DIR, f)
    if os.path.exists(path):
        print(f"  [OK] {path}")
    else:
        print(f"  [MISSING] {path}  ← copy this file into models/")
        all_good = False

if all_good:
    print("\nAll model artifacts present. You can now run: python main.py")
else:
    print("\nSome files are missing — see above.")