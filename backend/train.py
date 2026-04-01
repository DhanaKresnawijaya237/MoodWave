import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os
import json
from multiprocessing import Pool, cpu_count
from processor import load_dynamic_annotations, process_song
from model import MoodMLP

CHECKPOINT_FILE = "data/deam/checkpoint.json"


def load_checkpoint():
    """Load checkpoint: returns (processed_song_ids, samples_list)."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        return set(data["processed_ids"]), data["samples"]
    return set(), []


def save_checkpoint(processed_ids, samples):
    """Save checkpoint to disk."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "processed_ids": list(processed_ids),
            "samples": samples,
        }, f)


def extract_samples():
    """Load annotations, process songs (with checkpointing), return X, y_valence, y_arousal."""
    print("Loading dynamic annotations...")
    dynamic = load_dynamic_annotations()

    processed_ids, cached_samples = load_checkpoint()

    X, y_valence, y_arousal = [], [], []
    for entry in cached_samples:
        X.append(entry["vector"])
        y_valence.append(entry["valence"])
        y_arousal.append(entry["arousal"])

    remaining = {sid: samples for sid, samples in dynamic.items() if sid not in processed_ids}

    if remaining:
        print(f"Resuming: {len(processed_ids)} songs done, {len(remaining)} remaining...")
        num_workers = max(1, cpu_count() - 1)
        print(f"Processing with {num_workers} workers...")

        tasks = list(remaining.items())
        with Pool(num_workers) as pool:
            for i, (sid, results) in enumerate(pool.imap_unordered(process_song, tasks)):
                for vector, v, a in results:
                    X.append(vector)
                    y_valence.append(v)
                    y_arousal.append(a)
                    cached_samples.append({"vector": vector, "valence": v, "arousal": a})
                processed_ids.add(sid)

                if (i + 1) % 50 == 0:
                    print(f"  {len(processed_ids)}/{len(dynamic)} songs processed, saving checkpoint...")
                    save_checkpoint(processed_ids, cached_samples)

        save_checkpoint(processed_ids, cached_samples)
        print(f"All {len(processed_ids)} songs processed.")
    else:
        print(f"All {len(processed_ids)} songs already processed (loaded from checkpoint).")

    print(f"\nTotal training samples: {len(X)}")
    return (
        np.array(X, dtype=np.float32),
        np.array(y_valence, dtype=np.float32),
        np.array(y_arousal, dtype=np.float32),
    )


def split_and_scale(X, y_valence, y_arousal):
    """70/15/15 split + StandardScaler. Returns scaled splits and scaler."""
    X_trainval, X_test, yv_trainval, yv_test, ya_trainval, ya_test = train_test_split(
        X, y_valence, y_arousal, test_size=0.15, random_state=42
    )
    X_train, X_val, yv_train, yv_val, ya_train, ya_val = train_test_split(
        X_trainval, yv_trainval, ya_trainval, test_size=0.176, random_state=42
    )

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_val_s, X_test_s, yv_train, yv_val, yv_test, ya_train, ya_val, ya_test, scaler


def train_one_model(name, X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, device):
    """Train a single MLP for valence or arousal. Returns the trained model."""
    print(f"\n{'='*40}")
    print(f"Training {name} model on {device}...")
    print(f"{'='*40}")

    train_ds = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = MoodMLP(input_dim=X_train_s.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_epochs = 10
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(200):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(xb)
        val_loss /= len(val_ds)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {train_loss:.5f} | Val: {val_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= 25:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test_s).to(device)).cpu().numpy()

    print(f"  {name.capitalize()} - MAE: {mean_absolute_error(y_test, preds):.3f}, R2: {r2_score(y_test, preds):.3f}")

    return model


def save_models(valence_model, arousal_model, scaler, input_dim):
    """Save all model artifacts to disk."""
    os.makedirs("models", exist_ok=True)
    torch.save(valence_model.state_dict(), "models/valence_mlp.pth")
    torch.save(arousal_model.state_dict(), "models/arousal_mlp.pth")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(input_dim, "models/input_dim.pkl")
    print("\nModels saved:")
    print("  models/valence_mlp.pth")
    print("  models/arousal_mlp.pth")
    print("  models/scaler.pkl")


def train():
    X, y_valence, y_arousal = extract_samples()
    X_train_s, X_val_s, X_test_s, yv_train, yv_val, yv_test, ya_train, ya_val, ya_test, scaler = split_and_scale(X, y_valence, y_arousal)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valence_model = train_one_model("valence", X_train_s, X_val_s, X_test_s, yv_train, yv_val, yv_test, device)
    arousal_model = train_one_model("arousal", X_train_s, X_val_s, X_test_s, ya_train, ya_val, ya_test, device)

    save_models(valence_model, arousal_model, scaler, X_train_s.shape[1])


if __name__ == "__main__":
    train()
