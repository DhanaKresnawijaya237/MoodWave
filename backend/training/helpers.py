import copy

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.utils import _quadrant_label


class RegressionDataset(Dataset):
    def __init__(self, X, yv, ya, track_ids=None):
        self.X = X.astype(np.float32, copy=False)
        self.yv = yv.astype(np.float32, copy=False)
        self.ya = ya.astype(np.float32, copy=False)
        self.track_ids = track_ids

    def __len__(self):
        return len(self.yv)

    def __getitem__(self, index):
        out = [
            torch.from_numpy(self.X[index]),
            torch.tensor(self.yv[index], dtype=torch.float32),
            torch.tensor(self.ya[index], dtype=torch.float32),
        ]
        if self.track_ids is not None:
            out.append(torch.tensor(self.track_ids[index], dtype=torch.long))
        return tuple(out)


class SequenceRegressionDataset(Dataset):
    def __init__(self, X, lengths, yv, ya):
        self.X = X.astype(np.float32, copy=False)
        self.lengths = lengths.astype(np.int32, copy=False)
        self.yv = yv.astype(np.float32, copy=False)
        self.ya = ya.astype(np.float32, copy=False)

    def __len__(self):
        return len(self.yv)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.X[index]),
            self.lengths[index],
            torch.tensor(self.yv[index], dtype=torch.float32),
            torch.tensor(self.ya[index], dtype=torch.float32),
        )


def compute_feature_stats(X_train):
    mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = X_train.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    return {"mean": mean, "std": std}


def normalize_features(X, stats):
    return (X.astype(np.float32) - stats["mean"]) / stats["std"]


def compute_seq_feature_stats(X_train, lengths_train):
    _, t_max, _ = X_train.shape
    mask = np.arange(t_max) < lengths_train[:, None]
    masked = X_train[mask]
    mean = masked.mean(axis=0, keepdims=True).astype(np.float32)
    std = masked.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    return {"mean": mean, "std": std}


def normalize_seq_features(X, stats):
    return (X.astype(np.float32) - stats["mean"]) / stats["std"]


def _track_level_metrics(yv_true, ya_true, yv_pred, ya_pred, track_ids):
    tids = np.concatenate(track_ids)
    unique_tids = np.unique(tids)
    yv_true_track = []
    ya_true_track = []
    yv_pred_track = []
    ya_pred_track = []
    for utid in unique_tids:
        mask = tids == utid
        yv_true_track.append(yv_true[mask].mean())
        ya_true_track.append(ya_true[mask].mean())
        yv_pred_track.append(yv_pred[mask].mean())
        ya_pred_track.append(ya_pred[mask].mean())

    yv_true_track = np.array(yv_true_track)
    ya_true_track = np.array(ya_true_track)
    yv_pred_track = np.array(yv_pred_track)
    ya_pred_track = np.array(ya_pred_track)
    true_quadrants = np.array([_quadrant_label(v, a) for v, a in zip(yv_true_track, ya_true_track)], dtype=np.int64)
    pred_quadrants = np.array([_quadrant_label(v, a) for v, a in zip(yv_pred_track, ya_pred_track)], dtype=np.int64)

    return {
        "track_valence_mae": float(mean_absolute_error(yv_true_track, yv_pred_track)),
        "track_valence_r2": float(r2_score(yv_true_track, yv_pred_track)),
        "track_arousal_mae": float(mean_absolute_error(ya_true_track, ya_pred_track)),
        "track_arousal_r2": float(r2_score(ya_true_track, ya_pred_track)),
        "track_quadrant_acc": float(accuracy_score(true_quadrants, pred_quadrants)),
        "track_quadrant_weighted_f1": float(f1_score(true_quadrants, pred_quadrants, average="weighted", zero_division=0)),
        "track_quadrant_macro_f1": float(f1_score(true_quadrants, pred_quadrants, average="macro", zero_division=0)),
    }


def _regression_metrics(yv_true, ya_true, yv_pred, ya_pred):
    rmse_v = np.sqrt(np.mean((yv_true - yv_pred) ** 2))
    rmse_a = np.sqrt(np.mean((ya_true - ya_pred) ** 2))
    pcc_v, _ = pearsonr(yv_true, yv_pred)
    pcc_a, _ = pearsonr(ya_true, ya_pred)

    true_quadrants = np.array([_quadrant_label(v, a) for v, a in zip(yv_true, ya_true)], dtype=np.int64)
    pred_quadrants = np.array([_quadrant_label(v, a) for v, a in zip(yv_pred, ya_pred)], dtype=np.int64)

    return {
        "valence_mae": float(mean_absolute_error(yv_true, yv_pred)),
        "valence_r2": float(r2_score(yv_true, yv_pred)),
        "arousal_mae": float(mean_absolute_error(ya_true, ya_pred)),
        "arousal_r2": float(r2_score(ya_true, ya_pred)),
        "rmse_valence": float(rmse_v),
        "rmse_arousal": float(rmse_a),
        "rmse_avg": float((rmse_v + rmse_a) / 2),
        "pcc_valence": float(pcc_v),
        "pcc_arousal": float(pcc_a),
        "quadrant_acc": float(accuracy_score(true_quadrants, pred_quadrants)),
        "quadrant_weighted_f1": float(f1_score(true_quadrants, pred_quadrants, average="weighted", zero_division=0)),
        "quadrant_macro_f1": float(f1_score(true_quadrants, pred_quadrants, average="macro", zero_division=0)),
    }


def evaluate_vector_model(model, loader, device, loss_fn=None):
    model.eval()
    total_loss = 0.0
    true_valence = []
    true_arousal = []
    pred_valence = []
    pred_arousal = []
    track_ids = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                xb, yv, ya, tid = batch
                track_ids.append(tid.cpu().numpy())
            else:
                xb, yv, ya = batch
            xb = xb.to(device)
            yv = yv.to(device=device, dtype=torch.float32)
            ya = ya.to(device=device, dtype=torch.float32)

            pred_v, pred_a = model(xb)
            if loss_fn is not None:
                loss = 0.5 * (loss_fn(pred_v, yv) + loss_fn(pred_a, ya))
                total_loss += loss.item() * len(xb)

            true_valence.append(yv.cpu().numpy())
            true_arousal.append(ya.cpu().numpy())
            pred_valence.append(pred_v.cpu().numpy())
            pred_arousal.append(pred_a.cpu().numpy())

    yv_true = np.concatenate(true_valence)
    ya_true = np.concatenate(true_arousal)
    yv_pred = np.concatenate(pred_valence)
    ya_pred = np.concatenate(pred_arousal)
    metrics = _regression_metrics(yv_true, ya_true, yv_pred, ya_pred)

    if loss_fn is not None:
        metrics["loss"] = total_loss / len(loader.dataset)
    if track_ids:
        metrics.update(_track_level_metrics(yv_true, ya_true, yv_pred, ya_pred, track_ids))

    return metrics


def evaluate_sequence_model(model, loader, device, loss_fn=None):
    model.eval()
    total_loss = 0.0
    total_count = 0
    true_valence = []
    true_arousal = []
    pred_valence = []
    pred_arousal = []

    with torch.no_grad():
        for xb, lengths, yv, ya in loader:
            xb = xb.to(device)
            lengths = lengths.to(device)
            yv = yv.to(device=device, dtype=torch.float32)
            ya = ya.to(device=device, dtype=torch.float32)

            pred_v, pred_a = model(xb, lengths)
            mask = torch.arange(pred_v.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
            if loss_fn is not None:
                loss = 0.5 * (loss_fn(pred_v[mask], yv[mask]) + loss_fn(pred_a[mask], ya[mask]))
                valid_count = int(mask.sum().item())
                total_loss += loss.item() * valid_count
                total_count += valid_count

            true_valence.append(yv[mask].cpu().numpy())
            true_arousal.append(ya[mask].cpu().numpy())
            pred_valence.append(pred_v[mask].cpu().numpy())
            pred_arousal.append(pred_a[mask].cpu().numpy())

    metrics = _regression_metrics(
        np.concatenate(true_valence),
        np.concatenate(true_arousal),
        np.concatenate(pred_valence),
        np.concatenate(pred_arousal),
    )
    if loss_fn is not None:
        metrics["loss"] = total_loss / max(1, total_count)
    return metrics


def _log_epoch(metrics_logger, epoch, train_loss, val_metrics, lr):
    log_row = {
        "epoch": epoch + 1,
        "train_loss": round(train_loss, 6),
        "val_loss": round(val_metrics["loss"], 6),
        "valence_mae": round(val_metrics["valence_mae"], 6),
        "valence_r2": round(val_metrics["valence_r2"], 6),
        "arousal_mae": round(val_metrics["arousal_mae"], 6),
        "arousal_r2": round(val_metrics["arousal_r2"], 6),
        "rmse_valence": round(val_metrics["rmse_valence"], 6),
        "rmse_arousal": round(val_metrics["rmse_arousal"], 6),
        "rmse_avg": round(val_metrics["rmse_avg"], 6),
        "pcc_valence": round(val_metrics["pcc_valence"], 6),
        "pcc_arousal": round(val_metrics["pcc_arousal"], 6),
        "quadrant_acc": round(val_metrics["quadrant_acc"], 6),
        "lr": lr,
    }
    if "track_valence_r2" in val_metrics:
        log_row.update({
            "track_valence_mae": round(val_metrics["track_valence_mae"], 6),
            "track_valence_r2": round(val_metrics["track_valence_r2"], 6),
            "track_arousal_mae": round(val_metrics["track_arousal_mae"], 6),
            "track_arousal_r2": round(val_metrics["track_arousal_r2"], 6),
            "track_quadrant_acc": round(val_metrics["track_quadrant_acc"], 6),
        })
    metrics_logger.log(log_row)


def train_vector_model(
    X_train,
    yv_train,
    ya_train,
    X_val,
    yv_val,
    ya_val,
    device,
    model_factory,
    desc,
    epochs=80,
    patience=15,
    lr=1e-3,
    batch_size=64,
    weight_decay=1e-4,
    track_ids_val=None,
    quiet=False,
    metrics_logger=None,
):
    train_ds = RegressionDataset(X_train, yv_train, ya_train)
    val_ds = RegressionDataset(X_val, yv_val, ya_val, track_ids=track_ids_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, pin_memory=torch.cuda.is_available())

    model = model_factory().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=len(train_loader) * epochs,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    pbar = tqdm(range(epochs), desc=desc, unit="epoch", disable=quiet)
    for epoch in pbar:
        model.train()
        total_loss = 0.0
        for xb, yv, ya in train_loader:
            xb = xb.to(device)
            yv = yv.to(device=device, dtype=torch.float32)
            ya = ya.to(device=device, dtype=torch.float32)

            pred_v, pred_a = model(xb)
            loss = 0.5 * (loss_fn(pred_v, yv) + loss_fn(pred_a, ya))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(train_loader.dataset)
        val_metrics = evaluate_vector_model(model, val_loader, device, loss_fn=loss_fn)
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_metrics['loss']:.4f}",
            rmse=f"{val_metrics['rmse_avg']:.4f}",
            pcc_v=f"{val_metrics['pcc_valence']:.3f}",
            pcc_a=f"{val_metrics['pcc_arousal']:.3f}",
            best=f"{best_loss:.4f}",
            lr=f"{current_lr:.1e}",
        )

        if metrics_logger is not None:
            _log_epoch(metrics_logger, epoch, train_loss, val_metrics, current_lr)

        if val_metrics["loss"] < best_loss - 1e-4:
            best_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if not quiet:
                tqdm.write(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state is None:
        raise RuntimeError("Training finished without producing a valid checkpoint.")

    model.load_state_dict(best_state)
    if metrics_logger is not None:
        metrics_logger.close()
        plot_paths = metrics_logger.plot()
        if plot_paths and not quiet:
            tqdm.write(f"  Plots saved: {', '.join(plot_paths)}")
    return model


def train_sequence_model(
    X_train,
    lengths_train,
    yv_train,
    ya_train,
    X_val,
    lengths_val,
    yv_val,
    ya_val,
    device,
    model_factory,
    desc,
    epochs=80,
    patience=15,
    lr=1e-3,
    batch_size=32,
    weight_decay=1e-4,
    quiet=False,
    metrics_logger=None,
):
    train_ds = SequenceRegressionDataset(X_train, lengths_train, yv_train, ya_train)
    val_ds = SequenceRegressionDataset(X_val, lengths_val, yv_val, ya_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, pin_memory=torch.cuda.is_available())

    model = model_factory().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=len(train_loader) * epochs,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    pbar = tqdm(range(epochs), desc=desc, unit="epoch", disable=quiet)
    for epoch in pbar:
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, lengths, yv, ya in train_loader:
            xb = xb.to(device)
            lengths = lengths.to(device)
            yv = yv.to(device=device, dtype=torch.float32)
            ya = ya.to(device=device, dtype=torch.float32)

            pred_v, pred_a = model(xb, lengths)
            mask = torch.arange(pred_v.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
            loss = 0.5 * (loss_fn(pred_v[mask], yv[mask]) + loss_fn(pred_a[mask], ya[mask]))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            valid_count = int(mask.sum().item())
            total_loss += loss.item() * valid_count
            total_count += valid_count

        train_loss = total_loss / max(1, total_count)
        val_metrics = evaluate_sequence_model(model, val_loader, device, loss_fn=loss_fn)
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_metrics['loss']:.4f}",
            rmse=f"{val_metrics['rmse_avg']:.4f}",
            pcc_v=f"{val_metrics['pcc_valence']:.3f}",
            pcc_a=f"{val_metrics['pcc_arousal']:.3f}",
            best=f"{best_loss:.4f}",
            lr=f"{current_lr:.1e}",
        )

        if metrics_logger is not None:
            _log_epoch(metrics_logger, epoch, train_loss, val_metrics, current_lr)

        if val_metrics["loss"] < best_loss - 1e-4:
            best_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if not quiet:
                tqdm.write(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state is None:
        raise RuntimeError("Training finished without producing a valid checkpoint.")

    model.load_state_dict(best_state)
    if metrics_logger is not None:
        metrics_logger.close()
        plot_paths = metrics_logger.plot()
        if plot_paths and not quiet:
            tqdm.write(f"  Plots saved: {', '.join(plot_paths)}")
    return model
