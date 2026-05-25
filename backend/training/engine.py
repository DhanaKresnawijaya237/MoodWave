import copy

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from tqdm import tqdm

from features.dataset import build_loaders
from core.utils import _quadrant_label


def multitask_loss(pred_valence, pred_arousal, target_valence, target_arousal, loss_fn, pred_quadrant=None, target_quadrant=None, quadrant_weight=1.0):
    valence_loss = loss_fn(pred_valence, target_valence)
    arousal_loss = loss_fn(pred_arousal, target_arousal)
    total = 0.5 * (valence_loss + arousal_loss)
    quadrant_loss = None
    if pred_quadrant is not None and target_quadrant is not None:
        quadrant_loss = torch.nn.functional.cross_entropy(pred_quadrant, target_quadrant)
        total = total + quadrant_weight * quadrant_loss
    return total, valence_loss, arousal_loss, quadrant_loss


def _compute_track_level_metrics(yv_true, ya_true, yv_pred, ya_pred, track_ids):
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


def evaluate_loader(model, loader, device, loss_fn=None, use_quadrant_head=False):
    model.eval()
    total_loss = 0.0
    total_valence_loss = 0.0
    total_arousal_loss = 0.0
    total_quadrant_loss = 0.0
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

            model_out = model(xb)
            if use_quadrant_head:
                pred_v, pred_a, pred_q = model_out
            else:
                pred_v, pred_a = model_out
                pred_q = None

            if loss_fn is not None:
                if use_quadrant_head:
                    q_target = torch.tensor([_quadrant_label(v.item(), a.item()) for v, a in zip(yv, ya)], dtype=torch.long, device=device)
                    loss, loss_v, loss_a, loss_q = multitask_loss(pred_v, pred_a, yv, ya, loss_fn, pred_quadrant=pred_q, target_quadrant=q_target)
                    total_quadrant_loss += loss_q.item() * len(xb)
                else:
                    loss, loss_v, loss_a, _ = multitask_loss(pred_v, pred_a, yv, ya, loss_fn)
                total_loss += loss.item() * len(xb)
                total_valence_loss += loss_v.item() * len(xb)
                total_arousal_loss += loss_a.item() * len(xb)

            true_valence.append(yv.cpu().numpy())
            true_arousal.append(ya.cpu().numpy())
            pred_valence.append(pred_v.cpu().numpy())
            pred_arousal.append(pred_a.cpu().numpy())

    yv_true = np.concatenate(true_valence)
    ya_true = np.concatenate(true_arousal)
    yv_pred = np.concatenate(pred_valence)
    ya_pred = np.concatenate(pred_arousal)

    true_quadrants = np.array([_quadrant_label(v, a) for v, a in zip(yv_true, ya_true)], dtype=np.int64)
    pred_quadrants = np.array([_quadrant_label(v, a) for v, a in zip(yv_pred, ya_pred)], dtype=np.int64)

    metrics = {
        "valence_mae": float(mean_absolute_error(yv_true, yv_pred)),
        "valence_r2": float(r2_score(yv_true, yv_pred)),
        "arousal_mae": float(mean_absolute_error(ya_true, ya_pred)),
        "arousal_r2": float(r2_score(ya_true, ya_pred)),
        "quadrant_acc": float(accuracy_score(true_quadrants, pred_quadrants)),
        "quadrant_weighted_f1": float(f1_score(true_quadrants, pred_quadrants, average="weighted", zero_division=0)),
        "quadrant_macro_f1": float(f1_score(true_quadrants, pred_quadrants, average="macro", zero_division=0)),
        "yv_true": yv_true,
        "ya_true": ya_true,
        "yv_pred": yv_pred,
        "ya_pred": ya_pred,
    }
    if loss_fn is not None:
        size = len(loader.dataset)
        metrics["loss"] = total_loss / size
        metrics["valence_loss"] = total_valence_loss / size
        metrics["arousal_loss"] = total_arousal_loss / size
        if use_quadrant_head:
            metrics["quadrant_loss"] = total_quadrant_loss / size

    if track_ids:
        metrics.update(_compute_track_level_metrics(yv_true, ya_true, yv_pred, ya_pred, track_ids))

    return metrics


def train_model(
    X_train,
    yv_train,
    ya_train,
    X_val,
    yv_val,
    ya_val,
    device,
    args=None,
    n_mels=64,
    epochs=None,
    patience=None,
    lr=None,
    batch_size=None,
    weight_decay=None,
    model_kwargs=None,
    best_metric_mode="r2",
    quiet=False,
    num_workers=0,
    use_quadrant_head=False,
    track_ids_val=None,
    grad_clip=1.0,
    metrics_logger=None,
):
    max_epochs = epochs if epochs is not None else (args.epochs if args else 80)
    patience_val = patience if patience is not None else (args.patience if args else 15)
    lr_val = lr if lr is not None else (args.lr if args else 3e-4)
    bs = batch_size if batch_size is not None else (args.batch_size if args else 32)
    wd = weight_decay if weight_decay is not None else None
    if wd is None and args is not None:
        wd = getattr(args, "weight_decay", None)
    if wd is None:
        wd = 1e-4

    train_loader, val_loader = build_loaders(
        X_train,
        yv_train,
        ya_train,
        X_val,
        yv_val,
        ya_val,
        bs,
        num_workers=num_workers,
        track_ids_val=track_ids_val,
    )

    from core.model import MoodCNNBiGRU

    model = MoodCNNBiGRU(n_mels=n_mels, num_quadrant_classes=4 if use_quadrant_head else 0, **(model_kwargs or {})).to(device)

    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_val, weight_decay=wd)

    total_steps = len(train_loader) * max_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr_val,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    if best_metric_mode == "loss":
        best_metric = float("inf")

        def is_better(score, best):
            return score < best - 1e-4

        def get_score(vm):
            return vm["loss"]
    else:
        best_metric = -1e9

        def is_better(score, best):
            return score > best + 1e-4

        def get_score(vm):
            if "track_valence_r2" in vm:
                return 0.5 * (vm["track_valence_r2"] + vm["track_arousal_r2"])
            return 0.5 * (vm["valence_r2"] + vm["arousal_r2"])

    best_state = None
    patience_counter = 0
    pbar = tqdm(range(max_epochs), desc="CNN-BiGRU multitask", unit="epoch", disable=quiet)
    for epoch in pbar:
        model.train()
        total_loss = 0.0
        total_valence_loss = 0.0
        total_arousal_loss = 0.0

        for batch in train_loader:
            if len(batch) == 4:
                xb, yv, ya, _ = batch
            else:
                xb, yv, ya = batch
            xb = xb.to(device)
            yv = yv.to(device=device, dtype=torch.float32)
            ya = ya.to(device=device, dtype=torch.float32)
            model_out = model(xb)
            if use_quadrant_head:
                pred_v, pred_a, pred_q = model_out
                q_target = torch.tensor([_quadrant_label(v.item(), a.item()) for v, a in zip(yv, ya)], dtype=torch.long, device=device)
                loss, loss_v, loss_a, loss_q = multitask_loss(pred_v, pred_a, yv, ya, loss_fn, pred_quadrant=pred_q, target_quadrant=q_target, quadrant_weight=0.2)
            else:
                pred_v, pred_a = model_out
                loss, loss_v, loss_a, _ = multitask_loss(pred_v, pred_a, yv, ya, loss_fn)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(xb)
            total_valence_loss += loss_v.item() * len(xb)
            total_arousal_loss += loss_a.item() * len(xb)

        size = len(train_loader.dataset)
        train_loss = total_loss / size
        train_valence_loss = total_valence_loss / size
        train_arousal_loss = total_arousal_loss / size
        val_metrics = evaluate_loader(model, val_loader, device, loss_fn, use_quadrant_head=use_quadrant_head)

        score = get_score(val_metrics)

        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            train=f"{train_loss:.4f}",
            vloss=f"{train_valence_loss:.4f}",
            aloss=f"{train_arousal_loss:.4f}",
            vr2=f"{val_metrics['valence_r2']:.3f}",
            ar2=f"{val_metrics['arousal_r2']:.3f}",
            qacc=f"{val_metrics['quadrant_acc']:.3f}",
            best=f"{best_metric:.3f}",
            lr=f"{current_lr:.1e}",
        )

        # Log metrics
        if metrics_logger is not None:
            log_row = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_metrics.get("loss", 0), 6),
                "valence_mae": round(val_metrics["valence_mae"], 6),
                "valence_r2": round(val_metrics["valence_r2"], 6),
                "arousal_mae": round(val_metrics["arousal_mae"], 6),
                "arousal_r2": round(val_metrics["arousal_r2"], 6),
                "quadrant_acc": round(val_metrics["quadrant_acc"], 6),
                "lr": current_lr,
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

        if is_better(score, best_metric):
            best_metric = score
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_val:
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
