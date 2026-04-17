import copy

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from tqdm import tqdm

from features.dataset import build_loaders
from core.utils import _quadrant_label


def multitask_loss(pred_valence, pred_arousal, target_valence, target_arousal, loss_fn):
    valence_loss = loss_fn(pred_valence, target_valence)
    arousal_loss = loss_fn(pred_arousal, target_arousal)
    return 0.5 * (valence_loss + arousal_loss), valence_loss, arousal_loss


def evaluate_loader(model, loader, device, loss_fn=None):
    model.eval()
    total_loss = 0.0
    total_valence_loss = 0.0
    total_arousal_loss = 0.0
    true_valence = []
    true_arousal = []
    pred_valence = []
    pred_arousal = []

    with torch.no_grad():
        for xb, yv, ya in loader:
            xb = xb.to(device)
            yv = yv.to(device=device, dtype=torch.float32)
            ya = ya.to(device=device, dtype=torch.float32)
            pred_v, pred_a = model(xb)
            if loss_fn is not None:
                loss, loss_v, loss_a = multitask_loss(pred_v, pred_a, yv, ya, loss_fn)
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
    )

    from core.model import MoodCNNBiGRU

    model = MoodCNNBiGRU(n_mels=n_mels, **(model_kwargs or {})).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_val, weight_decay=wd)
    sched_mode = "min" if best_metric_mode == "loss" else "max"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=sched_mode, factor=0.5, patience=5)

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
            return 0.5 * (vm["valence_r2"] + vm["arousal_r2"])

    best_state = None
    patience_counter = 0
    pbar = tqdm(range(max_epochs), desc="CNN-BiGRU multitask", unit="epoch", disable=quiet)
    for epoch in pbar:
        model.train()
        total_loss = 0.0
        total_valence_loss = 0.0
        total_arousal_loss = 0.0

        for xb, yv, ya in train_loader:
            xb = xb.to(device)
            yv = yv.to(device=device, dtype=torch.float32)
            ya = ya.to(device=device, dtype=torch.float32)
            pred_v, pred_a = model(xb)
            loss, loss_v, loss_a = multitask_loss(pred_v, pred_a, yv, ya, loss_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            total_valence_loss += loss_v.item() * len(xb)
            total_arousal_loss += loss_a.item() * len(xb)

        size = len(train_loader.dataset)
        train_loss = total_loss / size
        train_valence_loss = total_valence_loss / size
        train_arousal_loss = total_arousal_loss / size
        val_metrics = evaluate_loader(model, val_loader, device, loss_fn)

        score = get_score(val_metrics)
        scheduler.step(score)

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
    return model
