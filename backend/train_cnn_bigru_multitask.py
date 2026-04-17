import argparse
import os
from multiprocessing import freeze_support

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from features.dataset import MelRegressionDataset
from core.model import MoodCNNBiGRU
from features.processor import MEL_N_MELS, load_or_extract_base_segments
from features.splits import split_indices
from training.engine import evaluate_loader, train_model
from core.utils import compute_mel_stats,  normalize_segments, set_seed, write_report

PREP_WORKERS = max(1, min(4, (os.cpu_count() or 1) - 1))
SEED = 42


def artifact_paths(split_mode):
    suffix = "paper" if split_mode == "paper" else "strict"
    return (
        f"models/cnn_bigru_multitask_{suffix}.pth",
        f"models/cnn_bigru_multitask_{suffix}_mel_stats.pkl",
        f"models/cnn_bigru_multitask_{suffix}_z_stats.pkl",
        f"models/cnn_bigru_multitask_{suffix}_report.txt",
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train a paper-style CNN-BiGRU multitask regressor.")
    parser.add_argument("--split", choices=("paper", "strict"), default="strict")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=PREP_WORKERS)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--augment", action="store_true", help="Enable audio augmentation (pitch shift, time stretch, noise) during segment extraction.")
    parser.add_argument("--use-lso", action="store_true", help="Enable Lion Swarm Optimization before final training.")
    parser.add_argument("--lso-pop-size", type=int, default=15)
    parser.add_argument("--lso-iterations", type=int, default=30)
    parser.add_argument("--lso-eval-epochs", type=int, default=10, help="Epochs per LSO candidate evaluation.")
    parser.add_argument("--lso-eval-patience", type=int, default=5, help="Early-stopping patience for LSO evals.")
    return parser.parse_args(args)


def main():
    args = parse_args()

    set_seed(SEED)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    X, yv, ya, track_ids = load_or_extract_base_segments(
        rebuild_cache=args.rebuild_cache,
        worker_count=args.workers,
        augment_cache=args.augment,
    )
    train_idx, val_idx, test_idx = split_indices(yv, ya, track_ids, args.split)

    X_train = X[train_idx]
    yv_train = yv[train_idx]
    ya_train = ya[train_idx]
    X_val = X[val_idx]
    yv_val = yv[val_idx]
    ya_val = ya[val_idx]
    X_test = X[test_idx]
    yv_test = yv[test_idx]
    ya_test = ya[test_idx]

    mel_stats = compute_mel_stats(X_train)
    X_train = normalize_segments(X_train, mel_stats)
    X_val = normalize_segments(X_val, mel_stats)
    X_test = normalize_segments(X_test, mel_stats)

    model_path, mel_stats_path, report_path = artifact_paths(args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nSplit: {args.split}")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Device: {device}")

    if args.eval_only:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing checkpoint at {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model_kwargs = checkpoint.get("model_kwargs", {})
        model = MoodCNNBiGRU(n_mels=MEL_N_MELS, **model_kwargs).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
    else:
        if args.use_lso:
            from training.lso import LSOSearch
            print("\n=== Running Lion Swarm Optimization (LSO) ===")
            lso = LSOSearch(
                X_train, yv_train, ya_train,
                X_val, yv_val, ya_val,
                device,
                pop_size=args.lso_pop_size,
                iterations=args.lso_iterations,
                eval_epochs=args.lso_eval_epochs,
                eval_patience=args.lso_eval_patience,
                batch_size=args.batch_size,
                n_mels=MEL_N_MELS,
                seed=SEED,
                num_workers=args.workers,
            )
            best_params = lso.optimize()
            print("\n=== Best hyperparameters from LSO ===")
            for k, v in best_params.items():
                print(f"  {k}: {v}")
            model_kwargs = {
                "conv1_filters": best_params["conv1_filters"],
                "conv2_filters": best_params["conv2_filters"],
                "kernel_size": best_params["kernel_size"],
                "hidden_size": best_params["hidden_size"],
                "dropout": best_params["dropout"],
            }
            final_lr = best_params["lr"]
            final_wd = best_params["weight_decay"]
        else:
            model_kwargs = {}
            final_lr = args.lr
            final_wd = 1e-4

        model = train_model(
            X_train, yv_train, ya_train,
            X_val, yv_val, ya_val,
            device, args=args, n_mels=MEL_N_MELS,
            model_kwargs=model_kwargs, lr=final_lr, weight_decay=final_wd, num_workers=args.workers,
        )
        os.makedirs("models", exist_ok=True)
        checkpoint = {
            "state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
        }
        torch.save(checkpoint, model_path)
        joblib.dump(mel_stats, mel_stats_path)
        print(f"\nSaved multitask checkpoint: {model_path}")
        print(f"Saved mel stats: {mel_stats_path}")

    test_loader = DataLoader(
        MelRegressionDataset(X_test, yv_test, ya_test, augment=False),
        batch_size=args.batch_size * 2,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    test_metrics = evaluate_loader(model, test_loader, device)

    write_report(
        f"CNN-BiGRU multitask {args.split} test metrics",
        test_metrics,
        report_path=report_path,
    )
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    freeze_support()
    main()
