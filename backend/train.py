"""Unified training entrypoint for MoodWave experiments.

Examples
--------
python train.py --encoder mel --classifier cnn-bigru --split strict
python train.py --encoder openl3 --classifier mlp --split paper --eval-only
python train.py --encoder muq --classifier bigru --epochs 80
python train.py --encoder clap --classifier mlp --rebuild-cache
"""

import argparse
import os
from dataclasses import dataclass
from multiprocessing import freeze_support

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.model import MoodCLAPMLP, MoodCNNBiGRU, MoodMuQBiGRU, MoodMuQMLP, MoodOpenL3MLP
from core.utils import compute_mel_stats, normalize_segments, set_seed, write_report
from features.clap_extractor import load_or_extract_clap_embeddings
from features.dataset import MelRegressionDataset
from features.muq_extractor import load_or_extract_muq_embeddings, load_or_extract_muq_sequences
from features.processor import MEL_N_MELS, load_or_extract_base_segments
from features.splits import split_indices
from training.engine import evaluate_loader, train_model
from training.helpers import (
    RegressionDataset,
    SequenceRegressionDataset,
    compute_feature_stats,
    compute_seq_feature_stats,
    evaluate_sequence_model,
    evaluate_vector_model,
    normalize_features,
    normalize_seq_features,
    train_sequence_model,
    train_vector_model,
)
from training.lso import LSOSearch

try:
    from training.logger import MetricsLogger
except ImportError:
    MetricsLogger = None

try:
    from features.openl3_extractor import load_or_extract_openl3_frames
except ImportError:
    load_or_extract_openl3_frames = None


PREP_WORKERS = max(1, min(4, (os.cpu_count() or 1) - 1))
SEED = 42


@dataclass(frozen=True)
class RawFeatures:
    kind: str
    sample_unit: str
    X: object
    yv: object
    ya: object
    track_ids: object
    lengths: object = None
    input_dim: int = None
    track_metrics: bool = False
    split_yv: object = None
    split_ya: object = None
    extra: tuple = ()


@dataclass(frozen=True)
class FeatureData:
    kind: str
    sample_unit: str
    X_train: object
    yv_train: object
    ya_train: object
    X_val: object
    yv_val: object
    ya_val: object
    X_test: object
    yv_test: object
    ya_test: object
    stats: object
    stats_label: str
    input_dim: int = None
    lengths_train: object = None
    lengths_val: object = None
    lengths_test: object = None
    track_ids_val: object = None
    track_ids_test: object = None
    extra: tuple = ()


@dataclass(frozen=True)
class EncoderSpec:
    label: str
    feature_kind: str
    loader: object


@dataclass(frozen=True)
class ClassifierSpec:
    feature_kind: str
    runner: object


@dataclass(frozen=True)
class ModelSpec:
    label: str
    artifact_prefix: str
    default_batch_size: int
    default_lr: float
    default_weight_decay: float
    model_factory: object = None
    log_metrics: bool = False


def _load_mel_segments(args):
    X, yv, ya, track_ids = load_or_extract_base_segments(
        rebuild_cache=args.rebuild_cache,
        worker_count=args.workers,
        augment_cache=args.augment,
    )
    return RawFeatures("mel_segment", "window", X, yv, ya, track_ids, track_metrics=True)


def _load_openl3_vectors(args):
    if load_or_extract_openl3_frames is None:
        raise ImportError("OpenL3 extraction requires the torchopenl3 dependency.")

    X, yv, ya, track_ids = load_or_extract_openl3_frames(
        rebuild_cache=args.rebuild_cache,
        worker_count=args.workers,
    )
    return RawFeatures(
        "vector",
        "window",
        X,
        yv,
        ya,
        track_ids,
        input_dim=512,
        track_metrics=True,
        extra=(("Frame step", "0.5s"),),
    )


def _load_muq_vectors(args):
    X, yv, ya, track_ids, input_dim = load_or_extract_muq_embeddings(
        rebuild_cache=args.rebuild_cache,
    )
    return RawFeatures(
        "vector",
        "window",
        X,
        yv,
        ya,
        track_ids,
        input_dim=input_dim,
        track_metrics=True,
        extra=(("MuQ embed dim", input_dim), ("Frame step", "0.5s")),
    )


def _load_muq_sequences(args):
    X, lengths, yv, ya, track_ids, input_dim = load_or_extract_muq_sequences(
        rebuild_cache=args.rebuild_cache,
    )
    split_yv = np.array([yv[index, :length].mean() for index, length in enumerate(lengths)], dtype=np.float32)
    split_ya = np.array([ya[index, :length].mean() for index, length in enumerate(lengths)], dtype=np.float32)
    return RawFeatures(
        "sequence",
        "song",
        X,
        yv,
        ya,
        track_ids,
        lengths=lengths,
        input_dim=input_dim,
        track_metrics=False,
        split_yv=split_yv,
        split_ya=split_ya,
        extra=(("MuQ embed dim", input_dim), ("Frame step", "0.5s"), ("Max song windows", X.shape[1])),
    )


def _load_clap_vectors(args):
    X, yv, ya, track_ids, input_dim = load_or_extract_clap_embeddings(
        rebuild_cache=args.rebuild_cache,
    )
    return RawFeatures(
        "vector",
        "window",
        X,
        yv,
        ya,
        track_ids,
        input_dim=input_dim,
        track_metrics=True,
        extra=(("CLAP embed dim", input_dim), ("Frame step", "0.5s")),
    )


def _openl3_mlp(data, args):
    return MoodOpenL3MLP(input_dim=data.input_dim or 512, hidden_dims=(256, 128), dropout=0.2)


def _muq_mlp(data, args):
    return MoodMuQMLP(input_dim=data.input_dim, hidden_dims=(512, 256), dropout=0.3)


def _clap_mlp(data, args):
    return MoodCLAPMLP(input_dim=data.input_dim, hidden_dims=(256, 128), dropout=0.3)


def _sequence_bigru(data, args):
    return MoodMuQBiGRU(input_dim=data.input_dim, hidden_size=128, num_layers=2, dropout=0.5)


ENCODERS = {
    "mel": {
        "mel_segment": EncoderSpec("Mel/MFCC 0.5s dynamic windows", "mel_segment", _load_mel_segments),
    },
    "openl3": {
        "vector": EncoderSpec("OpenL3 0.5s frame embeddings", "vector", _load_openl3_vectors),
    },
    "muq": {
        "vector": EncoderSpec("MuQ pooled 0.5s window embeddings", "vector", _load_muq_vectors),
        "sequence": EncoderSpec("MuQ song sequence of 0.5s window embeddings", "sequence", _load_muq_sequences),
    },
    "clap": {
        "vector": EncoderSpec("CLAP 0.5s window embeddings", "vector", _load_clap_vectors),
    },
}

CLASSIFIERS = {
    "mlp": ClassifierSpec("vector", lambda args, data, torch: _run_vector_sample_regression(args, data, torch)),
    "bigru": ClassifierSpec("sequence", lambda args, data, torch: _run_sequence_window_regression(args, data, torch)),
    "cnn-bigru": ClassifierSpec("mel_segment", lambda args, data, torch: _run_mel_window_cnn_bigru(args, data, torch)),
}

MODEL_SPECS = {
    ("mel", "cnn-bigru"): ModelSpec(
        label="CNN-BiGRU multitask",
        artifact_prefix="cnn_bigru_multitask",
        default_batch_size=32,
        default_lr=1e-4,
        default_weight_decay=1e-2,
    ),
    ("openl3", "mlp"): ModelSpec(
        label="OpenL3-MLP",
        artifact_prefix="openl3_mlp",
        default_batch_size=156,
        default_lr=5e-4,
        default_weight_decay=1e-4,
        model_factory=_openl3_mlp,
        log_metrics=True,
    ),
    ("muq", "mlp"): ModelSpec(
        label="MuQ-MLP",
        artifact_prefix="muq_mlp",
        default_batch_size=64,
        default_lr=1e-3,
        default_weight_decay=1e-4,
        model_factory=_muq_mlp,
        log_metrics=True,
    ),
    ("muq", "bigru"): ModelSpec(
        label="MuQ-BiGRU",
        artifact_prefix="muq_bigru",
        default_batch_size=32,
        default_lr=1e-3,
        default_weight_decay=1e-4,
        model_factory=_sequence_bigru,
        log_metrics=True,
    ),
    ("clap", "mlp"): ModelSpec(
        label="CLAP-MLP",
        artifact_prefix="clap_mlp",
        default_batch_size=64,
        default_lr=1e-3,
        default_weight_decay=1e-4,
        model_factory=_clap_mlp,
    ),
}

DEFAULT_CLASSIFIER_BY_ENCODER = {
    "mel": "cnn-bigru",
    "openl3": "mlp",
    "muq": "mlp",
    "clap": "mlp",
}


def _available_pairs():
    return ", ".join(f"{encoder}+{classifier}" for encoder, classifier in MODEL_SPECS)


def _resolve_experiment(args, parser):
    if args.classifier is None:
        args.classifier = DEFAULT_CLASSIFIER_BY_ENCODER[args.encoder]

    classifier = CLASSIFIERS[args.classifier]
    encoder = ENCODERS.get(args.encoder, {}).get(classifier.feature_kind)
    model = MODEL_SPECS.get((args.encoder, args.classifier))
    if encoder is None or model is None:
        parser.error(
            f"unsupported encoder/classifier pair: {args.encoder}+{args.classifier}. "
            f"Supported pairs: {_available_pairs()}"
        )

    if args.augment and classifier.feature_kind != "mel_segment":
        parser.error("--augment only applies to --encoder mel --classifier cnn-bigru")
    if args.use_lso and args.classifier != "cnn-bigru":
        parser.error("--use-lso only applies to --encoder mel --classifier cnn-bigru")
    if args.use_quadrant_head and args.classifier != "cnn-bigru":
        parser.error("--use-quadrant-head only applies to --encoder mel --classifier cnn-bigru")

    args.feature_kind = classifier.feature_kind
    args.encoder_label = encoder.label
    args.experiment_label = model.label
    args.artifact_prefix = model.artifact_prefix
    args.epochs = 80 if args.epochs is None else args.epochs
    args.patience = 15 if args.patience is None else args.patience
    args.batch_size = model.default_batch_size if args.batch_size is None else args.batch_size
    args.lr = model.default_lr if args.lr is None else args.lr
    args.weight_decay = model.default_weight_decay if args.weight_decay is None else args.weight_decay
    args.workers = PREP_WORKERS if args.workers is None else args.workers

    if classifier.feature_kind == "mel_segment":
        args.dropout = 0.7 if args.dropout is None else args.dropout
        args.conv1_filters = 16 if args.conv1_filters is None else args.conv1_filters
        args.conv2_filters = 32 if args.conv2_filters is None else args.conv2_filters
        args.hidden_size = 64 if args.hidden_size is None else args.hidden_size
        args.lso_pop_size = 15 if args.lso_pop_size is None else args.lso_pop_size
        args.lso_iterations = 30 if args.lso_iterations is None else args.lso_iterations
        args.lso_eval_epochs = 10 if args.lso_eval_epochs is None else args.lso_eval_epochs
        args.lso_eval_patience = 5 if args.lso_eval_patience is None else args.lso_eval_patience

    return args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train or evaluate a MoodWave model.",
        epilog=f"Supported pairs: {_available_pairs()}",
    )
    parser.add_argument("--encoder", choices=tuple(ENCODERS), default="mel")
    parser.add_argument(
        "--classifier",
        choices=tuple(CLASSIFIERS),
        default=None,
        help="Defaults to cnn-bigru for mel, otherwise mlp.",
    )
    parser.add_argument("--split", choices=("paper", "strict"), default="strict")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--eval-only", action="store_true")

    parser.add_argument("--augment", action="store_true", help="Enable mel audio augmentation during cache extraction.")
    parser.add_argument("--use-quadrant-head", action="store_true", help="Add a 4-class auxiliary head for mel CNN-BiGRU.")
    parser.add_argument("--use-lso", action="store_true", help="Run Lion Swarm Optimization before mel CNN-BiGRU training.")
    parser.add_argument("--lso-pop-size", type=int, default=None)
    parser.add_argument("--lso-iterations", type=int, default=None)
    parser.add_argument("--lso-eval-epochs", type=int, default=None, help="Epochs per LSO candidate evaluation.")
    parser.add_argument("--lso-eval-patience", type=int, default=None, help="Early-stopping patience for LSO evals.")
    parser.add_argument("--dropout", type=float, default=None, help="Mel CNN-BiGRU dropout rate.")
    parser.add_argument("--conv1-filters", type=int, default=None, help="Mel CNN layer 1 filter count.")
    parser.add_argument("--conv2-filters", type=int, default=None, help="Mel CNN layer 2 filter count.")
    parser.add_argument("--hidden-size", type=int, default=None, help="Mel BiGRU hidden size.")
    return _resolve_experiment(parser.parse_args(argv), parser)


def _prepare_runtime():
    set_seed(SEED)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    return torch


def _artifact_paths(args):
    suffix = "paper" if args.split == "paper" else "strict"
    stem = f"{args.artifact_prefix}_{suffix}"
    return (
        f"models/weights/{stem}.pth",
        f"models/metrics/{stem}_report.txt",
        f"models/metrics/{stem}_metrics.csv",
    )


def _ensure_artifact_dirs():
    os.makedirs("models/weights", exist_ok=True)
    os.makedirs("models/metrics", exist_ok=True)


def _checkpoint_path_for_read(model_path):
    if os.path.exists(model_path):
        return model_path
    legacy_path = os.path.join("models", os.path.basename(model_path))
    if os.path.exists(legacy_path):
        return legacy_path
    return model_path


def _checkpoint_stats(torch, stats):
    return {key: torch.as_tensor(value) for key, value in stats.items()}


def _prepare_features(args, encoder):
    raw = encoder.loader(args)
    split_yv = raw.split_yv if raw.split_yv is not None else raw.yv
    split_ya = raw.split_ya if raw.split_ya is not None else raw.ya
    train_idx, val_idx, test_idx = split_indices(split_yv, split_ya, raw.track_ids, args.split)

    X_train = raw.X[train_idx]
    yv_train = raw.yv[train_idx]
    ya_train = raw.ya[train_idx]
    X_val = raw.X[val_idx]
    yv_val = raw.yv[val_idx]
    ya_val = raw.ya[val_idx]
    X_test = raw.X[test_idx]
    yv_test = raw.yv[test_idx]
    ya_test = raw.ya[test_idx]

    lengths_train = lengths_val = lengths_test = None
    if raw.lengths is not None:
        lengths_train = raw.lengths[train_idx]
        lengths_val = raw.lengths[val_idx]
        lengths_test = raw.lengths[test_idx]

    if raw.kind == "mel_segment":
        stats = compute_mel_stats(X_train)
        X_train = normalize_segments(X_train, stats)
        X_val = normalize_segments(X_val, stats)
        X_test = normalize_segments(X_test, stats)
        stats_label = "mel stats"
    elif raw.kind == "vector":
        stats = compute_feature_stats(X_train)
        X_train = normalize_features(X_train, stats)
        X_val = normalize_features(X_val, stats)
        X_test = normalize_features(X_test, stats)
        stats_label = "feature stats"
    elif raw.kind == "sequence":
        stats = compute_seq_feature_stats(X_train, lengths_train)
        X_train = normalize_seq_features(X_train, stats)
        X_val = normalize_seq_features(X_val, stats)
        X_test = normalize_seq_features(X_test, stats)
        stats_label = "feature stats"
    else:
        raise ValueError(f"Unknown feature kind: {raw.kind}")

    return FeatureData(
        kind=raw.kind,
        sample_unit=raw.sample_unit,
        X_train=X_train,
        yv_train=yv_train,
        ya_train=ya_train,
        X_val=X_val,
        yv_val=yv_val,
        ya_val=ya_val,
        X_test=X_test,
        yv_test=yv_test,
        ya_test=ya_test,
        stats=stats,
        stats_label=stats_label,
        input_dim=raw.input_dim,
        lengths_train=lengths_train,
        lengths_val=lengths_val,
        lengths_test=lengths_test,
        track_ids_val=raw.track_ids[val_idx] if raw.track_metrics else None,
        track_ids_test=raw.track_ids[test_idx] if raw.track_metrics else None,
        extra=raw.extra,
    )


def _print_split(args, data, device):
    print(f"\nExperiment: {args.encoder}+{args.classifier} ({args.experiment_label})")
    print(f"Features: {args.encoder_label}")
    print(f"Sample unit: {data.sample_unit}")
    print(f"Split: {args.split}")
    print(f"  Train: {len(data.X_train)} | Val: {len(data.X_val)} | Test: {len(data.X_test)}")
    for label, value in data.extra:
        print(f"  {label}: {value}")
    print(f"  Device: {device}")


def _save_checkpoint(torch, model, path, args, data, model_kwargs=None):
    checkpoint = {
        "state_dict": model.state_dict(),
        "encoder": args.encoder,
        "classifier": args.classifier,
        "feature_stats": _checkpoint_stats(torch, data.stats),
        "feature_stats_label": data.stats_label,
        "sample_unit": data.sample_unit,
    }
    if data.input_dim is not None:
        checkpoint["input_dim"] = data.input_dim
    if model_kwargs is not None:
        checkpoint.update(model_kwargs)
    torch.save(checkpoint, path)


def _metrics_logger(csv_path):
    if MetricsLogger is None:
        raise ImportError("Metrics logging requires matplotlib. Install dependencies with `pip install -r requirements.txt`.")
    return MetricsLogger(csv_path=csv_path)


def _run_vector_sample_regression(args, data, torch):
    model_spec = MODEL_SPECS[(args.encoder, args.classifier)]
    model_path, report_path, metrics_path = _artifact_paths(args)
    _ensure_artifact_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print_split(args, data, device)

    if args.eval_only:
        read_path = _checkpoint_path_for_read(model_path)
        if not os.path.exists(read_path):
            raise FileNotFoundError(f"Missing checkpoint at {model_path}")
        checkpoint = torch.load(read_path, map_location=device)
        model = model_spec.model_factory(data, args).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
    else:
        metrics_logger = None
        if model_spec.log_metrics:
            metrics_logger = _metrics_logger(metrics_path)

        model = train_vector_model(
            data.X_train,
            data.yv_train,
            data.ya_train,
            data.X_val,
            data.yv_val,
            data.ya_val,
            device,
            model_factory=lambda: model_spec.model_factory(data, args),
            desc=args.experiment_label,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            track_ids_val=data.track_ids_val,
            metrics_logger=metrics_logger,
        )
        _save_checkpoint(torch, model, model_path, args, data)
        print(f"\nSaved {args.experiment_label} checkpoint: {model_path}")
        print(f"Embedded {data.stats_label} in checkpoint")

    test_ds = RegressionDataset(data.X_test, data.yv_test, data.ya_test, track_ids=data.track_ids_test)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, pin_memory=torch.cuda.is_available())
    test_metrics = evaluate_vector_model(model, test_loader, device, loss_fn=nn.MSELoss())
    write_report(f"{args.experiment_label} {args.split} test metrics", test_metrics, report_path=report_path)
    print(f"Report saved to {report_path}")


def _run_sequence_window_regression(args, data, torch):
    model_spec = MODEL_SPECS[(args.encoder, args.classifier)]
    model_path, report_path, metrics_path = _artifact_paths(args)
    _ensure_artifact_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print_split(args, data, device)

    if args.eval_only:
        read_path = _checkpoint_path_for_read(model_path)
        if not os.path.exists(read_path):
            raise FileNotFoundError(f"Missing checkpoint at {model_path}")
        checkpoint = torch.load(read_path, map_location=device)
        if args.encoder == "muq" and args.classifier == "bigru" and checkpoint.get("sample_unit") != "song":
            raise RuntimeError(
                "This MuQ-BiGRU checkpoint was saved for the old local-frame sequence model. "
                "Retrain with `python train.py --encoder muq --classifier bigru --split strict`."
            )
        model = model_spec.model_factory(data, args).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
    else:
        metrics_logger = None
        if model_spec.log_metrics:
            metrics_logger = _metrics_logger(metrics_path)

        model = train_sequence_model(
            data.X_train,
            data.lengths_train,
            data.yv_train,
            data.ya_train,
            data.X_val,
            data.lengths_val,
            data.yv_val,
            data.ya_val,
            device,
            model_factory=lambda: model_spec.model_factory(data, args),
            desc=args.experiment_label,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            metrics_logger=metrics_logger,
        )
        _save_checkpoint(torch, model, model_path, args, data)
        print(f"\nSaved {args.experiment_label} checkpoint: {model_path}")
        print(f"Embedded {data.stats_label} in checkpoint")

    test_ds = SequenceRegressionDataset(data.X_test, data.lengths_test, data.yv_test, data.ya_test)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, pin_memory=torch.cuda.is_available())
    test_metrics = evaluate_sequence_model(model, test_loader, device, loss_fn=nn.MSELoss())
    write_report(f"{args.experiment_label} {args.split} test metrics", test_metrics, report_path=report_path)
    print(f"Report saved to {report_path}")


def _run_mel_window_cnn_bigru(args, data, torch):
    model_path, report_path, metrics_path = _artifact_paths(args)
    _ensure_artifact_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print_split(args, data, device)

    if args.eval_only:
        read_path = _checkpoint_path_for_read(model_path)
        if not os.path.exists(read_path):
            raise FileNotFoundError(f"Missing checkpoint at {model_path}")
        checkpoint = torch.load(read_path, map_location=device)
        model_kwargs = dict(checkpoint.get("model_kwargs", {}))
        in_ch = checkpoint.get("in_channels", model_kwargs.pop("in_channels", 2))
        model_kwargs.pop("in_channels", None)
        use_qh = checkpoint.get("use_quadrant_head", args.use_quadrant_head)
        model = MoodCNNBiGRU(
            n_mels=MEL_N_MELS,
            in_channels=in_ch,
            num_quadrant_classes=4 if use_qh else 0,
            **model_kwargs,
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
    else:
        if args.use_lso:
            print("\n=== Running Lion Swarm Optimization (LSO) ===")
            lso = LSOSearch(
                data.X_train,
                data.yv_train,
                data.ya_train,
                data.X_val,
                data.yv_val,
                data.ya_val,
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
            for key, value in best_params.items():
                print(f"  {key}: {value}")
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
            model_kwargs = {
                "conv1_filters": args.conv1_filters,
                "conv2_filters": args.conv2_filters,
                "hidden_size": args.hidden_size,
                "dropout": args.dropout,
            }
            final_lr = args.lr
            final_wd = args.weight_decay

        model_kwargs["in_channels"] = 2
        metrics_logger = _metrics_logger(metrics_path)
        model = train_model(
            data.X_train,
            data.yv_train,
            data.ya_train,
            data.X_val,
            data.yv_val,
            data.ya_val,
            device,
            args=args,
            n_mels=MEL_N_MELS,
            model_kwargs=model_kwargs,
            lr=final_lr,
            weight_decay=final_wd,
            num_workers=args.workers,
            use_quadrant_head=args.use_quadrant_head,
            track_ids_val=data.track_ids_val,
            metrics_logger=metrics_logger,
        )
        checkpoint_model_kwargs = dict(model_kwargs)
        in_channels = checkpoint_model_kwargs.pop("in_channels", 2)
        _save_checkpoint(
            torch,
            model,
            model_path,
            args,
            data,
            model_kwargs={
                "model_kwargs": checkpoint_model_kwargs,
                "use_quadrant_head": args.use_quadrant_head,
                "in_channels": in_channels,
            },
        )
        print(f"\nSaved {args.experiment_label} checkpoint: {model_path}")
        print(f"Embedded {data.stats_label} in checkpoint")
        use_qh = args.use_quadrant_head

    test_loader = DataLoader(
        MelRegressionDataset(data.X_test, data.yv_test, data.ya_test, track_ids=data.track_ids_test, augment=False),
        batch_size=args.batch_size * 2,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    test_metrics = evaluate_loader(model, test_loader, device, use_quadrant_head=use_qh)
    write_report(f"{args.experiment_label} {args.split} test metrics", test_metrics, report_path=report_path)
    print(f"Report saved to {report_path}")


def main(argv=None):
    args = parse_args(argv)
    torch = _prepare_runtime()
    classifier = CLASSIFIERS[args.classifier]
    encoder = ENCODERS[args.encoder][classifier.feature_kind]
    data = _prepare_features(args, encoder)
    classifier.runner(args, data, torch)


if __name__ == "__main__":
    freeze_support()
    main()
