"""Lion Swarm Optimization (LSO) for CNN-BiGRU hyperparameter tuning."""

import numpy as np
import torch

from features.dataset import build_loaders 
from training.engine import evaluate_loader, train_model

SEARCH_SPACE = {
    "lr": {"type": "categorical", "choices": [5e-5, 1e-4, 1e-3]},
    "weight_decay": {"type": "continuous", "low": 1e-5, "high": 1e-2},
    "conv1_filters": {"type": "categorical", "choices": [32, 64]},
    "conv2_filters": {"type": "categorical", "choices": [64, 128]},
    "kernel_size": {"type": "categorical", "choices": [3, 5]},
    "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
    "dropout": {"type": "continuous", "low": 0.2, "high": 0.5},
}

NOMINAL_DEFAULTS = {
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "conv1_filters": 32,
    "conv2_filters": 64,
    "kernel_size": 3,
    "hidden_size": 128,
    "dropout": 0.3,
}


def _encode(param_dict):
    vec = []
    for key, spec in SEARCH_SPACE.items():
        val = param_dict[key]
        if spec["type"] == "categorical":
            try:
                idx = spec["choices"].index(val)
            except ValueError:
                idx = 0
            n = len(spec["choices"])
            vec.append(idx / (n - 1) if n > 1 else 0.5)
        elif spec["type"] == "continuous":
            low, high = spec["low"], spec["high"]
            vec.append((val - low) / (high - low))
    return np.array(vec, dtype=np.float32)


def _decode(vec):
    param_dict = {}
    for i, (key, spec) in enumerate(SEARCH_SPACE.items()):
        v = float(np.clip(vec[i], 0.0, 1.0))
        if spec["type"] == "categorical":
            n = len(spec["choices"])
            idx = int(round(v * (n - 1))) if n > 1 else 0
            param_dict[key] = spec["choices"][idx]
        elif spec["type"] == "continuous":
            param_dict[key] = v * (spec["high"] - spec["low"]) + spec["low"]
    return param_dict


def _build_model_kwargs(param_dict):
    return {
        "conv1_filters": param_dict["conv1_filters"],
        "conv2_filters": param_dict["conv2_filters"],
        "kernel_size": param_dict["kernel_size"],
        "hidden_size": param_dict["hidden_size"],
        "dropout": param_dict["dropout"],
    }


class LSOSearch:
    def __init__(
        self,
        X_train,
        yv_train,
        ya_train,
        X_val,
        yv_val,
        ya_val,
        device,
        pop_size=15,
        iterations=30,
        pride_ratio=0.7,
        rho=0.2,
        gamma=0.7,
        mutation_rate=0.05,
        eval_epochs=10,
        eval_patience=5,
        batch_size=32,
        n_mels=64,
        seed=42,
        num_workers=0,
    ):
        self.X_train = X_train
        self.yv_train = yv_train
        self.ya_train = ya_train
        self.X_val = X_val
        self.yv_val = yv_val
        self.ya_val = ya_val
        self.device = device
        self.pop_size = pop_size
        self.iterations = iterations
        self.pride_ratio = pride_ratio
        self.rho = rho
        self.gamma = gamma
        self.mutation_rate = mutation_rate
        self.eval_epochs = eval_epochs
        self.eval_patience = eval_patience
        self.batch_size = batch_size
        self.n_mels = n_mels
        self.rng = np.random.default_rng(seed)
        self.n_dims = len(SEARCH_SPACE)
        self.num_workers = num_workers

    def _evaluate(self, param_dict):
        model_kwargs = _build_model_kwargs(param_dict)
        lr = param_dict["lr"]
        wd = param_dict["weight_decay"]

        model = train_model(
            self.X_train,
            self.yv_train,
            self.ya_train,
            self.X_val,
            self.yv_val,
            self.ya_val,
            self.device,
            args=None,
            n_mels=self.n_mels,
            epochs=self.eval_epochs,
            patience=self.eval_patience,
            lr=lr,
            batch_size=self.batch_size,
            weight_decay=wd,
            model_kwargs=model_kwargs,
            best_metric_mode="loss",
            quiet=True,
            num_workers=self.num_workers,
        )

        _, val_loader = build_loaders(
            self.X_train,
            self.yv_train,
            self.ya_train,
            self.X_val,
            self.yv_val,
            self.ya_val,
            self.batch_size * 2,
        )
        loss_fn = torch.nn.MSELoss()
        metrics = evaluate_loader(model, val_loader, self.device, loss_fn=loss_fn)
        fitness = metrics["loss"]
        print(
            f"    LSO eval: lr={lr:.0e}, wd={wd:.0e}, c1={model_kwargs['conv1_filters']}, "
            f"c2={model_kwargs['conv2_filters']}, k={model_kwargs['kernel_size']}, "
            f"h={model_kwargs['hidden_size']}, do={model_kwargs['dropout']:.2f} -> loss={fitness:.4f}"
        )
        return fitness

    def optimize(self):
        print(f"\n[LSO] Initializing population (M={self.pop_size}, dims={self.n_dims})")
        positions = self.rng.random((self.pop_size, self.n_dims))
        positions_prev = positions.copy()
        fitness = np.array([self._evaluate(_decode(positions[i])) for i in range(self.pop_size)])

        for t in range(self.iterations):
            sorted_idx = np.argsort(fitness)
            positions = positions[sorted_idx]
            positions_prev = positions_prev[sorted_idx]
            fitness = fitness[sorted_idx]

            n_prides = max(1, int(self.pop_size * self.pride_ratio))
            prides = positions[:n_prides]
            prides_prev = positions_prev[:n_prides]
            pride_best = prides[0].copy()

            # Update pride members (Eq. 22 simplified)
            delta = prides - prides_prev
            rand_vec = self.rng.random(prides.shape)
            new_prides = pride_best + self.rho * rand_vec + self.gamma * delta
            new_prides = np.clip(new_prides, 0.0, 1.0)

            new_prides_fitness = np.array(
                [self._evaluate(_decode(new_prides[i])) for i in range(n_prides)]
            )

            # Generate offspring (Eq. 24)
            n_offspring = self.pop_size
            p1 = self.rng.integers(0, n_prides, size=n_offspring)
            p2 = self.rng.integers(0, n_prides, size=n_offspring)
            offspring = (
                0.5 * (prides[p1] + prides[p2])
                + self.mutation_rate * self.rng.standard_normal((n_offspring, self.n_dims))
            )
            offspring = np.clip(offspring, 0.0, 1.0)

            offspring_fitness = np.array(
                [self._evaluate(_decode(offspring[i])) for i in range(n_offspring)]
            )

            # Elitist selection: union of old pop, updated prides, and offspring -> keep top M
            union_positions = np.vstack([positions, new_prides, offspring])
            union_fitness = np.hstack([fitness, new_prides_fitness, offspring_fitness])
            top_idx = np.argsort(union_fitness)[: self.pop_size]
            new_positions = union_positions[top_idx]

            positions_prev = positions.copy()
            positions = new_positions
            fitness = union_fitness[top_idx]

            best_params = _decode(positions[0])
            print(
                f"[LSO] Iteration {t + 1}/{self.iterations}, best fitness={fitness[0]:.4f}, "
                f"best lr={best_params['lr']:.0e}, hidden={best_params['hidden_size']}, "
                f"dropout={best_params['dropout']:.2f}"
            )

        return _decode(positions[0])
