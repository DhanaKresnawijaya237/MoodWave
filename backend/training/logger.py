"""Per-epoch metrics logging + plotting for training scripts."""

import csv
import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


class MetricsLogger:
    """Logs per-epoch metrics to CSV and generates plots."""

    def __init__(self, csv_path, plot_dir=None):
        self.csv_path = csv_path
        self.plot_dir = plot_dir or os.path.splitext(csv_path)[0] + "_plots"
        os.makedirs(self.plot_dir, exist_ok=True)
        self.rows = []
        self._fieldnames = None
        self._csv_f = None
        self._writer = None

    def _open_csv(self):
        if self._csv_f is None:
            self._csv_f = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._csv_f, fieldnames=self._fieldnames)
            self._writer.writeheader()
            self._csv_f.flush()

    def log(self, row: dict):
        """Append a row of metrics. Call once per epoch."""
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            self._open_csv()
        # Ensure consistent field ordering
        ordered = {k: row.get(k, "") for k in self._fieldnames}
        self.rows.append(ordered)
        self._writer.writerow(ordered)
        self._csv_f.flush()

    def _save(self, fig, name):
        path = os.path.join(self.plot_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot(self):
        """Generate and save all plots from accumulated rows."""
        if not self.rows:
            return []

        epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(self.rows)]
        saved = []

        # ── Loss curves ──
        if "train_loss" in self.rows[0] or "val_loss" in self.rows[0]:
            fig, ax = plt.subplots(figsize=(8, 5))
            if "train_loss" in self.rows[0]:
                ax.plot(epochs, [r["train_loss"] for r in self.rows], label="Train Loss", linewidth=1.5)
            if "val_loss" in self.rows[0]:
                ax.plot(epochs, [r["val_loss"] for r in self.rows], label="Val Loss", linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training & Validation Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            saved.append(self._save(fig, "loss_curve.png"))

        # ── R² curves ──
        has_vr2 = "valence_r2" in self.rows[0]
        has_ar2 = "arousal_r2" in self.rows[0]
        if has_vr2 or has_ar2:
            fig, ax = plt.subplots(figsize=(8, 5))
            if has_vr2:
                ax.plot(epochs, [r["valence_r2"] for r in self.rows], label="Valence R²", linewidth=1.5)
            if has_ar2:
                ax.plot(epochs, [r["arousal_r2"] for r in self.rows], label="Arousal R²", linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("R²")
            ax.set_title("Validation R²")
            ax.legend()
            ax.grid(True, alpha=0.3)
            saved.append(self._save(fig, "r2_curve.png"))

        # ── MAE curves ──
        has_vmae = "valence_mae" in self.rows[0]
        has_amae = "arousal_mae" in self.rows[0]
        if has_vmae or has_amae:
            fig, ax = plt.subplots(figsize=(8, 5))
            if has_vmae:
                ax.plot(epochs, [r["valence_mae"] for r in self.rows], label="Valence MAE", linewidth=1.5)
            if has_amae:
                ax.plot(epochs, [r["arousal_mae"] for r in self.rows], label="Arousal MAE", linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MAE")
            ax.set_title("Validation MAE")
            ax.legend()
            ax.grid(True, alpha=0.3)
            saved.append(self._save(fig, "mae_curve.png"))

        # ── Quadrant accuracy ──
        if "quadrant_acc" in self.rows[0]:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, [r["quadrant_acc"] for r in self.rows], label="Quadrant Acc", linewidth=1.5, color="purple")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("Validation Quadrant Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)
            saved.append(self._save(fig, "quadrant_acc_curve.png"))

        # ── LR schedule ──
        if "lr" in self.rows[0]:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, [r["lr"] for r in self.rows], linewidth=1.5, color="orange")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            saved.append(self._save(fig, "lr_schedule.png"))

        # ── OpenL3-specific: RMSE + PCC ──
        has_rmse_v = "rmse_valence" in self.rows[0]
        has_rmse_a = "rmse_arousal" in self.rows[0]
        if has_rmse_v or has_rmse_a:
            fig, ax = plt.subplots(figsize=(8, 5))
            if has_rmse_v:
                ax.plot(epochs, [r["rmse_valence"] for r in self.rows], label="RMSE Valence", linewidth=1.5)
            if has_rmse_a:
                ax.plot(epochs, [r["rmse_arousal"] for r in self.rows], label="RMSE Arousal", linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("RMSE")
            ax.set_title("Validation RMSE")
            ax.legend()
            ax.grid(True, alpha=0.3)
            saved.append(self._save(fig, "rmse_curve.png"))

        has_pcc_v = "pcc_valence" in self.rows[0]
        has_pcc_a = "pcc_arousal" in self.rows[0]
        if has_pcc_v or has_pcc_a:
            fig, ax = plt.subplots(figsize=(8, 5))
            if has_pcc_v:
                ax.plot(epochs, [r["pcc_valence"] for r in self.rows], label="PCC Valence", linewidth=1.5)
            if has_pcc_a:
                ax.plot(epochs, [r["pcc_arousal"] for r in self.rows], label="PCC Arousal", linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Pearson r")
            ax.set_title("Validation Pearson Correlation")
            ax.legend()
            ax.grid(True, alpha=0.3)
            saved.append(self._save(fig, "pcc_curve.png"))

        # ── Track-level metrics (if present) ──
        has_tr_vmae = "track_valence_mae" in self.rows[0]
        if has_tr_vmae:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, [r["track_valence_mae"] for r in self.rows], label="Track V MAE", linewidth=1.5)
            ax.plot(epochs, [r["track_arousal_mae"] for r in self.rows], label="Track A MAE", linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MAE")
            ax.set_title("Track-Level Validation MAE")
            ax.legend()
            ax.grid(True, alpha=0.3)
            saved.append(self._save(fig, "track_mae_curve.png"))

        has_tr_vr2 = "track_valence_r2" in self.rows[0]
        if has_tr_vr2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, [r["track_valence_r2"] for r in self.rows], label="Track V R²", linewidth=1.5)
            ax.plot(epochs, [r["track_arousal_r2"] for r in self.rows], label="Track A R²", linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("R²")
            ax.set_title("Track-Level Validation R²")
            ax.legend()
            ax.grid(True, alpha=0.3)
            saved.append(self._save(fig, "track_r2_curve.png"))

        return saved

    def close(self):
        if self._csv_f is not None:
            self._csv_f.close()
            self._csv_f = None
