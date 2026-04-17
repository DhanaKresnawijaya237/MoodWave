import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MelRegressionDataset(Dataset):
    def __init__(self, X, yv, ya, augment=False):
        self.X = X
        self.yv = yv.astype(np.float32, copy=False)
        self.ya = ya.astype(np.float32, copy=False)
        self.augment = augment

    def __len__(self):
        return len(self.yv)

    def _augment(self, x):
        if torch.rand(1).item() < 0.5:
            x = x + 0.01 * torch.randn_like(x)

        if torch.rand(1).item() < 0.5:
            width = int(torch.randint(4, 10, (1,)).item())
            start = int(torch.randint(0, max(1, x.shape[-1] - width + 1), (1,)).item())
            x[:, :, start : start + width] = 0

        if torch.rand(1).item() < 0.5:
            width = int(torch.randint(4, 10, (1,)).item())
            start = int(torch.randint(0, max(1, x.shape[-2] - width + 1), (1,)).item())
            x[:, start : start + width, :] = 0

        return x

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index].astype(np.float32, copy=False))
        if self.augment:
            x = self._augment(x.clone())
        return (
            x,
            torch.tensor(self.yv[index], dtype=torch.float32),
            torch.tensor(self.ya[index], dtype=torch.float32),
        )


def build_loaders(X_train, yv_train, ya_train, X_val, yv_val, ya_val, batch_size, num_workers=0):
    train_ds = MelRegressionDataset(X_train, yv_train, ya_train, augment=True)
    val_ds = MelRegressionDataset(X_val, yv_val, ya_val, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
