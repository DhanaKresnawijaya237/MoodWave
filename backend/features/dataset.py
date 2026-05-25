import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MelRegressionDataset(Dataset):
    def __init__(self, X, yv, ya, track_ids=None, augment=False):
        self.X = X
        self.yv = yv.astype(np.float32, copy=False)
        self.ya = ya.astype(np.float32, copy=False)
        self.track_ids = track_ids
        self.augment = augment

    def __len__(self):
        return len(self.yv)

    def _augment(self, x):
        # Gaussian noise
        if torch.rand(1).item() < 0.5:
            x = x + 0.02 * torch.randn_like(x)

        # Time masking (horizontal) — wider for longer segments
        if torch.rand(1).item() < 0.5:
            max_w = max(1, x.shape[-1] // 5)
            width = int(torch.randint(max_w // 2, max_w + 1, (1,)).item())
            start = int(torch.randint(0, max(1, x.shape[-1] - width + 1), (1,)).item())
            x[:, :, start : start + width] = 0

        # Frequency masking (vertical)
        if torch.rand(1).item() < 0.5:
            max_w = max(1, x.shape[-2] // 4)
            width = int(torch.randint(max_w // 2, max_w + 1, (1,)).item())
            start = int(torch.randint(0, max(1, x.shape[-2] - width + 1), (1,)).item())
            x[:, start : start + width, :] = 0

        # Random time warping (subsample / stretch)
        if torch.rand(1).item() < 0.3 and x.shape[-1] > 20:
            src_len = x.shape[-1]
            dst_len = int(src_len * torch.empty(1).uniform_(0.8, 1.2).item())
            dst_len = max(10, min(dst_len, src_len * 2))
            x = torch.nn.functional.interpolate(
                x.unsqueeze(0), size=(x.shape[-2], dst_len), mode="bilinear", align_corners=False
            ).squeeze(0)
            if dst_len != src_len:
                # Center crop or pad back to original length
                if dst_len > src_len:
                    start = (dst_len - src_len) // 2
                    x = x[:, :, start : start + src_len]
                else:
                    pad_left = (src_len - dst_len) // 2
                    pad_right = src_len - dst_len - pad_left
                    x = torch.nn.functional.pad(x, (pad_left, pad_right), mode="replicate")

        return x

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index].astype(np.float32, copy=False))
        if self.augment:
            x = self._augment(x.clone())
        out = [
            x,
            torch.tensor(self.yv[index], dtype=torch.float32),
            torch.tensor(self.ya[index], dtype=torch.float32),
        ]
        if self.track_ids is not None:
            out.append(torch.tensor(self.track_ids[index], dtype=torch.long))
        return tuple(out)


def build_loaders(X_train, yv_train, ya_train, X_val, yv_val, ya_val, batch_size, num_workers=0, track_ids_train=None, track_ids_val=None):
    train_ds = MelRegressionDataset(X_train, yv_train, ya_train, track_ids=track_ids_train, augment=True)
    val_ds = MelRegressionDataset(X_val, yv_val, ya_val, track_ids=track_ids_val, augment=False)
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
