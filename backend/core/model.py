import torch
import torch.nn as nn


class MoodCNNBiGRU(nn.Module):
    """CNN-BiGRU for music emotion recognition (valence + arousal regression).

    Input:  (batch, 1, n_mels, n_frames) — default (B, 1, 64, 64)
    Output: valence (B,), arousal (B,)
    """

    def __init__(
        self,
        n_mels=64,
        conv1_filters=32,
        conv2_filters=64,
        kernel_size=3,
        pool1=(2, 2),
        pool2=(2, 1),
        hidden_size=128,
        dropout=0.3,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv2d(1, conv1_filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(conv1_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool1),
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(conv2_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool2),
        )

        # Dynamically compute GRU input dimension from actual CNN output shape
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, 64)
            feat = self.cnn(dummy)
            _, C, F, T = feat.shape
            gru_input_dim = C * F

        self.bigru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        head_in = hidden_size * 2
        self.valence_head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        self.arousal_head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        feat = self.cnn(x)
        B, C, F, T = feat.shape
        feat = feat.permute(0, 3, 1, 2).reshape(B, T, C * F)
        gru_out, _ = self.bigru(feat)
        pooled = self.dropout(gru_out.mean(dim=1))
        valence = self.valence_head(pooled).squeeze(-1)
        arousal = self.arousal_head(pooled).squeeze(-1)
        return valence, arousal
