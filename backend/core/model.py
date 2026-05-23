import torch
import torch.nn as nn


class MoodCNNBiGRU(nn.Module):
    """CNN-BiGRU for music emotion recognition (valence + arousal regression).

    Input:  (batch, 1, n_mels, n_frames) — default (B, 1, 64, 64)
    Output: valence (B,), arousal (B,) [+ quadrant logits (B, 4) if num_quadrant_classes > 0]
    """

    def __init__(
        self,
        n_mels=64,
        in_channels=2,
        conv1_filters=32,
        conv2_filters=64,
        kernel_size=3,
        pool1=(2, 2),
        pool2=(2, 1),
        hidden_size=128,
        dropout=0.5,
        num_quadrant_classes=0,
        proj_dim=256,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, conv1_filters, kernel_size=kernel_size, padding=padding),
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
            dummy = torch.zeros(1, in_channels, n_mels, 64)
            feat = self.cnn(dummy)
            _, C, F, T = feat.shape
            gru_input_dim = C * F

        self.proj = nn.Linear(gru_input_dim, proj_dim)
        self.bigru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Attention pooling over BiGRU outputs
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

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
        self.num_quadrant_classes = num_quadrant_classes
        if num_quadrant_classes > 0:
            self.quadrant_head = nn.Sequential(
                nn.Linear(head_in, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_quadrant_classes),
            )

    def forward(self, x):
        feat = self.cnn(x)
        B, C, F, T = feat.shape
        feat = feat.permute(0, 3, 1, 2).reshape(B, T, C * F)
        feat = self.proj(feat)
        gru_out, _ = self.bigru(feat)

        # Attention pooling
        attn_scores = self.attention(gru_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(gru_out * attn_weights, dim=1)

        pooled = self.dropout(pooled)
        valence = self.valence_head(pooled).squeeze(-1)
        arousal = self.arousal_head(pooled).squeeze(-1)
        if self.num_quadrant_classes > 0:
            quadrant = self.quadrant_head(pooled)
            return valence, arousal, quadrant
        return valence, arousal


class MoodOpenL3MLP(nn.Module):
    """OpenL3 + MLP for frame-level V/A regression.

    Replicates the paper architecture:
    - Two fully-connected hidden layers (256, 128)
    - ReLU activation
    - Dropout 0.2
    - 2-dim linear output (valence, arousal)

    Input:  (batch, 512) — OpenL3 embedding
    Output: valence (B,), arousal (B,)
    """

    def __init__(self, input_dim=512, hidden_dims=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 2)

    def forward(self, x):
        # x: (B, 512)
        h = self.hidden(x)
        out = self.output(h)
        return out[:, 0], out[:, 1]


class MoodMuQMLP(nn.Module):
    """MuQ + MLP for track-level V/A regression.

    Takes a mean-pooled MuQ embedding and predicts
    track-level valence and arousal.

    Input:  (batch, embed_dim) — MuQ embedding
    Output: valence (B,), arousal (B,)
    """

    def __init__(self, input_dim=768, hidden_dims=(512, 256), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.hidden = nn.Sequential(*layers)
        self.valence_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Tanh(),
        )
        self.arousal_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.hidden(x)
        valence = self.valence_head(h).squeeze(-1)
        arousal = self.arousal_head(h).squeeze(-1)
        return valence, arousal


class MoodCLAPMLP(nn.Module):
    """CLAP + MLP for track-level V/A regression.

    Takes a CLAP audio embedding and predicts
    track-level valence and arousal.

    Input:  (batch, embed_dim) — CLAP embedding
    Output: valence (B,), arousal (B,)
    """

    def __init__(self, input_dim=512, hidden_dims=(256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.hidden = nn.Sequential(*layers)
        self.valence_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Tanh(),
        )
        self.arousal_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.hidden(x)
        valence = self.valence_head(h).squeeze(-1)
        arousal = self.arousal_head(h).squeeze(-1)
        return valence, arousal


class MoodMuQBiGRU(nn.Module):
    """BiGRU over ordered 0.5s MuQ window embeddings.

    Returns one prediction per valid window; padded windows are masked by the trainer.

    Input:  (batch, time, embed_dim) — frame-level MuQ features
    Output: valence (B, time), arousal (B, time)
    """

    def __init__(self, input_dim=768, hidden_size=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        head_in = hidden_size * 2
        self.valence_head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        self.arousal_head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x, lengths):
        """Forward pass.

        Args:
            x: (B, T, input_dim) padded MuQ features.
            lengths: (B,) actual sequence lengths before padding.

        Returns:
            valence (B, T), arousal (B, T)
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.bigru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, total_length=x.size(1))
        gru_out = self.dropout(gru_out)
        valence = self.valence_head(gru_out).squeeze(-1)
        arousal = self.arousal_head(gru_out).squeeze(-1)
        return valence, arousal
