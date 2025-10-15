import torch
import torch.nn as nn
import torch.nn.functional as F
from models.video_model import MultiScaleTemporalShift 


def normalize_chunk(chunk):
    mean = chunk.mean(dim=1, keepdim=True)  # (B, 1)
    std = chunk.std(dim=1, keepdim=True)    # (B, 1)
    std = torch.where(std == 0, torch.ones_like(std), std)  # 避免除以0
    chunk = (chunk - mean) / std
    return chunk


# Attention mechanism for chunks
class ChunkAttention(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 8),
            nn.ReLU(),
            nn.Linear(embed_dim // 8, 1)
        )

    def forward(self, x):  # x: (B, N, embed_dim)
        B, N, C = x.shape
        attention_scores = self.attn_fc(x)  # (B, N, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (B, N, 1)
        weighted_output = (x * attention_weights).sum(dim=1)  # Weighted sum across chunks
        return weighted_output

    
class Conv1DAudioEncoder(nn.Module):
    def __init__(self, config, in_channels=1, embed_dim=512):
        super().__init__()
        layers = []
        prev_channels = in_channels
        for out_channels, kernel_size, stride in config:
            layers.append(nn.Conv1d(prev_channels, out_channels, kernel_size, stride, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            prev_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(prev_channels, embed_dim)

    def forward(self, x):  # x: (B, T)
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.conv(x)    # (B, C, T')
        x = self.pool(x).squeeze(-1)  # (B, C)
        x = self.fc(x)      # (B, embed_dim)
        return x


def get_audio_encoder(size="small", embed_dim=512):
    if size == "small":
        config = [
            (64, 5, 1),   # Conv1d(1 -> 64), kernel=5, stride=1
            (128, 5, 2),  # Conv1d(64 -> 128), kernel=5, stride=2
        ]
    elif size == "medium":
        config = [
            (64, 5, 1),   # Conv1d(1 -> 64)
            (128, 5, 2),  # Conv1d(64 -> 128)
            (256, 3, 2),  # Conv1d(128 -> 256)
        ]
    elif size == "large":
        config = [
            (64, 5, 1),    # Conv1d(1 -> 64)
            (128, 5, 2),   # Conv1d(64 -> 128)
            (256, 3, 2),   # Conv1d(128 -> 256)
            (512, 3, 2),   # Conv1d(256 -> 512)
        ]
    else:
        raise ValueError("Unknown model size")

    return Conv1DAudioEncoder(config=config, embed_dim=embed_dim)


class LSTMTemporalEncoder(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=256, num_layers=2, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, embed_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# LSTM Encoder for temporal modeling
class LSTMTemporalEncoder(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=256, num_layers=2, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, embed_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# Full Audio Model integrating Conv, Temporal Shift, Attention, and LSTM
class AudioModel(nn.Module):
    def __init__(self, size="small", embed_dim=512, hidden_dim=256, num_classes=50,
                 chunk_size=800, chunk_interval=800):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_interval = chunk_interval
        self.encoder = get_audio_encoder(size=size, embed_dim=embed_dim)
        self.temporal = LSTMTemporalEncoder(embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.temporal_shift = MultiScaleTemporalShift(n_div=4)  # Temporal Shift for chunks
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, T = x.shape
        chunks = []
        for start in range(0, T, self.chunk_interval):
            end = start + self.chunk_size
            chunk = x[:, start:end]
            if chunk.shape[1] < self.chunk_size:
                pad_size = self.chunk_size - chunk.shape[1]
                chunk = nn.functional.pad(chunk, (0, pad_size))

            chunk = normalize_chunk(chunk)
            chunks.append(chunk)

        chunks = torch.stack(chunks, dim=1)  # (B, N, chunk_size)
        B, N, C = chunks.shape
        chunks = chunks.view(B * N, C)

        # Extract features from chunks using convolutional encoder
        embeddings = self.encoder(chunks)  # (B * N, embed_dim)
        embeddings = embeddings.view(B, N, -1)  # (B, N, embed_dim)

        # temporal shift
        feat = embeddings.permute(0, 2, 1)  # (B, embed_dim, N)
        feat = self.temporal_shift(feat)
        embeddings = feat.permute(0, 2, 1)  # (B, N, embed_dim)

        # Apply LSTM for temporal modeling
        pooled = self.temporal(embeddings)  # (B, embed_dim)
        logits = self.classifier(pooled)
        return logits


class AudioEncoder(nn.Module):
    def __init__(self, size="small", embed_dim=512, hidden_dim=256, num_classes=50,
                 chunk_size=800, chunk_interval=800):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_interval = chunk_interval
        self.encoder = get_audio_encoder(size=size, embed_dim=embed_dim)
        self.temporal = LSTMTemporalEncoder(embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.temporal_shift = MultiScaleTemporalShift(n_div=4)  # Temporal Shift for chunks

    def forward(self, x):
        B, T = x.shape
        chunks = []
        for start in range(0, T, self.chunk_interval):
            end = start + self.chunk_size
            chunk = x[:, start:end]
            if chunk.shape[1] < self.chunk_size:
                pad_size = self.chunk_size - chunk.shape[1]
                chunk = nn.functional.pad(chunk, (0, pad_size))

            chunk = normalize_chunk(chunk)
            chunks.append(chunk)

        chunks = torch.stack(chunks, dim=1)  # (B, N, chunk_size)
        B, N, C = chunks.shape
        chunks = chunks.view(B * N, C)

        # Extract features from chunks using convolutional encoder
        embeddings = self.encoder(chunks)  # (B * N, embed_dim)
        embeddings = embeddings.view(B, N, -1)  # (B, N, embed_dim)

        # temporal shift
        feat = embeddings.permute(0, 2, 1)  # (B, embed_dim, N)
        feat = self.temporal_shift(feat)
        embeddings = feat.permute(0, 2, 1)  # (B, N, embed_dim)

        # Apply LSTM for temporal modeling
        pooled = self.temporal(embeddings)  # (B, embed_dim)
        
        return pooled


class AudioTemporalEncoder(nn.Module):
    def __init__(self, size="small", embed_dim=512, hidden_dim=256, num_classes=50,
                 chunk_size=800, chunk_interval=800):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_interval = chunk_interval
        self.temporal_shift = MultiScaleTemporalShift(n_div=4)  # Temporal Shift for chunks
        self.temporal = LSTMTemporalEncoder(embed_dim=embed_dim, hidden_dim=hidden_dim)

    def forward(self, x):
        B, N, D = x.shape
        x = x.view(B, N, -1)  # (B, N, embed_dim)

        # temporal shift
        x = x.permute(0, 2, 1)  # (B, embed_dim, N)
        x = self.temporal_shift(x)
        x = x.permute(0, 2, 1)  # (B, N, embed_dim)

        # Apply LSTM for temporal modeling
        x = self.temporal(x)  # (B, embed_dim)
        
        return x


class AudioTemporalClassifier(nn.Module):
    def __init__(self, size="small", embed_dim=512, hidden_dim=256, num_classes=50,
                 chunk_size=800, chunk_interval=800):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_interval = chunk_interval
        self.temporal_shift = MultiScaleTemporalShift(n_div=4)  # Temporal Shift for chunks
        self.temporal = LSTMTemporalEncoder(embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, N, D = x.shape
        x = x.view(B, N, -1)  # (B, N, embed_dim)

        # temporal shift
        x = x.permute(0, 2, 1)  # (B, embed_dim, N)
        x = self.temporal_shift(x)
        x = x.permute(0, 2, 1)  # (B, N, embed_dim)

        # Apply LSTM for temporal modeling
        x = self.temporal(x)  # (B, embed_dim)
        x = self.classifier(x)  # (B, num_classes)
        
        return x


class AudioModelBaseline(nn.Module):
    def __init__(self, input_len=19456, num_classes=50, embed_dim=512, hidden_dim=256, num_layers=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=4, padding=4),   # T // 4
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=4, padding=4),  # T // 16
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2), # T // 32
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):  # x: (B, T)
        x = normalize_chunk(x)            # (B, T)
        x = x.unsqueeze(1)                # (B, 1, T)
        x = self.conv(x)                  # (B, 256, T')
        x = x.permute(0, 2, 1)            # (B, T', 256)

        x, _ = self.lstm(x)               # (B, T', 2H)
        x = x.mean(dim=1)                 # mean-pooling over time: (B, 2H)
        x = self.fc(x)                    # (B, num_classes)

        return x


class AudioModelBaselineEncoder(nn.Module):
    def __init__(self, input_len=19456, num_classes=50, embed_dim=512, hidden_dim=256, num_layers=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=4, padding=4),   # T // 4
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=4, padding=4),  # T // 16
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2), # T // 32
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True)


    def forward(self, x):  # x: (B, T)
        x = normalize_chunk(x)            # (B, T)
        x = x.unsqueeze(1)                # (B, 1, T)
        x = self.conv(x)                  # (B, 256, T')
        x = x.permute(0, 2, 1)            # (B, T', 256)

        x, _ = self.lstm(x)               # (B, T', 2H)
        x = x.mean(dim=1)                 # mean-pooling over time: (B, 2H)

        return x


if __name__ == '__main__':
    from thop import profile
    x = torch.randn(1, 16000 * 2)  # 2s audio
    # for size in ['small', 'medium', 'large']:
    #     model = AudioModel(size=size)
    #     out = model(x)
    #     flops, params = profile(model, inputs=(x,))
    #     print(f"{size} model: output={out.shape}, FLOPs={flops/1e6:.2f}M, Params={params/1e6:.2f}M")
    model = FullSequenceLSTMAudioModel()
    out = model(x)
    flops, params = profile(model, inputs=(x,))
    print(f"model: output={out.shape}, FLOPs={flops/1e6:.2f}M, Params={params/1e6:.2f}M")
