import torch
import torch.nn as nn
import torch.nn.functional as F


class FastImageyAnalyzer(nn.Module):
    def __init__(self, input_shape=(1, 88, 88)):  # 输入可以改
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, x):  # x: (B, 1, H, W)
        return self.encoder(x).flatten(1)  # (B, 32)


class FastAudioAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: (B, T)
        x = x.unsqueeze(1)          # (B, 1, T)
        x = self.conv(x)            # (B, 256, T')
        x = self.pool(x)            # (B, 256, 1)
        return x.squeeze(-1)        # (B, 256)



class FastModalityTester(nn.Module):
    def __init__(self, image_feat_dim=32, audio_feat_dim=256, hidden=64, temperature=1.0):
        super().__init__()
        self.image_encoder = FastImageyAnalyzer()   # 任意轻量 CNN
        self.audio_encoder = FastAudioAnalyzer()    # 上面定义的结构

        self.fc = nn.Sequential(
            nn.Linear(image_feat_dim + audio_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)  # 对 image/audio 各给出是否启用的概率
        )
        self.temperature = temperature

    def forward(self, image_raw, audio_raw):
        img_feat = self.image_encoder(image_raw)    # (B, image_feat_dim)
        aud_feat = self.audio_encoder(audio_raw)    # (B, audio_feat_dim)
        fused_feat = torch.cat([img_feat, aud_feat], dim=1)
        logits = self.fc(fused_feat)                # (B, 2)
        return F.gumbel_softmax(logits, tau=self.temperature, hard=True)
