import torch
import torch.nn as nn

class SpatialFeatureExtractor(nn.Module):
    """ 2D CNN：提取空间特征 """
    def __init__(self, in_channels=3, out_channels=1024):
        super(SpatialFeatureExtractor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)  # (B, 1024)


class MobileNet2D_FeatureMemory(nn.Module):
    """ 2D CNN + 递归输入 """
    def __init__(self, num_classes=101, alpha=0.3, use_lstm=True):
        super(MobileNet2D_FeatureMemory, self).__init__()
        self.use_lstm = use_lstm
        self.alpha = alpha  # 图像混合比例

        # **2D CNN 提取空间特征**
        self.feature_extractor = SpatialFeatureExtractor(out_channels=1024)

        # **时间建模**
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        if self.use_lstm:
            self.temporal_lstm = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)
            self.temporal_fc = nn.Linear(512 * 2, num_classes)
        else:
            self.temporal_fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        # **初始化累积图像**
        memory = None
        spatial_features = []

        for t in range(T):
            frame = x[:, t, :, :, :]  # 当前帧

            # **图像混合：保持部分前一帧信息**
            if memory is None:
                memory = frame  # 第一帧不混合
            else:
                memory = self.alpha * memory + (1 - self.alpha) * frame  # 图像级递归融合

            # **提取特征**
            frame_feature = self.feature_extractor(memory)
            spatial_features.append(frame_feature)

        spatial_features = torch.stack(spatial_features, dim=1)  # (B, T, 1024)

        # **时序建模**
        temporal_out = spatial_features.permute(0, 2, 1)  # (B, 1024, T)
        temporal_out = self.temporal_conv(temporal_out)  # 1D Conv (B, 256, T)

        if self.use_lstm:
            temporal_out = temporal_out.permute(0, 2, 1)  # (B, T, 256)
            temporal_out, _ = self.temporal_lstm(temporal_out)  # LSTM (B, T, 512*2)
            temporal_out = temporal_out[:, -1, :]  # 取最后一个时间步

        logits = self.temporal_fc(temporal_out)  # (B, num_classes)
        return logits
