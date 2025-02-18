import torch
import torch.nn as nn
import torchvision.models as models


class SpatialFeatureExtractor(nn.Module):
    """ 改进的 2D CNN：增加 BatchNorm 和 Dropout """
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
            nn.Dropout(0.3)  # 适当加入 Dropout
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

        # 仍然使用全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class TemporalInteractionModule(nn.Module):
    """时间建模模块 (TIM) - 1D 时间卷积"""
    def __init__(self, channels, kernel_size=3):
        super(TemporalInteractionModule, self).__init__()
        self.temporal_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels  # 使用 Depthwise 卷积降低计算量
        )

    def forward(self, x):
        return self.temporal_conv(x)


class TEINet_Light(nn.Module):
    """轻量级 TEINet 适用于端到端训练"""
    def __init__(self, num_classes=101):
        super(TEINet_Light, self).__init__()
        self.in_channels = 64

        # 2D CNN Backbone 
        self.feature_extractor = SpatialFeatureExtractor(in_channels=3, out_channels=1024)

        # 时间建模
        self.tim = TemporalInteractionModule(1024, kernel_size=3)

        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类层
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        # **提取空间特征**
        spatial_features = []
        for t in range(T):
            feat = self.feature_extractor(x[:, t, :, :, :])  # (B, 512, H/32, W/32)
            feat = self.global_avg_pool(feat)  # 归一化
            feat = feat.view(feat.size(0), -1)  # (B, 512)
            spatial_features.append(feat)

        spatial_features = torch.stack(spatial_features, dim=1)  # (B, T, 512)

        # **时间建模**
        temporal_out = spatial_features.permute(0, 2, 1)  # (B, 512, T)
        temporal_out = self.tim(temporal_out)  # (B, 512, T)
        temporal_out = temporal_out.mean(dim=-1)  # (B, 512)
        return self.fc(temporal_out)  # (B, num_classes)

