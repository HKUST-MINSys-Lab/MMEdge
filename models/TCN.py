import torch
import torch.nn as nn


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
        x = self.global_pool(x)  # (B, 1024, 1, 1)
        return x.view(x.size(0), -1)  # 变为 (B, 1024)


class TemporalConvNet(nn.Module):
    """TCN 时间建模"""
    def __init__(self, input_size=1024, num_channels=[512, 512, 256], kernel_size=3):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i  # 2^0, 2^1, 2^2...
            layers.append(nn.Conv1d(input_size if i == 0 else num_channels[i-1],
                                    num_channels[i],
                                    kernel_size,
                                    stride=1,
                                    padding=dilation,
                                    dilation=dilation))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MobileNet2D_TCN(nn.Module):
    """ 2D CNN + TCN """
    def __init__(self, num_classes=101):
        super(MobileNet2D_TCN, self).__init__()

        self.feature_extractor = SpatialFeatureExtractor(out_channels=1024)
        self.temporal_model = TemporalConvNet(input_size=1024, num_channels=[512, 512, 256])
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        spatial_features = [self.feature_extractor(x[:, t, :, :, :]) for t in range(T)]
        spatial_features = torch.stack(spatial_features, dim=1)  # (B, T, 1024)

        temporal_out = spatial_features.permute(0, 2, 1)  # (B, 1024, T)
        temporal_out = self.temporal_model(temporal_out)  # TCN 时间建模
        temporal_out = torch.mean(temporal_out, dim=2)  # (B, 256)

        logits = self.fc(temporal_out)  # (B, num_classes)
        return logits
