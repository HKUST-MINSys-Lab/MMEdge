import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18FeatureExtractor(nn.Module):
    """ 预训练 ResNet18 作为 2D CNN 空间特征提取器 """
    def __init__(self, pretrained=True, out_channels=512):
        super(ResNet18FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # 去掉全连接层，保留卷积部分
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局池化

    def forward(self, x):
        x = self.feature_extractor(x)  # (B, 512, H', W')
        x = self.global_pool(x)  # (B, 512, 1, 1)
        return x.view(x.size(0), -1)  # (B, 512)


class MobileNet2D1D(nn.Module):
    """ **ResNet18 空间特征提取 + Conv1D/LSTM 进行时序建模** """
    def __init__(self, num_classes=101, use_lstm=True):
        super(MobileNet2D1D, self).__init__()
        self.use_lstm = use_lstm

        # **改用预训练 ResNet18 提取空间特征**
        self.feature_extractor = ResNet18FeatureExtractor(out_channels=512)

        # **时序建模（Conv1D + LSTM 结合）**
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        if self.use_lstm:
            self.temporal_lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
            self.temporal_fc = nn.Linear(256 * 2, num_classes)
        else:
            self.temporal_fc = nn.Linear(128, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        # **提取空间特征**
        spatial_features = [self.feature_extractor(x[:, t, :, :, :]) for t in range(T)]
        spatial_features = torch.stack(spatial_features, dim=1)  # (B, T, 512)

        # **时序建模**
        temporal_out = spatial_features.permute(0, 2, 1)  # (B, 512, T)
        temporal_out = self.temporal_conv(temporal_out)  # 1D Conv (B, 128, T)

        if self.use_lstm:
            temporal_out = temporal_out.permute(0, 2, 1)  # (B, T, 128)
            temporal_out, _ = self.temporal_lstm(temporal_out)  # LSTM (B, T, 256*2)
            temporal_out = temporal_out[:, -1, :]  # 取最后一个时间步

        logits = self.temporal_fc(temporal_out)  # (B, num_classes)
        return logits

