import torch
import torch.nn as nn
import torchvision.models as models


class Bottleneck2D(nn.Module):
    """2D CNN 版的 Bottleneck 结构"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(Bottleneck2D, self).__init__()
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Downsample shortcut
        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)


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

        # 2D CNN Backbone (ResNet18)
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # 移除全连接层

        # 时间建模
        self.tim = TemporalInteractionModule(512, kernel_size=3)

        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类层
        self.fc = nn.Linear(512, num_classes)

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




class TEINet_LSTMTemporal(nn.Module):
    """时间建模：LSTM"""
    def __init__(self, num_classes=101, hidden_size=512, num_layers=1):
        super(TEINet_LSTMTemporal, self).__init__()

        # 2D CNN Backbone (ResNet18)
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # 移除全连接层

        # LSTM 时间建模
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)

        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类层
        self.fc = nn.Linear(hidden_size, num_classes)

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

        # **时间建模（LSTM）**
        lstm_out, _ = self.lstm(spatial_features)  # (B, T, hidden_size)

        # 取最后一个时间步的输出
        temporal_out = lstm_out[:, -1, :]  # (B, hidden_size)

        return self.fc(temporal_out)  # (B, num_classes)


import timm


class TEINet_Light_MobileViT(nn.Module):
    """使用 MobileViT 替代 ResNet18 作为 Backbone，并减少过拟合"""
    def __init__(self, num_classes=101):
        super(TEINet_Light_MobileViT, self).__init__()

        # 加载 MobileViT 预训练模型
        mobilevit = timm.create_model("mobilevit_s", pretrained=True, features_only=True)  
        
        # 获取输出通道数
        feature_dim = mobilevit.feature_info.channels()[-1]  # 获取最后一层的通道数

        self.feature_extractor = mobilevit

        # 降维以减少计算量
        self.reduce_dim = nn.Conv1d(feature_dim, 256, kernel_size=1)  # 降维到 256

        # 时间建模 (TIM) - 增加 Dropout
        self.tim = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1, groups=256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类层
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        # **提取空间特征**
        spatial_features = []
        for t in range(T):
            feat = self.feature_extractor(x[:, t, :, :, :])[-1]  # 获取最后一层特征
            feat = self.global_avg_pool(feat)  # 全局池化，避免高维度
            feat = feat.view(feat.size(0), -1)  # (B, feature_dim)
            spatial_features.append(feat)

        spatial_features = torch.stack(spatial_features, dim=1)  # (B, T, feature_dim)

        # **降维并进行时间建模**
        spatial_features = self.reduce_dim(spatial_features.permute(0, 2, 1))  # (B, 256, T)
        spatial_features = self.tim(spatial_features)  # (B, 256, T)
        spatial_features = spatial_features.mean(dim=-1)  # (B, 256)

        return self.fc(spatial_features)  # (B, num_classes)


if __name__ == "__main__":
    from thop import profile

    model = TEINet_Light_MobileViT(num_classes=101)
    input = torch.randn(1, 3, 32, 224, 224)  # 模拟输入
    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 输出 GFLOP
    print(f"Parameters: {params / 1e6:.2f} M")  # 输出参数量（单位：百万）