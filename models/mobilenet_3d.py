import torch
import torch.nn as nn


class Bottleneck3D(nn.Module):
    """3D ResNet Bottleneck Block"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(Bottleneck3D, self).__init__()
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_channels)

        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Downsample shortcut
        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
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


class MobileNet3D(nn.Module):
    """基于 ResNet3D 设计的深层 3D 卷积网络"""
    def __init__(self, num_classes=101):
        super(MobileNet3D, self).__init__()
        self.in_channels = 64

        # 3D 输入层
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1,2,2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)  # 降采样，保持时间信息

        # 3D ResNet Block
        self.layer1 = self._make_layer(64, 128, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, 512, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, 1024, blocks=2, stride=2)

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 分类层
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(Bottleneck3D(in_channels, out_channels, stride=stride, downsample=True))
        for _ in range(1, blocks):
            layers.append(Bottleneck3D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # 初始卷积
        x = self.maxpool(x)  # 3D 池化

        x = self.layer1(x)  # 64 -> 128
        x = self.layer2(x)  # 128 -> 256
        x = self.layer3(x)  # 256 -> 512
        x = self.layer4(x)  # 512 -> 1024

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) 
        x = self.fc(x)

        return x




class TemporalAttention(nn.Module):
    """ 时间注意力机制 """
    def __init__(self, in_channels):
        super(TemporalAttention, self).__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入: [B, C, T, H, W]
        attention = self.conv(x)  # [B, 1, T, H, W]
        attention = self.sigmoid(attention)
        return x * attention  # 加权特征

class MultiScaleBlock(nn.Module):
    """ 多尺度特征提取块 """
    def __init__(self, in_planes, out_planes, stride=1, expansion=2):
        super(MultiScaleBlock, self).__init__()
        mid_planes = in_planes * expansion

        # 分支1: 常规3D卷积
        self.conv1 = nn.Conv3d(in_planes, mid_planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(mid_planes)

        # 分支2: 时间维度分离卷积
        self.conv2 = nn.Conv3d(in_planes, mid_planes, kernel_size=(3, 1, 1), stride=(1, stride, stride), padding=(1, 0, 0))
        self.bn2 = nn.BatchNorm3d(mid_planes)

        # 合并分支
        self.conv3 = nn.Conv3d(mid_planes * 2, out_planes, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        # 下采样
        self.downsample = None
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=(1, stride, stride)),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):
        identity = x

        # 分支1
        out1 = self.relu(self.bn1(self.conv1(x)))
        # 分支2
        out2 = self.relu(self.bn2(self.conv2(x)))
        # 合并
        out = torch.cat([out1, out2], dim=1)
        out = self.bn3(self.conv3(out))

        # 下采样
        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)

class X3D_MultiScale_TemporalAttention(nn.Module):
    """ 多尺度 + 时间注意力的 X3D 模型 """
    def __init__(self, num_classes=101):
        super(X3D_MultiScale_TemporalAttention, self).__init__()

        # 输入层
        self.conv1 = nn.Conv3d(3, 128, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # 多尺度特征提取层
        self.layer1 = self._make_layer(128, 256, blocks=2, stride=1)
        self.layer2 = self._make_layer(256, 512, blocks=2, stride=2)
        self.layer3 = self._make_layer(512, 1024, blocks=2, stride=2)

        # 时间注意力机制
        self.temporal_attn = TemporalAttention(1024)

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 分类层
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, in_planes, out_planes, blocks, stride):
        layers = []
        layers.append(MultiScaleBlock(in_planes, out_planes, stride=stride))
        for _ in range(1, blocks):
            layers.append(MultiScaleBlock(out_planes, out_planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)  # 128 -> 256
        x = self.layer2(x)  # 256 -> 512
        x = self.layer3(x)  # 512 -> 1024

        # 时间注意力
        x = self.temporal_attn(x)

        # 全局池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from thop import profile

    model = X3D_MultiScale_TemporalAttention(num_classes=101)
    input = torch.randn(1, 3, 32, 224, 224)  # 模拟输入
    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 输出 GFLOP
    print(f"Parameters: {params / 1e6:.2f} M")  # 输出参数量（单位：百万）