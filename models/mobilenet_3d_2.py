import torch
import torch.nn as nn


class DepthwiseConv3D(nn.Module):
    """ 3D 深度可分离卷积 """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseConv3D, self).__init__()
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class Light3DResNet(nn.Module):
    """ 轻量级 3D ResNet 结构 """
    def __init__(self, num_classes=101):
        super(Light3DResNet, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)

        # 3D 轻量级特征提取
        self.layer1 = DepthwiseConv3D(64, 128, stride=1)
        self.layer2 = DepthwiseConv3D(128, 256, stride=2)
        self.layer3 = DepthwiseConv3D(256, 512, stride=2)
        self.layer4 = DepthwiseConv3D(512, 1024, stride=2)

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 分类层
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)  # 64 -> 128
        x = self.layer2(x)  # 128 -> 256
        x = self.layer3(x)  # 256 -> 512
        x = self.layer4(x)  # 512 -> 1024

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

