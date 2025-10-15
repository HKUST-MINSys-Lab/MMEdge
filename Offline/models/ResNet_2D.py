import torch
import torch.nn as nn
import torch.nn.functional as F
# from .utils import load_state_dict_from_url
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

# import nets.PC_CAL as EF_zoo
from models.utils.Calibrator2D import GC_L33D, GC_T13D, GC_S23DD, GC_CLLD

__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    # 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    # 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    # 'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    # 'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth', 
}

# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_ef=False, cdiv=8, num_segments=8, loop_id=0):
        super(Bottleneck, self).__init__()
        self.use_ef = use_ef
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if self.use_ef:
            print('=> Using Partial Channel Calibrator with cdiv: {}'.format(cdiv))
            self.loop_id = loop_id
            self.eft_c = planes // cdiv
            self.eft1 = GC_L33D(self.eft_c, self.eft_c, num_segments)
            self.eft2 = GC_T13D(self.eft_c, self.eft_c, num_segments)
            self.eft3 = GC_S23DD(self.eft_c, self.eft_c, num_segments)
            self.eft4 = GC_CLLD(self.eft_c, self.eft_c, num_segments)
            self.eft = (self.eft_c, self.eft_c, num_segments)
            self.start_c1 = loop_id*self.eft_c
            self.end_c1 = self.start_c1 + self.eft_c
            loop_id2 = (loop_id+1)%cdiv
            self.start_c2 = loop_id2*self.eft_c
            self.end_c2 = self.start_c2 + self.eft_c
            loop_id3 = (loop_id+2)%cdiv
            self.start_c3 = loop_id3*self.eft_c
            self.end_c3 = self.start_c3 + self.eft_c
            loop_id4 = (loop_id+3)%cdiv
            self.start_c4 = loop_id4*self.eft_c
            self.end_c4 = self.start_c4 + self.eft_c
            print('loop_ids: [{}:({}-{}), {}:({}-{}), {}:({}-{}), {}:({}-{})]'.format(loop_id, self.start_c1, self.end_c1, \
                loop_id2, self.start_c2, self.end_c2, loop_id3, self.start_c3, self.end_c3, loop_id4, self.start_c4, self.end_c4))
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        # x = [bcz*n_seg, c, h, w]
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        #
        if self.use_ef:
            new_out = torch.zeros_like(out)
            BN, C_size, H_size, W_size = new_out.size()
            # new_out = out
            new_out[:, self.start_c1:self.end_c1, :, :] = self.eft1(out[:, self.start_c1:self.end_c1, :, :])
            new_out[:, self.start_c2:self.end_c2, :, :] = self.eft2(out[:, self.start_c2:self.end_c2, :, :])
            new_out[:, self.start_c3:self.end_c3, :, :] = self.eft3(out[:, self.start_c3:self.end_c3, :, :])
            new_out[:, self.start_c4:self.end_c4, :, :] = self.eft4(out[:, self.start_c4:self.end_c4, :, :])
            # new_out = torch.zeros_like(out)
            # new_out[:, :self.eft_c, :, :] = self.eft(out[:, :self.eft_c, :, :])
            if self.end_c4 > self.start_c1:
                if self.start_c1 > 0:
                    new_out[:, :self.start_c1:, :, :] = out[:, :self.start_c1:, :, :]
                if self.end_c4 < C_size:
                    new_out[:, self.end_c4:, :, :] = out[:, self.end_c4:, :, :]
            elif self.end_c4 < self.start_c1:
                new_out[:, self.end_c4:self.start_c1:, :, :] = out[:, self.end_c4:self.start_c1:, :, :]

            out = new_out
            # out[:, self.eft_c:, :, :] = out[:, self.eft_c:, :, :]
        #
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out


class ResNet(nn.Module):
 
    def __init__(self, block, layers, num_classes=1000, cdiv=2, num_segments=8, loop=False):
        self.inplanes = 64
        self.loop_id = 0
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.loop = loop
        self.layer1 = self._make_layer(block, 64, layers[0], cdiv=cdiv, n_seg=num_segments)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, cdiv=cdiv, n_seg=num_segments)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, cdiv=cdiv, n_seg=num_segments)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, cdiv=cdiv, n_seg=num_segments)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for name, m in self.named_modules():
            if 'eft' not in name:
                # if 'deconv' in name:
                #     nn.init.xavier_normal_(m.weight)
                # else:
                #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
 
    def _make_layer(self, block, planes, blocks, stride=1, cdiv=2, n_seg=8):
        print('=> Processing stage with {} blocks'.format(blocks))
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, False, cdiv=cdiv, num_segments=n_seg, loop_id=self.loop_id))
        self.inplanes = planes * block.expansion
        if self.loop:
            self.loop_id = (self.loop_id+1)%cdiv

        #
        n_round = 1
        if blocks >= 23:
            n_round = 2
            print('=> Using n_round {} to insert Element Filter -T'.format(n_round))
        #
        for i in range(1, blocks):
            if i % n_round == 0:
                use_ef = False
                layers.append(block(self.inplanes, planes, use_ef=use_ef, cdiv=cdiv, num_segments=n_seg, loop_id=self.loop_id))
                if self.loop:
                    self.loop_id = (self.loop_id+1)%cdiv
            else:
                use_ef = False
                layers.append(block(self.inplanes, planes, use_ef=use_ef, cdiv=cdiv, num_segments=n_seg))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
 
        return x
    
    
class Video_ResNet_2D(nn.Module):
    def __init__(self, pretrain=True, feature_dim=1000, num_class=50):
        super(Video_ResNet_2D, self).__init__()
        self.base_model = resnet50(pretrain)
        # 用一维卷积提取时序信息
        self.conv_temporal = nn.Sequential(
            nn.Conv1d(feature_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.AdaptiveAvgPool1d(1)  # 将时序维度降为1
        )

    def forward(self, x):
        # x 形状: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        spatial_feat = self.base_model(x)  # (B*T, feature_dim)
        temporal_feat = spatial_feat.view(B, T, -1)  # (B, T, feature_dim)
        # 转置为 (B, feature_dim, T) 以便做 1D 卷积
        temporal_feat = temporal_feat.permute(0, 2, 1)
        conv_out = self.conv_temporal(temporal_feat)  # (B, 128, 1)
        conv_out = conv_out.squeeze(2)  # (B, 128)
        
        return conv_out


class Video_Classification(nn.Module):
    def __init__(self, pretrain=True, feature_dim=1000, num_classes=50):
        super(Video_Classification, self).__init__()
        self.base_model = resnet50(pretrain)
        # 用一维卷积提取时序信息
        self.conv_temporal = nn.Sequential(
            nn.Conv1d(feature_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.AdaptiveAvgPool1d(1),  # 将时序维度降为1
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x 形状: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        spatial_feat = self.base_model(x)  # (B*T, feature_dim)
        temporal_feat = spatial_feat.view(B, T, -1)  # (B, T, feature_dim)
        # 转置为 (B, feature_dim, T) 以便做 1D 卷积
        temporal_feat = temporal_feat.permute(0, 2, 1)
        conv_out = self.conv_temporal(temporal_feat)  # (B, 128, 1)
        conv_out = conv_out.squeeze(2)  # (B, 128)
        output = self.fc(conv_out)
        return output


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)  # BasicBlock + layer config [3,4,6,3]
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet34'])
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


# ResNet-18 实现（更小）
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [2, 2, 2, 2], **kwargs)  # BasicBlock + layer config [2,2,2,2]
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # EF_name = getattr(EF_zoo, EF)
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet50'])
        # checkpoint_keys = list(checkpoint.keys())
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)

    return model
 
 
def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # EF_name = getattr(EF_zoo, EF)
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet101'])
        # checkpoint_keys = list(checkpoint.keys())
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model
 
 
def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet152'])
        # checkpoint_keys = list(checkpoint.keys())
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


class MultiScaleTemporalShift(nn.Module):
    def __init__(self, n_div=4):
        super().__init__()
        self.n_div = n_div

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        fold = C // self.n_div
        out = x.clone()

        if fold == 0:
            return x

        out[:, :fold, 1:] = x[:, :fold, :-1]
        out[:, fold:2 * fold, :-1] = x[:, fold:2 * fold, 1:]

        if T > 2:
            out[:, 2 * fold:3 * fold, 2:] = x[:, 2 * fold:3 * fold, :-2]
            out[:, 3 * fold:, :-2] = x[:, 3 * fold:, 2:]

        return out


class TemporalAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, in_dim // 8, 1),
            nn.ReLU(),
            nn.Conv1d(in_dim // 8, 1, 1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):  # x: (B, C, T)
        weights = self.attn(x)  # (B, 1, T)
        attended = (x * weights).sum(dim=2)  # (B, C)
        return attended


class ChannelSE(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        y = self.pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1)
        return x * y


class DualTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.short = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.long = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.relu = nn.ReLU(inplace=True)
        self.use_proj = in_channels != out_channels
        if self.use_proj:
            self.proj = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):  # x: (B, C, T)
        residual = x if not self.use_proj else self.proj(x)
        out = self.short(x) + self.long(x)
        return self.relu(out + residual)


class Video_ResNet_P3D(nn.Module):
    def __init__(self, model, feature_dim=2048, num_classes=50):
        super().__init__()
        self.feature_dim = feature_dim
        self.base_model = nn.Sequential(*list(model.children())[:-1])  # (B*T, 2048, 1, 1)

        self.temporal_shift = MultiScaleTemporalShift(n_div=4)

        self.temporal_conv = nn.Sequential(
            DualTemporalConv(in_channels=feature_dim, out_channels=512),
            DualTemporalConv(in_channels=512, out_channels=512)
        )

        self.channel_attn = ChannelSE(512)
        self.temporal_attn = TemporalAttention(512)

        self.motion_gate = nn.Linear(512, 512)  # 因为 motion_feat 是 concat 后的维度

        self.bottleneck = nn.Sequential(  # 类似于 transformer FFN
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.base_model(x).squeeze(-1).squeeze(-1)  # (B*T, 2048)

        feat = feat.view(B, T, self.feature_dim).permute(0, 2, 1)  # (B, C, T)

        feat = self.temporal_shift(feat)
        feat = self.temporal_conv(feat)
        feat = self.channel_attn(feat)

        pooled = self.temporal_attn(feat)  # (B, 512)

        # motion branch: 多尺度差分特征
        if T > 2:
            diff1 = feat[:, :, 1:] - feat[:, :, :-1]
            diff2 = feat[:, :, 2:] - feat[:, :, :-2]
            motion_feat = (
                F.adaptive_avg_pool1d(diff1, 1) +
                F.adaptive_avg_pool1d(diff2, 1)
            ).squeeze(-1)  # (B, 512)

            gate = torch.sigmoid(self.motion_gate(pooled))  # (B, 1024)
            pooled = pooled + gate * motion_feat  # motion gating 融合

        pooled = self.bottleneck(pooled)  # 增强表达能力

        logits = self.classifier(pooled)
        return logits



class Video_ResNet_P3D_Encoder(nn.Module):
    def __init__(self, pretrain=True, feature_dim=2048, num_classes=50):
        super().__init__()
        self.feature_dim = feature_dim
        resnet = resnet50(pretrained=pretrain)
        self.base_model = nn.Sequential(*list(resnet.children())[:-1])  # (B*T, 2048, 1, 1)

        self.temporal_shift = MultiScaleTemporalShift(n_div=4)

        self.temporal_conv = nn.Sequential(
            DualTemporalConv(in_channels=feature_dim, out_channels=512),
            DualTemporalConv(in_channels=512, out_channels=512)
        )

        self.channel_attn = ChannelSE(512)
        self.temporal_attn = TemporalAttention(512)

        self.motion_gate = nn.Linear(512, 512)  # 因为 motion_feat 是 concat 后的维度

        self.bottleneck = nn.Sequential(  # 类似于 transformer FFN
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.base_model(x).squeeze(-1).squeeze(-1)  # (B*T, 2048)

        feat = feat.view(B, T, self.feature_dim).permute(0, 2, 1)  # (B, C, T)

        feat = self.temporal_shift(feat)
        feat = self.temporal_conv(feat)
        feat = self.channel_attn(feat)

        pooled = self.temporal_attn(feat)  # (B, 512)

        # motion branch: 多尺度差分特征
        if T > 2:
            diff1 = feat[:, :, 1:] - feat[:, :, :-1]
            diff2 = feat[:, :, 2:] - feat[:, :, :-2]
            motion_feat = (
                F.adaptive_avg_pool1d(diff1, 1) +
                F.adaptive_avg_pool1d(diff2, 1)
            ).squeeze(-1)  # (B, 512)

            gate = torch.sigmoid(self.motion_gate(pooled))  # (B, 1024)
            pooled = pooled + gate * motion_feat  # motion gating 融合

        pooled = self.bottleneck(pooled)  # 增强表达能力

        return pooled

"""

Baseline Model

"""


class SimpleTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # depthwise
            nn.Conv1d(in_channels, out_channels, kernel_size=1),  # pointwise
            nn.ReLU(inplace=True)
        )

    def forward(self, x):  # x: (B, C, T)
        return self.conv(x)


class Video_ResNet_Baseline(nn.Module):
    def __init__(self, pretrain=True, feature_dim=2048, num_classes=50):
        super().__init__()
        self.feature_dim = feature_dim
        resnet = resnet50(pretrained=pretrain)
        self.base_model = nn.Sequential(*list(resnet.children())[:-1])  # (B*T, 2048, 1, 1)

        self.temporal_conv = SimpleTemporalConv(in_channels=feature_dim, out_channels=512)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.base_model(x).squeeze(-1).squeeze(-1)  # (B*T, 2048)

        feat = feat.view(B, T, self.feature_dim).permute(0, 2, 1)  # (B, C, T)

        feat = self.temporal_conv(feat)  # (B, 512, T)
        pooled = feat.mean(dim=2)        # mean pooling over time

        logits = self.classifier(pooled)  # (B, num_classes)
        return logits



"""

Only Multi Scale Temporal Shift

"""


class MultiScaleTSM(nn.Module):
    def __init__(self, n_div=4, scales=(1, 2, 3)):
        super().__init__()
        self.n_div = n_div
        self.scales = scales

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        fold = C // self.n_div
        out = x.clone()

        if fold == 0 or T <= max(self.scales):
            return x

        for i, s in enumerate(self.scales):
            ch_start = i * fold
            ch_end = (i + 1) * fold

            # forward shift
            if s < T:
                out[:, ch_start:ch_end, :-s] = x[:, ch_start:ch_end, s:]
            # backward shift
            if s < T:
                out[:, ch_end:ch_end + fold, s:] = x[:, ch_end:ch_end + fold, :-s]

        return out


class Video_ResNet_TSM(nn.Module):
    def __init__(self, model=None, feature_dim=2048, num_classes=50):
        super().__init__()
        self.feature_dim = feature_dim
        if model is None:
            model = resnet50(pretrained=True)
        self.base_model = nn.Sequential(*list(model.children())[:-1])  # (B*T, 2048, 1, 1)

        self.temporal_shift = MultiScaleTSM(n_div=4, scales=(1, 2, 3))  # 多尺度 shift

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(feature_dim, 512, kernel_size=1),  # 压缩通道
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.base_model(x).squeeze(-1).squeeze(-1)  # (B*T, 2048)

        feat = feat.view(B, T, self.feature_dim).permute(0, 2, 1)  # (B, C, T)

        feat = self.temporal_shift(feat)       # 多尺度 TSM
        feat = self.temporal_conv(feat)        # 压缩维度 (B, 512, T)
        pooled = feat.mean(dim=2)              # Temporal pooling

        logits = self.classifier(pooled)       # 分类输出
        return logits
