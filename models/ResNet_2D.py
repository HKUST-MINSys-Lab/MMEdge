import torch
import torch.nn as nn
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


class SlowFast2D_Cache(nn.Module):
    def __init__(self, base_model, feature_dim, num_class, alpha=4):
        super(SlowFast2D_Cache, self).__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.alpha = alpha

        reduced_dim = feature_dim // 2

        # Temporal fusion for fast pathway
        self.fast_temporal_fusion = nn.Sequential(
            nn.Conv1d(feature_dim, reduced_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(reduced_dim),
            nn.ReLU(inplace=True),
        )

        # Temporal fusion for slow pathway
        self.slow_temporal_fusion = nn.Sequential(
            nn.Conv1d(feature_dim, reduced_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(reduced_dim),
            nn.ReLU(inplace=True),
        )

        # Fusion of slow and fast pathways (simplified)
        self.fusion_layer = nn.Sequential(
            nn.Linear(reduced_dim * 2, reduced_dim),
            nn.ReLU(inplace=True),
        )

        # Temporal encoder (reduced complexity)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(reduced_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # 将时序维度降为1
        )

        self.fc = nn.Linear(64, num_class)

        # Cache for real-time inference
        self.register_buffer('cache', torch.zeros(1, feature_dim))

    def forward(self, x, use_cache=False, reset_cache=False):
        if use_cache:
            # Real-time inference (single frame)
            B = x.size(0)
            current_feat = self.base_model(x)

            if reset_cache or self.cache.size(0) != B:
                device = current_feat.device
                self.cache = torch.zeros(B, self.feature_dim).to(device)

            combined_feat = torch.cat([current_feat, self.cache], dim=-1)
            fused_feat = self.fusion_layer(combined_feat).unsqueeze(-1)

            temporal_out = self.temporal_encoder(fused_feat)
            pooled_feats = temporal_out.squeeze(-1)

            output = self.fc(pooled_feats)
            self.cache = current_feat.detach()

        else:
            # Training mode (full clip)
            B, T, C, H, W = x.shape

            # Slow Path
            slow_indices = torch.arange(0, T, self.alpha).long().to(x.device)
            slow_x = x[:, slow_indices, :, :, :]
            slow_B, slow_T, C, H, W = slow_x.shape
            slow_feats = self.base_model(slow_x.view(slow_B * slow_T, C, H, W)).view(slow_B, slow_T, -1)
            slow_feats = self.slow_temporal_fusion(slow_feats.permute(0, 2, 1))
            slow_feats = nn.functional.interpolate(slow_feats, size=T, mode='linear')

            # Fast Path
            fast_feats = self.base_model(x.view(B * T, C, H, W)).view(B, T, -1)
            fast_feats = self.fast_temporal_fusion(fast_feats.permute(0, 2, 1))

            # Fuse Slow and Fast paths
            combined_feats = torch.cat([slow_feats, fast_feats], dim=1)
            fused_feats = self.fusion_layer(combined_feats.permute(0, 2, 1)).permute(0, 2, 1)

            # Temporal Encoding
            temporal_out = self.temporal_encoder(fused_feats)
            pooled_feats = temporal_out.squeeze(-1)

            output = self.fc(pooled_feats)

        return output



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
