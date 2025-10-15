import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet34, resnet18


def get_resnet_backbone(name='resnet50', pretrained=True):
    model_fn = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}
    base = model_fn[name](pretrained=pretrained)
    feature_dim = 512 if name in ['resnet18', 'resnet34'] else 2048
    return base, feature_dim


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


class Spatial_Encoder(nn.Module):
    def __init__(self, model, feature_dim=2048):
        super().__init__()
        self.base_model = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(feature_dim, 512)
    
    def forward(self, x):
        x = self.base_model(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x


class Video_ResNet_P3D(nn.Module):
    def __init__(self, model, feature_dim=2048, num_classes=50):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_encoder = Spatial_Encoder(model, feature_dim)  # (B*T, 2048, 1, 1)

        self.temporal_shift = MultiScaleTemporalShift(n_div=4)

        self.temporal_conv = nn.Sequential(
            DualTemporalConv(in_channels=512, out_channels=512),
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
        x = x.reshape(B * T, C, H, W)
        feat = self.spatial_encoder(x) # (B*T, 2048)
        
        feat = feat.view(B, T, 512).permute(0, 2, 1)  # (B, C, T)

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



class Video_ResNet_P3D_Encoder(nn.Module):
    def __init__(self, model, feature_dim=2048, num_classes=50):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_encoder = Spatial_Encoder(model, feature_dim)  # (B*T, 2048, 1, 1)

        self.temporal_shift = MultiScaleTemporalShift(n_div=4)

        self.temporal_conv = nn.Sequential(
            DualTemporalConv(in_channels=512, out_channels=512),
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
        x = x.reshape(B * T, C, H, W)
        feat = self.spatial_encoder(x) # (B*T, 2048)
        
        feat = feat.view(B, T, 512).permute(0, 2, 1)  # (B, C, T)

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
    

class Video_ResNet_P3D_Temporal_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_shift = MultiScaleTemporalShift(n_div=4)

        self.temporal_conv = nn.Sequential(
            DualTemporalConv(in_channels=512, out_channels=512),
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

    def forward(self, x):
        B, T, C = x.shape

        x = self.temporal_shift(x)
        x = self.temporal_conv(x)
        feat = self.channel_attn(x)

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
        

if __name__ == '__main__':
    backbone, feat_dim = get_resnet_backbone('resnet18', pretrained=True)
    model = Video_ResNet_P3D(backbone, feature_dim=feat_dim)