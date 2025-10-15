import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ResNet_2D import resnet50


class SpatialEncoder(nn.Module):
    def __init__(self, feature_dim=1000, pretrain=True):
        super(SpatialEncoder, self).__init__()
        self.base_model = resnet50(pretrained=pretrain)
        self.feature_dim = feature_dim
        
    def forward(self, x):
        features = self.base_model(x)
        return features  # (B, feature_dim)

class TemporalEncoder(nn.Module):
    def __init__(self, feature_dim=1000):
        super(TemporalEncoder, self).__init__()
        reduced_dim = feature_dim // 2
        
        self.temporal_fusion = nn.Sequential(
            nn.Conv1d(feature_dim, reduced_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(reduced_dim),
            nn.ReLU(inplace=True),
        )
        
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(reduced_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
    def forward(self, x):
        x = self.temporal_fusion(x.permute(0, 2, 1))
        x = self.temporal_encoder(x)
        pooled_feats = x.squeeze(-1)
        return pooled_feats


class SlowFast_N_Branch(nn.Module):
    def __init__(self, branches=[1, 2, 4, 8], feature_dim=1000, num_classes=50, pretrain=True):
        super(SlowFast_N_Branch, self).__init__()
        self.train_branches = sorted(branches)  # Branches used during training
        self.spatial_encoder = SpatialEncoder(feature_dim, pretrain)
        
        # Each branch has its own temporal encoder
        self.temporal_encoders = nn.ModuleList([
            TemporalEncoder(feature_dim) for _ in branches
        ])
        
        # Global Temporal Encoder after combining branches
        self.global_temporal_encoder = TemporalEncoder(512)
        
        self.final_fc = nn.Linear(512, num_classes)
    
    def forward(self, x, test_branches=None):
        B, T, C, H, W = x.shape
        
        # Use specified test branches or default to training branches
        branches = self.train_branches if test_branches is None else sorted(test_branches)
        temporal_outputs = []
        
        for i, alpha in enumerate(branches):
            indices = torch.arange(0, T, alpha).long().to(x.device)
            selected_x = x[:, indices, :, :, :]
            selected_B, selected_T, C, H, W = selected_x.shape
            
            spatial_feats = self.spatial_encoder(selected_x.view(selected_B * selected_T, C, H, W))
            spatial_feats = spatial_feats.view(selected_B, selected_T, -1)
            
            # Each branch has its own temporal encoder
            temporal_feats = self.temporal_encoders[i](spatial_feats)
            temporal_outputs.append(temporal_feats)
        
        # Concatenate features from all branches
        combined_feats = torch.stack(temporal_outputs, dim=1)

        # Apply global temporal encoder
        global_feats = self.global_temporal_encoder(combined_feats).squeeze(-1)
        
        output = self.final_fc(global_feats)
        return output


class Branch_Video_Encoder(nn.Module):
    def __init__(self, branches=[1, 2, 4, 8], feature_dim=1000, num_classes=50, pretrain=True):
        super(Branch_Video_Encoder, self).__init__()
        self.train_branches = sorted(branches)  # Branches used during training
        self.spatial_encoder = SpatialEncoder(feature_dim, pretrain)
        
        # Each branch has its own temporal encoder
        self.temporal_encoders = nn.ModuleList([
            TemporalEncoder(feature_dim) for _ in branches
        ])
        
        # Global Temporal Encoder after combining branches
        self.global_temporal_encoder = TemporalEncoder(512)
        
            
    def forward(self, x, test_branches=None):
        B, T, C, H, W = x.shape
        
        # Use specified test branches or default to training branches
        branches = self.train_branches if test_branches is None else sorted(test_branches)
        temporal_outputs = []
        
        for i, alpha in enumerate(branches):
            indices = torch.arange(0, T, alpha).long().to(x.device)
            selected_x = x[:, indices, :, :, :]
            selected_B, selected_T, C, H, W = selected_x.shape
            
            spatial_feats = self.spatial_encoder(selected_x.view(selected_B * selected_T, C, H, W))
            spatial_feats = spatial_feats.view(selected_B, selected_T, -1)
            
            # Each branch has its own temporal encoder
            temporal_feats = self.temporal_encoders[i](spatial_feats)
            temporal_outputs.append(temporal_feats)
        
        # Concatenate features from all branches
        combined_feats = torch.stack(temporal_outputs, dim=1)
        
        # Apply global temporal encoder
        global_feats = self.global_temporal_encoder(combined_feats).squeeze(-1)
        
        return global_feats



class FrameModel(nn.Module):
    def __init__(self, feature_dim=1000, pretrain=True):
        super(FrameModel, self).__init__()
        self.spatial_encoder = SpatialEncoder(feature_dim, pretrain)
        # Global Temporal Encoder after combining branches
    def forward(self, x):
        x = self.spatial_encoder(x)

        return x


class SampleModel(nn.Module):
    def __init__(self, feature_dim=1000):
        super(SampleModel, self).__init__()
        # Global Temporal Encoder after combining branches
        self.temporal_encoder = TemporalEncoder(feature_dim)
        self.global_temporal_encoder = TemporalEncoder(512)
        
            
    def forward(self, features, branches=[1, 2, 4, 8]):
        global_temporal_feature = []
        for i in range(len(features)):
            x = features[i]
            x = self.temporal_encoder(x)
            x = x.unsqueeze(1)
            global_temporal_feature.append(x)
        global_temporal_feature = torch.stack(global_temporal_feature, dim=1)
        # Apply global temporal encoder
        x = self.global_temporal_encoder(x).squeeze(-1)
        
        return x