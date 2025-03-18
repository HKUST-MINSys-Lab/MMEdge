import torch
import torch.nn as nn
import torch.nn.functional as F


class BimodalFusion(nn.Module):
    def __init__(self, 
                 modal_dims=(2048, 1024),  # 两个模态的输入维度
                 hidden_dim=512,
                 num_classes=50):
        super().__init__()
        
        # 动态特征校准
        self.calibrate = nn.ModuleList([
            DynamicCalibration(modal_dims[0], hidden_dim),
            DynamicCalibration(modal_dims[1], hidden_dim)
        ])
        
        # 双向交叉注意力
        self.cross_attn = BidirectionalAttention(hidden_dim)
        
        # 特征合成器
        self.synthesizer = FeatureSynthesizer(hidden_dim)
        
        # 残差融合
        self.res_fusion = ResidualFusion(modal_dims, hidden_dim)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x1, x2):
        """
        输入: 
            x1: (B, D1)
            x2: (B, D2)
        输出: 
            fused: (B, hidden_dim)
        """
        # 特征校准
        h1 = self.calibrate[0](x1)  # (B, H)
        h2 = self.calibrate[1](x2)  # (B, H)
        
        # 交叉注意力交互
        attn1, attn2 = self.cross_attn(h1, h2)  # 各(B, H)
        
        # 特征合成
        fused = self.synthesizer(attn1 + attn2)  # (B, H)
        
        # 残差增强
        fused = fused + self.res_fusion(x1, x2)
        fused = fused.squeeze(1)

        logits = self.fc(fused)  # (B, num_classes)
        return logits


class DynamicCalibration(nn.Module):
    """动态特征校准器"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim*2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        projected = self.proj(x)
        gate = self.gate(x)
        return projected * gate


class BidirectionalAttention(nn.Module):
    """双向交叉注意力"""
    def __init__(self, dim):
        super().__init__()
        self.q1 = nn.Linear(dim, dim)
        self.k2 = nn.Linear(dim, dim)
        self.v2 = nn.Linear(dim, dim)
        
        self.q2 = nn.Linear(dim, dim)
        self.k1 = nn.Linear(dim, dim)
        self.v1 = nn.Linear(dim, dim)
        
    def forward(self, h1, h2):
        # h1 -> h2 注意力
        q1 = self.q1(h1).unsqueeze(1)  # (B,1,D)
        k2 = self.k2(h2).unsqueeze(2)   # (B,D,1)
        attn_weights = F.softmax(torch.bmm(q1, k2), dim=-1)  # (B,1,1)
        attn2 = attn_weights * self.v2(h2).unsqueeze(1)  # (B,1,D)
        
        # h2 -> h1 注意力
        q2 = self.q2(h2).unsqueeze(1)    # (B,1,D)
        k1 = self.k1(h1).unsqueeze(2)    # (B,D,1)
        attn_weights = F.softmax(torch.bmm(q2, k1), dim=-1)  # (B,1,1)
        attn1 = attn_weights * self.v1(h1).unsqueeze(1)  # (B,1,D)
        
        return attn1.squeeze(1), attn2.squeeze(1)


class FeatureSynthesizer(nn.Module):
    """特征合成模块"""
    def __init__(self, dim):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Linear(dim*4, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        return self.transform(x + x.sin())  # 引入周期性激活


class ResidualFusion(nn.Module):
    """残差增强模块"""
    def __init__(self, modal_dims, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(sum(modal_dims), hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x1, x2):
        return self.alpha * self.mlp(torch.cat([x1, x2], dim=-1))

# 使用示例
if __name__ == "__main__":
    modal1 = torch.randn(32, 2048)  # 模态1特征
    modal2 = torch.randn(32, 1024)  # 模态2特征
    
    model = BimodalFusion(modal_dims=(2048, 1024))
    fused_feature = model(modal1, modal2)
    
    print(fused_feature.shape)  # 输出 torch.Size([32, 512])