import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbed(nn.Module):
    """ 视频 Patch 分割 & 嵌入 """
    def __init__(self, img_size=224, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')  # 合并 batch & 时间维度
        x = self.proj(x)  # (B*T, embed_dim, H//patch, W//patch)
        x = x.flatten(2).transpose(1, 2)  # (B*T, num_patches, embed_dim)
        return x, T


class Mlp(nn.Module):
    """ MLP 前馈网络 """
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features * 2
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Attention(nn.Module):
    """ Self Attention (用于时空 Transformer) """
    def __init__(self, dim, num_heads=6, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, N, heads, C//heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    """ Transformer Block：时空自注意力 + MLP """
    def __init__(self, dim, num_heads, mlp_ratio=2., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniVideoTransformer(nn.Module):
    """ 小型 Video Transformer（适合端侧） """
    def __init__(self, img_size=224, patch_size=8, num_classes=101, num_frames=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=2.):
        super().__init__()

        # Patch 嵌入
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        # 可学习的 CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置编码
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B, C, T, H, W = x.shape

        # Patch Embed
        x, T = self.patch_embed(x)  # (B*T, num_patches, embed_dim)

        # 添加 CLS Token
        cls_tokens = self.cls_token.expand(B * T, -1, -1)  # (B*T, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B*T, num_patches+1, embed_dim)

        # 位置编码
        x = x + self.pos_embed
        x = x.view(B, T, -1, x.shape[-1])  # (B, T, num_patches+1, embed_dim)
        x = x.mean(dim=2)  # 在空间维度上取平均 (B, T, embed_dim)

        # Transformer Encoder
        for blk in self.blocks:
            x = blk(x)

        # 归一化
        x = self.norm(x)

        # 取 CLS Token 的特征进行分类
        x = x[:, 0, :]  # (B, embed_dim)
        return self.head(x)  # (B, num_classes)
