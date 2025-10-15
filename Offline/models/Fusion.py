import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalFusion(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, fusion_dim=512, num_classes=50):
        """
        Deep Multi-modal fusion model using Self-Attention and Cross-modal Gating.

        :param input_dim: Dimension of input features (default: 512 for both video and audio)
        :param hidden_dim: Hidden layer dimension for deep fusion processing
        :param fusion_dim: Intermediate fusion dimension
        :param num_classes: Number of output classes
        """
        super(MultiModalFusion, self).__init__()

        # Self-Attention Layers (Deep MLP for learning better attention)
        self.attn_video = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.attn_audio = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Cross-Modal Gating (让两个模态相互影响)
        self.gate_video = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.gate_audio = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        # **Transformer-style Cross Attention Layer**
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, dropout=0.1)

        # **Deeper Fusion Layer**
        self.fusion_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU()
        )

        # **Deeper Classification Head**
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_dim // 4, num_classes)
        )

    def forward(self, video_feat, audio_feat):
        """
        Forward pass for fusion model with deep architecture and cross-modal attention.

        :param video_feat: Video feature tensor of shape (batch, 512)
        :param audio_feat: Audio feature tensor of shape (batch, 512)
        :return: Classification output
        """
        # Compute attention weights
        attn_v = self.attn_video(video_feat)  # (batch, 1)
        attn_a = self.attn_audio(audio_feat)  # (batch, 1)

        # Compute cross-modal gating
        gate_v = self.gate_video(video_feat)  # (batch, 512)
        gate_a = self.gate_audio(audio_feat)  # (batch, 512)

        # Apply attention & gating
        video_feat = attn_v * gate_v * video_feat
        audio_feat = attn_a * gate_a * audio_feat

        # **Apply Cross-Modal Self-Attention**
        fused_feat = torch.stack([video_feat, audio_feat], dim=0)  # (2, batch, 512)
        fused_feat, _ = self.cross_attention(fused_feat, fused_feat, fused_feat)
        fused_feat = torch.mean(fused_feat, dim=0)  # (batch, 512)

        # **Deep Fusion**
        fused_feat = self.fusion_layer(fused_feat)

        # **Classification**
        output = self.classifier(fused_feat)
        return output


# Example usage
if __name__ == "__main__":
    batch_size = 8
    video_input = torch.randn(batch_size, 512)
    audio_input = torch.randn(batch_size, 512)

    model = MultiModalFusion()
    output = model(video_input, audio_input)
    print(output.shape)  # Expected: (batch_size, num_classes)
