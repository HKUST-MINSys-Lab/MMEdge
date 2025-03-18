import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models


class AudioResNetEncoder(nn.Module):
    def __init__(self, resnet_depth=18, embed_dim=512, chunk_size=800, chunk_interval=400, sample_rate=16000, n_mels=64):
        """
        Args:
            resnet_depth: 选择 ResNet 结构（18/34/50）
            embed_dim: ResNet 提取的特征维度
            chunk_size: 每个 chunk 的样本数
            chunk_interval: 滑动窗口步长
            sample_rate: 采样率
            n_mels: Mel 频谱的维度
        """
        super(AudioResNetEncoder, self).__init__()
        
        self.chunk_size = chunk_size
        self.chunk_interval = chunk_interval
        self.sample_rate = sample_rate

        # Mel-Spectrogram 提取
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=400,
            hop_length=160,
            window_fn=torch.hann_window
        )
        
        # ResNet 特征提取
        self.resnet = self._make_resnet(resnet_depth, embed_dim)

    def _make_resnet(self, depth, embed_dim):
        """ 创建 ResNet 并去掉分类层 """
        if depth == 18:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif depth == 34:
            resnet = models.resnet34(pretrained=True)
        elif depth == 50:
            resnet = models.resnet50(pretrained=True)
        else:
            raise ValueError("ResNet depth not supported!")

        # 修改第一层 Conv 以适应 1 通道 Mel-Spectrogram 输入
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 去掉 ResNet 的全连接层
        resnet.fc = nn.Linear(resnet.fc.in_features, embed_dim)
        return resnet

    def forward(self, x):
        """
        Args:
            x: 输入音频数据 (B, T)，T 是时间维度
        Returns:
            chunk_embeddings: (B, Num_chunks, embed_dim)
        """
        B, T = x.shape  # (Batch, Time)

        # 计算可以提取的 chunk 数量
        chunks = []
        for start in range(0, T, self.chunk_interval):
            end = start + self.chunk_size
            chunk = x[:, start:end]  # 取出 chunk

            # 处理 chunk 长度不足 chunk_size 的情况
            if chunk.shape[1] < self.chunk_size:
                pad_size = self.chunk_size - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))  # 右侧填充 0

            chunks.append(chunk)

        # 拼接为 (B, Num_chunks, chunk_size)
        chunks = torch.stack(chunks, dim=1)  # (B, Num_chunks, chunk_size)

        # 转换为 Mel-Spectrogram (B, Num_chunks, 1, F, T)
        mel_chunks = [self.mel_spectrogram(c).unsqueeze(1) for c in chunks.unbind(dim=1)]
        mel_chunks = torch.stack(mel_chunks, dim=1)  # (B, Num_chunks, 1, F, T')

        # ResNet 处理 (B * Num_chunks, 1, F, T')
        B, Num_chunks, C, H, W = mel_chunks.shape
        mel_chunks = mel_chunks.view(B * Num_chunks, C, H, W)
        chunk_embeddings = self.resnet(mel_chunks)  # (B * Num_chunks, embed_dim)

        # 重新调整形状 (B, Num_chunks, embed_dim)
        chunk_embeddings = chunk_embeddings.view(B, Num_chunks, -1)
        return chunk_embeddings  # 输出局部编码的 chunk 特征


class LSTMTemporalEncoder(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=512, num_layers=2, bidirectional=True):
        """
        LSTM 编码器，处理 chunk 级别的时间依赖
        """
        super(LSTMTemporalEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, Num_chunks, embed_dim)
        Returns:
            final feature: (B, embed_dim)
        """
        lstm_out, _ = self.lstm(x)  # (B, Num_chunks, hidden_dim * 2)
        final_feature = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的特征
        return final_feature


class AudioEncoderModel(nn.Module):
    def __init__(self, resnet_depth=18, embed_dim=512, chunk_size=800, chunk_interval=400, sample_rate=16000, n_mels=64, hidden_dim=512, num_layers=2, bidirectional=True):
        """
        组合 AudioResNetEncoder + LSTMTemporalEncoder
        """
        super(AudioEncoderModel, self).__init__()
        self.feature_extractor = AudioResNetEncoder(
            resnet_depth=resnet_depth,
            embed_dim=embed_dim,
            chunk_size=chunk_size,
            chunk_interval=chunk_interval,
            sample_rate=sample_rate,
            n_mels=n_mels
        )
        self.temporal_encoder = LSTMTemporalEncoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional
        )

    def forward(self, audio_waveform):
        """
        Args:
            audio_waveform: (B, T)
        Returns:
            final feature: (B, embed_dim)
        """
        chunk_features = self.feature_extractor(audio_waveform)  # (B, Num_chunks, embed_dim)
        final_feature = self.temporal_encoder(chunk_features)  # (B, embed_dim)
        return final_feature

# # 使用示例
if __name__ == "__main__":
    from thop import profile
    input = torch.randn(1, 19456)  # 10秒音频
    model = AudioEncoderModel(resnet_depth=18, chunk_size=800, chunk_interval=400, sample_rate=16000)
    output = model(input)
    print(output.shape)  # (B, embed_dim)
    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 输出 GFLOP
    print(f"Parameters: {params / 1e6:.2f} M")  # 输出参数量（单位：百万）
