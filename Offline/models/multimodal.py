import torch
import torch.nn as nn

from models.ResNet_2D import Video_ResNet_2D
from models.Audio_ResNet import AudioEncoderModel
from models.Fusion import BimodalFusion

class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        self.video_model = Video_ResNet_2D()
        self.audio_model = AudioEncoderModel(resnet_depth=18, chunk_size=800, chunk_interval=400, sample_rate=16000)
        self.fusion_model = BimodalFusion((512, 512), num_classes=50)  # 二模态融合模型    
    
    def forward(self, video_inputs, audio_inputs):
        video_feature = self.video_model(video_inputs)
        audio_feature = self.audio_model(audio_inputs)
        outputs = self.fusion_model(video_feature, audio_feature)
        return outputs  
    