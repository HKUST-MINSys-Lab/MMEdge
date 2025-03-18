import torch
from thop import profile
from models.ResNet_2D import ResNet_2D

if __name__ == "__main__":
    # 例如，使用 w1_0 配置
    input = torch.rand(1, 16, 3, 224, 224) #[btz, channel, T, H, W]
    num_frames = 16 # 设置视频帧数
    # model = VideoMobileViTv2(classifier_num=48, num_frames=num_frames)
    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 输出 GFLOP
    print(f"Parameters: {params / 1e6:.2f} M")  # 输出参数量（单位：百万）