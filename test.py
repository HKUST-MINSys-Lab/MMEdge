import torch
from thop import profile
from models.ResNet_2D import Video_ResNet_2D
from models.Fusion import BimodalFusion

if __name__ == "__main__":
    # # 例如，使用 w1_0 配置
    # input = torch.rand(1, 16, 3, 224, 224) #[btz, channel, T, H, W]
    # num_frames = 16 # 设置视频帧数
    # model = Video_ResNet_2D()
    # output = model(input)
    # print(output.shape)
    # flops, params = profile(model, inputs=(input,))
    # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 输出 GFLOP
    # print(f"Parameters: {params / 1e6:.2f} M")  # 输出参数量（单位：百万）


    modal1 = torch.randn(1, 512)  # 模态1特征
    modal2 = torch.randn(1, 512)  # 模态2特征
    
    model = EnhancedBimodalFusion(modal_dims=(512, 512))
    fused_feature = model(modal1, modal2)
    
    print(fused_feature.shape)

    flops, params = profile(model, inputs=(modal1, modal2))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 输出 GFLOP
    print(f"Parameters: {params / 1e6:.2f} M")  # 输出参数量（单位：百万）