import torch
from thop import profile
from models.ResNet_2D import Video_ResNet_P3D, Video_Classification as Video_Classification_2D, resnet18, resnet34, resnet50, Video_ResNet_TSM, Video_ResNet_Baseline
from models.Fusion import MultiModalFusion
from models.BranchModel import SlowFast_N_Branch
from models.LSTM import AudioLSTM
from models.Audio_ResNet import AudioModel, AudioEncoder
from models.ResNet_3D import Video_Classification as Video_Classification_3D
from models.Video_ResNet import Video_ResNet_P3D, get_resnet_backbone
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torchinfo import summary
from models.Video_ResNet_Ablation import Video_ResNet_P3D_Frame_Shift, Video_ResNet_P3D_Without_Diff, Video_ResNet_P3D_Without_SE


if __name__ == "__main__":
    # 初始化模型（你可以换成你自己的模型）
    input_shape = (1, 30, 3, 88, 88)
    # input_shape = (1, 30, 2048)
    # model = Video_Classification_3D()
    # input_shape = (1, 19456)
    
    backbone, feat_dim = get_resnet_backbone('resnet50', pretrained=True)
    model = Video_ResNet_P3D_Without_SE(backbone, feature_dim=feat_dim)
    # model = Video_ResNet_Baseline()
    # model = AudioEncoder(size='small')
    model.eval()

    # 构造输入：(B, C, T, H, W)
    input_tensor = torch.randn(input_shape).to('cuda')

    print("======= TorchInfo Summary =======")
    summary(model, input_size=(input_shape))

    print("\n======= fvcore Analysis =======")
    # 参数量
    print(parameter_count_table(model))

    # FLOPs（注意是 multiply-add，不是 MACs）
    flops = FlopCountAnalysis(model, input_tensor)
    print(f"Total FLOPs: {flops.total():,} ({flops.total() / 1e9:.2f} GFLOPs)")

    # 每层 FLOPs（可选）
    # print(flops.by_module())

    
