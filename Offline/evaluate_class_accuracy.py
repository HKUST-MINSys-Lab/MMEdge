import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
# os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'
import csv
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from data.lipreading_dataset import LipreadingDataset
from models.Video_ResNet import Video_ResNet_P3D_Encoder, get_resnet_backbone
from models.Audio_ResNet import AudioEncoder, AudioModelBaselineEncoder
from models.Fusion import MultiModalFusion
from models.Video_ResNet_Ablation import Video_ResNet_Baseline_Encoder, Video_ResNet_P3D_Without_Shift, Video_ResNet_P3D_Without_Diff, Video_ResNet_P3D_Without_SE
from models.ResNet_3D import Video_3D_Encoder


def truncate_and_pad_video(video_inputs, n):
    """
    video_inputs: Tensor of shape (B, T, C, H, W)
    n: number of frames to keep
    Returns:
        padded_video: same shape as video_inputs
    """
    B, T, C, H, W = video_inputs.shape
    device = video_inputs.device

    n = min(n, T)
    kept = video_inputs[:, :n, :, :, :]  # (B, n, C, H, W)

    if n < T:
        zeros = torch.zeros((B, T - n, C, H, W), device=device, dtype=video_inputs.dtype)
        padded_video = torch.cat([kept, zeros], dim=1)  # (B, T, C, H, W)
    else:
        padded_video = kept

    return padded_video


def sample_audio_chunks(audio_inputs, chunk_size=800, num_chunks=3):
    """
        audio_inputs: Tensor of shape (B, L)
        Returns: Tensor of shape (B, num_chunks * chunk_size)
    """
    B, L = audio_inputs.shape
    device = audio_inputs.device
    dtype = audio_inputs.dtype

    total_chunks = (L + chunk_size - 1) // chunk_size  # ceil
    padded_len = total_chunks * chunk_size

    # Padding
    if padded_len > L:
        pad = torch.zeros((B, padded_len - L), dtype=dtype, device=device)
        audio_inputs = torch.cat([audio_inputs, pad], dim=1)

    # Split into chunks: (B, total_chunks, chunk_size)
    audio_chunks = audio_inputs.view(B, total_chunks, chunk_size)

    # For each sample in the batch, randomly select `num_chunks`
    selected = []
    for b in range(B):
        idx = torch.randperm(total_chunks)[:num_chunks]
        selected_chunks = audio_chunks[b, idx, :]  # (num_chunks, chunk_size)
        selected.append(selected_chunks.reshape(-1))  # flatten

    # Stack back to (B, num_chunks * chunk_size)
    result = torch.stack(selected, dim=0)
    return result  # (B, num_chunks * chunk_size)



def evenly_spaced_indices(total_range, num_indices):
    if num_indices > total_range:
        raise ValueError("num_indices cannot be greater than total_range")
    return np.linspace(0, total_range - 1, num=num_indices, dtype=int)


def evaluate(args, models, val_loader, criterion, device):
    audio_model, video_model, fusion_model = models
    video_model.eval()
    audio_model.eval()
    fusion_model.eval()

    overall_loss = 0.0
    overall_correct = 0
    overall_total = 0

    num_classes = 50  # 修改为你实际的数据集类别数
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    progress_bar = tqdm(val_loader, desc="Validating", unit="batch")

    with torch.no_grad():
        for video_inputs, audio_inputs, labels in progress_bar:
            video_inputs, audio_inputs, labels = video_inputs.to(device), audio_inputs.to(device), labels.to(device)
            
            audio_model.chunk_size = args.audio_chunk_size
            audio_model.chunk_interval = args.audio_chunk_size
            video_indices = evenly_spaced_indices(total_range=29, num_indices=args.video_fps)
            video_inputs = video_inputs[:, video_indices, :, :, :]
            # video_inputs = truncate_and_pad_video(video_inputs, n=13)
            # audio_inputs = sample_audio_chunks(audio_inputs, chunk_size=args.audio_chunk_size, num_chunks=25)
            # audio_inputs = audio_inputs[:, -16800:]
            
            video_feature = video_model(video_inputs)
            audio_feature = audio_model(audio_inputs)
            outputs = fusion_model(video_feature, audio_feature)
            
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            overall_loss += loss.item() * labels.size(0)
            overall_correct += (predicted == labels).sum().item()
            overall_total += labels.size(0)

            # 更新 tqdm 进度条，显示完整模型的结果
            current_loss = overall_loss / overall_total
            current_acc = (overall_correct / overall_total) * 100.0
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
            
    overall_loss /= overall_total
    overall_acc = (overall_correct / overall_total) * 100.0
    print(f"Full Model: Loss = {overall_loss:.4f}, Accuracy = {overall_acc:.2f}%\n")
    print("Class-wise Accuracy:")
    for i in range(num_classes):
        if class_total[i] == 0:
            acc = 0.0
        else:
            acc = 100.0 * class_correct[i] / class_total[i]
        print(f"  Class {i:02d}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    result_file = "results.csv"
    with open(result_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([
            args.audio_model,
            args.audio_chunk_size,
            args.video_model,
            args.video_fps,
            f"{overall_acc:.2f}"
        ])


    return overall_loss, overall_acc


if __name__ == '__main__':
    # **参数设置**
    parser = argparse.ArgumentParser(description="Run RealSense experiment with specific FPS.")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints')
    parser.add_argument("--audio_sample_rate", type=int, default=16000)
    parser.add_argument("--audio_chunk_size", type=int, default=800)
    parser.add_argument("--audio_model", type=str, default='large')
    parser.add_argument("--video_fps", type=int, default=29)
    parser.add_argument("--video_model", type=int, default=50)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    data_root = '/data/rxhuang/lipread_feature'
    label_file = './data/selected_words.txt'

    test_transform = transforms.Compose([
        T.ToTensor(),
        T.CenterCrop((88, 88)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

   
    test_dataset = LipreadingDataset(root_dir=data_root, label_file=label_file, video_transform=test_transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=32)


    num_classes = 50 
    
    model_config_video = args.video_model
    model_config_audio = args.audio_model
    model_config_fusion =  str(model_config_video) + '_' + model_config_audio

    print(f"Model Config: Video: ResNet-{model_config_video}")
    print(f"Model Config: Audio: ResNet-{model_config_audio}")

    audio_model = AudioEncoder(size=model_config_audio)
    audio_model = AudioModelBaselineEncoder()
    backbone, feat_dim = get_resnet_backbone(f'resnet{model_config_video}', pretrained=True)
    # video_model = Video_ResNet_P3D_Encoder(backbone, feature_dim=feat_dim)
    # video_model = Video_ResNet_Baseline_Encoder(backbone, feature_dim=feat_dim)
    # video_model = Video_ResNet_P3D_Without_Shift(backbone, feature_dim=feat_dim)
    video_model = Video_3D_Encoder(pretrain=True, feature_dim=512)
    # video_model = Video_ResNet_P3D_Without_Diff(backbone, feature_dim=feat_dim)
    # video_model = Video_ResNet_P3D_Without_SE(backbone, feature_dim=feat_dim)
    fusion_model = MultiModalFusion()

    # audio_pretrained_dict = torch.load(f'checkpoints/audio/audio_{model_config_audio}.pth')
    audio_pretrained_dict = torch.load(f'checkpoints/baselines/audio_baseline.pth')
    audio_pretrained_dict = {k.replace('module.', ''): v for k, v in audio_pretrained_dict.items()}
    audio_filtered_dict = {k: v for k, v in audio_pretrained_dict.items() if not k.startswith('fc.')}
    # video_pretrained_dict = torch.load(f'checkpoints/video/video_resnet_{model_config_video}.pth')
    video_pretrained_dict = torch.load(f'checkpoints/baselines/video_resnet50_3D.pth')
    # video_pretrained_dict = torch.load(f'checkpoints/baselines/video_resnet50_baseline_scratch.pth')
    # video_pretrained_dict = torch.load(f'checkpoints/ablation_study/video_resnet50_without_shift_scratch.pth')
    video_pretrained_dict = {k.replace('module.', ''): v for k, v in video_pretrained_dict.items()}
    video_filtered_dict = {k: v for k, v in video_pretrained_dict.items() if not k.startswith('fc.')}
    fusion_pretrained_dict = torch.load(f'./checkpoints/baselines/fusion.pth')
    # fusion_pretrained_dict = torch.load(f'./checkpoints/fusion/fusion_{model_config_fusion}.pth')
    # fusion_pretrained_dict = torch.load(f'checkpoints/baselines/fusion_baseline_scratch.pth')
    # fusion_pretrained_dict = torch.load(f'checkpoints/ablation_study/fusion_shift_scratch.pth')
    fusion_filtered_dict = {k.replace('module.', ''): v for k, v in fusion_pretrained_dict.items()}
    # fusion_filtered_dict = {k: v for k, v in fusion_pretrained_dict.items() if not k.startswith('fc.')}

    audio_model.load_state_dict(audio_filtered_dict, strict=False)
    video_model.load_state_dict(video_filtered_dict, strict=False)
    fusion_model.load_state_dict(fusion_filtered_dict, strict=False)

    models = [audio_model, video_model, fusion_model]

    for i in range(len(models)):
        models[i] = models[i].to(device)
        
    # **损失函数**
    criterion = nn.CrossEntropyLoss()

    # **学习率调度器**
    epochs = 100
    
    # 训练
    evaluate(args, models, test_loader, criterion, device)
