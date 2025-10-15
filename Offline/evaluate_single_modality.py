import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import time
import torch
import argparse
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from data.lipreading_dataset import LipreadingDataset
from models.Audio_ResNet import get_audio_encoder, LSTMTemporalEncoder, AudioTemporalEncoder, AudioTemporalClassifier, AudioModel
from models.Video_ResNet import Video_ResNet_P3D_Encoder, get_resnet_backbone, Video_ResNet_P3D_Temporal_Encoder
from models.Fusion import MultiModalFusion
from optimizer import AccuracyPredictor, Optimizer, latency_table


video_idx_map = {'18': 0, '34': 1, '50': 2}
audio_idx_map = {'small': 0, 'medium': 1, 'large': 2}


def normalize_chunk(chunk):
    mean = chunk.mean(dim=1, keepdim=True)  # (B, 1)
    std = chunk.std(dim=1, keepdim=True)    # (B, 1)
    std = torch.where(std == 0, torch.ones_like(std), std)  # 避免除以0
    chunk = (chunk - mean) / std
    return chunk


def load_models(args):
    video_spatial_models = []
    video_temporal_models = []
    audio_chunk_models = []
    audio_temporal_models = []
    fusion_models = []
    
    for size in ["18", "34", "50"]:
        backbone, feat_dim = get_resnet_backbone(f'resnet{size}', pretrained=True)
        video_model = Video_ResNet_P3D_Encoder(backbone, feature_dim=feat_dim).to(args.device)
        checkpoint_path = os.path.join(args.checkpoint_path, f'video/video_resnet_{size}.pth')
        video_pretrained_dict = torch.load(checkpoint_path)
        video_pretrained_dict = {k.replace('module.', ''): v for k, v in video_pretrained_dict.items()}
        video_filtered_dict = {k: v for k, v in video_pretrained_dict.items() if not k.startswith('fc.')}
        video_model.load_state_dict(video_filtered_dict, strict=False)
        video_spatial_model = video_model.spatial_encoder
    
        video_temporal_model = Video_ResNet_P3D_Temporal_Encoder().to(args.device)
        video_temporal_model.load_state_dict(video_filtered_dict, strict=False)
        
        video_spatial_models.append(video_spatial_model)
        video_temporal_models.append(video_temporal_model)

    for size in ['small', 'medium', 'large']:
        audio_model = AudioModel(size=size).to(args.device)
        checkpoint_path = os.path.join(args.checkpoint_path, f'audio/audio_{size}_2.pth')
        audio_pretrained_dict = torch.load(checkpoint_path)
        # audio_pretrained_dict = {k.replace('module.', ''): v for k, v in audio_pretrained_dict.items()}
        # audio_filtered_dict = {k: v for k, v in audio_pretrained_dict.items() if not k.startswith('fc.')}
        audio_model.load_state_dict(audio_pretrained_dict, strict=False)
        audio_chunk_model = audio_model.encoder
        
        audio_temporal_encoder = AudioTemporalClassifier().to(args.device)
        audio_temporal_encoder.load_state_dict(audio_pretrained_dict, strict=False)
        
        audio_chunk_models.append(audio_chunk_model)
        audio_temporal_models.append(audio_temporal_encoder)
    
    for video in ["18", "34", "50"]:
        for audio in ['small', 'medium', 'large']:     
            fusion_model = MultiModalFusion().to(args.device)
            checkpoint_name = f"fusion/fusion_{video}_{audio}.pth"
            checkpoint_path = os.path.join(args.checkpoint_path, checkpoint_name)
            fusion_pretrained_dict = torch.load(checkpoint_path)
            fusion_pretrained_dict = {k.replace('module.', ''): v for k, v in fusion_pretrained_dict.items()}
            fusion_filtered_dict = {k: v for k, v in fusion_pretrained_dict.items() if not k.startswith('fc.')}
            fusion_model.load_state_dict(fusion_filtered_dict, strict=False)
            fusion_models.append(fusion_model)

    return video_spatial_models, video_temporal_models, audio_chunk_models, audio_temporal_models, fusion_models



def evaluate(models, val_loader, criterion, device):
    video_spatial_model, video_temporal_model, audio_chunk_model, audio_temporal_model, fusion_model = models
    overall_loss = 0.0
    overall_correct = 0
    overall_total = 0

    progress_bar = tqdm(val_loader, desc="Validating", unit="batch")

    with torch.no_grad():
        for video_inputs, audio_inputs, labels in progress_bar:
            video_inputs, audio_inputs, labels = video_inputs.to(device), audio_inputs.to(device), labels.to(device)
            
            # fps = 25
            # chunk = 1600
            # audio_model.chunk_size = chunk
            # video_indices = evenly_spaced_indices(total_range=29, num_indices=fps)
            # video_inputs = video_inputs[:, video_indices, :, :, :]
            
            # print(f"Video fps: {fps}, Audio Chunk: {chunk}")

            chunk_size = 800
            
            B, T = audio_inputs.shape
            chunks = []
            for start in range(0, T, chunk_size):
                end = start + chunk_size
                chunk = audio_inputs[:, start:end]
                if chunk.shape[1] < chunk_size:
                    pad_size = chunk_size - chunk.shape[1]
                    chunk = nn.functional.pad(chunk, (0, pad_size))
                chunk = normalize_chunk(chunk)
                chunks.append(chunk)

            chunks = torch.stack(chunks, dim=1)  # (B, N, chunk_size)
            B, N, C = chunks.shape
            chunks = chunks.view(B * N, C)
            audio_embeddings = audio_chunk_model(chunks)  # (B * N, embed_dim)
            audio_embeddings = audio_embeddings.view(B, N, -1)
            
            # audio_feature = audio_temporal_model(audio_embeddings)

            outputs = audio_temporal_model(audio_embeddings)
            
            # B, T, C, H, W = video_inputs.shape
            # video_inputs = video_inputs.reshape(B * T, C, H, W)
            # video_spatial_features = video_spatial_model(video_inputs) # (B*T, 2048)
            
            # video_spatial_features = video_spatial_features.view(B, T, 512).permute(0, 2, 1) 
            # video_feature = video_temporal_model(video_spatial_features)  # (B, C, T)

            # outputs = fusion_model(video_feature, audio_feature)
            
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            overall_loss += loss.item() * labels.size(0)
            overall_correct += (predicted == labels).sum().item()
            overall_total += labels.size(0)

            # 更新 tqdm 进度条，显示完整模型的结果
            current_loss = overall_loss / overall_total
            current_acc = (overall_correct / overall_total) * 100.0
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")
            
    overall_loss /= overall_total
    overall_acc = (overall_correct / overall_total) * 100.0
    print(f"Full Model: Loss = {overall_loss:.4f}, Accuracy = {overall_acc:.2f}%\n")

    return overall_loss, overall_acc


def main():
    parser = argparse.ArgumentParser(description="Run RealSense experiment with specific FPS.")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints')
    args = parser.parse_args()
    
    # Load dataset
    data_root = '/data/rxhuang/lipread_feature'
    label_file = './data/selected_words.txt'

    test_transform = transforms.Compose([
        T.ToTensor(),
        T.CenterCrop((88, 88)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = LipreadingDataset(root_dir=data_root, label_file=label_file, video_transform=test_transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=32)
    
    video_spatial_models, video_temporal_models, audio_chunk_models, audio_temporal_models, fusion_models = load_models(args)
    a_idx = 2
    v_idx = 2
    
    models = [
        video_spatial_models[v_idx].eval(), 
        video_temporal_models[v_idx].eval(), 
        audio_chunk_models[a_idx].eval(), 
        audio_temporal_models[a_idx].eval(), 
        fusion_models[v_idx * 3 + a_idx].eval()
    ]

    # **损失函数**
    criterion = nn.CrossEntropyLoss()

    evaluate(models, test_loader, criterion, args.device)

if __name__ == '__main__':
    main()