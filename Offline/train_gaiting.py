import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from data.lipreading_dataset import LipreadingDataset
from models.Video_ResNet import Video_ResNet_P3D, Video_ResNet_P3D_Encoder, get_resnet_backbone, Video_ResNet_P3D_Temporal_Encoder
from models.Audio_ResNet import AudioModel, AudioEncoder
from models.Fusion import MultiModalFusion


class EarlyExitClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # [B]


def evenly_spaced_indices(total_range, num_indices):
    if num_indices > total_range:
        raise ValueError("num_indices cannot be greater than total_range")
    return np.linspace(0, total_range - 1, num=num_indices, dtype=int)


# accuracy lookup tables based on your figure (rounded for clarity)
# === 替换为真实准确率表查值 ===
video_model_map = ['18', '34', '50']
audio_model_map = ['small', 'medium', 'large']


def load_models(args):
    video_models = []
    video_spatial_models = []
    video_temporal_models = []
    audio_models = []
    fusion_models = []
    
    for size in ["18", "34", "50"]:
        backbone, feat_dim = get_resnet_backbone(f'resnet{size}', pretrained=True)
        video_model = Video_ResNet_P3D_Encoder(backbone, feature_dim=feat_dim)
        checkpoint_path = os.path.join(args.checkpoint_path, f'video/video_resnet_{size}.pth')
        video_pretrained_dict = torch.load(checkpoint_path)
        video_pretrained_dict = {k.replace('module.', ''): v for k, v in video_pretrained_dict.items()}
        video_filtered_dict = {k: v for k, v in video_pretrained_dict.items() if not k.startswith('fc.')}
        video_model.load_state_dict(video_filtered_dict, strict=False)
        video_spatial_model = video_model.spatial_encoder
    
        video_temporal_model = Video_ResNet_P3D_Temporal_Encoder()
        video_temporal_model.load_state_dict(video_pretrained_dict, strict=False)
        
        video_model = nn.DataParallel(video_model).to(args.device)
        video_spatial_model = nn.DataParallel(video_spatial_model).to(args.device)
        video_temporal_model = nn.DataParallel(video_temporal_model).to(args.device)
        
        video_model.eval()
        video_spatial_model.eval()
        video_temporal_model.eval()
        
        video_models.append(video_model)
        video_spatial_models.append(video_spatial_model)
        video_temporal_models.append(video_temporal_model)

    for size in ['small', 'medium', 'large']:
        audio_model = AudioEncoder(size=size)
        checkpoint_path = os.path.join(args.checkpoint_path, f'audio/audio_{size}.pth')
        audio_pretrained_dict = torch.load(checkpoint_path)
        audio_pretrained_dict = {k.replace('module.', ''): v for k, v in audio_pretrained_dict.items()}
        audio_filtered_dict = {k: v for k, v in audio_pretrained_dict.items() if not k.startswith('fc.')}
        audio_model.load_state_dict(audio_filtered_dict, strict=False)
        
        audio_model = nn.DataParallel(audio_model).to(args.device)
        audio_model.eval()
        audio_models.append(audio_model)
    
    for video in ["18", "34", "50"]:
        for audio in ['small', 'medium', 'large']:     
            fusion_model = MultiModalFusion()
            checkpoint_name = f"fusion/fusion_{video}_{audio}.pth"
            checkpoint_path = os.path.join(args.checkpoint_path, checkpoint_name)
            fusion_pretrained_dict = torch.load(checkpoint_path)
            fusion_pretrained_dict = {k.replace('module.', ''): v for k, v in fusion_pretrained_dict.items()}
            # fusion_filtered_dict = {k: v for k, v in fusion_pretrained_dict.items() if not k.startswith('fc.')}
            fusion_model.load_state_dict(fusion_pretrained_dict, strict=False)
            
            fusion_model = nn.DataParallel(fusion_model).to(args.device)
            fusion_model.eval()
            fusion_models.append(fusion_model)

    return video_spatial_models, video_temporal_models, video_models, audio_models, fusion_models



def model_forward(batch, video_spatial_encoder, lidar_spatial_encoder, video_temporal_encoder, lidar_temporal_encoder, llm_model, device, video_frames, lidar_frames):
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in batch.items()}
        ground_truth = batch['answer'].view(-1)  # Flatten the ground truth tensor
        
        lidar_indexes = evenly_spaced_indices(batch['lidar'].shape[1], lidar_frames)
        video_indexes = evenly_spaced_indices(batch['CAM_FRONT'].shape[1], video_frames)

        lidar_input = batch['lidar'][:, lidar_indexes, :, :, :]  # [B, T, C, H, W]
        video_input = torch.cat([batch[cam][:, video_indexes, :, :, :] for cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']], dim=0)
        
        # print(lidar_input.shape, video_input.shape)
        
        V = 6  # 视角数
        B = lidar_input.shape[0]
        T_rgb = video_input.shape[1]
        C = batch['CAM_FRONT'].shape[2]
        H = batch['CAM_FRONT'].shape[3]
        W = batch['CAM_FRONT'].shape[4]

        T_lidar = lidar_input.shape[1]
        
        lidar_input = lidar_input.view(B * T_lidar, -1, lidar_input.size(-2), lidar_input.size(-1))  # Flatten batch and time
        lidar_features = lidar_spatial_encoder(lidar_input)
        lidar_features = lidar_features.view(B, T_lidar, lidar_features.size(-1)).permute(0, 2, 1)  # Reshape back to (B, T, D)
        lidar_tokens = lidar_temporal_encoder(lidar_features)  # (B, T, D)
        
        # # 重新排列为 [B, V, T, C, H, W]
        video_input = video_input.view(V, B, T_rgb, C, H, W).permute(1, 0, 2, 3, 4, 5)
        # # 拉平成帧级图像 [B * V * T, C, H, W]
        video_stack = video_input.reshape(B * V * T_rgb, C, H, W)
        # # video_features: [B*V*T, D] → [B, V, T, D]
        video_features = video_spatial_encoder(video_stack)
        D = video_features.shape[1]
        video_features = video_features.view(B, V, T_rgb, D)
        video_features = video_features.permute(0, 1, 3, 2).reshape(B * V, D, T_rgb)  # [B*V, D, T]

        # Temporal encoder 输出 [B*V, T', D']
        video_tokens_all = video_temporal_encoder(video_features)  # [B*V, T’, D’]
        feature_dim = video_tokens_all.shape[-1]
        video_tokens = video_tokens_all.mean(dim=1).view(B, V, feature_dim)  
        # [B*V, D’]            # [B, V*D’]
        outputs = llm_model(video_tokens, lidar_tokens, batch["ques_ix"])
    
    return outputs


def train_gaiting_classifier(video_spatial_models, video_temporal_models, video_models, audio_models, fusion_models, test_loader, device):
    gaiting_classifier = EarlyExitClassifier().to(device)
    optimizer = torch.optim.Adam(gaiting_classifier.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    gaiting_classifier.train()
    
    min_loss = float('inf')

    for epoch in range(50):
        total_loss = 0.0
        sample_count = 0
        correct_total = 0.0
        progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}", unit="batch")
        for i, (video_input, audio_input, ground_truth) in enumerate(progress_bar):
            video_input, audio_input, ground_truth = video_input.to(device), audio_input.to(device), ground_truth.to(device)
    
            for v_idx in range(3):
                video_model = video_models[v_idx]
                for video_fps in [20, 25, 29]:
                    for a_idx in range(3):
                        for audio_chunk_size in [1200, 1000, 800]:
                            audio_model = audio_models[a_idx]
                            audio_model.chunk_size = audio_chunk_size
                            audio_model.chunk_interval = audio_chunk_size
                            fusion_model = fusion_models[a_idx * 3 + v_idx]
                            for video_ratio in [0.5, 0.7, 0.9]:
                                video_indices = evenly_spaced_indices(total_range=29, num_indices=video_fps)
                                partial_len = max(1, int(video_fps * video_ratio))
                                indices = video_indices[:partial_len]
                                video_partial = video_input[:, indices, :, :, :]
                                
                                B = video_partial.shape[0]
                                
                                with torch.no_grad():
                                    video_partial_feature = video_model(video_partial)
                                    audio_feature = audio_model(audio_input)
                                    outputs = fusion_model(video_partial_feature, audio_feature)
                                    _, predicted = torch.max(outputs, 1)
                                    label = (predicted == ground_truth).float()
                                
                                input_features = torch.cat([audio_feature, video_partial_feature], dim=1)
                                gaiting_outputs = gaiting_classifier(input_features)
                                    
                                loss = criterion(gaiting_outputs, label)
                                
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                                total_loss += loss.item() * B
                                sample_count += B
                                
                                with torch.no_grad():
                                    pred_exit = (gaiting_outputs >= 0.5).float()  # threshold = 0.5
                                    correct = (pred_exit == label).float().sum()
                                    correct_total += correct.item()

            avg_loss = total_loss / sample_count
            gaiting_acc = correct_total / sample_count * 100
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{gaiting_acc:.4f}")

        print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}, Total Loss = {total_loss:.4f}")
        if avg_loss < min_loss:
            min_loss = avg_loss
            print(f"New best model found at epoch {epoch + 1} with loss {min_loss:.4f}")
            # Save the model
            torch.save(gaiting_classifier.state_dict(), "./checkpoints/gaiting/gaiting_classifier.pth")
            print("Best gaiting classifier saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run RealSense experiment with specific FPS.")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints')
    parser.add_argument("--audio_sample_rate", type=int, default=16000)
    parser.add_argument("--audio_chunk_size", type=int, default=800)
    parser.add_argument("--audio_model", type=str, default='large')
    parser.add_argument("--video_fps", type=int, default=29)
    parser.add_argument("--video_model", type=int, default=50)
    parser.add_argument("--audio_exit_threshold", type=float, default=0.8)
    args = parser.parse_args()

    device = args.device
    # 强制初始化CUDA上下文
    torch.cuda.current_device()
    torch.zeros(1).cuda()
  
    data_root = '/data/rxhuang/lipread_feature'
    label_file = './data/selected_words.txt'

    test_transform = transforms.Compose([
        T.ToTensor(),
        T.CenterCrop((88, 88)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

   
    test_dataset = LipreadingDataset(root_dir=data_root, label_file=label_file, video_transform=test_transform, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=192, shuffle=False, num_workers=32)
    
    video_spatial_models, video_temporal_models, video_models, audio_models, fusion_models = load_models(args)
 
    # 训练
    train_gaiting_classifier(video_spatial_models, video_temporal_models, video_models, audio_models, fusion_models, test_loader, device)
