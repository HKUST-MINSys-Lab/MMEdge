import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'
import json
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision.transforms.v2 as T
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from data.lipreading_dataset import LipreadingDataset
from models.Video_ResNet import Video_ResNet_P3D_Encoder, get_resnet_backbone
from models.Audio_ResNet import AudioEncoder
from models.Fusion import MultiModalFusion


# class AccuracyPredictor(nn.Module):
#     def __init__(self, input_dim=6, hidden_dim=64, dropout=0.3):  # ⬅️ 修改 input_dim=6
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, x):
#         return self.net(x)


class AccuracyPredictor(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_classes=50, dropout=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.logits_head = nn.Linear(hidden_dim, num_classes)
        self.accuracy_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        rep = self.shared(x)
        logits = self.logits_head(rep)
        accuracy = self.accuracy_head(rep).squeeze(1)
        return logits, accuracy
    

def evenly_spaced_indices(total_range, num_indices):
    if num_indices > total_range:
        raise ValueError("num_indices cannot be greater than total_range")
    return np.linspace(0, total_range - 1, num=num_indices, dtype=int)


# accuracy lookup tables based on your figure (rounded for clarity)
# === 替换为真实准确率表查值 ===
audio_size_map = {800: 0, 1600: 1, 2000: 2}
video_size_map = {0: '18', 1: '34', 2: '50'}
audio_size = ['small', 'medium', 'large']


def compute_cosine_similarity(audio_feat, video_feat):
    audio_feat = F.normalize(audio_feat, dim=-1)
    video_feat = F.normalize(video_feat, dim=-1)
    return (audio_feat * video_feat).sum(dim=-1)  # shape: (B,)


def evaluate_accuracy_predictor(audio_models, video_models, test_loader, accuracy_table, device):
    predictor = AccuracyPredictor(input_dim=6, hidden_dim=64, dropout=0.3).to(device)
    predictor_dict = torch.load(f'checkpoints/accuracy_predictor/accuracy_predictor_logits_2.pth')
    predictor.load_state_dict(predictor_dict, strict=False)
    
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    predictor.eval()

    total_loss = 0.0
    sample_count = 0
    
    all_conf = []
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(test_loader, desc=f"Evaluating...", unit="batch")
    for i, (video_input, audio_input, ground_truth) in enumerate(progress_bar):
        video_input, audio_input, ground_truth = video_input.to(device), audio_input.to(device), ground_truth.to(device)
        
        audio_encoder = audio_models[0].encoder
        video_encoder = video_models[0].spatial_encoder

        # Feature extraction (1st frame)
        audio_chunk = audio_input[:, :800]
        audio_feat = audio_encoder(audio_chunk).detach()

        video_frame = video_input[:, 0:1]  # use first frame only
        B, _, C, H, W = video_frame.shape
        video_feat = video_encoder(video_frame.view(B, C, H, W)).squeeze(-1).squeeze(-1).detach()

        cos_sim = compute_cosine_similarity(audio_feat, video_feat)
        modality_diff = 1 - cos_sim

        for v_idx, video_model in enumerate(video_models):
            for video_fps in [20, 25, 29]:
                for a_idx, audio_model in enumerate(audio_models):
                    for audio_chunk_size in [1200, 1000, 800]:
                        audio_encoder = audio_model.encoder
    
                        chunk_index = [1200, 1000, 800].index(audio_chunk_size) 
                        fps_index = [20, 25, 29].index(video_fps)
                        
                        input_vec = torch.stack([
                                cos_sim,
                                modality_diff,
                                torch.full_like(cos_sim, float(v_idx)),
                                torch.full_like(cos_sim, float(fps_index)),
                                torch.full_like(cos_sim, float(a_idx)),
                                torch.full_like(cos_sim, float(chunk_index))
                            ], dim=1)

                        audio_size_str = audio_size[a_idx]  # "Small", "Medium", "Large"
                        # fps_index = [20, 25, 29].index(video_fps)
                        label = []  
                        for i in range(B):
                            acc = accuracy_table[video_size_map[v_idx]][str(video_fps)][audio_size_str][str(audio_chunk_size)][str(ground_truth[i].item())]
                            label.append(float(acc))
                        label = torch.tensor(label, dtype=torch.float32, device=device)

                        # label = torch.full((video_input.size(0),), acc, dtype=torch.float32, device=device)
                        # label = acc
                        
                        # Predict and update
                        with torch.no_grad():
                            _, pred = predictor(input_vec)
                        
                        pred = pred.squeeze()
                        # print(ground_truth, pred, acc)
                        loss = criterion(pred, label)
                        
                        all_preds.append(pred.detach().cpu())
                        all_labels.append(label.detach().cpu())


                        total_loss += loss.item() * label.size(0)
                        sample_count += label.size(0)

            avg_loss = total_loss / sample_count
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
            
    torch.save({
        "preds": torch.cat(all_preds, dim=0), 
        "labels": torch.cat(all_labels, dim=0)
    }, "./outputs/predictor_outputs_logits.pt")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    data_root = '/data/rxhuang/lipread_feature'
    label_file = './data/selected_words.txt'

    test_transform = transforms.Compose([
        T.ToTensor(),
        T.CenterCrop((88, 88)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

   
    test_dataset = LipreadingDataset(root_dir=data_root, label_file=label_file, video_transform=test_transform, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=32)


    num_classes = 50 
    audio_models = []
    video_models = []
    fusion_models = []
    
    for size in ['small', 'medium', 'large']:
        audio_model = AudioEncoder(size=size)
        audio_pretrained_dict = torch.load(f'checkpoints/audio/audio_{size}.pth')
        audio_pretrained_dict = {k.replace('module.', ''): v for k, v in audio_pretrained_dict.items()}
        audio_filtered_dict = {k: v for k, v in audio_pretrained_dict.items() if not k.startswith('fc.')}
        audio_model.load_state_dict(audio_filtered_dict, strict=False)
        audio_model = audio_model.to(device)
        audio_model.eval()
        audio_models.append(audio_model)
        
    for size in ['18', '34', '50']:
        backbone, feat_dim = get_resnet_backbone(f'resnet{size}', pretrained=True)
        video_model = Video_ResNet_P3D_Encoder(backbone, feature_dim=feat_dim)
        video_pretrained_dict = torch.load(f'checkpoints/video/video_resnet_{size}.pth')
        video_pretrained_dict = {k.replace('module.', ''): v for k, v in video_pretrained_dict.items()}
        video_filtered_dict = {k: v for k, v in video_pretrained_dict.items() if not k.startswith('fc.')}
        video_model.load_state_dict(video_filtered_dict, strict=False)
        video_model = video_model.to(device)
        video_model.eval()
        video_models.append(video_model)
        
    for video in ['18', '34', '50']:
        for audio in ['small', 'medium', 'large']:
            fusion_model = MultiModalFusion()
            fusion_pretrained_dict = torch.load(f'checkpoints/fusion/fusion_{video}_{audio}.pth')
            fusion_filtered_dict = {k.replace('module.', ''): v for k, v in fusion_pretrained_dict.items()}
            fusion_model.load_state_dict(fusion_filtered_dict, strict=False)
            fusion_model = fusion_model.to(device) 
            fusion_model.eval()
            fusion_models.append(fusion_model)
        
    # **损失函数**
    criterion = nn.CrossEntropyLoss()
    
    accuracy_table = json.load(open("./data/accuracy_table.json", "r"))

    # **学习率调度器**
    epochs = 100
    
    # 训练
    evaluate_accuracy_predictor(audio_models, video_models, test_loader, accuracy_table, device)
