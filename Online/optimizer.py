import time
import json
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linprog


class AccuracyPredictor_Ablation(nn.Module):
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


def load_latency_table(path='data/latency_table.json'):
    with open(path, 'r') as f:
        return json.load(f)
 
def evenly_spaced_indices(total_range, num_indices):
    if num_indices > total_range:
        raise ValueError("num_indices cannot be greater than total_range")
    return np.linspace(0, total_range - 1, num=num_indices, dtype=int)


def compute_cosine_similarity(audio_feat, video_feat):
    audio_feat = F.normalize(audio_feat, dim=-1)
    video_feat = F.normalize(video_feat, dim=-1)
    return (audio_feat * video_feat).sum(dim=-1)  # shape: (B,)



def is_valid_combo(audio_size_str, video_model_size, fps, latency, acc, T_max):
    if latency > T_max:
        return False
    if acc < 90.0:
        return False
    if audio_size_str == "Small" and video_model_size == "ResNet50":
        return False
    if fps == 29 and video_model_size == "ResNet50":
        return False
    return True



audio_sensing_map = {800: 0, 1600: 1, 2000: 2}
video_size_map = {0: '18', 1: '34', 2: '50'}
audio_size = ['small', 'medium', 'large']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Optimizer:
    def __init__(self, args, predictor, path, T_max=90, ablation=False):
        self.predictor = predictor
        self.latency_table = load_latency_table(path)
        self.T_max = T_max

        self.audio_size = ['small', 'medium', 'large']
        self.audio_sensing_map = {1200: 0, 1000: 1, 800: 2}
        self.video_size_map = {0: '18', 1: '34', 2: '50'}

        if ablation:
            predictor_dict = torch.load(f'checkpoints/ablation_study/accuracy_predictor.pth')
        else:
            predictor_dict = torch.load(f'checkpoints/accuracy_predictor/accuracy_predictor_logits.pth')
        self.predictor = predictor.to(args.device).eval()
        self.predictor.load_state_dict(predictor_dict, strict=False)
        self.ablation = ablation

        # ---- Warmup the accuracy predictor ----
        with torch.no_grad():
            if self.ablation:
                dummy_input = torch.randn(1, 1028, device=args.device)  # 输入维度是6
            else:
                dummy_input = torch.randn(1, 6, device=args.device)  # 输入维度是6
            _ = self.predictor(dummy_input)
        print("[Warmup] Accuracy predictor warmup done.")

    def compute_cosine_similarity(self, audio_feat, video_feat):
        audio_feat = F.normalize(audio_feat, dim=-1)
        video_feat = F.normalize(video_feat, dim=-1)
        return (audio_feat * video_feat).sum(dim=-1).item()

    def optimize(self, audio_feat, video_feat):
        cos_val = self.compute_cosine_similarity(audio_feat, video_feat)
        modality_diff = 1.0 - cos_val

        input_vec_np_list = []
        latency_list = []
        config_list = []

        # for a_idx in range(3):
        #     for v_idx in range(3):
        #         for fps in [20, 25, 29]:
        #             for chunk in [1200, 1000, 800]:
        for a_idx in range(9):
            for v_idx in range(9):
                for fps in [20] * 9:
                    for chunk in [1200] * 9:
                        a_idx = 2
                        v_idx = 2
                        audio_size_str = self.audio_size[a_idx]
                        video_model_size = self.video_size_map[v_idx]
                        
                        chunk_index = [1200, 1000, 800].index(chunk) 
                        fps_index = [20, 25, 29].index(fps)

                        latency = self.latency_table[video_model_size][str(fps)][audio_size_str][str(chunk)]

                        if self.ablation:
                            B = audio_feat.shape[0]
                            metadata_v_idx = torch.full((B, 1), float(v_idx)).to(device)  # 形状 (256, 1)
                            metadata_fps = torch.full((B, 1), float(fps_index)).to(device)
                            metadata_a_idx = torch.full((B, 1), float(a_idx)).to(device)
                            metadata_chunk = torch.full((B, 1), float(chunk_index)).to(device)

                            # 沿特征维度 (dim=1) 拼接
                            input_vec = torch.cat([
                                audio_feat,          # (256, 512)
                                video_feat,          # (256, 512)
                                metadata_v_idx,      # (256, 1)
                                metadata_fps,        # (256, 1)
                                metadata_a_idx,      # (256, 1)
                                metadata_chunk       # (256, 1)
                            ], dim=1)                # 结果形状: (256, 512 + 512 + 4) = (256, 1028)
                        else:
                            input_vec_np_list.append([
                                cos_val,
                                modality_diff,
                                a_idx,
                                v_idx,
                                chunk,
                                fps
                            ])
                        latency_list.append(latency)
                        config_list.append({
                            'audio_model_id': a_idx,
                            'audio_chunk_size': chunk,
                            'video_model_id': v_idx,
                            'video_fps': fps
                        })
        
        if self.ablation:
            input_batch = input_vec.to(audio_feat.device)
        else:
            input_batch = torch.tensor(input_vec_np_list, dtype=torch.float32).to(audio_feat.device)
        with torch.no_grad():
            _, acc_preds = self.predictor(input_batch)
            acc_preds = acc_preds.cpu().numpy()

        valid_configs = []
        for idx, acc in enumerate(acc_preds):
            if latency_list[idx] <= self.T_max:
                valid_configs.append((acc, latency_list[idx], config_list[idx]))

        if valid_configs:
            best_acc, best_latency, best_config = max(valid_configs, key=lambda x: x[0])
            return best_config, best_acc, best_latency
        else:
            return None, None, None
