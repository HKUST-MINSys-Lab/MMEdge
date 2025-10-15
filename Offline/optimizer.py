import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linprog


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


latency_table = {
    'small': {
        800:  [[64.04, 60.26, 62.08], [72.85, 73.16, 74.67], [70.88, 80.22, 95.45]],
        1600: [[61.57, 63.16, 63.97], [71.81, 70.26, 73.68], [80.73, 68.44, 91.22]],
        2000: [[61.66, 63.25, 63.11], [73.15, 80.25, 71.77], [69.28, 68.15, 88.09]],
    },
    'medium': {
        800:  [[59.39, 63.07, 62.99], [74.05, 76.39, 75.85], [75.56, 73.76, 116.58]],
        1600: [[62.16, 58.45, 62.71], [80.71, 70.04, 70.69], [72.53, 69.36, 105.21]],
        2000: [[59.97, 65.55, 58.64], [67.19, 82.44, 76.11], [67.62, 81.60, 110.84]],
    },
    'large': {
        800:  [[60.62, 60.49, 59.48], [71.72, 75.73, 79.15], [78.88, 75.03, 91.86]],
        1600: [[58.55, 63.02, 66.78], [71.31, 73.25, 72.66], [72.41, 71.94, 95.56]],
        2000: [[60.49, 60.74, 58.93], [75.86, 69.04, 69.38], [73.53, 75.35, 84.36]],
    }
}


predict_table = {
    'small': {
        800:  [[93.20, 95.40, 95.60], [92.12, 94.20, 95.48], [91.68, 95.16, 95.88]],
        1600: [[84.64, 88.16, 89.64], [82.48, 86.52, 89.32], [84.36, 90.12, 92.44]],
        2000: [[75.64, 80.96, 83.56], [73.48, 80.12, 83.76], [75.36, 84.32, 88.12]]
    },
    'medium': {
        800:  [[94.80, 95.76, 96.64], [94.88, 96.36, 96.76], [93.32, 96.16, 97.12]],
        1600: [[88.52, 91.20, 92.32], [89.28, 92.20, 93.76], [87.76, 92.40, 94.60]],
        2000: [[80.20, 85.20, 87.60], [82.68, 87.52, 89.92], [81.40, 88.32, 91.64]]
    },
    'large': {
        800:  [[95.20, 96.64, 96.96], [95.24, 96.36, 96.84], [94.60, 96.72, 97.36]],
        1600: [[87.00, 89.96, 91.20], [88.56, 91.00, 92.48], [86.32, 91.32, 93.72]],
        2000: [[78.84, 84.08, 85.32], [81.48, 85.84, 88.32], [79.84, 86.44, 89.76]]
    }
}



class Optimizer:
    def __init__(self, args, predictor, latency_table, T_max=90):
        self.predictor = predictor
        self.latency_table = latency_table
        self.T_max = T_max

        self.audio_size = ['small', 'medium', 'large']
        self.audio_sensing_map = {800: 0, 1600: 1, 2000: 2}
        self.video_size_map = {0: '18', 1: '34', 2: '50'}

        predictor_dict = torch.load(f'checkpoints/accuracy_predictor_logits.pth')
        self.predictor = predictor.to(args.device).eval()
        self.predictor.load_state_dict(predictor_dict, strict=False)

        # ---- Warmup the accuracy predictor ----
        with torch.no_grad():
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

        for a_idx in range(3):
            for v_idx in range(3):
                for fps in [20, 25, 29]:
                    for chunk in [800, 1600, 2000]:
                        audio_size_str = self.audio_size[a_idx]
                        video_model_size = f"ResNet{self.video_size_map[v_idx]}"
                        fps_index = [20, 25, 29].index(fps)

                        latency = self.latency_table[audio_size_str][chunk][v_idx][fps_index]

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
