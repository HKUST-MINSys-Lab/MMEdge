import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import argparse
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
from models.Audio_ResNet import AudioModel
from models.Fusion import MultiModalFusion


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


def compute_cosine_similarity(audio_feat, video_feat):
    audio_feat = F.normalize(audio_feat, dim=-1)
    video_feat = F.normalize(video_feat, dim=-1)
    return (audio_feat * video_feat).sum(dim=-1)


video_model_map = ['18', '34', '50']
audio_model_map = ['small', 'medium', 'large']

def train_accuracy_predictor(audio_models, video_models, fusion_models, test_loader, device, accuracy_table, args):
    predictor = AccuracyPredictor(input_dim=6, hidden_dim=64, dropout=0.3).to(device)
    if args.load_ckpt is not None:
        print(f"Loading checkpoint from {args.load_ckpt}...")
        ckpt = torch.load(args.load_ckpt, map_location=device)
        predictor.load_state_dict(ckpt, strict=False)

    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    predictor.train()

    min_loss = float('inf')

    for epoch in range(args.epochs):
        total_loss = 0.0
        sample_count = 0
        progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}", unit="batch")
        for i, (video_input, audio_input, ground_truth) in enumerate(progress_bar):
            video_input, audio_input, ground_truth = video_input.to(device), audio_input.to(device), ground_truth.to(device)

            audio_encoder = audio_models[0].encoder
            video_encoder = video_models[0].spatial_encoder

            audio_chunk = audio_input[:, :800]
            audio_feat = audio_encoder(audio_chunk).detach()

            video_frame = video_input[:, 0:1]
            B, _, C, H, W = video_frame.shape
            video_feat = video_encoder(video_frame.view(B, C, H, W)).squeeze(-1).squeeze(-1).detach()

            cos_sim = compute_cosine_similarity(audio_feat, video_feat)
            modality_diff = 1 - cos_sim

            for v_idx, video_model in enumerate(video_models):
                for video_fps in [20, 25, 29]:
                    for a_idx, audio_model in enumerate(audio_models):
                        for audio_chunk_size in [1200, 1000, 800]:
                            with torch.no_grad():
                                audio_model.classifier = nn.Identity()
                                video_model.classifier = nn.Identity()
                                audio_feature = audio_model(audio_input)
                                video_feature = video_model(video_input)

                                fusion_model = fusion_models[a_idx * 3 + v_idx]
                                logits_ground_truth = fusion_model(video_feature, audio_feature).unsqueeze(1)

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

                            logits_pred, acc_pred = predictor(input_vec)
                            B = ground_truth.shape[0]

                            accuracy_labels = torch.tensor([
                                float(
                                    accuracy_table
                                    .get(video_model_map[v_idx], {})
                                    .get(str(video_fps), {})
                                    .get(audio_model_map[a_idx], {})
                                    .get(str(audio_chunk_size), {})
                                    .get(str(ground_truth[i].item()), 0.0)
                                ) for i in range(B)
                            ], dtype=torch.float32).unsqueeze(1).to(device)

                            acc_loss = F.mse_loss(acc_pred, accuracy_labels) * 0.7
                            logits_loss = F.mse_loss(logits_pred, logits_ground_truth) * 0.3
                            loss = acc_loss + logits_loss

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            total_loss += loss.item() * B
                            sample_count += B

        avg_loss = total_loss / sample_count
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}, Total Loss = {total_loss:.4f}")
        if avg_loss < min_loss:
            min_loss = avg_loss
            print(f"New best model found at epoch {epoch + 1} with loss {min_loss:.4f}")
            torch.save(predictor.state_dict(), args.save_path)
            print(f"Best accuracy predictor saved at {args.save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./checkpoints/accuracy_predictor_logits.pth')
    parser.add_argument('--load_ckpt', type=str, default=None)
    parser.add_argument('--data_root', type=str, default='./data/lipread_feature')
    parser.add_argument('--label_file', type=str, default='./data/selected_words.txt')
    parser.add_argument('--acc_table', type=str, default='./data/accuracy_table.json')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        T.ToTensor(),
        T.CenterCrop((88, 88)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = LipreadingDataset(
        root_dir=args.data_root,
        label_file=args.label_file,
        video_transform=test_transform,
        mode='val'
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32)

    accuracy_table = json.load(open(args.acc_table, "r"))

    num_classes = 50
    audio_models = []
    video_models = []
    fusion_models = []

    for size in ['small', 'medium', 'large']:
        audio_model = AudioModel(size=size)
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

    for audio in ['small', 'medium', 'large']:
        for video in ['18', '34', '50']:
            fusion_model = MultiModalFusion()
            fusion_pretrained_dict = torch.load(f'checkpoints/fusion/fusion_{video}_{audio}.pth')
            fusion_filtered_dict = {k.replace('module.', ''): v for k, v in fusion_pretrained_dict.items()}
            fusion_model.load_state_dict(fusion_filtered_dict, strict=False)
            fusion_model = fusion_model.to(device)
            fusion_model.eval()
            fusion_models.append(fusion_model)

    train_accuracy_predictor(audio_models, video_models, fusion_models, test_loader, device, accuracy_table, args)
