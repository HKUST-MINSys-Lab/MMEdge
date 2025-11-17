import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from data.lipreading_dataset import LipreadingDataset
from models.ResNet_3D import Video_Encoder
from models.Audio_ResNet import AudioEncoder
from models.Fusion import MultiModalFusion


# -----------------------------------------------------
# Early-Exit Classifier
# -----------------------------------------------------
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
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# -----------------------------------------------------
# evenly spaced selection
# -----------------------------------------------------
def evenly_spaced_indices(total_range, num_indices):
    return np.linspace(0, total_range - 1, num=num_indices, dtype=int)


# -----------------------------------------------------
# Load Video Encoders (18/34/50)
# -----------------------------------------------------
def load_video_models(ckpt_dir, device):
    models = []
    for backbone in ["resnet18", "resnet34", "resnet50"]:
        model = Video_Encoder(backbone, pretrain=False)

        ckpt_path = f"{ckpt_dir}/video/video_{backbone}.pth"
        ckpt = torch.load(ckpt_path)
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        ckpt = {k: v for k, v in ckpt.items() if not k.startswith("fc.")}

        model.load_state_dict(ckpt, strict=False)
        model = nn.DataParallel(model).to(device).eval()
        models.append(model)

    return models


# -----------------------------------------------------
# Load Audio Encoders (small/medium/large)
# -----------------------------------------------------
def load_audio_models(ckpt_dir, device):
    models = []
    for size in ["small", "medium", "large"]:
        model = AudioEncoder(size=size)

        ckpt_path = f"{ckpt_dir}/audio/audio_{size}.pth"
        ckpt = torch.load(ckpt_path)
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        ckpt = {k: v for k, v in ckpt.items() if not k.startswith("fc.")}

        model.load_state_dict(ckpt, strict=False)
        model = nn.DataParallel(model).to(device).eval()
        models.append(model)

    return models


# -----------------------------------------------------
# Load Fusion Models (9 combinations)
# -----------------------------------------------------
def load_fusion_models(ckpt_dir, device):
    fusion_models = []
    for video in ["resnet18", "resnet34", "resnet50"]:
        for audio in ["small", "medium", "large"]:
            model = MultiModalFusion()

            ckpt_path = f"{ckpt_dir}/fusion/fusion_{video}_{audio}.pth"
            ckpt = torch.load(ckpt_path)
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

            model.load_state_dict(ckpt, strict=False)
            model = nn.DataParallel(model).to(device).eval()
            fusion_models.append(model)

    return fusion_models


# -----------------------------------------------------
# Train Gating Classifier
# -----------------------------------------------------
def train_gating(video_models, audio_models, fusion_models, test_loader, device):

    gating = EarlyExitClassifier().to(device)
    optimizer = optim.Adam(gating.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    gating.train()

    best_loss = float("inf")

    for epoch in range(50):
        total_loss = 0.0
        correct_total = 0
        sample_total = 0

        progress = tqdm(test_loader, desc=f"Gating Epoch {epoch+1}", unit="batch")

        for video_input, audio_input, ground_truth in progress:
            video_input = video_input.to(device)
            audio_input = audio_input.to(device)
            ground_truth = ground_truth.to(device)

            B = video_input.size(0)

            for vid_idx, video_model in enumerate(video_models):
                for fps in [20, 25, 29]:

                    full_indices = evenly_spaced_indices(29, fps)

                    for ratio in [0.5, 0.7, 0.9]:
                        partial_len = max(1, int(fps * ratio))
                        indices = full_indices[:partial_len]
                        video_partial = video_input[:, indices]

                        for aud_idx, audio_model in enumerate(audio_models):
                            for chunk in [800, 1000, 1200]:

                                # dynamic chunking
                                audio_model.module.chunk_size = chunk
                                audio_model.module.chunk_interval = chunk

                                fusion_idx = vid_idx * 3 + aud_idx
                                fusion_model = fusion_models[fusion_idx]

                                with torch.no_grad():
                                    v_feat = video_model(video_partial)
                                    a_feat = audio_model(audio_input)
                                    logits = fusion_model(v_feat, a_feat)

                                    _, pred = logits.max(1)
                                    label = (pred == ground_truth).float()

                                gating_input = torch.cat([v_feat, a_feat], dim=1)
                                gating_output = gating(gating_input)

                                loss = criterion(gating_output, label)

                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                                total_loss += loss.item() * B
                                sample_total += B

                                pred_exit = (gating_output >= 0.5).float()
                                correct_total += (pred_exit == label).float().sum().item()

            avg_loss = total_loss / sample_total
            gating_acc = correct_total / sample_total * 100

            progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{gating_acc:.2f}%")

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={gating_acc:.2f}%")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(gating.state_dict(), "./checkpoints/gating/gating_classifier.pth")
            print(f"✔ Saved best gating classifier at epoch {epoch+1}")


# -----------------------------------------------------
# Main
# -----------------------------------------------------
if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints")

    args = parser.parse_args()

    device = args.device

    # Dataset
    test_transform = transforms.Compose([
        T.ToTensor(),
        T.CenterCrop((88, 88)),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    data_root = "/data/rxhuang/lipread_feature"
    label_file = "./data/selected_words.txt"

    test_dataset = LipreadingDataset(
        root_dir=data_root,
        label_file=label_file,
        video_transform=test_transform,
        mode="val"
    )
    test_loader = DataLoader(test_dataset, batch_size=192, shuffle=False, num_workers=32)

    # Load only what we need
    video_models = load_video_models(args.checkpoint_path, device)
    audio_models = load_audio_models(args.checkpoint_path, device)
    fusion_models = load_fusion_models(args.checkpoint_path, device)

    # Train Gating
    train_gating(video_models, audio_models, fusion_models, test_loader, device)
