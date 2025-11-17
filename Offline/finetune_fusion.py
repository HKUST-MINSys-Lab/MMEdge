import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from data.lipreading_dataset import LipreadingDataset
from models.ResNet_3D import Video_Encoder
from models.Audio_ResNet import (
    AudioEncoder,
    AudioModelLightweightEncoder,
)
from models.Fusion import MultiModalFusion


# -----------------------------------------------------
# Weight Init
# -----------------------------------------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# -----------------------------------------------------
# Train One Epoch
# -----------------------------------------------------
def train_epoch(models, loader, criterion, optimizer, device):
    audio_model, video_model, fusion_model = models

    audio_model.train()
    video_model.train()
    fusion_model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Training", unit="batch")

    for video_inputs, audio_inputs, labels in progress:
        video_inputs = video_inputs.to(device)
        audio_inputs = audio_inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # feature extraction
        video_feat = video_model(video_inputs)
        audio_feat = audio_model(audio_inputs)

        outputs = fusion_model(video_feat, audio_feat)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(
            loss=f"{running_loss/total:.4f}",
            acc=f"{100 * correct/total:.2f}%"
        )

    return running_loss / total, 100 * correct / total


# -----------------------------------------------------
# Evaluation
# -----------------------------------------------------
def evaluate(models, loader, criterion, device):
    audio_model, video_model, fusion_model = models

    audio_model.eval()
    video_model.eval()
    fusion_model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Validating", unit="batch")

    with torch.no_grad():
        for video_inputs, audio_inputs, labels in progress:
            video_inputs = video_inputs.to(device)
            audio_inputs = audio_inputs.to(device)
            labels = labels.to(device)

            video_feat = video_model(video_inputs)
            audio_feat = audio_model(audio_inputs)
            outputs = fusion_model(video_feat, audio_feat)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)

            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            progress.set_postfix(
                loss=f"{running_loss/total:.4f}",
                acc=f"{100 * correct/total:.2f}%"
            )

    return running_loss / total, 100 * correct / total


# -----------------------------------------------------
# Full Training Loop
# -----------------------------------------------------
def train(models, train_loader, val_loader, criterion,
          optimizer, scheduler, device, epochs, save_path):

    best_acc = 0.0

    for epoch in range(epochs):
        print(f"\n===== Epoch [{epoch+1}/{epochs}] =====")

        train_loss, train_acc = train_epoch(models, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(models, val_loader, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(models[-1].state_dict(), save_path)
            print(f"✔ Best Fusion Model Saved: {best_acc:.2f}%")

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Best Acc: {best_acc:.2f}%"
        )

        scheduler.step()

    print("\nTraining Complete.")


# -----------------------------------------------------
# Audio Backbone Loader
# -----------------------------------------------------
def build_audio_model(name, num_classes):
    if name == "small":
        return AudioModelLightweightEncoder()
    elif name == "medium":
        return AudioEncoder(size="medium", num_classes=num_classes)
    elif name == "large":
        return AudioEncoder(size="large", num_classes=num_classes)
    else:
        raise ValueError(f"Unknown audio backbone: {name}")


# -----------------------------------------------------
# Video Backbone Loader
# -----------------------------------------------------
def build_video_model(name, num_classes):
    if name == "resnet18":
        return Video_Encoder("resnet18", pretrain=False, feature_dim=512, num_classes=num_classes)
    elif name == "resnet34":
        return Video_Encoder("resnet34", pretrain=False, feature_dim=512, num_classes=num_classes)
    elif name == "resnet50":
        return Video_Encoder("resnet50", pretrain=False, feature_dim=2048, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown video backbone: {name}")


# -----------------------------------------------------
# Main
# -----------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data/lipread_feature")
    parser.add_argument("--label_file", type=str, default="./data/selected_words.txt")

    parser.add_argument("--audio_backbone", type=str,
                        choices=["small", "medium", "large"],
                        default="medium")
    parser.add_argument("--video_backbone", type=str,
                        choices=["resnet18", "resnet34", "resnet50"],
                        default="resnet18")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=50)

    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()

    if args.save_path is None:
        os.makedirs("./checkpoints/fusion", exist_ok=True)
        args.save_path = f"./checkpoints/fusion/fusion_{args.video_backbone}_{args.audio_backbone}.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------
    # Transforms
    # -------------------------------------------------
    train_transform = T.Compose([
        T.ToTensor(),
        T.RandomCrop((88, 88)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        T.ToTensor(),
        T.CenterCrop((88, 88)),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------
    train_dataset = LipreadingDataset(
        root_dir=args.data_root,
        label_file=args.label_file,
        mode="train",
        video_transform=train_transform,
        sample_cnt=500,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )

    val_dataset = LipreadingDataset(
        root_dir=args.data_root,
        label_file=args.label_file,
        mode="val",
        video_transform=test_transform,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    # -------------------------------------------------
    # Models
    # -------------------------------------------------
    audio_model = build_audio_model(args.audio_backbone, args.num_classes)
    video_model = build_video_model(args.video_backbone, args.num_classes)
    fusion_model = MultiModalFusion()

    # ---- load pretrained audio ----
    audio_pre = torch.load(f"checkpoints/audio/audio_{args.audio_backbone}.pth")
    audio_pre = {k.replace("module.", ""): v for k, v in audio_pre.items()}
    audio_pre = {k: v for k, v in audio_pre.items() if not k.startswith("fc.")}
    audio_model.load_state_dict(audio_pre, strict=False)

    # ---- load pretrained video ----
    video_pre = torch.load(f"checkpoints/video/video_{args.video_backbone}.pth")
    video_pre = {k: v for k, v in video_pre.items() if not k.startswith("fc.")}
    video_model.load_state_dict(video_pre, strict=False)

    models = [
        audio_model.to(device),
        video_model.to(device),
        fusion_model.to(device),
    ]

    # fusion-only optimizer
    optimizer = optim.Adam(fusion_model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # -------------------------------------------------
    # Train
    # -------------------------------------------------
    train(
        models, train_loader, val_loader, criterion,
        optimizer, scheduler, device,
        epochs=args.epochs, save_path=args.save_path
    )
