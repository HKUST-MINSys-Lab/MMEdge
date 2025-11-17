import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from data.lipreading_dataset import LipreadingDataset
from models.Video_ResNet import Video_ResNet_P3D, get_resnet_backbone


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for video_inputs, _, labels in progress_bar:
        video_inputs, labels = video_inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(video_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        current_loss = running_loss / total
        current_acc = (correct / total) * 100.0
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    return running_loss / total, correct / total * 100.0


def evaluate(model, val_loader, criterion, device):
    model.eval()
    overall_loss = 0.0
    overall_correct = 0
    overall_total = 0
    progress_bar = tqdm(val_loader, desc="Validating", unit="batch")

    with torch.no_grad():
        for video_inputs, _, labels in progress_bar:
            video_inputs, labels = video_inputs.to(device), labels.to(device)
            outputs = model(video_inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            overall_loss += loss.item() * labels.size(0)
            overall_correct += (predicted == labels).sum().item()
            overall_total += labels.size(0)
            current_loss = overall_loss / overall_total
            current_acc = (overall_correct / overall_total) * 100.0
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    return overall_loss / overall_total, (overall_correct / overall_total) * 100.0


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=10, save_path=None):
    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with accuracy: {best_val_acc:.2f}%")

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Best Acc: {best_val_acc:.2f}%")
        scheduler.step()

    print("\nTraining Complete.")


def get_model(backbone_name, num_classes):
    backbone, feat_dim = get_resnet_backbone(backbone_name, pretrained=False)
    return Video_ResNet_P3D(backbone, feature_dim=feat_dim, num_classes=num_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/lipread_feature')
    parser.add_argument('--label_file', type=str, default='./data/selected_words.txt')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'resnet34', 'resnet50'], default='resnet50')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_classes', type=int, default=50)
    args = parser.parse_args()

    if args.save_path is None:
        os.makedirs('./checkpoints', exist_ok=True)
        suffix = args.backbone[-2:] if args.backbone != 'resnet50' else '50'
        args.save_path = f'./checkpoints/video_{suffix}.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = T.Compose([
        T.ToTensor(),
        T.RandomCrop((88, 88)),
        T.RandomHorizontalFlip(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        T.ToTensor(),
        T.CenterCrop((88, 88)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = LipreadingDataset(root_dir=args.data_root, label_file=args.label_file,
                                      mode='train', video_transform=train_transform, sample_cnt=500)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_dataset = LipreadingDataset(root_dir=args.data_root, label_file=args.label_file,
                                     video_transform=test_transform, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_model(args.backbone, num_classes=args.num_classes)
    model = model.to(device)
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    train(model, train_loader, test_loader, criterion, optimizer, scheduler, device,
          epochs=args.epochs, save_path=args.save_path)
