import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from ucf101_dataset import VideoDataset
from models.mobilenet_3d import MobileNet3D, X3D_MultiScale_TemporalAttention


# 训练函数
def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 计算 loss
        running_loss += loss.item() * inputs.size(0)  # 计算整个 batch 的 loss 总和

        # 计算 accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # 计算总体累计 loss 和 accuracy（基于所有 batch）
        current_loss = running_loss / total
        current_acc = (correct / total) * 100.0  # 计算所有已处理样本的累计准确率

        # 更新 tqdm 进度条
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    train_loss = running_loss / total
    train_acc = correct / total * 100.0
    return train_loss, train_acc


def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(val_loader, desc="Validating", unit="batch")

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 计算所有 batch 的累计 loss 和 accuracy
            current_loss = running_loss / total
            current_acc = (correct / total) * 100.0

            # 更新 tqdm 进度条
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    val_loss = running_loss / total
    val_acc = correct / total * 100.0
    return val_loss, val_acc


# 训练循环
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=10, save_path="model.pth"):
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with accuracy: {best_val_acc:.4f}")

    print("\nTraining Complete.")
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 示例
    root_dir = "/data/rxhuang/UCF-101"  # 数据集根目录
    train_file_list = "/data/rxhuang/ucfTrainTestlist/trainlist01.txt"  # 包含视频路径和标签的文件
    test_file_list = "/data/rxhuang/ucfTrainTestlist/testlist01.txt"
    label_file = "/data/rxhuang/ucfTrainTestlist/classInd.txt"
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = VideoDataset(file_list=train_file_list, root_dir=root_dir, frames_per_clip=32, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16)
    
    test_dataset = VideoDataset(file_list=test_file_list, root_dir=root_dir, frames_per_clip=32, transform=test_transform, label_file=label_file)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=16)

    # 模型
    model = X3D_MultiScale_TemporalAttention(num_classes=101).to(device)

    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # **学习率调度器**
    epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # 训练
    train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=epochs, save_path="./checkpoints/x3d_attention.pth")
