import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
# os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from data.lipreading_dataset import LipreadingDataset
from models.Audio_ResNet import AudioModel, AudioModelBaseline, AudioModelLightweightEncoder


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)  # 正交初始化更适合深层网络
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# 训练函数
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for _, audio_inputs, labels in progress_bar:
        audio_inputs, labels =  audio_inputs.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()
        
        outputs = model(audio_inputs)
        
        # 计算损失（这里 fusion_criterion 主要用于最终分类任务）
        loss = criterion(outputs, labels)
        loss.backward()

        # 反向传播 + 更新权重
        optimizer.step()

        total_grad = 0

        # 计算 loss
        running_loss += loss.item() * labels.size(0)  # 计算整个 batch 的 loss 总和

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
        for _, audio_inputs, labels in progress_bar:
            audio_inputs, labels = audio_inputs.to(device), labels.to(device)
            
            outputs = model(audio_inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
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
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=10, save_path=None):
    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with accuracy: {best_val_acc:.2f}%")

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Best Acc: {best_val_acc:.2f}%")
        
        scheduler.step()

    print("\nTraining Complete.")
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    data_root = '/data/rxhuang/lipread_feature'
    label_file = './data/selected_words.txt'


    train_transform = T.Compose([
        T.ToTensor(),
        T.RandomCrop((88, 88)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        T.ToTensor(),
        T.CenterCrop((88, 88)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = LipreadingDataset(root_dir=data_root, label_file=label_file, mode='train', video_transform=train_transform, sample_cnt=500)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
    
    test_dataset = LipreadingDataset(root_dir=data_root, label_file=label_file, video_transform=test_transform, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16)


    num_classes = 50
    
    # model = AudioEncoder(size="small", num_classes=num_classes)
    # model = AudioModelWithLSTM(num_classes=50)
    model = AudioModelLightweightEncoder(num_classes=num_classes)
    # model = AudioModelLightweightEncoder(num_classes=num_classes)
    # model = FullSequenceLSTMAudioModel()
    # model.apply(init_weights)

    model = model.to(device)
    # model = nn.DataParallel(model)  
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # **损失函数**
    criterion = nn.CrossEntropyLoss()

    # **学习率调度器**
    epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # 训练
    train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=epochs, save_path="./checkpoints/baselines/model_selection/audio_model.pth")
