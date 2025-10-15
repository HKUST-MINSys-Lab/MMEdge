import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'
os.environ["CUDA_VISIBLE_DEVICES"] = '4, 6'
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from data.lipreading_dataset import LipreadingDataset
from models.Video_ResNet import Video_ResNet_P3D, get_resnet_backbone
from models.Video_ResNet_Ablation import Video_ResNet_P3D_Without_Shift, Video_ResNet_P3D_Without_Diff, Video_ResNet_P3D_Without_SE, Video_ResNet_Baseline
from models.ResNet_3D import Video_Classification


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

    for video_inputs, _, labels in progress_bar:
        video_inputs, labels = video_inputs.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()
        
        outputs = model(video_inputs)
        # 计算损失（这里 fusion_criterion 主要用于最终分类任务）
        loss = criterion(outputs, labels)
        loss.backward()

        # 反向传播 + 更新权重
        optimizer.step()

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

    overall_loss = 0.0
    overall_correct = 0
    overall_total = 0

    progress_bar = tqdm(val_loader, desc="Validating", unit="batch")

    with torch.no_grad():
        for video_inputs, _, labels in progress_bar:
            video_inputs, labels = video_inputs.to(device), labels.to(device)

            # 计算完整模型的loss和accuracy
            outputs = model(video_inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            overall_loss += loss.item() * labels.size(0)
            overall_correct += (predicted == labels).sum().item()
            overall_total += labels.size(0)

            # 更新 tqdm 进度条，显示完整模型的结果
            current_loss = overall_loss / overall_total
            current_acc = (overall_correct / overall_total) * 100.0
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    # 计算并打印所有分支的最终结果
    overall_loss /= overall_total
    overall_acc = (overall_correct / overall_total) * 100.0

    return overall_loss, overall_acc


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
        # T.ColorJitter(brightness=0.2, contrast=0.2),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        T.ToTensor(),
        T.CenterCrop((88, 88)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = LipreadingDataset(root_dir=data_root, label_file=label_file, mode='train', video_transform=train_transform, sample_cnt=500)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=32)
    
    test_dataset = LipreadingDataset(root_dir=data_root, label_file=label_file, video_transform=test_transform, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=32)


    num_classes = 50
    
    # model = Video_ResNet_Baseline(resnet50(pretrained=True), feature_dim=2048, num_classes=num_classes)
    # backbone, feat_dim = get_resnet_backbone('resnet50', pretrained=False)
    # model = Video_ResNet_Baseline(backbone, feature_dim=feat_dim, num_classes=num_classes)
    # model = Video_ResNet_P3D(backbone, feature_dim=feat_dim, num_classes=num_classes)
    # model = Video_ResNet_P3D_Without_SE(backbone, feature_dim=feat_dim, num_classes=num_classes)
    # model = Video_ResNet_P3D_Without_Diff(backbone, feature_dim=feat_dim, num_classes=num_classes)
    # model = Video_ResNet_P3D_Frame_Shift(backbone, feature_dim=feat_dim, num_classes=num_classes)
    
    # model.apply(init_weights)
    model = Video_Classification(pretrain=False, feature_dim=512, num_classes=num_classes)
    # video_pretrained_dict = torch.load('checkpoints/video_only_branch.pth')
    # new_state_dict = {k.replace('module.', ''): v for k, v in video_pretrained_dict.items()}
    # model.load_state_dict(new_state_dict)

    model = model.to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    model = nn.DataParallel(model)

    # **损失函数**
    criterion = nn.CrossEntropyLoss()

    # **学习率调度器**
    epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # 训练
    train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=epochs, save_path="./checkpoints/baselines/model_selection/video_model.pth")
