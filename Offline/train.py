import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from data.lipreading_dataset import LipreadingDataset
from models.ResNet_2D import Video_ResNet_2D
from models.Audio_ResNet import AudioEncoderModel
from models.Fusion import BimodalFusion


# 训练函数
def train_epoch(models, train_loader, criterion, optimizer, device):
    video_model, audio_model, fusion_model = models
    video_criterion, audio_criterion, fusion_criterion = criterions
    video_optimizer, audio_optimizer, fusion_optimizer = optimizers
    
    video_model.train()
    audio_model.train()
    fusion_model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for video_inputs, audio_inputs, labels in progress_bar:
        video_inputs, audio_inputs, labels = video_inputs.to(device), audio_inputs.to(device), labels.to(device)

        # 梯度清零（分别针对不同的优化器）
        video_optimizer.zero_grad()
        audio_optimizer.zero_grad()
        fusion_optimizer.zero_grad()
        
        video_feature = video_model(video_inputs)
        audio_feature = audio_model(audio_inputs)
        outputs = fusion_model(video_feature, audio_feature)
        print(f"Train outputs: {outputs.shape}: {outputs}")
        
        # 计算损失（这里 fusion_criterion 主要用于最终分类任务）
        loss = fusion_criterion(outputs, labels)
        loss.backward()

        # 反向传播 + 更新权重
        video_optimizer.step()
        audio_optimizer.step()
        fusion_optimizer.step()

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


def evaluate(models, val_loader, criterion, device):
    video_model, audio_model, fusion_model = models
    video_model.eval()
    audio_model.eval()
    fusion_model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(val_loader, desc="Validating", unit="batch")

    with torch.no_grad():
        for video_inputs, audio_inputs, labels in progress_bar:
            video_inputs, audio_inputs, labels = video_inputs.to(device), audio_inputs.to(device), labels.to(device)
            
            video_feature = video_model(video_inputs)
            audio_feature = audio_model(audio_inputs)
            outputs = fusion_model(video_feature, audio_feature)
            print(f"Val outputs: {outputs.shape}: {outputs}")
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
def train(models, train_loader, val_loader, criterions, optimizers, schedulers, device, epochs=10, save_paths=["video.pth", "audio.pth", "fusion.pth"]):
    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(models, train_loader, criterions, optimizers, device)
        val_loss, val_acc = evaluate(models, val_loader, criterions[-1], device)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            for model, save_path in zip(models, save_paths):
                torch.save(model.state_dict(), save_path)
            print(f"Best model saved with accuracy: {best_val_acc:.2f}%")

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Best Acc: {best_val_acc:.2f}%")
        
        for scheduler in schedulers:
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
        T.ColorJitter(brightness=0.5, contrast=0.5),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        T.ToTensor(),
        T.CenterCrop((88, 88)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = LipreadingDataset(root_dir=data_root, label_file=label_file, mode='train', video_transform=train_transform, sample_cnt=500)
    train_loader = DataLoader(train_dataset, batch_size=192, shuffle=True, num_workers=16)
    
    test_dataset = LipreadingDataset(root_dir=data_root, label_file=label_file, video_transform=test_transform, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=192, shuffle=False, num_workers=16)


    num_classes = 50  # SS-V2 dataset
    
    video_model = Video_ResNet_2D()
    audio_model = AudioEncoderModel(resnet_depth=18, chunk_size=800, chunk_interval=400, sample_rate=16000)
    fusion_model = BimodalFusion(modal_dims=(512, 512), num_classes=num_classes)

    models = [video_model, audio_model, fusion_model]
    for i in range(len(models)):
        models[i] = models[i].to(device)
        models[i] = nn.DataParallel(models[i])  
        
    # **分别定义优化器**
    video_optimizer = optim.Adam(video_model.parameters(), lr=3e-4, weight_decay=1e-5)
    audio_optimizer = optim.Adam(audio_model.parameters(), lr=1e-4, weight_decay=1e-5)
    fusion_optimizer = optim.Adam(fusion_model.parameters(), lr=5e-4, weight_decay=1e-5)

    optimizers = [video_optimizer, audio_optimizer, fusion_optimizer]

    # **损失函数**
    criterions = [nn.CrossEntropyLoss()] * 3

    # **学习率调度器**
    epochs = 100
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(video_optimizer, T_max=epochs, eta_min=1e-5),
        torch.optim.lr_scheduler.CosineAnnealingLR(audio_optimizer, T_max=epochs, eta_min=1e-5),
        torch.optim.lr_scheduler.CosineAnnealingLR(fusion_optimizer, T_max=epochs, eta_min=1e-5),
    ]

    save_paths = ["./checkpoints/video_pretrain.pth", "./checkpoints/audio_pretrain.pth", "./checkpoints/fusion_pretrain.pth"]
    
    # 训练
    train(models, train_loader, test_loader, criterions, optimizers, schedulers, device, epochs=epochs, save_paths=save_paths)
