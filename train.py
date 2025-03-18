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
    video_model.train()
    audio_model.train()
    fusion_model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for video_inputs, audio_inputs, labels in progress_bar:
        video_inputs, audio_inputs, labels = video_inputs.to(device), audio_inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        video_feature = video_model(video_inputs)
        audio_feature = audio_model(audio_inputs)
        outputs = fusion_model(video_feature, audio_feature)
        
        loss = criterion(outputs, labels)
        loss.backward()
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
def train(models, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=10, save_paths=["video.pth", "audio.pth", "fusion.pth"]):
    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(models, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(models, val_loader, criterion, device)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            for model, save_path in zip(models, save_paths):
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
        models[i] = nn.DataParallel(models[i])  # 直接修改列表元素

    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(video_model.parameters()) + list(audio_model.parameters()) + list(fusion_model.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )

    # **学习率调度器**
    epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    save_paths = ["./checkpoints/video_pretrain.pth", "./checkpoints/audio_pretrain.pth", "./checkpoints/fusion_pretrain.pth"]
    
    # 训练
    train(models, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=epochs, save_paths=save_paths)
