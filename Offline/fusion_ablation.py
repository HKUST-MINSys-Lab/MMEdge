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
from models.Video_ResNet import Video_ResNet_P3D_Encoder, get_resnet_backbone
from models.Audio_ResNet import AudioEncoder, AudioModelBaselineEncoder
from models.Fusion import MultiModalFusion
from models.ResNet_3D import Video_Classification
from models.Video_ResNet_Ablation import Video_ResNet_Baseline_Encoder, Video_ResNet_P3D_Without_Shift, Video_ResNet_P3D_Without_SE, Video_ResNet_P3D_Without_Diff


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
def train_epoch(models, train_loader, criterion, optimizer, device):
    audio_model, video_model, fusion_model = models
    video_model.train()
    audio_model.train()
    fusion_model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for video_inputs, audio_inputs, labels in progress_bar:
        video_inputs, audio_inputs, labels = video_inputs.to(device), audio_inputs.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        with torch.no_grad():
            video_feature = video_model(video_inputs)
            audio_feature = audio_model(audio_inputs)
        video_feature = video_model(video_inputs)
        audio_feature = audio_model(audio_inputs)
        outputs = fusion_model(video_feature, audio_feature)
        
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


def evaluate(models, val_loader, criterion, device):
    audio_model, video_model, fusion_model = models
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
def train(models, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=10, save_path=None):
    audio_model, video_model, fusion_model = models
    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(models, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(models, val_loader, criterion, device)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(models[-1].state_dict(), save_path)
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
    train_loader = DataLoader(train_dataset, batch_size=192, shuffle=True, num_workers=32)
    
    test_dataset = LipreadingDataset(root_dir=data_root, label_file=label_file, video_transform=test_transform, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=192, shuffle=False, num_workers=32)


    num_classes = 50  

    # model_config_video = 18
    model_config_video = 'ablation'
    model_config_audio = 'large'
    model_config_fusion =  str(model_config_video) + '_' + model_config_audio

    print(f"Model Config: Video: ResNet-{model_config_video}")
    print(f"Model Config: Audio: ResNet-{model_config_audio}")
    
    ablation = 'diff'
    pretrain = 'scratch'
    print(f"{ablation} model with {pretrain} weights")
    
    audio_model = AudioEncoder(size=model_config_audio)
    # backbone, feat_dim = get_resnet_backbone(f'resnet{model_config_video}', pretrained=True)
    # video_model = Video_ResNet_P3D_Encoder(backbone, feature_dim=feat_dim)
    # fusion_model = MultiModalFusion()

    audio_pretrained_dict = torch.load(f'checkpoints/audio/audio_{model_config_audio}.pth')
    audio_pretrained_dict = {k.replace('module.', ''): v for k, v in audio_pretrained_dict.items()}
    audio_filtered_dict = {k: v for k, v in audio_pretrained_dict.items() if not k.startswith('fc.')}
    # video_pretrained_dict = torch.load(f'checkpoints/video_resnet_{model_config_video}.pth')
    # video_pretrained_dict = {k.replace('module.', ''): v for k, v in video_pretrained_dict.items()}
    # video_filtered_dict = {k: v for k, v in video_pretrained_dict.items() if not k.startswith('fc.')}
    
    # audio_model = AudioModelBaselineEncoder()
    # video_model = Video_Classification()
    backbone, feat_dim = get_resnet_backbone(f'resnet50', pretrained=False)
    # video_model = Video_ResNet_Baseline_Encoder(backbone)
    # video_model = Video_ResNet_P3D_Without_SE(backbone, feature_dim=feat_dim)
    video_model = Video_ResNet_P3D_Without_Diff(backbone, feature_dim=feat_dim)
    fusion_model = MultiModalFusion()
    
    # audio_pretrained_dict = torch.load(f'checkpoints/baselines/audio_baseline.pth')
    # audio_filtered_dict = {k: v for k, v in audio_pretrained_dict.items() if not k.startswith('fc.')}
    video_pretrained_dict = torch.load(f'checkpoints/ablation_study/video_resnet50_without_{ablation}_{pretrain}.pth')
    video_filtered_dict = {k: v for k, v in video_pretrained_dict.items() if not k.startswith('fc.')}

    audio_model.load_state_dict(audio_filtered_dict, strict=False)
    video_model.load_state_dict(video_filtered_dict, strict=False)

    models = [audio_model, video_model, fusion_model]

    for i in range(len(models)):
        models[i] = models[i].to(device)
        models[i] = nn.DataParallel(models[i])
        
    optimizer = optim.Adam(fusion_model.parameters(), lr=1e-4, weight_decay=1e-5)

    # **损失函数**
    criterion = nn.CrossEntropyLoss()

    # **学习率调度器**
    epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # 训练
    # save_path = f"./checkpoints/fusion/fusion_{model_config_fusion}.pth"
    save_path = f"./checkpoints/ablation_study/fusion_{ablation}_{pretrain}.pth"
    train(models, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=epochs, save_path=save_path)
