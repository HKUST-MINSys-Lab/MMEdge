import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from data.lipreading_dataset import LipreadingDataset
# from models.ResNet_2D import Video_Classification, SlowFast_2_Branch
from models.ResNet_3D import Video_Classification as Video_Classification_3D
from models.Video_ResNet import Video_ResNet_P3D_Encoder, get_resnet_backbone, Video_ResNet_P3D
from models.Audio_ResNet import AudioEncoder
from models.Video_ResNet_Ablation import Video_ResNet_Baseline


def evenly_spaced_indices(total_range, num_indices):
    if num_indices > total_range:
        raise ValueError("num_indices cannot be greater than total_range")
    return np.linspace(0, total_range - 1, num=num_indices, dtype=int)


def evaluate(model, val_loader, criterion, device):
    model.eval()

    overall_loss = 0.0
    overall_correct = 0
    overall_total = 0

    progress_bar = tqdm(val_loader, desc="Validating", unit="batch")

    with torch.no_grad():
        for video_inputs, _, labels in progress_bar:
            video_inputs, labels = video_inputs.to(device), labels.to(device)
            
            indices = evenly_spaced_indices(total_range=29, num_indices=29)

            video_inputs = video_inputs[:, indices, :, :, :]

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
            
    overall_loss /= overall_total
    overall_acc = (overall_correct / overall_total) * 100.0
    print(f"Full Model: Loss = {overall_loss:.4f}, Accuracy = {overall_acc:.2f}%\n")

    return overall_loss, overall_acc


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    data_root = '/data/rxhuang/lipread_feature'
    label_file = './data/selected_words.txt'

    test_transform = transforms.Compose([
        T.ToTensor(),
        T.CenterCrop((88, 88)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

   
    test_dataset = LipreadingDataset(root_dir=data_root, label_file=label_file, video_transform=test_transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=32)


    num_classes = 500
    
    backbone, feat_dim = get_resnet_backbone('resnet50', pretrained=False)
    model = Video_ResNet_Baseline(backbone, feature_dim=feat_dim, num_classes=num_classes)
    # model = Video_ResNet_P3D(backbone, feature_dim=feat_dim)
    # model.apply(init_weights)
    # model = Video_Classification_3D(feature_dim=512, num_classes=num_classes)
    video_pretrained_dict = torch.load('checkpoints/LRW/baselines/video_3D_baseline.pth')
    # video_pretrained_dict = torch.load('checkpoints/video/video_resnet_50.pth')
    # video_pretrained_dict = torch.load('checkpoints/baselines/video_resnet50_3D.pth')
    video_pretrained_dict = {k.replace('module.', ''): v for k, v in video_pretrained_dict.items()}
    # video_filtered_dict = {k: v for k, v in video_pretrained_dict.items() if not k.startswith('fc.')}
    model.load_state_dict(video_pretrained_dict)

    model = model.to(device)
    model = nn.DataParallel(model)  
        
    # **损失函数**
    criterion = nn.CrossEntropyLoss()

    # **学习率调度器**
    epochs = 100
    
    # 训练
    evaluate(model, test_loader, criterion, device)
