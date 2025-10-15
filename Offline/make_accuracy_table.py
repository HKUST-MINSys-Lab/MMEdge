import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'
import csv
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from data.lipreading_dataset import LipreadingDataset
from models.Video_ResNet import Video_ResNet_P3D_Encoder, get_resnet_backbone
from models.Audio_ResNet import AudioEncoder
from models.Fusion import MultiModalFusion


def load_accuracy_table(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return {}


def set_accuracy(table, v_model, fps, a_model, a_chunk, acc):
    v_model = str(v_model)
    fps = str(fps)
    a_chunk = str(a_chunk)

    if v_model not in table:
        table[v_model] = {}
    if fps not in table[v_model]:
        table[v_model][fps] = {}
    if a_model not in table[v_model][fps]:
        table[v_model][fps][a_model] = {}
    
    table[v_model][fps][a_model][a_chunk] = acc
    

def evenly_spaced_indices(total_range, num_indices):
    if num_indices > total_range:
        raise ValueError("num_indices cannot be greater than total_range")
    return np.linspace(0, total_range - 1, num=num_indices, dtype=int)


def evaluate(args, models, val_loader, criterion, num_classes, accuracy_table):
    device = args.device
    audio_model, video_model, fusion_model = models
    video_model.eval()
    audio_model.eval()
    fusion_model.eval()

    overall_loss = 0.0
    overall_correct = 0
    overall_total = 0
    
    # 初始化每个类别的统计
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    progress_bar = tqdm(val_loader, desc="Validating", unit="batch")

    with torch.no_grad():
        for video_inputs, audio_inputs, labels in progress_bar:
            video_inputs, audio_inputs, labels = video_inputs.to(device), audio_inputs.to(device), labels.to(device)
            
            audio_model.chunk_size = args.audio_chunk_size
            audio_model.chunk_interval = args.audio_chunk_size
            video_indices = evenly_spaced_indices(total_range=29, num_indices=args.video_fps)
            video_inputs = video_inputs[:, video_indices, :, :, :]
            
            # print(f"Video fps: {args.video_fps}, Audio Chunk: {args.audio_chunk_size}")
            
            video_feature = video_model(video_inputs)
            audio_feature = audio_model(audio_inputs)
            outputs = fusion_model(video_feature, audio_feature)
            
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            overall_loss += loss.item() * labels.size(0)
            overall_correct += (predicted == labels).sum().item()
            overall_total += labels.size(0)
            
            # 更新类别统计
            for label, pred in zip(labels, predicted):
                label_idx = label.item()
                class_total[label_idx] += 1
                if pred == label:
                    class_correct[label_idx] += 1

            # 更新 tqdm 进度条，显示完整模型的结果
            current_loss = overall_loss / overall_total
            current_acc = (overall_correct / overall_total) * 100.0
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")
            
    overall_loss /= overall_total
    overall_acc = (overall_correct / overall_total) * 100.0
    print(f"Full Model: Loss = {overall_loss:.4f}, Accuracy = {overall_acc:.2f}%\n")
    
    # 生成类别准确率字典
    class_accuracy = {str(i): (class_correct[i]/class_total[i])*100 if class_total[i] > 0 else 0.0 
                     for i in range(num_classes)}
    
    # 加载并更新准确率表
    accuracy_table_path = "accuracy_table.json"
    accuracy_table = load_accuracy_table(accuracy_table_path)
    
    # 保存到JSON结构
    set_accuracy(
        accuracy_table,
        args.video_model,
        args.video_fps,
        args.audio_model,
        args.audio_chunk_size,
        {k: f"{v:.2f}" for k, v in class_accuracy.items()}
    )
    
    # 保存更新后的表
    with open(accuracy_table_path, 'w') as f:
        json.dump(accuracy_table, f, indent=4)

    print(f"\nFull Model: Loss = {overall_loss:.4f}, Accuracy = {overall_acc:.2f}%")
    print("Class Accuracies:")
    for i in range(num_classes):
        print(f"Class {i}: {class_accuracy[str(i)]:.2f}% ({class_correct[i]}/{class_total[i]})")


    return overall_loss, overall_acc


if __name__ == '__main__':
    # **参数设置**
    parser = argparse.ArgumentParser(description="Run RealSense experiment with specific FPS.")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints')
    parser.add_argument("--audio_sample_rate", type=int, default=16000)
    parser.add_argument("--audio_chunk_size", type=int, default=800)
    parser.add_argument("--audio_model", type=str, default='large')
    parser.add_argument("--video_fps", type=int, default=29)
    parser.add_argument("--video_model", type=int, default=50)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    data_root = '/data/rxhuang/lipread_feature'
    label_file = './data/selected_words.txt'

    test_transform = transforms.Compose([
        T.ToTensor(),
        T.CenterCrop((88, 88)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    accuracy_table_path = "accuracy_table.json"
    accuracy_table = load_accuracy_table(accuracy_table_path)

    test_dataset = LipreadingDataset(root_dir=data_root, label_file=label_file, video_transform=test_transform, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=32)


    num_classes = 50 
    
    model_config_video = args.video_model
    model_config_audio = args.audio_model
    model_config_fusion =  str(model_config_video) + '_' + model_config_audio

    print(f"Model Config: Video: ResNet-{model_config_video}")
    print(f"Model Config: Audio: ResNet-{model_config_audio}")

    audio_model = AudioEncoder(size=model_config_audio)
    backbone, feat_dim = get_resnet_backbone(f'resnet{model_config_video}', pretrained=True)
    video_model = Video_ResNet_P3D_Encoder(backbone, feature_dim=feat_dim)
    fusion_model = MultiModalFusion()

    audio_pretrained_dict = torch.load(f'checkpoints/audio/audio_{model_config_audio}.pth')
    audio_pretrained_dict = {k.replace('module.', ''): v for k, v in audio_pretrained_dict.items()}
    audio_filtered_dict = {k: v for k, v in audio_pretrained_dict.items() if not k.startswith('fc.')}
    video_pretrained_dict = torch.load(f'checkpoints/video/video_resnet_{model_config_video}.pth')
    video_pretrained_dict = {k.replace('module.', ''): v for k, v in video_pretrained_dict.items()}
    video_filtered_dict = {k: v for k, v in video_pretrained_dict.items() if not k.startswith('fc.')}
    fusion_pretrained_dict = torch.load(f'./checkpoints/fusion/fusion_{model_config_fusion}.pth')
    fusion_pretrained_dict = {k.replace('module.', ''): v for k, v in fusion_pretrained_dict.items()}
    fusion_filtered_dict = {k: v for k, v in fusion_pretrained_dict.items() if not k.startswith('fc.')}

    audio_model.load_state_dict(audio_filtered_dict, strict=False)
    video_model.load_state_dict(video_filtered_dict, strict=False)
    fusion_model.load_state_dict(fusion_filtered_dict, strict=False)

    models = [audio_model, video_model, fusion_model]

    for i in range(len(models)):
        models[i] = models[i].to(device)
        
    # **损失函数**
    criterion = nn.CrossEntropyLoss()

    # **学习率调度器**
    epochs = 100
    
    # 训练
    evaluate(args, models, test_loader, criterion, num_classes, accuracy_table)
