import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from data.lipreading_dataset import LipreadingDataset
from models.Audio_ResNet import AudioModelBaselineEncoder
from models.Fusion import MultiModalFusion
from models.ResNet_3D import Video_3D_Encoder
from models.SMG import FastModalityTester


def segment_video_to_frames(video_tensor, num_segments=5):
    """
    将视频张量划分为 num_segments 段，每段取中心帧。
    输入:
        video_tensor: (T, C, H, W)
    输出:
        frames: (num_segments, C, H, W)
    """
    total_frames = video_tensor.shape[0]
    segment_size = total_frames // num_segments
    frames = []
    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_segments - 1 else total_frames
        center_idx = (start + end) // 2
        center_idx = min(center_idx, total_frames - 1)
        frames.append(video_tensor[center_idx])  # (C, H, W)
    return torch.stack(frames, dim=0)  # (num_segments, C, H, W)


def segment_audio_to_chunks(audio_tensor, num_segments=5):
    """
    将音频波形张量均匀划分为 num_segments 段。
    输入:
        audio_tensor: (L,)
    输出:
        chunks: (num_segments, chunk_len), 自动 padding 保证对齐
    """
    total_samples = audio_tensor.shape[0]
    boundaries = torch.linspace(0, total_samples, steps=num_segments + 1, dtype=torch.int32)

    chunks = []
    max_len = 0
    for i in range(num_segments):
        start = boundaries[i].item()
        end = boundaries[i + 1].item()
        chunk = audio_tensor[start:end]
        max_len = max(max_len, chunk.shape[0])
        chunks.append(chunk)

    # padding 到相同长度
    padded_chunks = []
    for chunk in chunks:
        if chunk.shape[0] < max_len:
            pad = torch.zeros(max_len - chunk.shape[0], dtype=chunk.dtype, device=chunk.device)
            chunk = torch.cat([chunk, pad], dim=0)
        padded_chunks.append(chunk)

    return torch.stack(padded_chunks, dim=0)  # (num_segments, chunk_len)



def freeze_fmt(fmt):
    for p in fmt.parameters():
        p.requires_grad = False

def unfreeze_fmt(fmt):
    for p in fmt.parameters():
        p.requires_grad = True


def train_epoch(models, train_loader, criterion, optimizer, device, num_segments=5, penalty_weight=10):
    """
    训练一个 epoch，参考 SMG 中的分类损失 + 效率损失设计。
    """
    audio_model, video_model, fusion_model, fmt = models
    fmt.train()
    video_model.train()
    audio_model.train()
    fusion_model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for video_inputs, audio_inputs, labels in progress_bar:
        batch_size = video_inputs.shape[0]
        video_inputs = video_inputs.to(device)  # (1, T, C, H, W)
        audio_inputs = audio_inputs.to(device)  # (1, L)
        labels = labels.to(device)              # (B,)
        optimizer.zero_grad()


        video_sample = video_inputs.squeeze(0) # (T, C, H, W)
        audio_sample = audio_inputs.squeeze(0)  # (L,)
        label_sample = labels   # (1,)

        T_total = video_sample.shape[0]
        L_total = audio_sample.shape[0]
        seg_len = T_total // num_segments
        chunk_len = L_total // num_segments

        video_selected_segments = []
        audio_selected_segments = []
        segment_decisions = []

        for t in range(num_segments):
            v_start = t * seg_len
            v_end = (t + 1) * seg_len if t < num_segments - 1 else T_total
            a_start = t * chunk_len
            a_end = (t + 1) * chunk_len if t < num_segments - 1 else L_total
            center_frame = video_sample[v_start + (v_end - v_start) // 2].unsqueeze(0)  # (1, C, H, W)

            audio_seg = audio_sample[a_start:a_end]
            if audio_seg.shape[0] < chunk_len:
                pad = torch.zeros(chunk_len - audio_seg.shape[0], dtype=audio_seg.dtype, device=audio_seg.device)
                audio_seg = torch.cat([audio_seg, pad], dim=0)
            audio_seg = audio_seg.unsqueeze(0)  # (1, L)
            
            decision = fmt(center_frame, audio_seg)  # (1, 2)
            segment_decisions.append(decision)

            if decision[0, 0] > 0.5:
                video_selected_segments.append(video_sample[v_start:v_end])
            if decision[0, 1] > 0.5:
                audio_selected_segments.append(audio_sample[a_start:a_end])

        if video_selected_segments:
            selected_video = torch.cat(video_selected_segments, dim=0).unsqueeze(0)
            feat_v = video_model(selected_video)
        else:
            feat_v = torch.zeros(1, 512).to(device)

        if audio_selected_segments:
            selected_audio = torch.cat(audio_selected_segments, dim=0).unsqueeze(0)
            feat_a = audio_model(selected_audio)
        else:
            feat_a = torch.zeros(1, 512).to(device)

        outputs = fusion_model(feat_v, feat_a)
        loss_cls = criterion(outputs, label_sample)

        stm = torch.stack(segment_decisions).squeeze(1)
        efficiency_loss = torch.sum(stm) / (num_segments * 2)

        loss = loss_cls + penalty_weight * efficiency_loss
        loss.backward()

        _, predicted = torch.max(outputs, 1)
        optimizer.step()

        total_loss += loss.item()
        total_correct += (predicted == label_sample).sum().item()
        total_samples += 1
        
        current_loss = total_loss / total_samples
        current_acc = (total_correct / total_samples) * 100.0
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    train_loss = total_loss / total_samples
    train_acc = total_correct / total_samples * 100.0
    
    return train_loss, train_acc


def evaluate_model(models, dataloader, criterion, device, num_segments=5):
    audio_model, video_model, fusion_model, fmt = models
    fmt.eval()
    video_model.eval()
    audio_model.eval()
    fusion_model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")

    with torch.no_grad():
        for video_inputs, audio_inputs, labels in progress_bar:
            batch_size = video_inputs.shape[0]
            video_inputs = video_inputs.to(device)  # (B, T, C, H, W)
            audio_inputs = audio_inputs.to(device)  # (B, L)
            labels = labels.to(device)              # (B,)

            video_sample = video_inputs.squeeze(0)  # (T, C, H, W)
            audio_sample = audio_inputs.squeeze(0)  # (L,)
            label_sample = labels  # (1,)

            T_total = video_sample.shape[0]
            L_total = audio_sample.shape[0]
            seg_len = T_total // num_segments
            chunk_len = L_total // num_segments

            video_selected_segments = []
            audio_selected_segments = []
            segment_decisions = []

            for t in range(num_segments):
                v_start = t * seg_len
                v_end = (t + 1) * seg_len if t < num_segments - 1 else T_total
                a_start = t * chunk_len
                a_end = (t + 1) * chunk_len if t < num_segments - 1 else L_total

                center_frame = video_sample[v_start + (v_end - v_start) // 2].unsqueeze(0)  # (1, C, H, W)

                audio_seg = audio_sample[a_start:a_end]
                if audio_seg.shape[0] < chunk_len:
                    pad = torch.zeros(chunk_len - audio_seg.shape[0], dtype=audio_seg.dtype, device=audio_seg.device)
                    audio_seg = torch.cat([audio_seg, pad], dim=0)
                audio_seg = audio_seg.unsqueeze(0)  # (1, L)

                decision = fmt(center_frame, audio_seg)  # (1, 2)
                segment_decisions.append(decision)

                if decision[0, 0] > 0.5:
                    video_selected_segments.append(video_sample[v_start:v_end])
                if decision[0, 1] > 0.5:
                    audio_selected_segments.append(audio_sample[a_start:a_end])

            if video_selected_segments:
                selected_video = torch.cat(video_selected_segments, dim=0).unsqueeze(0)
                feat_v = video_model(selected_video)
            else:
                feat_v = torch.zeros(1, 512).to(device)

            if audio_selected_segments:
                selected_audio = torch.cat(audio_selected_segments, dim=0).unsqueeze(0)
                feat_a = audio_model(selected_audio)
            else:
                feat_a = torch.zeros(1, 512).to(device)

            logits = fusion_model(feat_v, feat_a)
            loss_cls = criterion(logits, label_sample)
            pred = torch.argmax(logits, dim=1)
            
            total_loss += loss_cls.item()
            total_correct += (pred == label_sample).sum().item()
            total_samples += 1
            current_loss = total_loss / total_samples
            current_acc = (total_correct / total_samples) * 100.0
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
    print(f"[Eval] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}")
    return avg_loss, accuracy


# 训练循环
def train(models, train_loader, val_loader, criterion, device, epochs=10, save_dir=None):
    audio_model, video_model, fusion_model, fmt_model = models
    
    # optimizer = optim.Adam(
    #     list(fusion_model.parameters()) + list(fmt_model.parameters()),
    #     lr=1e-4,
    #     weight_decay=1e-5
    # )
    optimizer = optim.Adam(
        fmt_model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(models, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(models, val_loader, criterion, device)

        # 保存最佳模型（分别保存 fmt 和 fusion_model）
        if val_acc > best_val_acc:     
            best_val_acc = val_acc
            fmt_path = os.path.join(save_dir, 'smg_fmt_only.pth')
            torch.save(fmt_model.state_dict(), fmt_path)
            print(f"Best models saved at epoch {epoch+1} with accuracy: {best_val_acc:.2f}%")

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
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=16)
    
    test_dataset = LipreadingDataset(root_dir=data_root, label_file=label_file, video_transform=test_transform, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)


    num_classes = 50  

    # 3D + 3D Models
    audio_model = AudioModelBaselineEncoder()
    video_model = Video_3D_Encoder()
    fusion_model = MultiModalFusion()
    fmt_model = FastModalityTester()
    
    audio_pretrained_dict = torch.load(f'checkpoints/baselines/audio_baseline.pth')
    audio_filtered_dict = {k: v for k, v in audio_pretrained_dict.items() if not k.startswith('fc.')}
    video_pretrained_dict = torch.load(f'checkpoints/baselines/video_resnet50_3D.pth')
    video_pretrained_dict = {k.replace('module.', ''): v for k, v in video_pretrained_dict.items()}
    video_filtered_dict = {k: v for k, v in video_pretrained_dict.items() if not k.startswith('fc.')}
    fusion_pretrained_dict = torch.load(f'checkpoints/baselines/fusion.pth')
    fusion_filtered_dict = {k.replace('module.', ''): v for k, v in fusion_pretrained_dict.items()}

    audio_model.load_state_dict(audio_filtered_dict, strict=False)
    video_model.load_state_dict(video_filtered_dict, strict=False)
    fusion_model.load_state_dict(fusion_filtered_dict, strict=False)

    models = [audio_model, video_model, fusion_model, fmt_model]

    for i in range(len(models)):
        # models[i] = nn.DataParallel(models[i])
        models[i] = models[i].to(device)
        
    # **损失函数**
    criterion = nn.CrossEntropyLoss()
    epochs = 100    
    
    # 训练
    save_dir = f"./checkpoints/baselines"
    train(models, train_loader, test_loader, criterion, device, epochs=epochs, save_dir=save_dir)
