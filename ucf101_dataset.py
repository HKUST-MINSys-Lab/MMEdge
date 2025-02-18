import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as F

class VideoDataset(Dataset):
    def __init__(self, file_list, root_dir, frames_per_clip=16, transform=None, label_file=None):
        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.video_info = []
        self.name_to_label = dict()

        # 解析文件列表
        if label_file is not None:
            with open(label_file, 'r') as f:
                for line in f:
                    label, name = str(line).split(' ')
                    self.name_to_label[name[:-1]] = label
            with open(file_list, 'r') as f:
                for line in f:
                    path = line.strip().rsplit(' ', 1)[0]
                    name = path.split('/')[0]
                    label = self.name_to_label[name]
                    self.video_info.append((path, int(label)))
        else:
            with open(file_list, 'r') as f:
                for line in f:
                    path, label = line.strip().rsplit(' ', 1)
                    self.video_info.append((path, int(label)))

    def __len__(self):
        return len(self.video_info)

    def __getitem__(self, idx):
        video_path, label = self.video_info[idx]
        video_full_path = os.path.join(self.root_dir, video_path)

        # 读取视频
        video, _, _ = read_video(video_full_path, pts_unit="sec")
        video_frame = video[0].permute(2, 0, 1)
        
        T, H, W, C = video.shape

        # 补齐帧
        # if T < self.frames_per_clip:
        #     pad_frames = torch.zeros((self.frames_per_clip - T, H, W, C), dtype=torch.uint8)
        #     video = torch.cat((video, pad_frames), dim=0)

        # # 随机选帧
        # start_frame = torch.randint(0, max(1, T - self.frames_per_clip + 1), (1,)).item()
        # video_clip = video[start_frame:start_frame + self.frames_per_clip]
        
        indices = torch.linspace(0, T - 1, self.frames_per_clip).long()
        video_clip = video[indices]

        # 对每帧进行 transform
        frames = []
        for t in range(video_clip.shape[0]):
            frame = video_clip[t].permute(2, 0, 1).float() / 255.0  # (H, W, C) -> (C, H, W)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        # 合并为视频张量 (C, T, H, W)
        video_clip = torch.stack(frames, dim=1)
        
        return video_clip, label - 1


# # 示例
# root_dir = "/data/rxhuang/UCF-101"  # 数据集根目录
# file_list = "/data/rxhuang/ucfTrainTestlist/testlist01.txt"  # 包含视频路径和标签的文件
# label_file = "/data/rxhuang/ucfTrainTestlist/classInd.txt"
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# dataset = VideoDataset(file_list=file_list, root_dir=root_dir, frames_per_clip=32, transform=transform, label_file=label_file)
# loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8)


# # # 测试数据加载
# # cnt = 0
# for videos, labels in loader:
#     print(f"Video batch shape: {videos.shape}")  # (B, C, T, H, W)
#     print(f"Labels: {labels}")
#     break
#     cnt += 1
#     if cnt == 10:
#         break
