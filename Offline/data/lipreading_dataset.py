import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from data.utils import  add_gaussian_noise


def load_label_map(label_file):
    """ 加载标签映射 """
    label_df = pd.read_csv(label_file, sep='\t', header=None, names=['word', 'label'])
    label_map = {row['word']: row['label'] for _, row in label_df.iterrows()}
    return label_map

class LipreadingDataset(Dataset):
    def __init__(self, root_dir, label_file, mode='train', video_transform=None, sample_cnt=None):
        """
        Args:
            root_dir (str): 数据根目录 (/data/rxhuang/lipreading_feature)
            label_file (str): 词汇和标签的映射文件 (select_words.txt)
            mode (str): 选择 'train', 'val', 'test'
            sample_cnt (int, optional): 仅在训练集选择前 sample_cnt 个单词
        """
        assert mode in ['train', 'val', 'test'], "mode 必须是 'train', 'val' 或 'test'"
        self.root_dir = root_dir
        self.mode = mode
        self.video_transform = video_transform
        self.sample_cnt = sample_cnt
        self.label_map = load_label_map(label_file)
        self.data = self._load_data()

    def _load_data(self):
        """加载符合选择单词的数据"""
        data_list = []
        for word in self.label_map.keys():
            video_path = os.path.join(self.root_dir, 'video', word, self.mode)
            audio_path = os.path.join(self.root_dir, 'audio', word, self.mode)
            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                continue
            
            if self.sample_cnt is None:
                samples = os.listdir(video_path)
            else:
                samples = os.listdir(video_path)[:self.sample_cnt]
            for file in samples:
                if file.endswith('.npz'):
                    video_file = os.path.join(video_path, file)
                    audio_file = os.path.join(audio_path, file)
                    if os.path.exists(audio_file):
                        data_list.append((video_file, audio_file, self.label_map[word]))
        return data_list

    def normalisation(self, inputs):
        inputs_std = np.std(inputs)
        if inputs_std == 0.:
            inputs_std = 1.
        return (inputs - np.mean(inputs))/inputs_std
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_file, audio_file, label = self.data[idx]
        
        # 加载 npz 数据
        video_data = np.load(video_file)['data'].astype(np.float32) / 255.
        audio_data = np.load(audio_file)['data'].astype(np.float32)

        transformed_frames = []
        if self.video_transform:
            for frame in video_data:
                transformed_frame = self.video_transform(frame)
                transformed_frames.append(transformed_frame)
            video_tensor = torch.stack(transformed_frames)
        else:
            video_tensor = torch.from_numpy(video_data)
            
        if self.mode == 'train':
            # 随机添加高斯噪声
            audio_data = add_gaussian_noise(audio_data)
        
        # 转换为 PyTorch Tensor
        audio_tensor = torch.from_numpy(audio_data)  # 形状: (C, 音频时间步)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return video_tensor, audio_tensor, label_tensor

if __name__ == '__main__':
# 示例用法:
    dataset = LipreadingDataset(root_dir='/data/rxhuang/lipread_feature',
                                label_file='all_words.txt',
                                mode='train',
                                sample_cnt=None)

    print(f"数据集大小: {len(dataset)}")

