import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def load_label_map(label_file):
    """ 加载标签映射 """
    label_df = pd.read_csv(label_file, sep='\t', header=None, names=['word', 'label'])
    label_map = {row['word']: row['label'] for _, row in label_df.iterrows()}
    return label_map

class LipreadingDataset(Dataset):
    def __init__(self, root_dir, label_file, video_transform=None, sample_cnt=None):
        """
        Args:
            root_dir (str): 数据根目录 (/data/rxhuang/lipreading_feature)
            label_file (str): 词汇和标签的映射文件 (select_words.txt)
            mode (str): 选择 'train', 'val', 'test'
            sample_cnt (int, optional): 仅在训练集选择前 sample_cnt 个单词
        """
        self.root_dir = root_dir
        self.video_transform = video_transform
        self.sample_cnt = sample_cnt
        self.label_map = load_label_map(label_file)
        self.data = self._load_data()

    def _load_data(self):
        """加载符合选择单词的数据"""
        data_list = []
        for word in self.label_map.keys():
            video_path = os.path.join(self.root_dir, 'lipread_mp4', word, 'test')
            audio_path = os.path.join(self.root_dir, 'lipread_wav', word, 'test')
            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                continue
            
            cnt = 0
            for file in os.listdir(video_path):
                if file.endswith('.mp4'):
                    video_file = os.path.join(video_path, file)
                    audio_file = os.path.join(audio_path, file.replace('.mp4', '.wav'))
                    if os.path.exists(audio_file):
                        data_list.append((video_file, audio_file, self.label_map[word]))
                    cnt += 1
                    if self.sample_cnt is not None and cnt >= self.sample_cnt:
                        break
        return data_list
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_file, audio_file, label = self.data[idx]
        return video_file, audio_file, label

# 示例用法:
# dataset = LipreadingDataset(root_dir='/home/jetson/Dataset',
#                             label_file='selected_words.txt',
#                             sample_cnt=5)

# print(f"数据集大小: {len(dataset)}")

