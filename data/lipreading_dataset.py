import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

def load_label_map(label_file, max_words=None):
    """ 加载标签映射 """
    label_df = pd.read_csv(label_file, sep='\t', header=None, names=['word', 'label'])
    label_map = {row['word']: row['label'] for _, row in label_df.iterrows()}
    if max_words is not None:
        selected_words = list(label_map.keys())[:max_words]
        label_map = {word: label_map[word] for word in selected_words}
    return label_map

class LipreadingDataset(Dataset):
    def __init__(self, root_dir, label_file, mode='train', max_words=None):
        """
        Args:
            root_dir (str): 数据根目录 (/data/rxhuang/lipreading_feature)
            label_file (str): 词汇和标签的映射文件 (select_words.txt)
            mode (str): 选择 'train', 'val', 'test'
            max_words (int, optional): 仅在训练集选择前 max_words 个单词
        """
        assert mode in ['train', 'val', 'test'], "mode 必须是 'train', 'val' 或 'test'"
        self.root_dir = root_dir
        self.mode = mode
        self.label_map = load_label_map(label_file, max_words)
        self.data = self._load_data()

    def _load_data(self):
        """加载符合选择单词的数据"""
        data_list = []
        for word in self.label_map.keys():
            video_path = os.path.join(self.root_dir, 'video', word, self.mode)
            audio_path = os.path.join(self.root_dir, 'audio', word, self.mode)
            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                continue
            
            for file in os.listdir(video_path):
                if file.endswith('.npz'):
                    video_file = os.path.join(video_path, file)
                    audio_file = os.path.join(audio_path, file)
                    if os.path.exists(audio_file):
                        data_list.append((video_file, audio_file, self.label_map[word]))
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_file, audio_file, label = self.data[idx]
        
        # 加载 npz 数据
        video_data = np.load(video_file)['data'].astype(np.float32) 
        audio_data = np.load(audio_file)['data'].astype(np.float32)
        
        # 转换为 PyTorch Tensor
        video_tensor = torch.from_numpy(video_data)  # 形状: (T, C, H, W)
        audio_tensor = torch.from_numpy(audio_data)  # 形状: (C, 音频时间步)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return video_tensor, audio_tensor, label_tensor

# 示例用法:
# dataset = LipreadingDataset(root_dir='/data/rxhuang/lipread_feature',
#                             label_file='selected_words.txt',
#                             mode='train',
#                             max_words=5)

