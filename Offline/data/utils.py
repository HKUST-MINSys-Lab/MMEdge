# coding: utf-8
import random
import cv2
import numpy as np


def CenterCrop(batch_img, size):
    """
    向量化实现 CenterCrop，不使用显式循环。
    
    参数:
        batch_img: numpy 数组，形状 (B, F, H, W, C)
        size: 裁剪大小 (th, tw)
    返回:
        裁剪后的数组，形状 (B, F, th, tw, C)
    """
    B, F, H, W, C = batch_img.shape
    th, tw = size
    y1 = int(round((H - th) / 2))
    x1 = int(round((W - tw) / 2))
    return batch_img[:, :, y1:y1+th, x1:x1+tw, :]


def RandomCrop(batch_img, size):
    """
    对每个样本随机裁剪，再用 np.stack 合并。
    
    参数:
        batch_img: numpy 数组，形状 (B, F, H, W, C)
        size: 裁剪大小 (th, tw)
    返回:
        裁剪后的数组，形状 (B, F, th, tw, C)
    """
    B, F, H, W, C = batch_img.shape
    th, tw = size
    crops = []
    for i in range(B):
        # 随机生成裁剪起点，确保不会超出范围
        y1 = np.random.randint(0, H - th + 1)
        x1 = np.random.randint(0, W - tw + 1)
        crops.append(batch_img[i, :, y1:y1+th, x1:x1+tw, :])
    return np.stack(crops, axis=0)


def HorizontalFlip(batch_img):
    for i in range(len(batch_img)):
        if random.random() > 0.5:
            for j in range(len(batch_img[i])):
                batch_img[i][j] = cv2.flip(batch_img[i][j], 1)
    return batch_img


def ColorNormalize(batch_img):
    mean = 0.413621
    std = 0.1700239
    batch_img = (batch_img - mean) / std
    return batch_img


def add_gaussian_noise(audio, noise_level=0.01):
    """
    在音频数据上添加高斯噪声
    Args:
        audio: np.array, 形状为 (B, T)
        noise_level: 控制噪声的标准差
    Returns:
        加噪音后的音频数据
    """
    noise = np.random.normal(0, noise_level, audio.shape).astype(np.float32) 
    return audio + noise