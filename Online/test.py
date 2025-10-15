import os
import cv2
import time
import torch
import subprocess
import numpy as np
import torchvision.transforms as T
from models.audio_model import AudioModelBaselineEncoder
from models.video_model_3d import Video_Classification

# model = Video_Classification().eval().to('cuda')
model = AudioModelBaselineEncoder().eval().to('cuda')
encoding_latency_list = []

for _ in range(10):
    dummy_video = torch.randn(1, 16800).to('cuda')
    start_time = time.time()
    pred = model(dummy_video) # (2, 400)
    end_time = time.time()
    encoding_latency = (end_time - start_time) * 1000
    encoding_latency_list.append(encoding_latency)
    print(f"Encoding Latency: {encoding_latency:.3f}")

avg_latency = sum(encoding_latency_list[2:]) / len(encoding_latency_list[2:])
print(f"Avg Encoding Latency: {avg_latency:.3f}")