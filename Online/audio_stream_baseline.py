import os
import time
import torch
import queue
import subprocess
import numpy as np
import soundfile as sf
import scipy.signal

from utils import average


def normalize_chunk(chunk):
    mean = chunk.mean(dim=1, keepdim=True)  # (B, 1)
    std = chunk.std(dim=1, keepdim=True)    # (B, 1)
    std = torch.where(std == 0, torch.ones_like(std), std)  # 避免除以0
    chunk = (chunk - mean) / std
    return chunk


def load_audio_tensor(audio_path, target_sample_rate=16000):
    waveform, sample_rate = sf.read(audio_path)  # waveform: (T,) or (T, 1)
    if waveform.ndim > 1:
        waveform = waveform[:, 0]  # 只取第一个通道
    if sample_rate != target_sample_rate:
        duration = len(waveform) / sample_rate
        new_length = int(duration * target_sample_rate)
        waveform = scipy.signal.resample(waveform, new_length)
    # waveform = waveform[-16800:]
    waveform = waveform[-16800:]
    waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)  # (1, T)
    return waveform


def audio_stream(args, audio_data_queue, audio_feature_queue, audio_model, audio_model_bad=None):
    pid = os.getpid()   
    print(f"[Audio Stream] Process {pid} started")
    if args.cpu_limitation:
        cgroup_path = "/sys/fs/cgroup/cpu/my_cgroup"
        try:
            subprocess.run(
                ["sudo", "tee", f"{cgroup_path}/cgroup.procs"],
                input=f"{pid}\n", 
                text=True,
                check=True
            )
            print(f"[Audio Stream] Process {pid} has been added to cgroup {cgroup_path}")
        except subprocess.CalledProcessError as e:
            print(f"[Audio Stream] Failed to add process {pid} to cgroup: {e}")
            
    audio_model = audio_model.to(args.device)
    if audio_model_bad is not None:
        audio_model_bad = audio_model_bad.to(args.device)
    
    while True:
        request = audio_data_queue.get()
        sample_id = request["sample_id"]
        audio_path = request["audio_path"]
        start_time = request["start_time"]

        waveform = load_audio_tensor(audio_path, args.audio_sample_rate)  # (1, T)
        chunk_size = 800
        sampling_start_time = time.time()

        audio_chunk_size = args.audio_chunk_size
        sensing_interval = audio_chunk_size / args.audio_sample_rate
     
        frame_count = int(np.ceil(waveform.shape[1] / audio_chunk_size))
        # print(f"[Audio Stream] Frame count: {frame_count}, chunk size: {audio_chunk_size}, sensing interval: {sensing_interval:.3f} s")

        chunk_list = []
        
        for i in range(frame_count):
            scheduled_time = sampling_start_time + i * sensing_interval
            now = time.time()
            wait_time = max(0, scheduled_time - now)

            if wait_time > 0:
                # print(f"[Audio Stream] Waiting for {wait_time * 1000:.3f} ms for chunk {i}...")
                time.sleep(wait_time)

            offset = i * audio_chunk_size
            chunk = waveform[:, offset:offset + audio_chunk_size].to(args.device)
            if chunk.shape[1] < audio_chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, audio_chunk_size - chunk.shape[1]))
            
            chunk_list.append(chunk)
        sensing_end_time = time.time()
            

        audio_encode_start_time = time.time()
        chunks = torch.Tensor(waveform[-16800:]).to(args.device)  # (1, T, D)
        with torch.no_grad():
            if audio_model_bad is not None and sample_id % 2 == 0:
                audio_feature = audio_model_bad(chunks).squeeze(0).cpu().numpy()
            else:
                audio_feature = audio_model(chunks).squeeze(0).cpu().numpy()
        audio_encode_end_time = time.time()
        audio_encode_latency = (audio_encode_end_time - audio_encode_start_time) * 1000

        finish_timestamp = time.time()

        audio_feature_queue.put({
            "sample_id": sample_id,
            "stage": "full",
            "feature": audio_feature,
            "end_time": finish_timestamp,
            "sensing_end_time": sensing_end_time,
            "audio_encode_latency": audio_encode_latency
        })
