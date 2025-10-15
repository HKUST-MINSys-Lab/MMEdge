import os
import time
import torch
import subprocess
import numpy as np
import queue
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
    waveform = waveform[-16800:]
    waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)  # (1, T)
    return waveform


def audio_stream(args, audio_data_queue, config_queue, audio_feature_queue, exit_queue, audio_chunk_models, audio_temporal_models, early_exit_event, barrier):
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
    
    for idx in range(len(audio_chunk_models)):
        audio_chunk_models[idx] = audio_chunk_models[idx].to(args.device)
        audio_temporal_models[idx] = audio_temporal_models[idx].to(args.device)
    
    while True:
        request = audio_data_queue.get()
        sample_id = request["sample_id"]
        audio_path = request["audio_path"]
        start_time = request["start_time"]

        chunk_size = 800
        waveform = load_audio_tensor(audio_path, args.audio_sample_rate)  # (1, T)
        barrier.wait()
        sampling_start_time = time.time()
        chunk_0 = waveform[:, :chunk_size].to(args.device)

        with torch.no_grad():
            feat_0 = audio_chunk_models[0](chunk_0).squeeze(0).cpu()
            first_feature = feat_0.numpy()
        print(f"[Audio Stream] Loaded chunk 0 for sample {sample_id} successfully, start time: {(sampling_start_time - start_time) * 1000:.3f} ms")

        audio_feature_queue.put({
            "sample_id": sample_id,
            "stage": "early",
            "feature": first_feature
        })

        # === Phase 2: Wait for config from main process ===
        config_msg = config_queue.get()
        config = config_msg[0]

        if waveform is None:
            continue

        audio_chunk_size = config['audio_chunk_size']
        audio_model = config['audio_model_id']

        sample_rate = args.audio_sample_rate
        sensing_interval = audio_chunk_size / sample_rate

        audio_chunk_model = audio_chunk_models[audio_model]
        audio_temporal_model = audio_temporal_models[audio_model]

        frame_count = int(np.ceil(waveform.shape[1] / audio_chunk_size))
        # print(f"[Audio Stream] Frame count: {frame_count}, chunk size: {audio_chunk_size}, sensing interval: {sensing_interval:.3f} s")

        collected_feats = []
        audio_chunk_encode_latency_list = []
        sensing_end_time = None

        for i in range(frame_count):
            scheduled_time = sampling_start_time + i * sensing_interval
            now = time.time()
            wait_time = max(0, scheduled_time - now)

            if wait_time > 0:
                # print(f"[Audio Stream] Waiting for {wait_time * 1000:.3f} ms for chunk {i}...")
                time.sleep(wait_time)

            offset = i * audio_chunk_size
            chunk = waveform[:, offset:offset + audio_chunk_size].to(args.device)
            sensing_end_time = time.time()
            if chunk.shape[1] < audio_chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, audio_chunk_size - chunk.shape[1]))
            chunk = normalize_chunk(chunk)
            try:
                encode_start_time = time.time()
                with torch.no_grad():
                    feat = audio_chunk_model(chunk).squeeze(0).cpu()
                encode_end_time = time.time()
                audio_chunk_encode_latency = (encode_end_time - encode_start_time) * 1000
                audio_chunk_encode_latency_list.append(audio_chunk_encode_latency)
                collected_feats.append(feat)
            except Exception as e:
                print(f"[Audio Stream] Error processing chunk {i}: {e}")
                collected_feats.append(torch.zeros(512))
        avg_audio_chunk_encode_latency = average(audio_chunk_encode_latency_list)

        audio_temporal_encode_start_time = time.time()
        feat_stack = torch.stack(collected_feats).unsqueeze(0).to(args.device)  # (1, T, D)
        
        with torch.no_grad():
            temporal_feat = audio_temporal_model(feat_stack)
            # audio_logits = audio_classifiers[audio_model](temporal_feat) 
            # audio_probs = torch.softmax(audio_logits, dim=1)
            # top1_confidence, _ = torch.max(audio_probs, dim=1)
            temporal_feat = temporal_feat.squeeze(0).cpu().numpy()
            
        audio_temporal_encode_end_time = time.time()
        audio_temporal_encode_latency = (audio_temporal_encode_end_time - audio_temporal_encode_start_time) * 1000

        finish_timestamp = time.time()
        
        if args.gaiting:
            print(f"[Audio Stream] Early exit triggered for sample {sample_id}, at {(finish_timestamp - start_time) * 1000:.3f} ms.")
            exit_queue.put(temporal_feat)
            early_exit_event.set()
        
        if args.blocking:
            print(f"[Audio Stream] Finished for sample {sample_id}, at {(finish_timestamp - start_time) * 1000:.3f} ms.")
            early_exit_event.set()

        audio_feature_queue.put({
            "sample_id": sample_id,
            "stage": "full",
            "feature": temporal_feat,
            "end_time": finish_timestamp,
            'sensing_end_time': sensing_end_time,
            "chunk_encode_latency": avg_audio_chunk_encode_latency,
            "temporal_encode_latency": audio_temporal_encode_latency
        })
