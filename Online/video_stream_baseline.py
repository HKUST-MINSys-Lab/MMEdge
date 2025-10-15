import os
import cv2
import time
import torch
import subprocess
import numpy as np
import torchvision.transforms as T

from utils import average


def evenly_spaced_indices(total_range, num_indices):
    if num_indices > total_range:
        raise ValueError("num_indices cannot be greater than total_range")
    return np.linspace(0, total_range - 1, num=num_indices, dtype=int)


def load_video_tensor(video_path, max_frames=32):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[..., ::-1][115:211, 79:175]  # BGR to RGB
        frames.append(frame)
        count += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Failed to read video from {video_path}")

    video_np = np.stack(frames)  # (T, H, W, C)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)

    # 应用 transform
    transform = T.Compose([
        T.CenterCrop((88, 88)),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # 注意：要 frame by frame 处理
    video_tensor = torch.stack([transform(frame) for frame in video_tensor])  # (T, C, 88, 88)

    return video_tensor


def video_stream(args, video_data_queue, video_feature_queue, video_model, video_model_bad=None):
    pid = os.getpid()
    print(f"[Video Stream] Process {pid} started")
    if args.cpu_limitation:
        cgroup_path = "/sys/fs/cgroup/cpu/my_cgroup"
        try:
            subprocess.run(
                ["sudo", "tee", f"{cgroup_path}/cgroup.procs"],
                input=f"{pid}\n",   
                text=True,
                check=True
            )
            print(f"[Video Stream] Process {pid} has been added to cgroup {cgroup_path}")
        except subprocess.CalledProcessError as e:
            print(f"[Video Stream] Failed to add process {pid} to cgroup: {e}")
            
    video_model = video_model.to(args.device)
    if video_model_bad is not None:
        video_model_bad = video_model_bad.to(args.device)
        
    while True:
        # === Phase 1: receive video path and extract 1st frame ===
        request = video_data_queue.get()
        sample_id = request['sample_id']
        video_path = request['video_path']
        start_time = request['start_time']

        all_frames = load_video_tensor(video_path)  # (T, C, H, W)
        sampling_start_time = time.time()

        video_fps = args.video_fps
        
        # print(f"[Video Stream] Selecting {video_fps} frames from {all_frames.shape[0]} total frames.")
        selected_indices = evenly_spaced_indices(total_range=all_frames.shape[0], num_indices=video_fps)
        selected_frames = all_frames[selected_indices]  # (T_sel, C, H, W)

        frames = []  # Initialize with the first frame feature

        # === Phase 3: simulate real-time sampling aligned to t0 ===
        for idx in range(len(selected_frames)):
            frame = selected_frames[idx]  # (C, H, W)
            scheduled_time = sampling_start_time + idx / video_fps
            now = time.time()
            sleep_time = max(0, scheduled_time - now)
            if sleep_time > 0:
                # print(f"[Video Stream] Sleeping for {sleep_time * 1000:.3f} ms before processing frame {idx}")
                time.sleep(sleep_time)

            frames.append(frame)  # Append the current frame to the list
            if args.imputation and idx == 25:
                break
            
        print(f"[Video Stream] Using {len(frames)} frames for inference")
        sensing_end_time = time.time()
        # === Phase 4: temporal encoder & send back full feature ===
        video_encode_start_time = time.time()
        feat_stack = torch.stack(frames).unsqueeze(0).to(args.device)  # (1, T, D)
        with torch.no_grad():
            if video_model_bad is not None and sample_id % 2 == 0:
                temporal_feat = video_model_bad(feat_stack).squeeze(0).cpu().numpy()
            else:
                temporal_feat = video_model(feat_stack).squeeze(0).cpu().numpy()
        video_encode_end_time = time.time()
        video_encode_latency = (video_encode_end_time - video_encode_start_time) * 1000

        finish_timestamp = time.time()

        video_feature_queue.put({
            "sample_id": sample_id,
            "stage": "full",
            "feature": temporal_feat,
            "end_time": finish_timestamp,
            "sensing_end_time": sensing_end_time,
            "video_encode_latency": video_encode_latency
        })
