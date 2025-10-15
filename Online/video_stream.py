import os
import cv2
import time
import torch
import subprocess
import numpy as np
import torchvision.transforms as T

from models.cross_modal_gaiting import EarlyExitClassifier
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


def video_stream(args, video_data_queue, config_queue, video_feature_queue, exit_queue, video_spatial_models, video_temporal_models, early_exit_event, barrier):
    
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
            
    if args.gaiting:
        gaiting_classifier = EarlyExitClassifier().to(args.device)
        gaiting_classifier.load_state_dict(torch.load('./checkpoints/gaiting/gaiting_classifier.pth'))
    for idx in range(len(video_spatial_models)):
        video_spatial_models[idx] = video_spatial_models[idx].to(args.device)
        video_temporal_models[idx] = video_temporal_models[idx].to(args.device)
    while True:
        # === Phase 1: receive video path and extract 1st frame ===
        request = video_data_queue.get()
        sample_id = request['sample_id']
        video_path = request['video_path']
        start_time = request['start_time']

        all_frames = load_video_tensor(video_path)  # (T, C, H, W)
        barrier.wait()
        sampling_start_time = time.time()
        frame_0 = all_frames[0].unsqueeze(0).to(args.device)  # (1, C, 1, H, W)

        with torch.no_grad():
            # Use the first spatial model (ResNet-18) for first frame feature extraction
            feat_0 = video_spatial_models[0](frame_0).squeeze(0).cpu() 
            first_feature = feat_0.numpy()  
        print(f"[Video Stream] Loaded first frame for sample {sample_id} successfully, start time: {(sampling_start_time - start_time) * 1000:.3f} ms")

        # send early feature to main process
        video_feature_queue.put({
            "sample_id": sample_id,
            "stage": "early",
            "feature": first_feature
        })

        # === Phase 2: wait for sensing/model config ===
        config_msg = config_queue.get()
        config = config_msg[0]
        if all_frames is None:
            continue

        video_fps = config['video_fps']
        video_model = config['video_model_id']
        
        video_spatial_model = video_spatial_models[video_model]  # 可根据 config 选择轻量模型
        video_temporal_model = video_temporal_models[video_model] 
        
        # select frames based on evenly spaced indices
        # print(f"[Video Stream] Selecting {video_fps} frames from {all_frames.shape[0]} total frames.")
        selected_indices = evenly_spaced_indices(total_range=all_frames.shape[0], num_indices=video_fps)
        selected_frames = all_frames[selected_indices]  # (T_sel, C, H, W)

        collected_feats = []  # Initialize with the first frame feature

        # === Phase 3: simulate real-time sampling aligned to t0 ===
        spatial_encode_latency_list = []
        
        frame_cnt = 0
        audio_feat = None
        sensing_end_time = None
        
        BATCH_SIZE = 2
        batch_frames = []
        batch_indices = []
        gaiting = False
        gaiting_latency_total = 0.0 
        gaiting_position = None
        for idx in range(len(selected_frames)):
            condition = (frame_cnt == int(0.5 * video_fps)) or (frame_cnt == int(0.7 * video_fps)) or (frame_cnt == int(0.9 * video_fps)) or (BATCH_SIZE == 4 and frame_cnt in [16, 20, 24, 28]) or (BATCH_SIZE == 3 and frame_cnt in [18, 24, 27])
            if args.gaiting and condition and early_exit_event.is_set():
                gaiting_start_time = time.time()
                if audio_feat is None:
                    audio_feat = torch.tensor(exit_queue.get()).unsqueeze(0).to(args.device)  # (1, C, H, W)
                with torch.no_grad():
                    feat_stack = torch.stack(collected_feats).unsqueeze(0).to(args.device)  # (1, T, D)
                    video_feat = video_temporal_model(feat_stack)
                    input_features = torch.cat([audio_feat, video_feat], dim=1)
                    gating_outputs = gaiting_classifier(input_features)
                    exit_decision = (gating_outputs.item() >= 0.5)
                    
                    gaiting_end_time = time.time()
                    gaiting_latency = (gaiting_end_time - gaiting_start_time) * 1000
                    gaiting_latency_total += gaiting_latency
                    
                    if exit_decision:
                        print(f"[Video Stream] Early exit decision made for sample {sample_id} with {frame_cnt} frames, the score {gating_outputs.item():.4f}, at {(gaiting_start_time - start_time) * 1000:.3f} ms")
                        gaiting = True
                        gaiting_position = idx
                        break
                    else:
                        print(f"[Video Stream] No early exit decision made for sample {sample_id} with {frame_cnt} frames, the score {gating_outputs.item():.4f}.")
            
            if args.blocking and early_exit_event.is_set():
                sensing_end_time = time.time()
                print(f"[Video Stream] Blocking for sample {sample_id}, at {(sensing_end_time - start_time) * 1000:.3f} ms.")
                break
                
            frame = selected_frames[idx]  # (C, H, W)
            scheduled_time = sampling_start_time + idx / video_fps
            now = time.time()
            sleep_time = max(0, scheduled_time - now)
            if sleep_time > 0:
                # print(f"[Video Stream] Sleeping for {sleep_time * 1000:.3f} ms before processing frame {idx}")
                time.sleep(sleep_time)

            spatial_encode_start_time = time.time()
            frame = frame.unsqueeze(0).to(args.device)  # (1, C, H, W)
            sensing_end_time = time.time()
            
            if args.batch:
                batch_frames.append(frame)
                batch_indices.append(idx)
                if len(batch_frames) == BATCH_SIZE or idx == len(selected_frames) - 1:
                    batch_tensor = torch.cat(batch_frames, dim=0).to(args.device)
                    with torch.no_grad():
                        feats = video_spatial_model(batch_tensor)  # (B, D)
                    for i in range(len(batch_frames)):
                        collected_feats.append(feats[i].cpu())
                    batch_frames.clear()
                    batch_indices.clear()
            else:
                with torch.no_grad():
                    feat = video_spatial_model(frame).squeeze(0).cpu()
                collected_feats.append(feat)
                
            spatial_encode_end_time = time.time()
            spatial_encode_latency = (spatial_encode_end_time - spatial_encode_start_time) * 1000
            spatial_encode_latency_list.append(spatial_encode_latency)
            frame_cnt += 1

        average_spatial_encode_latency = average(spatial_encode_latency_list)

        # === Phase 4: temporal encoder & send back full feature ===
        video_temporal_encode_start_time = time.time()
        feat_stack = torch.stack(collected_feats).unsqueeze(0).to(args.device)  # (1, T, D)
        with torch.no_grad():
            temporal_feat = video_temporal_model(feat_stack).squeeze(0).cpu().numpy()
        video_temporal_encode_end_time = time.time()
        video_temporal_encode_latency = (video_temporal_encode_end_time - video_temporal_encode_start_time) * 1000

        finish_timestamp = time.time()
        
        print(f"[Video Stream] Finish time: {(finish_timestamp - start_time) * 1000:.3f} ms, with {frame_cnt} frames")

        video_feature_queue.put({
            "sample_id": sample_id,
            "stage": "full",
            "feature": temporal_feat,
            "end_time": finish_timestamp,
            'sensing_end_time': sensing_end_time,
            "spatial_encode_latency": average_spatial_encode_latency,
            "temporal_encode_latency": video_temporal_encode_latency,
            "gaiting": gaiting,
            "gaiting_latency_total": gaiting_latency_total,
            "gaiting_position": gaiting_position
        })
