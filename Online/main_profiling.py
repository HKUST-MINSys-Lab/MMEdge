import os
import time
import json
import torch
import argparse
import subprocess
import torch.multiprocessing as mp
import torch.nn.functional as F

from data.lipreading_dataset import LipreadingDataset
from video_stream import video_stream
from audio_stream import audio_stream

from utils import average
from models.audio_model import AudioEncoder, AudioTemporalEncoder, AudioTemporalClassifier
from models.video_model import Video_ResNet_P3D_Encoder, get_resnet_backbone, Video_ResNet_P3D_Temporal_Encoder
from models.fusion_model import MultiModalFusion
from optimizer import AccuracyPredictor, Optimizer


video_idx_map = {'18': 0, '34': 1, '50': 2}
audio_idx_map = {'small': 0, 'medium': 1, 'large': 2}


def load_latency_table(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return {}


def set_latency(table, v_model, fps, a_model, a_chunk, latency):
    v_model = str(v_model)
    fps = str(fps)
    a_chunk = str(a_chunk)

    if v_model not in table:
        table[v_model] = {}
    if fps not in table[v_model]:
        table[v_model][fps] = {}
    if a_model not in table[v_model][fps]:
        table[v_model][fps][a_model] = {}
    
    table[v_model][fps][a_model][a_chunk] = latency
    

def load_models(args):
    video_spatial_models = []
    video_temporal_models = []
    audio_chunk_models = []
    audio_temporal_models = []
    fusion_models = []
    
    for size in ["18", "34", "50"]:
        backbone, feat_dim = get_resnet_backbone(f'resnet{size}', pretrained=True)
        video_model = Video_ResNet_P3D_Encoder(backbone, feature_dim=feat_dim).eval()
        checkpoint_path = os.path.join(args.checkpoint_path, f'video/video_resnet_{size}.pth')
        video_pretrained_dict = torch.load(checkpoint_path)
        video_pretrained_dict = {k.replace('module.', ''): v for k, v in video_pretrained_dict.items()}
        video_model.load_state_dict(video_pretrained_dict, strict=False)
        video_spatial_model = video_model.spatial_encoder
    
        video_temporal_model = Video_ResNet_P3D_Temporal_Encoder().eval()
        video_temporal_model.load_state_dict(video_pretrained_dict, strict=False)
        
        video_spatial_models.append(video_spatial_model)
        video_temporal_models.append(video_temporal_model)

    for size in ['small', 'medium', 'large']:
        audio_model = AudioEncoder(size=size).eval()
        checkpoint_path = os.path.join(args.checkpoint_path, f'audio/audio_{size}.pth')
        audio_pretrained_dict = torch.load(checkpoint_path)
        audio_model.load_state_dict(audio_pretrained_dict, strict=False)
        audio_chunk_model = audio_model.encoder
        
        audio_temporal_encoder = AudioTemporalEncoder().eval()
        audio_temporal_encoder.load_state_dict(audio_pretrained_dict, strict=False)
        
        audio_chunk_models.append(audio_chunk_model)
        audio_temporal_models.append(audio_temporal_encoder)
    
    for video in ["18", "34", "50"]:
        for audio in ['small', 'medium', 'large']:     
            fusion_model = MultiModalFusion().to(args.device).eval()
            checkpoint_name = f"fusion/fusion_{video}_{audio}.pth"
            checkpoint_path = os.path.join(args.checkpoint_path, checkpoint_name)
            fusion_pretrained_dict = torch.load(checkpoint_path)
            fusion_pretrained_dict = {k.replace('module.', ''): v for k, v in fusion_pretrained_dict.items()}
            fusion_filtered_dict = {k: v for k, v in fusion_pretrained_dict.items() if not k.startswith('fc.')}
            fusion_model.load_state_dict(fusion_filtered_dict, strict=False)
            fusion_models.append(fusion_model)

    return video_spatial_models, video_temporal_models, audio_chunk_models, audio_temporal_models, fusion_models


def main():
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Run RealSense experiment with specific FPS.")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints')
    parser.add_argument("--audio_sample_rate", type=int, default=16000)
    parser.add_argument("--audio_chunk_size", type=int, default=800)
    parser.add_argument("--audio_model", type=str, default='large')
    parser.add_argument("--video_fps", type=int, default=29)
    parser.add_argument("--video_model", type=int, default=50)
    parser.add_argument("--gaiting", action='store_true', help="Enable gaiting module if set")
    parser.add_argument("--cpu_limitation", action='store_true', help="Enable cpu limitation if set")
    parser.add_argument("--avg_start", type=int, default=2)
    parser.add_argument("--blocking", action='store_true', help="Enable blocking if set")
    parser.add_argument("--batch", action='store_true', help="Enable batch if set")
    args = parser.parse_args()

    if args.cpu_limitation:
        pid = os.getpid()
        print(f"[Main Process] Process {pid} started")
        cgroup_path = "/sys/fs/cgroup/cpu/my_cgroup"
        try:
            subprocess.run(
                ["sudo", "tee", f"{cgroup_path}/cgroup.procs"],
                input=f"{pid}\n",
                text=True,
                check=True
            )
            print(f"[Main Process] Process {pid} has been added to cgroup {cgroup_path}")
        except subprocess.CalledProcessError as e:
            print(f"[Main Process] Failed to add process {pid} to cgroup: {e}")

    device = args.device

    # Queues for data and result communication
    video_data_queue = mp.Queue()
    audio_data_queue = mp.Queue()
    video_config_queue = mp.Queue()
    audio_config_queue = mp.Queue()
    video_feature_queue = mp.Queue()
    audio_feature_queue = mp.Queue()
    exit_queue = mp.Queue()

    # Load accuracy predictor and optimizer
    predictor = AccuracyPredictor().to(device).eval()
    optimizer = Optimizer(args=args, predictor=predictor, path='./data/latency_table.json', T_max=200)
    
    # Load models
    video_spatial_models, video_temporal_models, audio_chunk_models, audio_temporal_models, fusion_models = load_models(args)

    # Load dataset
    dataset = LipreadingDataset(root_dir='/home/jetson/Dataset',
                                label_file='./data/selected_words.txt',
                                sample_cnt=1)

    # Start worker processes
    early_exit_event = mp.Event()
    barrier = mp.Barrier(2)
    video_process = mp.Process(target=video_stream, args=(args, video_data_queue, video_config_queue, video_feature_queue, exit_queue, video_spatial_models, video_temporal_models, early_exit_event, barrier))
    audio_process = mp.Process(target=audio_stream, args=(args, audio_data_queue, audio_config_queue, audio_feature_queue, exit_queue, audio_chunk_models, audio_temporal_models, early_exit_event, barrier))

    video_process.start()
    audio_process.start()

    correct = 0
    total = 0

    optimizer_latency_list = []
    end_to_end_latency_list = []
    video_spatial_encode_latency_list = []
    video_temporal_encode_latency_list = []
    audio_chunk_encode_latency_list = []
    audio_temporal_encode_latency_list = []
    fusion_latency_list = []
    video_communication_latency_list = []
    audio_communication_latency_list = []
    
    for sample_id, (video_path, audio_path, label) in enumerate(dataset):
        # Step 1: Send video/audio path for first-frame feature extraction
        early_exit_event.clear()
        start_time = time.time()
        video_data_queue.put({"sample_id": sample_id, "video_path": video_path, "start_time": start_time})
        audio_data_queue.put({"sample_id": sample_id, "audio_path": audio_path, "start_time": start_time})

        video_early, audio_early = None, None
        
        while video_early is None or audio_early is None:
            video_first = video_feature_queue.get()
            video_early = torch.tensor(video_first['feature']).unsqueeze(0).to(device)
            audio_first = audio_feature_queue.get()
            audio_early = torch.tensor(audio_first['feature']).unsqueeze(0).to(device)

        optimize_start_time = time.time()
        predicted_config = optimizer.optimize(video_early, audio_early)
        predicted_config = [{
                            'audio_model_id': audio_idx_map[str(args.audio_model)],
                            'audio_chunk_size': args.audio_chunk_size,
                            'video_model_id': video_idx_map[str(args.video_model)],
                            'video_fps': args.video_fps,
                        }]
        optimize_end_time = time.time()
        optimizer_latency = (optimize_end_time - optimize_start_time) * 1000
        print(f"[Main Process] Optimization Time: {optimizer_latency:.3f} ms")
        print(f"[Main Process] Optimized Config: {predicted_config}")
        video_config_queue.put(predicted_config)
        audio_config_queue.put(predicted_config)
        
        audio_package = audio_feature_queue.get()
        audio_waiting_time = time.time()
        video_package = video_feature_queue.get()
        video_waiting_time = time.time()

        if video_package is None or audio_package is None:
            break

        video_feature = video_package['feature']
        video_end_timestamp = video_package['end_time']
        video_spatial_encode_latency = video_package['spatial_encode_latency']
        video_temporal_encode_latency = video_package['temporal_encode_latency']
        video_sensing_end_time = video_package['sensing_end_time']
        audio_feature = audio_package['feature']
        audio_end_timestamp = audio_package['end_time']
        audio_chunk_encode_latency = audio_package['chunk_encode_latency']
        audio_temporal_encode_latency = audio_package['temporal_encode_latency']
        audio_sensing_end_time = audio_package['sensing_end_time']

        video_communication_latency = (video_waiting_time - video_end_timestamp) * 1000
        audio_communication_latency = (audio_waiting_time - audio_end_timestamp) * 1000
        print(f"[Main Process] Video Communication Latency: {video_communication_latency:.2f} ms, Audio Communication Latency: {audio_communication_latency:.2f} ms")

        video_idx = predicted_config[0]['video_model_id']
        audio_idx = predicted_config[0]['audio_model_id']

        video_feature = torch.from_numpy(video_feature).float().unsqueeze(0).to(device)
        audio_feature = torch.from_numpy(audio_feature).float().unsqueeze(0).to(device)
        
        fusion_model = fusion_models[video_idx * 3 + audio_idx]

        fusion_start_time = time.time()
        fused = fusion_model(video_feature, audio_feature)
        prediction = torch.argmax(fused, dim=1)
        fusion_end_time = time.time()
        fusion_latency = (fusion_end_time - fusion_start_time) * 1000

        correct += (prediction.item() == label)
        total += 1
        
        end_time = time.time()
        end_to_end_latency = (end_time - min(video_sensing_end_time, audio_sensing_end_time)) * 1000

        optimizer_latency_list.append(optimizer_latency)
        end_to_end_latency_list.append(end_to_end_latency)
        video_spatial_encode_latency_list.append(video_spatial_encode_latency)
        video_temporal_encode_latency_list.append(video_temporal_encode_latency)
        audio_chunk_encode_latency_list.append(audio_chunk_encode_latency)
        audio_temporal_encode_latency_list.append(audio_temporal_encode_latency)
        fusion_latency_list.append(fusion_latency)
        video_communication_latency_list.append(video_communication_latency)
        audio_communication_latency_list.append(audio_communication_latency)
        
        print(f"[Main Process] Sample {sample_id} - Video Timestamp: {(video_end_timestamp - start_time) * 1000:.3f}, Audio Timestamp: {(audio_end_timestamp - start_time) * 1000:.3f}, End-to-End Latency: {end_to_end_latency:.3f} ms")
        print(f"[Main Process] Video Spatial Encode Latency: {video_spatial_encode_latency:.3f} ms, Video Temporal Encode Latency: {video_temporal_encode_latency:.3f} ms")
        print(f"[Main Process] Audio Chunk Encode Latency: {audio_chunk_encode_latency:.3f} ms, Audio Temporal Encode Latency: {audio_temporal_encode_latency:.3f} ms")
        print(f"[Main Process] Fusion Latency: {fusion_latency:.3f} ms")
        Accuracy = (correct / total) * 100
        print(f"Prediction: {prediction.item()}, Ground Truth: {label}, Accuracy: {Accuracy:.3f}%")
        if len(end_to_end_latency_list) > args.avg_start:
            current_avg_latency = average(end_to_end_latency_list[args.avg_start:])
            print(f"Average End-to-End Latency: {current_avg_latency:.3f} ms")
        print(f"------------------------------------------------------------------------------------------------")
    
    video_process.terminate()
    audio_process.terminate()
    video_process.join()
    audio_process.join()

    avg_optimizer_latency = average(optimizer_latency_list[2:])
    avg_end_to_end_latency = average(end_to_end_latency_list[2:])
    avg_video_spatial_encode_latency = average(video_spatial_encode_latency_list[2:])
    avg_video_temporal_encode_latency = average(video_temporal_encode_latency_list[2:])
    avg_audio_chunk_encode_latency = average(audio_chunk_encode_latency_list[2:])
    avg_audio_temporal_encode_latency = average(audio_temporal_encode_latency_list[2:])
    avg_fusion_latency = average(fusion_latency_list[2:])
    avg_video_communication_latency = average(video_communication_latency_list[2:])
    avg_audio_communication_latency = average(audio_communication_latency_list[2:])

    print(f"Average Video Spatial Encode Latency: {avg_video_spatial_encode_latency:.3f} ms")
    print(f"Average Video Temporal Encode Latency: {avg_video_temporal_encode_latency:.3f} ms")
    print(f"Average Audio Chunk Encode Latency: {avg_audio_chunk_encode_latency:.3f} ms")
    print(f"Average Audio Temporal Encode Latency: {avg_audio_temporal_encode_latency:.3f} ms")
    print(f"Average Fusion Latency: {avg_fusion_latency:.3f} ms")
    print(f"Average Optimizer Latency: {avg_optimizer_latency:.3f} ms")
    print(f"Average End-to-End Latency: {avg_end_to_end_latency:.3f} ms")
    print(f"Average Video Communication Latency: {avg_video_communication_latency:.3f} ms")
    print(f"Average Audio Communication Latency: {avg_audio_communication_latency:.3f} ms")
    print(f"Accuracy: {correct / total * 100:.2f}% ({correct}/{total})")
    print(f"[Result] V{args.video_model}@{args.video_fps}FPS + {args.audio_model}({args.audio_chunk_size}): "
      f"{round(avg_end_to_end_latency, 2)} ms")
    
    latency_table_path = "latency_table.json"
    latency_table = load_latency_table(latency_table_path)
    set_latency(latency_table, args.video_model, args.video_fps, args.audio_model, args.audio_chunk_size, round(avg_end_to_end_latency, 2))
    
    with open(latency_table_path, 'w') as f:
        json.dump(latency_table, f, indent=4)


if __name__ == "__main__":
    main()
