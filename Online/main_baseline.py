import os
import time
import torch
import argparse
import subprocess
import torch.multiprocessing as mp
import torch.nn.functional as F

from data.lipreading_dataset import LipreadingDataset
from video_stream_baseline import video_stream
from audio_stream_baseline import audio_stream

from utils import average
from models.audio_model import AudioModelBaselineEncoder, AudioEncoder
from models.video_model_3d import Video_Encoder, Video_Classification
from models.fusion_model import MultiModalFusion
from models.imputation import Generator


video_idx_map = {'18': 0, '34': 1, '50': 2}
audio_idx_map = {'small': 0, 'medium': 1, 'large': 2}


def load_models(args):
    if args.model_selection:
        video_model_bad = Video_Encoder(pretrain=False).eval()
        checkpoint_path_bad = os.path.join(args.checkpoint_path, f'baselines/model_selection/video_model.pth')
        video_pretrained_dict_bad = torch.load(checkpoint_path_bad)
        video_pretrained_dict_bad = {k.replace('module.', ''): v for k, v in video_pretrained_dict_bad.items()}
        video_model_bad.load_state_dict(video_pretrained_dict_bad, strict=False)
        
        audio_model_bad = AudioEncoder('medium').eval()
        checkpoint_path_bad = os.path.join(args.checkpoint_path, f'baselines/model_selection/audio_model.pth')
        audio_pretrained_dict_bad = torch.load(checkpoint_path_bad)
        audio_pretrained_dict_bad = {k.replace('module.', ''): v for k, v in audio_pretrained_dict_bad.items()}
        audio_model_bad.load_state_dict(audio_pretrained_dict_bad, strict=False)
        
        fusion_model_bad = MultiModalFusion().to(args.device).eval()
        checkpoint_path_bad = os.path.join(args.checkpoint_path, f'baselines/model_selection/fusion_model_2.pth')
        fusion_pretrained_dict_bad = torch.load(checkpoint_path_bad)
        fusion_pretrained_dict_bad = {k.replace('module.', ''): v for k, v in fusion_pretrained_dict_bad.items()}
        fusion_model_bad.load_state_dict(fusion_pretrained_dict_bad, strict=False)
        
    video_model = Video_Classification(feature_dim=512).eval()
    checkpoint_path = os.path.join(args.checkpoint_path, f'baselines/video_resnet50_3D.pth')
    video_pretrained_dict = torch.load(checkpoint_path)
    video_pretrained_dict = {k.replace('module.', ''): v for k, v in video_pretrained_dict.items()}
    video_model.load_state_dict(video_pretrained_dict, strict=False)
    
    audio_model = AudioModelBaselineEncoder().eval()
    checkpoint_path = os.path.join(args.checkpoint_path, f'baselines/audio_baseline.pth')
    audio_pretrained_dict = torch.load(checkpoint_path)
    audio_model.load_state_dict(audio_pretrained_dict, strict=False)
     
    fusion_model = MultiModalFusion().to(args.device).eval()
    if args.model_selection:
        checkpoint_name = f"baselines/model_selection/fusion_model_2.pth"
    else:
        checkpoint_name = f"baselines/fusion.pth"
    checkpoint_path = os.path.join(args.checkpoint_path, checkpoint_name)
    fusion_pretrained_dict = torch.load(checkpoint_path)
    fusion_pretrained_dict = {k.replace('module.', ''): v for k, v in fusion_pretrained_dict.items()}
    fusion_model.load_state_dict(fusion_pretrained_dict, strict=False)

    if args.model_selection:
        return audio_model, video_model, fusion_model, audio_model_bad, video_model_bad, fusion_model_bad
    else:
        return audio_model, video_model, fusion_model


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
    parser.add_argument("--avg_start", type=int, default=10)
    parser.add_argument("--cpu_limitation", action='store_true', help="Enable cpu limitation if set")
    parser.add_argument("--sample_cnt", type=int, default=500)
    parser.add_argument("--imputation", action='store_true', help="Enable imputation if set")
    parser.add_argument("--model_selection", action='store_true', help="Enable model selection if set")
    args = parser.parse_args()

    device = args.device
    
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

    # Queues for data and result communication
    video_data_queue = mp.Queue()
    audio_data_queue = mp.Queue()
    video_config_queue = mp.Queue()
    audio_config_queue = mp.Queue()
    video_feature_queue = mp.Queue()
    audio_feature_queue = mp.Queue()

    # Load models
    if args.model_selection:
        audio_model, video_model, fusion_model, audio_model_bad, video_model_bad, fusion_model_bad = load_models(args)
    else:
        audio_model, video_model, fusion_model = load_models(args)
        audio_model_bad = None
        video_model_bad = None
        
    if args.imputation:
        generator = Generator().to(device).eval()
        checkpoint_path = os.path.join(args.checkpoint_path, f'baselines/imputation/generator_best.pth')
        generator_pretrained_dict = torch.load(checkpoint_path)
        generator_pretrained_dict = {k.replace('module.', ''): v for k, v in generator_pretrained_dict.items()}
        generator.load_state_dict(generator_pretrained_dict, strict=False)

    # Load dataset
    dataset = LipreadingDataset(root_dir='/home/jetson/Dataset',
                                label_file='./data/selected_words.txt',
                                sample_cnt=args.sample_cnt)

    # Start worker processes
    video_process = mp.Process(target=video_stream, args=(args, video_data_queue, video_feature_queue, video_model, video_model_bad))
    audio_process = mp.Process(target=audio_stream, args=(args, audio_data_queue, audio_feature_queue, audio_model, audio_model_bad))

    video_process.start()
    audio_process.start()

    correct = 0
    total = 0

    end_to_end_latency_list = []
    video_encode_latency_list = []
    audio_encode_latency_list = []
    fusion_latency_list = []
    video_communication_latency_list = []
    audio_communication_latency_list = []
    waiting_latency_list = []
    
    hard_latency_list = []
    easy_latency_list = []
    hard_correct = 0
    easy_correct = 0
    hard_total = 0
    easy_total = 0
    
    easy_classes = [36, 48, 7, 47, 29, 30, 27, 17, 43, 13]
    hard_classes = [23, 33, 3, 9, 37, 46, 26, 18, 32, 11]

    for sample_id, (video_path, audio_path, label) in enumerate(dataset):
        # Step 1: Send video/audio path for first-frame feature extraction
        start_time = time.time()
        video_data_queue.put({"sample_id": sample_id, "video_path": video_path, "start_time": start_time})
        audio_data_queue.put({"sample_id": sample_id, "audio_path": audio_path, "start_time": start_time})
        
        audio_package = audio_feature_queue.get()
        audio_waiting_time = time.time()
        video_package = video_feature_queue.get()
        video_waiting_time = time.time()
        
        if video_package is None or audio_package is None:
            break

        video_feature = video_package['feature']
        video_end_timestamp = video_package['end_time']
        video_encode_latency = video_package['video_encode_latency']
        video_sensing_end_time = video_package['sensing_end_time']
        audio_feature = audio_package['feature']
        audio_end_timestamp = audio_package['end_time']
        audio_encode_latency = audio_package['audio_encode_latency']
        audio_sensing_end_time = audio_package['sensing_end_time']

        video_communication_latency = (video_waiting_time - video_end_timestamp) * 1000
        audio_communication_latency = (audio_waiting_time - audio_end_timestamp) * 1000
        waiting_latency = (video_sensing_end_time - audio_sensing_end_time) * 1000
        waiting_latency_list.append(waiting_latency)
        print(f"[Main Process] Video Communication Latency: {video_communication_latency:.2f} ms, Audio Communication Latency: {audio_communication_latency:.2f} ms, Waiting Latency: {waiting_latency:.2f} ms")

        
        video_feature = torch.from_numpy(video_feature).float().unsqueeze(0).to(device)
        audio_feature = torch.from_numpy(audio_feature).float().unsqueeze(0).to(device)
        if args.imputation:
            video_feature = generator(video_feature, audio_feature)
        
        fusion_start_time = time.time()
        if args.model_selection:
            fused = fusion_model_bad(video_feature, audio_feature)
        else:
            fused = fusion_model(video_feature, audio_feature)
        prediction = torch.argmax(fused, dim=1)
        fusion_end_time = time.time()
        fusion_latency = (fusion_end_time - fusion_start_time) * 1000

        correct += (prediction.item() == label)
        total += 1
        
        end_time = time.time()
        end_to_end_latency = (end_time - min(video_sensing_end_time, audio_sensing_end_time)) * 1000

        end_to_end_latency_list.append(end_to_end_latency)
        video_encode_latency_list.append(video_encode_latency)
        audio_encode_latency_list.append(audio_encode_latency)
        fusion_latency_list.append(fusion_latency)
        video_communication_latency_list.append(video_communication_latency)
        audio_communication_latency_list.append(audio_communication_latency)
        
        if label in easy_classes:
            easy_latency_list.append(end_to_end_latency)
            easy_correct += 1 if prediction.item() == label else 0
            easy_total += 1
        elif label in hard_classes:
            hard_latency_list.append(end_to_end_latency)
            hard_correct += 1 if prediction.item() == label else 0
            hard_total += 1
            
        print(f"[Main Process] Sample {sample_id} - Video Timestamp: {(video_end_timestamp - start_time) * 1000:.2f}, Audio Timestamp: {(audio_end_timestamp - start_time) * 1000:.2f}, End-to-End Latency: {end_to_end_latency:.2f} ms")
        print(f"[Main Process] Audio Encode Latency: {audio_encode_latency:.2f} ms, Video Encoding Latency: {video_encode_latency:.2f} ms")
        print(f"[Main Process] Fusion Latency: {fusion_latency:.2f} ms")
        Accuracy = (correct / total) * 100
        easy_Accuracy = (easy_correct / easy_total) * 100 if easy_total > 0 else 0
        hard_Accuracy = (hard_correct / hard_total) * 100 if hard_total > 0 else 0
        print(f"Prediction: {prediction.item()}, Ground Truth: {label}, Accuracy: {Accuracy:.2f}%")
        print(f"Easy Accuracy: {easy_Accuracy:.2f}% ({easy_correct}/{easy_total})")
        print(f"Hard Accuracy: {hard_Accuracy:.2f}% ({hard_correct}/{hard_total})")
        if sample_id >= args.avg_start:
            current_avg_latency = average(end_to_end_latency_list[args.avg_start:])
            print(f"Average End-to-End Latency: {current_avg_latency:.2f} ms")
        print(f"------------------------------------------------------------------------------------------------")
    
    video_process.terminate()
    audio_process.terminate()
    video_process.join()
    audio_process.join()

    avg_end_to_end_latency = average(end_to_end_latency_list[args.avg_start:])
    avg_video_encode_latency = average(video_encode_latency_list[args.avg_start:])
    avg_audio_encode_latency = average(audio_encode_latency_list[args.avg_start:])
    avg_fusion_latency = average(fusion_latency_list[args.avg_start:])
    avg_video_communication_latency = average(video_communication_latency_list[args.avg_start:])
    avg_audio_communication_latency = average(audio_communication_latency_list[args.avg_start:])
    avg_waiting_latency = average(waiting_latency_list[args.avg_start:])
    avg_easy_latency = average(easy_latency_list[args.avg_start:])
    avg_hard_latency = average(hard_latency_list[args.avg_start:])
    
    print(f"Average Video Encode Latency: {avg_video_encode_latency:.2f} ms")
    print(f"Average Audio Encode Latency: {avg_audio_encode_latency:.2f} ms")
    print(f"Average Fusion Latency: {avg_fusion_latency:.2f} ms")
    print(f"Average End-to-End Latency: {avg_end_to_end_latency:.2f} ms")
    print(f"Average Video Communication Latency: {avg_video_communication_latency:.2f} ms")
    print(f"Average Audio Communication Latency: {avg_audio_communication_latency:.2f} ms")
    print(f"Average Waiting Latency: {avg_waiting_latency:.2f} ms")
    print(f"Accuracy: {correct / total * 100:.2f}% ({correct}/{total})")
    print(f"Easy Accuracy: {easy_Accuracy:.2f}% ({easy_correct}/{easy_total})")
    print(f"Hard Accuracy: {hard_Accuracy:.2f}% ({hard_correct}/{hard_total})")
    print(f"Average Easy Latency: {avg_easy_latency:.2f} ms")
    print(f"Average Hard Latency: {avg_hard_latency:.2f} ms")

if __name__ == "__main__":
    main()
