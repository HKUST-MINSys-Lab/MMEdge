# import torch
# import os
# from models.Audio_ResNet import AudioEncoder

# # 输入完整模型检查点目录
# checkpoint_dir = "./checkpoints/audio"
# # 输出路径（分别为 chunk encoder 和 temporal encoder）
# chunk_output_dir = "./checkpoints/audio_chunk_encoders"
# temporal_output_dir = "./checkpoints/audio_temporal_encoders"

# # 确保目录存在
# os.makedirs(chunk_output_dir, exist_ok=True)
# os.makedirs(temporal_output_dir, exist_ok=True)

# for size in ['small', 'medium', 'large']:
#     print(f"Processing {size} model...")

#     # 加载完整 AudioModel 的 state_dict
#     checkpoint_path = os.path.join(checkpoint_dir, f"audio_{size}_2.pth")
#     raw_state_dict = torch.load(checkpoint_path, map_location='cpu')

#     # 移除 classifier 相关权重（即 fc 层）
#     filtered_state_dict = {k: v for k, v in raw_state_dict.items() if not k.startswith("classifier.")}

#     # 创建结构匹配的 AudioEncoder，并加载过滤后的权重
#     model = AudioEncoder(size=size)
#     model.load_state_dict(filtered_state_dict, strict=False)

#     # 拆分
#     chunk_encoder = model.encoder
#     temporal_encoder = model.temporal

#     # 保存 chunk encoder
#     chunk_path = os.path.join(chunk_output_dir, f"audio_{size}_chunk_encoder.pth")
#     torch.save(chunk_encoder.state_dict(), chunk_path)
#     print(f"Saved chunk encoder to: {chunk_path}")

#     # 保存 temporal encoder
#     temporal_path = os.path.join(temporal_output_dir, f"audio_{size}_temporal_encoder.pth")
#     torch.save(temporal_encoder.state_dict(), temporal_path)
#     print(f"Saved temporal encoder to: {temporal_path}")

# print("All models processed and saved to separate directories.")


import json

file = json.load(open("./accuracy_table.json", "r"))

print(float(file['18']['20']['small']['1200']['0']))