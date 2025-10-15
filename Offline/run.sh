#!/bin/bash


# 配置
audio_models=("small" "medium" "large")
audio_chunks=(1200 1000 800)
video_models=(18 34 50)
video_fps=(20 25 29)

# 遍历所有组合
for a_model in "${audio_models[@]}"; do
  for chunk in "${audio_chunks[@]}"; do
    for v_model in "${video_models[@]}"; do
      for fps in "${video_fps[@]}"; do
        echo "Running Audio=$a_model Chunk=$chunk Video=ResNet$v_model FPS=$fps"

        python make_accuracy_table.py \
          --audio_model "$a_model" \
          --audio_chunk_size "$chunk" \
          --video_model "$v_model" \
          --video_fps "$fps"
      done
    done
  done
done

echo "All runs complete. Results saved to accuracy_table.json"
