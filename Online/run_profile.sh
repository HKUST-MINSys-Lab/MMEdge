#!/bin/bash

# 临时日志文件
LOG_FILE="end_to_end_latency_log.txt"
> $LOG_FILE  # 清空旧日志

# 最终 CSV 输出文件
CSV_FILE="end_to_end_latency_results.csv"
echo "video_model,video_fps,audio_model,audio_chunk_size,end_to_end_latency_ms" > $CSV_FILE

# 配置组合
video_models=(18 34 50)
video_fps=(20 25 29)
audio_models=("small" "medium" "large")
audio_chunks=(800 1000 1200)

# 遍历所有组合
for v_model in "${video_models[@]}"; do
  for fps in "${video_fps[@]}"; do
    for a_model in "${audio_models[@]}"; do
      for chunk in "${audio_chunks[@]}"; do
        echo "Running: Video=ResNet$v_model, FPS=$fps, Audio=$a_model, Chunk=$chunk"
        
        # 运行 main.py，将输出写入日志文件
        python main_profiling.py \
          --cpu_limitation \
          --video_model $v_model \
          --video_fps $fps \
          --audio_model $a_model \
          --audio_chunk_size $chunk >> $LOG_FILE 
          
        # grep 最后一条 [Result] 并提取字段
        result_line=$(grep "\[Result\]" $LOG_FILE | tail -n 1)

        if [[ $result_line =~ V([0-9]+)@([0-9]+)FPS[[:space:]]+\+[[:space:]]+([a-z]+)\(([0-9]+)\):[[:space:]]+([0-9.]+)[[:space:]]+ms ]]; then
          echo "${BASH_REMATCH[1]},${BASH_REMATCH[2]},${BASH_REMATCH[3]},${BASH_REMATCH[4]},${BASH_REMATCH[5]}" >> $CSV_FILE
        else
          echo "Failed to parse result for: V$v_model, FPS=$fps, A=$a_model, chunk=$chunk"
        fi
      done
    done
  done
done

echo "All configurations completed. Results saved to $CSV_FILE"

kill -9 20104
