#!/bin/bash

# 所有 video backbones
VIDEO_LIST=("resnet18" "resnet34" "resnet50")

# 所有 audio backbones
AUDIO_LIST=("small" "medium" "large")

SAVE_DIR="./checkpoints/fusion"
mkdir -p ${SAVE_DIR}

# 遍历所有组合
for v in "${VIDEO_LIST[@]}"; do
    for a in "${AUDIO_LIST[@]}"; do

        echo "============================================"
        echo " Training Fusion Model: Video=${v}, Audio=${a}"
        echo "============================================"

        python train_fusion.py \
            --video_backbone ${v} \
            --audio_backbone ${a} \
            --data_root /data/rxhuang/lipread_feature \
            --label_file ./data/selected_words.txt \
            --epochs 100 \
            --batch_size 128 \
            --num_workers 16 \
            --num_classes 50 \
            --save_path ${SAVE_DIR}/fusion_${v}_${a}.pth

        echo ""
    done
done
