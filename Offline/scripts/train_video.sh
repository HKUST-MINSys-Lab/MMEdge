#!/bin/bash

# 遍历 video backbone: 18 / 34 / 50
VIDEO_BACKBONES=("resnet18" "resnet34" "resnet50")

# 输出目录
SAVE_DIR="./checkpoints/video"
mkdir -p ${SAVE_DIR}

# 遍历训练
for backbone in "${VIDEO_BACKBONES[@]}"; do
    echo "======================================"
    echo " Training Video Model: ${backbone}"
    echo "======================================"

    python train_video.py \
        --video_backbone ${backbone} \
        --data_root /data/rxhuang/lipread_feature \
        --label_file ./data/selected_words.txt \
        --epochs 100 \
        --batch_size 128 \
        --num_workers 16 \
        --num_classes 50 \
        --save_path ${SAVE_DIR}/video_${backbone}.pth

    echo ""
done
