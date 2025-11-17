#!/bin/bash

# 遍历 audio backbone: small / medium / large
AUDIO_BACKBONES=("small" "medium" "large")

# 输出目录
SAVE_DIR="./checkpoints/audio"
mkdir -p ${SAVE_DIR}

# 遍历训练
for backbone in "${AUDIO_BACKBONES[@]}"; do
    echo "======================================"
    echo " Training Audio Model: ${backbone}"
    echo "======================================"

    python train_audio.py \
        --backbone ${backbone} \
        --data_root /data/rxhuang/lipread_feature \
        --label_file ./data/selected_words.txt \
        --epochs 100 \
        --batch_size 128 \
        --num_workers 16 \
        --num_classes 50 \
        --save_path ${SAVE_DIR}/audio_${backbone}.pth

    echo ""
done
