#!/bin/bash
# 正式重跑 VQA Step1（经济档）：随机anchor + 解耦2万候选池 + K=32 + beam3
# + InfoScore 增益 reward + teacher-forcing 修 −∞。
# 输出写到 *_v2 文件，保留旧实验便于前后对比。断点每 200 anchor 保存一次。
cd /home/jiyi/lizhiheng/Lever-LM || exit 1
conda run -n leverlm python generate_data.py \
    --anchor_random \
    --pool_size 20000 \
    --max_samples 5000 \
    --K 32 \
    --beam_size 3 \
    --shot_num 2 \
    --score_batch_size 16 \
    --reward_mode info \
    --pool_out data/vqav2_pool_v2.json \
    --pool_images_dir data/pool_images_v2 \
    --output_file data/vqav2_train_beams_v2.json \
    --save_every 200 \
    --seed 42
