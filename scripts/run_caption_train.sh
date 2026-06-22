#!/bin/bash
# Caption 数据生成完成后执行：切分 train/val + 训练 selector（池负样本）
set -e
cd /home/jiyi/lizhiheng/Lever-LM

BEAMS=data/caption_train_beams_v2.json
if [ ! -f "$BEAMS" ]; then
  echo "缺少 $BEAMS，请先跑完 generate_data.py --task caption"
  exit 1
fi

echo "=== 切分 train/val (1800/200) ==="
conda run -n leverlm python -c "
import json, random
d=json.load(open('$BEAMS'))
print('total', len(d))
rng=random.Random(42)
idx=list(range(len(d))); rng.shuffle(idx)
n_val=200
val_idx=set(idx[:n_val])
train=[d[i] for i in range(len(d)) if i not in val_idx]
val=[d[i] for i in range(len(d)) if i in val_idx]
json.dump(train, open('data/caption_train_beams_v2_train.json','w'))
json.dump(val, open('data/caption_train_beams_v2_val.json','w'))
print('train', len(train), 'val', len(val))
"

echo "=== 训练 Caption selector ==="
setsid nohup env PYTHONPATH=. conda run -n leverlm python train.py \
  --train_file data/caption_train_beams_v2_train.json \
  --val_file data/caption_train_beams_v2_val.json \
  --d_model 1536 --K 32 --shot_num 2 --max_beams 3 \
  --loss_mode rce --neg_weight 1.0 --num_neg 16 \
  --pool_file data/caption_pool_v2.json --num_distractor 64 \
  --batch_size 32 --num_workers 0 --gpus 1 --precision bf16-mixed \
  --max_epochs 80 --max_steps 8000 --warmup_steps 200 --lr 2e-4 \
  --run_name lever_lm_caption_poolneg \
  > data/train_caption.log 2>&1 < /dev/null &
echo "训练已启动 PID=$!  日志: data/train_caption.log"
