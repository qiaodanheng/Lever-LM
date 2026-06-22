#!/bin/bash
# VQA Step3 ICL 评测：Lever-LM vs random vs 0-shot（v2 配置）
set -e
cd /home/jiyi/lizhiheng/Lever-LM
export PYTHONPATH=.
CKPT=checkpoints/lever_lm_v2_poolneg/epoch003-valloss7.0768.ckpt
POOL=data/vqav2_pool_v2.json
N=${MAX_SAMPLES:-500}
K=32
SHOT=2

run_one() {
  local tag=$1; shift
  echo "========== $tag $(date) =========="
  conda run -n leverlm python icl_inference.py \
    --pool_file "$POOL" --K "$K" --shot_num "$SHOT" \
    --split validation --max_samples "$N" \
    --output_file "results/icl_v2_${tag}_n${N}.json" \
    "$@" 2>&1 | tail -5
}

run_one lever   --ckpt_path "$CKPT"
run_one random  --baseline random
run_one zeroshot --baseline zeroshot

echo "=== 汇总 ==="
conda run -n leverlm python -c "
import json, glob
for f in sorted(glob.glob('results/icl_v2_*_n${N}.json')):
    d=json.load(open(f)); print(f.split('/')[-1], 'acc=', round(d['accuracy'],4))
"
