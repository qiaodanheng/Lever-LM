#!/bin/bash
# 统一评测协议：三方法均 short-answer suffix + extract + strict match
set -e
cd /home/jiyi/lizhiheng/Lever-LM
export PYTHONPATH=.
CKPT=checkpoints/lever_lm_v2_poolneg/epoch003-valloss7.0768.ckpt
POOL=data/vqav2_pool_v2.json
N=${MAX_SAMPLES:-500}
K=32
SHOT=2
PROTO=unified
TAG="unified_n${N}"

run_one() {
  local name=$1; shift
  echo "========== ${name} $(date) =========="
  conda run -n leverlm python icl_inference.py \
    --pool_file "$POOL" --K "$K" --shot_num "$SHOT" \
    --split validation --max_samples "$N" \
    --eval_protocol "$PROTO" \
    --output_file "results/icl_v2_${name}_${TAG}.json" \
    "$@" 2>&1 | tee -a "data/icl_${TAG}.log" | tail -8
}

run_one lever   --ckpt_path "$CKPT"
run_one random  --baseline random
run_one zeroshot --baseline zeroshot

echo "=== 统一协议汇总 (protocol=${PROTO}) ==="
conda run -n leverlm python -c "
import json, glob
for f in sorted(glob.glob('results/icl_v2_*_${TAG}.json')):
    d=json.load(open(f))
    print(f.split('/')[-1], 'acc=', round(d['accuracy'],4), 'protocol=', d.get('eval_protocol'))
"
