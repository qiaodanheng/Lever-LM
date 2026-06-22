#!/bin/bash
# 对齐原版 Lever-LM：全 train 候选池 (~443k) + ICL K=32
# Phase A: embed 全库（可 --resume，后台跑 ~30-40h）
# Phase B: val5000 unified ICL（Lever / Random / 0-shot）
set -euo pipefail
cd /home/jiyi/lizhiheng/Lever-LM
export PYTHONPATH=.

POOL_BASE=data/vqav2_pool_full
POOL_JSONL="${POOL_BASE}.jsonl"
POOL_PT="${POOL_BASE}.pt"
IMG_DIR=data/pool_images_full
LOG=data/fullpool_align.log
CKPT=checkpoints/lever_lm_v2_poolneg/epoch003-valloss7.0768.ckpt
K=32
N=5000
PROTO=unified

exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " Full-pool align pipeline  $(date)"
echo " Pool: ${POOL_BASE}  K=${K}  val=${N}"
echo "============================================================"

# ── Phase A: build full pool ──
TOTAL=$(conda run -n leverlm python -c "from lever_lm.tasks import get_task; print(get_task('vqa').total('train'))")
DONE=0
if [[ -f "$POOL_JSONL" ]]; then
  DONE=$(wc -l < "$POOL_JSONL")
fi
echo "Train total=${TOTAL}  embedded=${DONE}"

if [[ "$DONE" -lt "$TOTAL" ]]; then
  echo ">>> Phase A: embedding pool ${DONE}/${TOTAL}  $(date)"
  conda run -n leverlm python scripts/build_pool_only.py \
    --task vqa --split train \
    --pool_out "$POOL_BASE" \
    --pool_images_dir "$IMG_DIR" \
    --embed_batch 16 --load_chunk 64 --checkpoint_every 500 \
    --resume
else
  echo ">>> Phase A: pool already complete (${DONE} rows)"
fi

# ── Phase B: ICL val5000 K=32 ──
run_icl() {
  local name=$1; shift
  local out="results/icl_fullpool_${name}_unified_n${N}_k${K}.json"
  echo ""
  echo "========== ICL ${name} fullpool K=${K} n=${N} $(date) =========="
  conda run -n leverlm python icl_inference.py \
    --pool_file "$POOL_JSONL" --K "$K" --shot_num 2 \
    --split validation --max_samples "$N" \
    --eval_protocol "$PROTO" \
    --output_file "$out" \
    "$@"
}

echo ""
echo ">>> Phase B: ICL val${N} unified protocol  $(date)"
run_icl lever   --ckpt_path "$CKPT"
run_icl random  --baseline random
run_icl zeroshot --baseline zeroshot

echo ""
echo "=== Summary $(date) ==="
conda run -n leverlm python - <<PY
import json, glob
from pathlib import Path
for f in sorted(glob.glob("results/icl_fullpool_*_unified_n${N}_k${K}.json")):
    d = json.load(open(f))
    print(f"  {Path(f).name}: acc={d['accuracy']:.4f}")
PY

echo ""
echo "============================================================"
echo " DONE $(date)"
echo "============================================================"
