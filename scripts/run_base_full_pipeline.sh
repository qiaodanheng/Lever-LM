#!/usr/bin/env bash
# Qwen2-VL-2B Base 全流程：Step1 → 训练 Selector → ICL 评测
# 与 Instruct 实验对齐，并复用已 embed 的 base 池（跳过重 embed）
set -euo pipefail
cd /home/jiyi/lizhiheng/Lever-LM
export PYTHONPATH=.

QWEN_BASE="/home/jiyi/.cache/modelscope/qwen/Qwen2-VL-2B"
POOL_BASE="data/vqav2_pool_v2_base"
POOL_JSON="${POOL_BASE}.json"
BEAMS="data/vqav2_train_beams_v2_base.json"
BEAMS_TRAIN="data/vqav2_train_beams_v2_base_train.json"
BEAMS_VAL="data/vqav2_train_beams_v2_base_val.json"
CKPT_DIR="checkpoints/lever_lm_v2_base_poolneg"
LOG="data/base_full_pipeline.log"

exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " Base 全流程 pipeline  $(date)"
echo " model=$QWEN_BASE"
echo " log=$LOG"
echo "============================================================"

# ── 准备：jsonl+pt → json（train 用 pool_file）──
if [[ ! -f "$POOL_JSON" ]]; then
  echo ">>> Export pool json for training $(date)"
  conda run -n leverlm python scripts/export_pool_jsonl_to_json.py \
    --base "$POOL_BASE" --out "$POOL_JSON"
fi

# ── Step1: Beam 数据（复用 base 池 embed，不重复 embed 2 万条）──
N_DONE=0
if [[ -f "$BEAMS" ]]; then
  N_DONE=$(python3 -c "import json; print(len(json.load(open('$BEAMS'))))")
fi
if [[ "$N_DONE" -lt 5000 ]]; then
  echo ""
  echo ">>> STEP1: generate beams Base ($N_DONE/5000) $(date)"
  conda run -n leverlm python generate_data.py \
    --task vqa \
    --qwen_model "$QWEN_BASE" \
    --anchor_random \
    --pool_size 20000 \
    --max_samples 5000 \
    --K 32 \
    --beam_size 3 \
    --shot_num 2 \
    --score_batch_size 16 \
    --reward_mode info \
    --pool_in "$POOL_BASE" \
    --pool_out "$POOL_JSON" \
    --pool_images_dir data/pool_images_v2_base \
    --output_file "$BEAMS" \
    --save_every 200 \
    --resume \
    --seed 42
else
  echo ">>> STEP1: skip ($N_DONE beams done)"
fi

# ── 切分 train/val 4500/500（与 Instruct 一致）──
echo ""
echo ">>> Split train/val 4500/500 $(date)"
conda run -n leverlm python - <<PY
import json, random
d=json.load(open("$BEAMS"))
assert len(d) >= 5000, f"need 5000 beams, got {len(d)}"
rng=random.Random(42)
idx=list(range(len(d))); rng.shuffle(idx)
val_idx=set(idx[:500])
train=[d[i] for i in range(len(d)) if i not in val_idx]
val=[d[i] for i in range(len(d)) if i in val_idx]
json.dump(train, open("$BEAMS_TRAIN","w"))
json.dump(val, open("$BEAMS_VAL","w"))
print("train", len(train), "val", len(val))
PY

# ── Step2: 训练 Selector ──
if ! compgen -G "$CKPT_DIR/*.ckpt" > /dev/null; then
  echo ""
  echo ">>> STEP2: train PointerSelector on Base beams $(date)"
  conda run -n leverlm python train.py \
    --train_file "$BEAMS_TRAIN" \
    --val_file "$BEAMS_VAL" \
    --d_model 1536 --K 32 --shot_num 2 --max_beams 3 \
    --loss_mode rce --neg_weight 1.0 --num_neg 16 \
    --pool_file "$POOL_JSON" --num_distractor 64 \
    --batch_size 32 --num_workers 0 --gpus 1 --precision bf16-mixed \
    --max_epochs 80 --max_steps 8000 --warmup_steps 200 --lr 2e-4 \
    --run_name lever_lm_v2_base_poolneg
else
  echo ">>> STEP2: ckpt exists in $CKPT_DIR, skip train"
fi

# 选最佳 ckpt（优先 epoch003 样式，否则 last）
CKPT=$(ls -1 "$CKPT_DIR"/epoch*.ckpt 2>/dev/null | sort -V | head -1 || true)
CKPT="${CKPT:-$CKPT_DIR/last.ckpt}"
echo "Using ckpt: $CKPT"

# ── Step3: ICL 评测（20k base 池，独立结果前缀 icl_basefull_*）──
run_icl() {
  local name=$1 out=$2
  shift 2
  if [[ -f "$out" ]] && [[ "${FORCE:-0}" != "1" ]]; then
    echo "[SKIP] $out"
    return 0
  fi
  echo ""
  echo "========== ICL $name $(date) =========="
  conda run -n leverlm python icl_inference.py \
    --pool_file "${POOL_BASE}.jsonl" \
    --qwen_model "$QWEN_BASE" \
    --K 32 --shot_num "${SHOT:-2}" \
    --split validation --max_samples 5000 \
    --eval_protocol unified \
    --output_file "$out" \
    "$@"
}

echo ""
echo ">>> STEP3: ICL val5000 shot=2 $(date)"
run_icl lever   results/icl_basefull_20k_lever_unified_n5000.json   --ckpt_path "$CKPT"
run_icl random  results/icl_basefull_20k_random_unified_n5000.json  --baseline random
run_icl zeroshot results/icl_basefull_20k_zeroshot_unified_n5000.json --baseline zeroshot

# 改进：额外跑 1-shot（Instruct sweep 显示 1-shot 常优于 2-shot）
echo ""
echo ">>> STEP3b: 1-shot Lever/Random $(date)"
SHOT=1
run_icl lever_1shot  results/icl_basefull_20k_lever_1shot_unified_n5000.json  --ckpt_path "$CKPT"
run_icl random_1shot results/icl_basefull_20k_random_1shot_unified_n5000.json --baseline random

echo ""
echo "=== Summary $(date) ==="
conda run -n leverlm python - <<'PY'
import json, glob
from pathlib import Path
print("\n--- Base 全流程 ICL ---")
for f in sorted(glob.glob("results/icl_basefull_20k_*_unified_n5000.json")):
    d=json.load(open(f))
    print(f"  {Path(f).name}: {d['accuracy']*100:.2f}%")
print("\n--- Base 半套（Instruct ckpt，对照）---")
for f in sorted(glob.glob("results/icl_base_20k_*_unified_n5000.json")):
    if ".sweep" in f: continue
    d=json.load(open(f))
    print(f"  {Path(f).name}: {d['accuracy']*100:.2f}%")
print("\n--- Instruct 全流程 ---")
for f in sorted(glob.glob("results/icl_v2_*_unified_n5000.json")):
    if ".sweep" in f: continue
    d=json.load(open(f))
    print(f"  {Path(f).name}: {d['accuracy']*100:.2f}%")
PY

echo ""
echo "============================================================"
echo " Base 全流程 DONE $(date)"
echo "============================================================"
touch data/base_full_pipeline.DONE
