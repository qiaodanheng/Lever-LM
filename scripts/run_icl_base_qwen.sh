#!/usr/bin/env bash
# Qwen2-VL-2B (base, 非 Instruct) 复现与 Instruct 相同的 ICL 实验
# 输出独立前缀 icl_base_* / vqav2_pool_*_base，不会覆盖 icl_v2_* / icl_fullpool_*
# Phase 1: 重 embed 候选池 → Phase 2: val5000 ICL → Phase 3: shot sweep → Phase 4: 443k 池
set -euo pipefail
cd /home/jiyi/lizhiheng/Lever-LM
export PYTHONPATH=.

QWEN_BASE="/home/jiyi/.cache/modelscope/qwen/Qwen2-VL-2B"
CKPT=checkpoints/lever_lm_v2_poolneg/epoch003-valloss7.0768.ckpt
N=5000
K=32
SHOT=2
PROTO=unified
LOG=data/icl_base_qwen_pipeline.log

# 20k 池（与 icl_v2_* 对齐）
POOL_20K=data/vqav2_pool_v2_base
# 443k 全量池（与 icl_fullpool_* 对齐）
POOL_FULL=data/vqav2_pool_full_base

exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " Qwen2-VL-2B BASE ICL pipeline  $(date)"
echo " model=$QWEN_BASE"
echo "============================================================"

# ── Phase 0: re-embed pools with base model ──
embed_pool() {
  local src=$1 out=$2
  local n_done=0
  if [[ -f "${out}.jsonl" ]]; then n_done=$(wc -l < "${out}.jsonl"); fi
  local n_total
  if [[ "$src" == *.json ]]; then
    n_total=$(python -c "import json; print(len(json.load(open('$src'))))")
  else
    n_total=$(wc -l < "$src")
  fi
  if [[ "$n_done" -ge "$n_total" ]] && [[ -f "${out}.pt" ]] && [[ "${FORCE_EMBED:-0}" != "1" ]]; then
    echo "[SKIP embed] ${out} complete (${n_done}/${n_total})"
    return 0
  fi
  echo ">>> Re-embed ${out} from ${src} (${n_done}/${n_total}) $(date)"
  conda run -n leverlm python scripts/reembed_pool.py \
    --pool_file "$src" --pool_out "$out" \
    --qwen_model "$QWEN_BASE" --embed_batch 32 --resume
}

echo ""
echo ">>> PHASE 0: embed 20k pool $(date)"
embed_pool data/vqav2_pool_v2.json "$POOL_20K"

run_eval() {
  local pool=$1 name=$2 out=$3
  shift 3
  if [[ -f "$out" ]] && [[ "${FORCE:-0}" != "1" ]]; then
    echo "[SKIP] $out exists"
    return 0
  fi
  echo ""
  echo "========== EVAL ${name} pool=$(basename $pool) $(date) =========="
  conda run -n leverlm python icl_inference.py \
    --pool_file "${pool}.jsonl" --qwen_model "$QWEN_BASE" \
    --K "$K" --shot_num "$SHOT" \
    --split validation --max_samples "$N" \
    --eval_protocol "$PROTO" \
    --output_file "$out" \
    "$@"
}

run_sweep() {
  local pool=$1 name=$2 out=$3
  shift 3
  local summary="${out%.json}.sweep_summary.json"
  if [[ -f "$summary" ]] && [[ "${FORCE:-0}" != "1" ]]; then
    echo "[SKIP] $summary exists"
    return 0
  fi
  echo ""
  echo "========== SWEEP ${name} $(date) =========="
  conda run -n leverlm python icl_inference.py \
    --pool_file "${pool}.jsonl" --qwen_model "$QWEN_BASE" \
    --K "$K" --sweep \
    --split validation --max_samples "$N" \
    --eval_protocol "$PROTO" \
    --output_file "$out" \
    "$@"
}

summarize() {
  conda run -n leverlm python - <<'PY'
import json, glob
from pathlib import Path

print("\n--- Qwen2-VL-2B BASE val5000 shot=2 ---")
for tag in ["icl_base_20k", "icl_base_fullpool"]:
    for f in sorted(glob.glob(f"results/{tag}_*_unified_n5000*.json")):
        if ".sweep" in f:
            continue
        d = json.load(open(f))
        print(f"  {Path(f).name}: acc={d['accuracy']:.4f}")

print("\n--- BASE sweep avg_1_8 (20k pool) ---")
for f in sorted(glob.glob("results/icl_base_20k_*_unified_n5000.sweep_summary.json")):
    d = json.load(open(f))
    print(f"  {Path(f).name}: Avg1~8={d.get('avg_1_8',0):.4f}")

print("\n--- Instruct baseline (for comparison) ---")
for f in sorted(glob.glob("results/icl_v2_*_unified_n5000.json")):
    if ".sweep" in f:
        continue
    d = json.load(open(f))
    print(f"  {Path(f).name}: acc={d['accuracy']:.4f}")
for f in sorted(glob.glob("results/icl_fullpool_*_unified_n5000_k32.json")):
    d = json.load(open(f))
    print(f"  {Path(f).name}: acc={d['accuracy']:.4f}")
PY
}

# ── Phase 1: val5000 shot=2 (20k pool) ──
echo ""
echo ">>> PHASE 1: val5000 shot=2 on 20k pool $(date)"
run_eval "$POOL_20K" lever   results/icl_base_20k_lever_unified_n5000.json   --ckpt_path "$CKPT"
run_eval "$POOL_20K" random  results/icl_base_20k_random_unified_n5000.json  --baseline random
run_eval "$POOL_20K" zeroshot results/icl_base_20k_zeroshot_unified_n5000.json --baseline zeroshot
summarize

# ── Phase 2: shot sweep 1~8 (20k pool) ──
echo ""
echo ">>> PHASE 2: shot sweep on 20k pool $(date)"
run_sweep "$POOL_20K" lever   results/icl_base_20k_lever_unified_n5000.sweep.json   --ckpt_path "$CKPT"
run_sweep "$POOL_20K" random  results/icl_base_20k_random_unified_n5000.sweep.json  --baseline random
run_sweep "$POOL_20K" zeroshot results/icl_base_20k_zeroshot_unified_n5000.sweep.json --baseline zeroshot
summarize

# ── Phase 3: full pool (443k) if embedded ──
if [[ "${SKIP_FULLPOOL:-0}" != "1" ]]; then
  echo ""
  echo ">>> PHASE 3: embed + ICL full pool $(date)"
  embed_pool data/vqav2_pool_full.jsonl "$POOL_FULL"
  run_eval "$POOL_FULL" lever   results/icl_base_fullpool_lever_unified_n5000_k32.json   --ckpt_path "$CKPT"
  run_eval "$POOL_FULL" random  results/icl_base_fullpool_random_unified_n5000_k32.json  --baseline random
  run_eval "$POOL_FULL" zeroshot results/icl_base_fullpool_zeroshot_unified_n5000_k32.json --baseline zeroshot
  summarize
fi

echo ""
echo "============================================================"
echo " BASE pipeline DONE $(date)"
echo "============================================================"
