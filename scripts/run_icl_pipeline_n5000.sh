#!/bin/bash
# 按顺序后台跑：统一协议 val5000 (shot=2) → shot sweep 1~8 (5000)
# 用法: nohup bash scripts/run_icl_pipeline_n5000.sh &
set -euo pipefail
cd /home/jiyi/lizhiheng/Lever-LM
export PYTHONPATH=.

CKPT=checkpoints/lever_lm_v2_poolneg/epoch003-valloss7.0768.ckpt
POOL=data/vqav2_pool_v2.json
N=5000
K=32
SHOT=2
PROTO=unified
LOG=data/icl_pipeline_n5000.log
STAMP=$(date +%Y%m%d_%H%M%S)

exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " ICL pipeline start: $(date)  N=${N}  protocol=${PROTO}"
echo " PID=$$  log=$LOG"
echo "============================================================"

run_eval() {
  local name=$1; shift
  local out="results/icl_v2_${name}_unified_n${N}.json"
  if [[ -f "$out" ]] && [[ "${FORCE:-0}" != "1" ]]; then
    echo "[SKIP] $out exists ($(date))"
    return 0
  fi
  echo ""
  echo "========== EVAL ${name} n=${N} shot=${SHOT} $(date) =========="
  conda run -n leverlm python icl_inference.py \
    --pool_file "$POOL" --K "$K" --shot_num "$SHOT" \
    --split validation --max_samples "$N" \
    --eval_protocol "$PROTO" \
    --output_file "$out" \
    "$@"
}

run_sweep() {
  local name=$1; shift
  local out="results/icl_v2_${name}_unified_n${N}.sweep.json"
  local summary="${out%.json}.sweep_summary.json"
  if [[ -f "$summary" ]] && [[ "${FORCE:-0}" != "1" ]]; then
    echo "[SKIP] $summary exists ($(date))"
    return 0
  fi
  echo ""
  echo "========== SWEEP ${name} n=${N} shots=1..8 $(date) =========="
  conda run -n leverlm python icl_inference.py \
    --pool_file "$POOL" --K "$K" --sweep \
    --split validation --max_samples "$N" \
    --eval_protocol "$PROTO" \
    --output_file "$out" \
    "$@"
}

summarize() {
  echo ""
  echo "=== 汇总 $(date) ==="
  conda run -n leverlm python - <<'PY'
import json, glob
from pathlib import Path

print("\n--- val5000 shot=2 ---")
for f in sorted(glob.glob("results/icl_v2_*_unified_n5000.json")):
    if ".sweep" in f:
        continue
    d = json.load(open(f))
    print(f"  {Path(f).name}: acc={d['accuracy']:.4f}")

print("\n--- sweep avg_1_8 ---")
for f in sorted(glob.glob("results/icl_v2_*_unified_n5000.sweep_summary.json")):
    d = json.load(open(f))
    avg = d.get("avg_1_8", 0)
    shots = d.get("shot_accs", {})
    print(f"  {Path(f).name}: Avg1~8={avg:.4f}  per_shot={shots}")
PY
}

# ── Phase 1: val 5000, shot=2 ──
echo ""
echo ">>> PHASE 1: unified val5000 (shot=2) $(date)"
run_eval lever   --ckpt_path "$CKPT"
run_eval random  --baseline random
run_eval zeroshot --baseline zeroshot
summarize

# ── Phase 2: shot sweep 1~8 on 5000 ──
echo ""
echo ">>> PHASE 2: shot sweep 1~8 on n=${N} $(date)"
run_sweep lever   --ckpt_path "$CKPT"
run_sweep random  --baseline random
run_sweep zeroshot --baseline zeroshot
summarize

echo ""
echo "============================================================"
echo " ICL pipeline DONE: $(date)"
echo "============================================================"
