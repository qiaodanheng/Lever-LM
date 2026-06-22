#!/usr/bin/env bash
# scripts/run_eval.sh  –  ICL evaluation with trained PointerSelector
#
# Usage:
#   bash scripts/run_eval.sh <ckpt_path> [extra_args...]
#
# Examples:
#   bash scripts/run_eval.sh checkpoints/lever_lm_pointer_rce/last.ckpt
#   bash scripts/run_eval.sh checkpoints/lever_lm_pointer_rce/last.ckpt --sweep
#   BASELINE=random bash scripts/run_eval.sh none   # random retrieval baseline

set -euo pipefail

CKPT_PATH="${1:-}"
shift || true

POOL_FILE="${POOL_FILE:-data/vqav2_pool_v2.json}"
QWEN_MODEL="${QWEN_MODEL:-/home/jiyi/.cache/modelscope/qwen/Qwen2-VL-2B-Instruct}"
SHOT_NUM="${SHOT_NUM:-2}"
K="${K:-32}"
SPLIT="${SPLIT:-validation}"
OUTPUT="${OUTPUT:-results/eval_results.json}"
BASELINE="${BASELINE:-}"

CKPT_ARG=""
BASELINE_ARG=""

if [ -n "$BASELINE" ]; then
    BASELINE_ARG="--baseline ${BASELINE}"
else
    if [ -z "$CKPT_PATH" ] || [ "$CKPT_PATH" = "none" ]; then
        echo "ERROR: provide <ckpt_path> or set BASELINE=random"
        exit 1
    fi
    CKPT_ARG="--ckpt_path ${CKPT_PATH}"
fi

echo "=== ICL Inference Evaluation ==="
echo "  checkpoint: ${CKPT_PATH:-<none>}"
echo "  baseline:   ${BASELINE:-<none>}"
echo "  split:      $SPLIT"
echo "  shot_num:   $SHOT_NUM"

python icl_inference.py \
    --pool_file    "$POOL_FILE"   \
    --qwen_model   "$QWEN_MODEL"  \
    --shot_num     "$SHOT_NUM"    \
    --K            "$K"           \
    --split        "$SPLIT"       \
    --output_file  "$OUTPUT"      \
    $CKPT_ARG                     \
    $BASELINE_ARG                 \
    "$@"
