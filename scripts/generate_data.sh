#!/usr/bin/env bash
# scripts/generate_data.sh  –  Generate beam ICD data for VQAv2
#
# Usage:
#   bash scripts/generate_data.sh [split] [output_prefix] [extra_args...]
#
# Examples:
#   bash scripts/generate_data.sh train data/vqav2_train_beams.json
#   bash scripts/generate_data.sh validation data/vqav2_val_beams.json --max_samples 5000
#   bash scripts/generate_data.sh train data/vqav2_train_beams.json --vqav2_path /data/vqav2

set -euo pipefail

SPLIT="${1:-train}"
OUTPUT="${2:-data/vqav2_${SPLIT}_beams.json}"
shift 2 || true

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
SHOT_NUM="${SHOT_NUM:-2}"
BEAM_SIZE="${BEAM_SIZE:-5}"
K="${K:-64}"
DEVICE="${DEVICE:-auto}"
VQAV2_PATH="${VQAV2_PATH:-}"

VQAV2_ARG=""
if [ -n "$VQAV2_PATH" ]; then
    VQAV2_ARG="--vqav2_path ${VQAV2_PATH}"
fi

echo "=== Generating ICD beam data ==="
echo "  Split:      $SPLIT"
echo "  Output:     $OUTPUT"
echo "  Qwen model: $QWEN_MODEL"
echo "  shot_num:   $SHOT_NUM"
echo "  beam_size:  $BEAM_SIZE"
echo "  K:          $K"

python generate_data.py \
    --split       "$SPLIT"      \
    --qwen_model  "$QWEN_MODEL" \
    --shot_num    "$SHOT_NUM"   \
    --beam_size   "$BEAM_SIZE"  \
    --K           "$K"          \
    --output_file "$OUTPUT"     \
    --device      "$DEVICE"     \
    $VQAV2_ARG                  \
    "$@"

echo "Done: $OUTPUT"
