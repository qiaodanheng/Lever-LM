#!/usr/bin/env bash
# scripts/run_train.sh  –  Train PointerSelector with multi-beam RCE
#
# Usage:
#   bash scripts/run_train.sh [extra_args...]
#
# Key environment overrides:
#   LOSS_MODE=ce  bash scripts/run_train.sh          # ablation: single-target CE
#   GPUS=4        bash scripts/run_train.sh          # multi-GPU DDP
#   BATCH_SIZE=128 bash scripts/run_train.sh

set -euo pipefail

LOSS_MODE="${LOSS_MODE:-rce}"
GPUS="${GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
MAX_STEPS="${MAX_STEPS:-10000}"
PRECISION="${PRECISION:-bf16-mixed}"
TRAIN_FILE="${TRAIN_FILE:-data/vqav2_train_beams.json}"
VAL_FILE="${VAL_FILE:-data/vqav2_val_beams.json}"
RUN_NAME="${RUN_NAME:-lever_lm_pointer_${LOSS_MODE}}"

echo "=== Training Lever-LM PointerSelector ==="
echo "  loss_mode:  $LOSS_MODE"
echo "  GPUs:       $GPUS"
echo "  batch_size: $BATCH_SIZE"
echo "  lr:         $LR"
echo "  max_epochs: $MAX_EPOCHS"

python train.py \
    --train_file   "$TRAIN_FILE"   \
    --val_file     "$VAL_FILE"     \
    --loss_mode    "$LOSS_MODE"    \
    --gpus         "$GPUS"         \
    --batch_size   "$BATCH_SIZE"   \
    --lr           "$LR"           \
    --max_epochs   "$MAX_EPOCHS"   \
    --max_steps    "$MAX_STEPS"    \
    --precision    "$PRECISION"    \
    --run_name     "$RUN_NAME"     \
    "$@"

echo "Training complete. Checkpoints in checkpoints/$RUN_NAME/"
