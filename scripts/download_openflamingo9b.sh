#!/usr/bin/env bash
# OpenFlamingo-9B 下载脚本（使用 Hugging Face 国内镜像 hf-mirror.com）
set -euo pipefail

# ── 国内镜像 ──
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0   # 部分环境 hf_transfer 与镜像不兼容，关闭更稳

# ── 统一存放路径（与 Qwen 模型同级）──
BASE="${MODELSCOPE_CACHE:-/home/jiyi/.cache/modelscope}"
OF_DIR="${BASE}/openflamingo/OpenFlamingo-9B-vitl-mpt7b"
MPT_DIR="${BASE}/mpt/mpt-7b"
LOG="${LOG:-/home/jiyi/lizhiheng/Lever-LM/data/openflamingo9b_download.log}"

mkdir -p "$OF_DIR" "$MPT_DIR" "$(dirname "$LOG")"

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "OpenFlamingo checkpoint → $OF_DIR"
echo "MPT-7B backbone       → $MPT_DIR"
echo "Log                   → $LOG"
echo ""

{
  echo "=== $(date '+%F %T') [1/2] OpenFlamingo-9B checkpoint (5.5G) ==="
  hf download openflamingo/OpenFlamingo-9B-vitl-mpt7b \
    --local-dir "$OF_DIR"

  echo "=== $(date '+%F %T') [2/2] MPT-7B backbone (26.6G, OpenFlamingo 依赖) ==="
  hf download anas-awadalla/mpt-7b \
    --local-dir "$MPT_DIR"

  echo "=== $(date '+%F %T') DONE ==="
  echo ""
  echo "文件检查："
  ls -lh "$OF_DIR/checkpoint.pt" 2>/dev/null || echo "  checkpoint.pt 未找到"
  ls -lh "$MPT_DIR"/pytorch_model*.bin 2>/dev/null | head -5 || echo "  MPT-7B 权重未找到"
} 2>&1 | tee -a "$LOG"
