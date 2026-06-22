#!/bin/bash
# 数据下载看护脚本：完全脱离 SSH/IDE 会话运行，确保 COCO + SST-2 全部下完。
# 用 hf-mirror.com 国内镜像；hf download 支持断点续传，重复运行是幂等的。

export HF_ENDPOINT=https://hf-mirror.com
cd /home/jiyi/lizhiheng/Lever-LM || exit 1

LOG=data/download_watchdog.log
PRIMARY_PID=666874   # 之前已在跑的那个 COCO 下载进程
echo "================ watchdog start $(date) ================" >> "$LOG"

# 1) 等之前已在运行的 COCO 下载进程自然结束，避免并发锁冲突
while kill -0 "$PRIMARY_PID" 2>/dev/null; do
  echo "[$(date)] primary download (pid $PRIMARY_PID) still running, wait 30s..." >> "$LOG"
  sleep 30
done
echo "[$(date)] primary download finished, start verify/resume loop" >> "$LOG"

# 2) 循环续传 COCO val（已完成的分片会自动跳过；未完成的断点续传）
ok=0
for i in $(seq 1 80); do
  conda run -n leverlm hf download lmms-lab/COCO-Caption2017 \
      --repo-type dataset --include "data/val-*.parquet" \
      --local-dir data/coco_caption2017 >> "$LOG" 2>&1
  rc=$?
  n=$(ls data/coco_caption2017/data/val-*.parquet 2>/dev/null | wc -l)
  inc=$(find data/coco_caption2017 -name "*.incomplete" 2>/dev/null | wc -l)
  echo "[$(date)] COCO attempt $i: rc=$rc parquet=$n incomplete=$inc" >> "$LOG"
  if [ "$rc" -eq 0 ] && [ "$n" -ge 2 ] && [ "$inc" -eq 0 ]; then
    echo "[$(date)] >>> COCO val DONE" >> "$LOG"
    ok=1
    break
  fi
  sleep 15
done
[ "$ok" -eq 0 ] && echo "[$(date)] !!! COCO val NOT complete after retries" >> "$LOG"

# 3) 校验 SST-2（之前已下完，缺了就补）
if [ ! -s data/sst2/train.jsonl ]; then
  echo "[$(date)] SST-2 missing, re-download" >> "$LOG"
  for i in $(seq 1 20); do
    conda run -n leverlm hf download SetFit/sst2 --repo-type dataset \
        --local-dir data/sst2 >> "$LOG" 2>&1
    [ -s data/sst2/train.jsonl ] && break
    sleep 10
  done
fi
echo "[$(date)] SST-2 train.jsonl size: $(stat -c%s data/sst2/train.jsonl 2>/dev/null)" >> "$LOG"

# 4) 汇总
echo "---- FINAL STATE $(date) ----" >> "$LOG"
ls -la data/coco_caption2017/data/ >> "$LOG" 2>&1
ls -la data/sst2/*.jsonl >> "$LOG" 2>&1
echo "================ watchdog done $(date) ================" >> "$LOG"
