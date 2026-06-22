#!/usr/bin/env bash
# 守护进程：SSH 断开或子进程崩溃后自动续跑 Base 全流程
# 用法: setsid nohup bash scripts/base_pipeline_watchdog.sh >> data/base_pipeline_watchdog.log 2>&1 &
set -euo pipefail
cd /home/jiyi/lizhiheng/Lever-LM

LOG=data/base_pipeline_watchdog.log
PIPELINE=scripts/run_base_full_pipeline.sh
DONE=data/base_full_pipeline.DONE
INTERVAL="${WATCHDOG_INTERVAL:-300}"  # 5 分钟检查一次

is_done() {
  [[ -f "$DONE" ]]
}

is_running() {
  pgrep -f "run_base_full_pipeline.sh" >/dev/null 2>&1 && return 0
  pgrep -f "generate_data.py.*vqav2_train_beams_v2_base" >/dev/null 2>&1 && return 0
  pgrep -f "train.py.*lever_lm_v2_base_poolneg" >/dev/null 2>&1 && return 0
  pgrep -f "icl_inference.py.*icl_basefull" >/dev/null 2>&1 && return 0
  return 1
}

launch() {
  echo "[$(date '+%F %T')] launch pipeline"
  setsid nohup bash "$PIPELINE" >> data/base_full_pipeline.log 2>&1 < /dev/null &
  echo "[$(date '+%F %T')] pipeline pid=$!"
}

echo "[$(date '+%F %T')] watchdog start (interval=${INTERVAL}s)"

while ! is_done; do
  if is_running; then
    echo "[$(date '+%F %T')] pipeline running, sleep ${INTERVAL}s"
  else
    echo "[$(date '+%F %T')] pipeline NOT running, restarting..."
    launch
  fi
  sleep "$INTERVAL"
done

echo "[$(date '+%F %T')] watchdog exit: $DONE exists"
