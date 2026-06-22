# 待办：扩池 Ablation

**更新：** 2026-06-06 — 跳过 10 万中间档，**直接全量 443,757 + K=32**，后台已启动。

## 当前状态 🔄

- **Phase A 进行中：** embed 全 train → `data/vqav2_pool_full.jsonl` + `.pt`
- **Phase B 排队：** 建池完成后自动跑 val5000 ICL（Lever/Random/0-shot，K=32，unified）
- **日志：** `data/fullpool_align.log`
- **进度：** `wc -l data/vqav2_pool_full.jsonl`（目标 443757）

## 已完成

- [x] `scripts/build_pool_only.py`（jsonl+pt，--resume）
- [x] `VQAv2CandidatePool` 支持 `.jsonl` + `.pt` 大池
- [x] `scripts/run_fullpool_align.sh`
- [x] 后台 pipeline 启动

## 后续（视 Phase B 结果）

- [ ] 若 Lever 仍 < 0-shot 但差距缩小 → Step1 用全库重生成 + 微调 selector
- [ ] 若 Lever ≥ 0-shot → 写进报告，可选 shot sweep @ full pool
- [ ] Caption / SST-2 ICL（独立任务）

## 关键命令

```bash
tail -f data/fullpool_align.log
wc -l data/vqav2_pool_full.jsonl
ls -lh data/vqav2_pool_full.pt
```
