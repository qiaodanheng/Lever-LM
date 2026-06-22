# Lever-LM (Qwen2-VL)

Reproduction and improvement of [Lever-LM](https://github.com/ForJadeForest/Lever-LM) (ICLR 2024) using **Qwen2-VL-2B** as the frozen LVLM and a trainable **autoregressive PointerSelector** for ordered in-context demonstration (ICD) selection.

## Key improvements (v2)

| | Original Lever-LM | This repo (v2) |
|---|---|---|
| **LVLM** | Flamingo / IDEFICS | **Qwen2-VL-2B** (Instruct & Base) |
| **Selector** | Permutation-invariant logits sum | **Autoregressive pointer decoder** (`[3,9] ≠ [9,3]`) |
| **Scoring** | Substring match (many −∞ rewards) | **Teacher-forcing InfoScore** |
| **Training** | Single-target CE | **Multi-beam RCE + pool InfoNCE** |
| **Tasks** | VQA + Caption + SST-2 | Same three tasks |
| **Embedding dim** | CLIP 768 | **Qwen hidden 1536** |

## Project structure

```
Lever-LM/
├── lever_lm/
│   ├── pointer_selector.py    # Autoregressive PointerSelector
│   ├── qwen_vl_scorer.py      # Qwen-VL scoring, embedding, InfoScore
│   ├── lever_lm_module.py     # PyTorch Lightning training module
│   ├── dataset.py             # Beam dataset + candidate pool
│   └── tasks.py               # VQA / Caption / SST-2 task registry
├── generate_data.py           # Step 1: beam ICD data generation
├── train.py                   # Step 2: PointerSelector training
├── icl_inference.py           # Step 3: ICL evaluation
├── scripts/                   # Pipeline & utility scripts
├── configs/                   # YAML configs
└── docs/                      # Experiment notes (Chinese)
```

## Setup

```bash
git clone https://github.com/qiaodanheng/Lever-LM.git
cd Lever-LM

conda create -n leverlm python=3.10 && conda activate leverlm
pip install -r requirements.txt

cp .env.example .env
# Edit .env: VQAV2_PATH, QWEN_MODEL_NAME, etc.
```

**Note:** Large artifacts (`data/`, `checkpoints/`, `results/`) are not tracked in git. Download VQAv2 / COCO / SST-2 and run the pipeline locally, or use HuggingFace datasets online.

## Quick start (3-step pipeline)

### Step 1 – Generate beam data

```bash
python generate_data.py \
    --task vqa \
    --reward_mode info \
    --anchor_random \
    --num_anchors 5000 \
    --pool_size 20000 \
    --beam_size 3 \
    --K 32 \
    --output_file data/vqav2_train_beams_v2.json \
    --resume
```

### Step 2 – Train PointerSelector

```bash
python train.py \
    --train_file data/vqav2_train_beams_v2_train.json \
    --val_file   data/vqav2_train_beams_v2_val.json \
    --pool_file  data/vqav2_pool_v2.json \
    --d_model 1536 \
    --K 32 --shot_num 2 --max_beams 3 \
    --loss_mode rce --neg_weight 1.0 \
    --run_name lever_lm_v2_poolneg
```

### Step 3 – ICL evaluation

```bash
python icl_inference.py \
    --ckpt_path checkpoints/lever_lm_v2_poolneg/last.ckpt \
    --pool_file data/vqav2_pool_v2.json \
    --K 32 --shot_num 2 \
    --split validation --max_samples 5000 \
    --eval_protocol unified \
    --output_file results/icl_v2_lever_unified_n5000.json
```

## Key hyper-parameters

| param | value | notes |
|---|---|---|
| `d_model` | **1536** | Must match Qwen-VL hidden size |
| `hidden_dim` | 256 | Pointer attention dimension |
| `K` | 32 | Retrieved candidates per query |
| `shot_num` | 2 | ICDs per query |
| `beam_size` | 3 | Beams per anchor in Step 1 |
| `loss_mode` | rce | `rce` or `ce` |

## Results (VQA val 5000, unified protocol, Qwen2-VL-2B-Instruct)

| Method | Acc (shot=2) | Avg:1~8 |
|---|---|---|
| **Lever** | **63.4%** | **62.0%** |
| Random | 61.7% | 61.2% |
| 0-shot | 70.5% | 70.5% |

Lever consistently beats Random (+1.7% at shot=2). See `docs/项目总结_完整版_2026-06-05.md` for full experiment notes.

## Citation

```bibtex
@inproceedings{leverlm2024,
  title     = {Lever-LM: Configuring In-Context Sequence for Leveraging Demostration Learning},
  author    = {ForJadeForest and others},
  booktitle = {ICLR},
  year      = {2024}
}
```
