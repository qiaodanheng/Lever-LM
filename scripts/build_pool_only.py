#!/usr/bin/env python3
"""
Build VQAv2 candidate pool only (embed train split, no beam search).

Outputs (base = --pool_out, e.g. data/vqav2_pool_full):
  {base}.jsonl   metadata: id, image path, question, answer  (append-friendly)
  {base}.pt      float32 embeddings [N, D]

Supports --resume: skip already-embedded rows (by line count in jsonl).

Usage (full train ~443757, align with original Lever-LM):
  python scripts/build_pool_only.py \\
    --pool_out data/vqav2_pool_full \\
    --pool_images_dir data/pool_images_full \\
    --split train --resume --checkpoint_every 500

Legacy small pool (single JSON with inline embeddings) still works via generate_data.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lever_lm.qwen_vl_scorer import QwenVLScorer
from lever_lm.tasks import get_task


def _save_image(item: Dict, images_dir: Path) -> Optional[str]:
    img = item.get("image")
    if img is None:
        return None
    p = images_dir / f"{item['id']}.jpg"
    if not p.exists():
        if isinstance(img, str):
            img = Image.open(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(p, format="JPEG", quality=85)
    return str(p.resolve())


def _load_checkpoint(base: Path) -> Tuple[List[Dict], torch.Tensor]:
    jsonl = Path(str(base) + ".jsonl")
    pt = Path(str(base) + ".pt")
    records: List[Dict] = []
    if jsonl.exists():
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if pt.exists() and records:
        embs = torch.load(pt, map_location="cpu", weights_only=True)
        if embs.shape[0] != len(records):
            raise RuntimeError(
                f"Checkpoint mismatch: {len(records)} jsonl rows vs {embs.shape[0]} emb rows"
            )
    elif records:
        raise RuntimeError(f"Found {jsonl} but missing {pt}")
    else:
        embs = torch.zeros(0, 1536)
    return records, embs


def _atomic_save_pt(tensor: torch.Tensor, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, tmp)
    os.replace(tmp, path)


def build_pool(args: argparse.Namespace) -> None:
    base = Path(args.pool_out)
    jsonl_path = Path(str(base) + ".jsonl")
    pt_path = Path(str(base) + ".pt")
    images_dir = Path(args.pool_images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    base.parent.mkdir(parents=True, exist_ok=True)

    task = get_task(args.task)
    total = task.total(args.split)
    if args.pool_size is not None:
        total = min(args.pool_size, total)
    indices = list(range(total))

    records, embs = [], torch.zeros(0, 1536)
    start_at = 0
    if args.resume:
        records, embs = _load_checkpoint(base)
        start_at = len(records)
        if start_at > 0:
            print(f"[resume] {start_at}/{total} already embedded → continue from index {start_at}")

    scorer = QwenVLScorer(model_name=args.qwen_model, device=args.device)

    pending_meta: List[Dict] = []
    pending_emb: List[torch.Tensor] = []

    def flush_pending(force: bool = False) -> None:
        nonlocal records, embs, pending_meta, pending_emb
        if not pending_meta:
            return
        if not force and len(pending_meta) < args.checkpoint_every:
            return
        with open(jsonl_path, "a") as f:
            for rec in pending_meta:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        new_embs = torch.stack(pending_emb, dim=0)
        embs = torch.cat([embs, new_embs], dim=0) if embs.numel() else new_embs
        _atomic_save_pt(embs, pt_path)
        records.extend(pending_meta)
        pending_meta, pending_emb = [], []
        print(f"  [checkpoint] saved {len(records)}/{total} → {jsonl_path.name} {tuple(embs.shape)}")

    idx_iter = indices[start_at:]
    for i in tqdm(range(0, len(idx_iter), args.load_chunk), desc="Embed pool"):
        chunk_idx = idx_iter[i: i + args.load_chunk]
        batch_items = task.load(args.split, indices=chunk_idx)

        for j in range(0, len(batch_items), args.embed_batch):
            sub = batch_items[j: j + args.embed_batch]
            batch_embs = scorer.embed_batch(
                [s["image"] for s in sub],
                [s["question"] for s in sub],
            )
            for item, emb in zip(sub, batch_embs):
                image_ref = _save_image(item, images_dir)
                pending_meta.append({
                    "id": item["id"],
                    "image": image_ref,
                    "question": item["question"],
                    "answer": item["answer"],
                })
                pending_emb.append(emb)
            flush_pending()

    flush_pending(force=True)
    print(f"Done. Pool size={len(records)}  jsonl={jsonl_path}  pt={pt_path}  ({embs.shape})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build candidate pool embeddings only")
    p.add_argument("--task", default="vqa")
    p.add_argument("--split", default="train")
    p.add_argument("--pool_size", type=int, default=None,
                   help="Subsample pool (default: full split, ~443757 for VQA train)")
    p.add_argument("--pool_out", required=True,
                   help="Output base path, writes {base}.jsonl and {base}.pt")
    p.add_argument("--pool_images_dir", required=True)
    p.add_argument("--qwen_model", default="/home/jiyi/.cache/modelscope/qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--embed_batch", type=int, default=16)
    p.add_argument("--load_chunk", type=int, default=64,
                   help="Parquet rows loaded per chunk (memory control)")
    p.add_argument("--checkpoint_every", type=int, default=500)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Delete existing checkpoint and rebuild from scratch")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base = Path(args.pool_out)
    if args.force:
        for p in [Path(str(base) + ".jsonl"), Path(str(base) + ".pt")]:
            if p.exists():
                p.unlink()
                print(f"Removed {p}")
    elif not args.resume:
        jsonl = Path(str(base) + ".jsonl")
        if jsonl.exists():
            raise SystemExit(
                f"{jsonl} exists. Use --resume to continue or --force to rebuild."
            )
    build_pool(args)
