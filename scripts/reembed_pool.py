#!/usr/bin/env python3
"""Re-embed candidate pool metadata with a different Qwen-VL model."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lever_lm.qwen_vl_scorer import QwenVLScorer


def _load_meta(pool_file: str) -> List[Dict]:
    path = Path(pool_file)
    if path.suffix == ".json":
        with open(path) as f:
            items = json.load(f)
        return [{k: v for k, v in it.items() if k != "embedding"} for it in items]
    if path.suffix == ".jsonl":
        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    jsonl = path.with_suffix(".jsonl")
    if jsonl.exists():
        return _load_meta(str(jsonl))
    raise ValueError(f"Cannot load pool metadata from {pool_file}")


def _atomic_save_pt(tensor: torch.Tensor, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, tmp)
    os.replace(tmp, path)


def reembed(args: argparse.Namespace) -> None:
    items = _load_meta(args.pool_file)
    base = Path(args.pool_out)
    jsonl_path = Path(str(base) + ".jsonl")
    pt_path = Path(str(base) + ".pt")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    start = 0
    embs = torch.zeros(0, 1536)
    if args.resume and jsonl_path.exists() and pt_path.exists():
        with open(jsonl_path) as f:
            start = sum(1 for line in f if line.strip())
        if start:
            embs = torch.load(pt_path, map_location="cpu", weights_only=True).float()
            if embs.shape[0] != start:
                raise RuntimeError(
                    f"Resume mismatch: {start} jsonl rows vs {embs.shape[0]} embeddings"
                )
            print(f"[resume] {start}/{len(items)} already embedded")

    if args.force:
        start = 0
        embs = torch.zeros(0, 1536)
        if jsonl_path.exists():
            jsonl_path.unlink()
        if pt_path.exists():
            pt_path.unlink()

    if start >= len(items):
        print(f"Already complete: {start} embeddings")
        return

    scorer = QwenVLScorer(model_name=args.qwen_model, device=args.device)
    pending_meta: List[Dict] = []
    pending_emb: List[torch.Tensor] = []

    def flush(force: bool = False) -> None:
        nonlocal embs, pending_meta, pending_emb
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
        print(f"  [checkpoint] {embs.shape[0]}/{len(items)} → {pt_path.name}")
        pending_meta, pending_emb = [], []

    for i in tqdm(range(start, len(items), args.embed_batch), desc="Re-embed"):
        sub = items[i: i + args.embed_batch]
        batch_embs = scorer.embed_batch(
            [s["image"] for s in sub],
            [s["question"] for s in sub],
        )
        for item, emb in zip(sub, batch_embs):
            pending_meta.append(item)
            pending_emb.append(emb)
        flush()

    flush(force=True)
    print(f"Done: {len(items)} items → {jsonl_path} + {pt_path} {tuple(embs.shape)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-embed pool with new Qwen model")
    p.add_argument("--pool_file", required=True, help="Source .json or .jsonl (metadata only)")
    p.add_argument("--pool_out", required=True, help="Output base path → {base}.jsonl + {base}.pt")
    p.add_argument("--qwen_model", required=True)
    p.add_argument("--embed_batch", type=int, default=32)
    p.add_argument("--checkpoint_every", type=int, default=500)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    reembed(parse_args())
