"""
build_pool.py  –  Build the candidate pool file for icl_inference.py evaluation.

This script reconstructs the pool from:
  1. Embeddings already computed in vqav2_train_beams.json (query_emb fields)
  2. Original images/questions/answers from the VQAv2 parquet data

Output format per item:
    {
        "id":        str,
        "image":     str,  # absolute path to saved JPEG
        "question":  str,
        "answer":    str,
        "embedding": [float, ...]  # 1536-dim Qwen-VL embedding
    }

Usage:
    python build_pool.py \
        --beams_file data/vqav2_train_beams.json \
        --output_file data/vqav2_pool.json \
        --images_dir data/pool_images
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm


PARQUET_SNAP = "/home/jiyi/lizhiheng/Lever-LM/data/datasets--lmms-lab--VQAv2/snapshots/32665d35052eb4a6d4414851c3c829a72754915a/data"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--beams_file",   default="data/vqav2_train_beams.json")
    p.add_argument("--output_file",  default="data/vqav2_pool.json")
    p.add_argument("--images_dir",   default="data/pool_images")
    p.add_argument("--max_samples",  type=int, default=5000)
    return p.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1: load embeddings from beam records (query_emb = pool embedding)
    # ------------------------------------------------------------------ #
    print("Loading beam records …")
    with open(args.beams_file) as f:
        beams = json.load(f)

    id_to_emb = {r["query_id"]: r["query_emb"] for r in beams}
    print(f"  {len(id_to_emb)} embeddings loaded")

    # ------------------------------------------------------------------ #
    # Step 2: load training data from parquet (PIL images)
    # ------------------------------------------------------------------ #
    print("Loading VQAv2 train from parquet …")
    from datasets import load_dataset
    ds = load_dataset(
        "parquet",
        data_files={"train": f"{PARQUET_SNAP}/train-*.parquet"},
        split="train",
    )
    if args.max_samples and args.max_samples < len(ds):
        ds = ds.select(range(args.max_samples))
    print(f"  {len(ds)} samples loaded")

    # ------------------------------------------------------------------ #
    # Step 3: match, save images, build pool records
    # ------------------------------------------------------------------ #
    pool = []
    missing = 0
    for row in tqdm(ds, desc="Building pool"):
        qid = str(row["question_id"])
        if qid not in id_to_emb:
            missing += 1
            continue

        # Save PIL image to disk
        img_path = images_dir / f"{qid}.jpg"
        if not img_path.exists():
            pil_img = row["image"]
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_img.save(img_path, format="JPEG", quality=85)

        pool.append({
            "id":        qid,
            "image":     str(img_path.resolve()),
            "question":  row["question"],
            "answer":    row["multiple_choice_answer"],
            "embedding": id_to_emb[qid],
        })

    print(f"Pool size: {len(pool)}  (missing embeddings: {missing})")

    # ------------------------------------------------------------------ #
    # Step 4: save
    # ------------------------------------------------------------------ #
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(pool, f)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
