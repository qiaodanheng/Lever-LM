#!/usr/bin/env python3
"""Export {base}.jsonl + {base}.pt → single .json with inline embeddings (for train.py pool_file)."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export(base: str, out: str) -> None:
    base_path = Path(base)
    jsonl = base_path if base_path.suffix == ".jsonl" else Path(str(base_path) + ".jsonl")
    pt = jsonl.with_suffix(".pt")
    out_path = Path(out)
    if out_path.exists():
        print(f"[skip] {out_path} exists")
        return
    items = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    embs = torch.load(pt, map_location="cpu", weights_only=True).float()
    if embs.shape[0] != len(items):
        raise RuntimeError(f"size mismatch: {len(items)} vs {embs.shape[0]}")
    records = []
    for i, it in enumerate(items):
        records.append({**it, "embedding": embs[i].tolist()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f)
    print(f"Exported {len(records)} → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Pool base path (jsonl+pt)")
    p.add_argument("--out", required=True, help="Output .json path")
    export(**vars(p.parse_args()))
