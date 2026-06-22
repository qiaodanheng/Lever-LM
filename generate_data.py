"""
generate_data.py  –  Build the beam-search ICD sequence dataset for Lever-LM.

For each query in the VQAv2 training set:
  1. Retrieve K nearest candidates from the pool using random / BM25 / embedding
     similarity (configurable; default: FAISS cosine on Qwen-VL embeddings).
  2. Run beam-search over those K candidates: at each step score each candidate
     with Qwen-VL (Eq. 3) to extend existing partial sequences.
  3. Persist all `beam_size` complete sequences + their rewards to JSON.

The resulting JSON is consumed directly by VQAv2BeamDataset / DataModule.

Usage:
    python generate_data.py \
        --split train \
        --qwen_model Qwen/Qwen2-VL-2B-Instruct \
        --shot_num 2 \
        --beam_size 5 \
        --K 64 \
        --output_file data/vqav2_train_beams.json \
        [--vqav2_path /path/to/vqav2]      # local; omit to use HF online
        [--max_samples 1000]               # small-scale debug
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from lever_lm.qwen_vl_scorer import QwenVLScorer
from lever_lm.dataset import load_vqav2_hf
from lever_lm.tasks import get_task


# --------------------------------------------------------------------------- #
#  VQAv2 local / online loader
# --------------------------------------------------------------------------- #

def load_vqav2_local(root: str, split: str) -> List[Dict]:
    """
    Minimal local VQAv2 loader.  Expects:
        <root>/v2_OpenEnded_mscoco_{split}2014_questions.json
        <root>/v2_mscoco_{split}2014_annotations.json  (for GT answers)
        <root>/images/{split}2014/COCO_{split}2014_*.jpg
    """
    q_file = os.path.join(
        root, f"v2_OpenEnded_mscoco_{split}2014_questions.json"
    )
    a_file = os.path.join(
        root, f"v2_mscoco_{split}2014_annotations.json"
    )
    with open(q_file) as f:
        questions = {q["question_id"]: q for q in json.load(f)["questions"]}
    with open(a_file) as f:
        annotations = {
            a["question_id"]: a for a in json.load(f)["annotations"]
        }

    img_dir = os.path.join(root, "images", f"{split}2014")
    samples = []
    for qid, q in questions.items():
        ann = annotations.get(qid, {})
        img_path = os.path.join(
            img_dir,
            f"COCO_{split}2014_{q['image_id']:012d}.jpg",
        )
        samples.append({
            "id": str(qid),
            "image": img_path,
            "question": q["question"],
            "answer": ann.get("multiple_choice_answer", ""),
        })
    return samples


PARQUET_SNAP = "/home/jiyi/lizhiheng/Lever-LM/data/datasets--lmms-lab--VQAv2/snapshots/32665d35052eb4a6d4414851c3c829a72754915a/data"


def _open_vqav2_parquet(split: str):
    """Open the VQAv2 parquet split lazily (images NOT decoded yet)."""
    from datasets import load_dataset
    return load_dataset(
        "parquet",
        data_files={split: f"{PARQUET_SNAP}/{split}-*.parquet"},
        split=split,
    )


def vqav2_len(split: str) -> int:
    """Total number of rows in the parquet split (no image decode)."""
    return len(_open_vqav2_parquet(split))


def load_vqav2_parquet(
    split: str,
    max_samples: Optional[int] = None,
    indices: Optional[List[int]] = None,
) -> List[Dict]:
    """Load VQAv2 from local parquet (lmms-lab format). Slices BEFORE decoding.

    If `indices` is given, exactly those rows are loaded (used for random
    anchor / decoupled-pool sampling). Otherwise the first `max_samples`.
    """
    print(f"Loading VQAv2 {split} from local parquet …")
    ds = _open_vqav2_parquet(split)
    if indices is not None:
        ds = ds.select(indices)
        print(f"  → selecting {len(indices)} sampled rows")
    elif max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))
        print(f"  → using first {max_samples} samples")
    print(f"  → {len(ds)} samples, converting to list …")
    return [
        {
            "id": str(row["question_id"]),
            "image": row["image"],          # PIL Image
            "question": row["question"],
            "answer": row["multiple_choice_answer"],
        }
        for row in ds
    ]


def load_vqav2(split: str, local_path: Optional[str], max_samples: Optional[int] = None) -> List[Dict]:
    # 优先用本地 parquet（在 HF 解码前就切片，避免 OOM）
    if os.path.isdir(PARQUET_SNAP):
        return load_vqav2_parquet(split, max_samples=max_samples)
    if local_path and os.path.isdir(local_path):
        print(f"Loading VQAv2 from local path: {local_path}")
        data = load_vqav2_local(local_path, split)
        return data[:max_samples] if max_samples else data
    print("Loading VQAv2 from HuggingFace Hub …")
    ds = load_vqav2_hf(split=split)
    rows = [
        {
            "id": str(row["question_id"]),
            "image": row["image"],
            "question": row["question"],
            "answer": row["multiple_choice_answer"],
        }
        for row in ds
    ]
    return rows[:max_samples] if max_samples else rows


# --------------------------------------------------------------------------- #
#  Candidate retrieval
# --------------------------------------------------------------------------- #

def build_candidate_pool(
    samples: List[Dict],
    scorer: QwenVLScorer,
    pool_size: Optional[int] = None,
    embed_batch: int = 16,
) -> Tuple[List[Dict], torch.Tensor]:
    """
    Embed all samples and return (pool_items, pool_embeddings [N, D]).
    If pool_size is given, randomly subsample the pool.
    """
    pool = samples if pool_size is None else random.sample(samples, min(pool_size, len(samples)))
    print(f"Embedding {len(pool)} candidates …")
    embs = []
    for i in tqdm(range(0, len(pool), embed_batch), desc="Embed pool"):
        batch = pool[i: i + embed_batch]
        batch_embs = scorer.embed_batch(
            [s["image"] for s in batch],
            [s["question"] for s in batch],
        )
        embs.append(batch_embs)
    embeddings = torch.cat(embs, dim=0)  # [N, D]
    return pool, embeddings


def retrieve_top_k(
    query_emb: torch.Tensor,    # [D]
    pool_embs: torch.Tensor,    # [N, D]
    K: int,
    exclude_id: Optional[str] = None,
    pool_ids: Optional[List[str]] = None,
) -> List[int]:
    """Return indices of the K most similar candidates (cosine similarity)."""
    q = torch.nn.functional.normalize(query_emb.unsqueeze(0), dim=-1)
    p = torch.nn.functional.normalize(pool_embs, dim=-1)
    sims = (q @ p.T).squeeze(0)       # [N]

    if exclude_id is not None and pool_ids is not None:
        for i, pid in enumerate(pool_ids):
            if pid == exclude_id:
                sims[i] = float("-inf")

    _, top_k = sims.topk(K)
    return top_k.tolist()


# --------------------------------------------------------------------------- #
#  Beam search over candidate ICD sequences
# --------------------------------------------------------------------------- #

BeamState = Tuple[List[int], float]    # (selected_indices, cumulative_log_prob)


def beam_search_icd(
    scorer: QwenVLScorer,
    candidate_pool: List[Dict],
    candidate_indices: List[int],
    query_item: Dict,
    shot_num: int,
    beam_size: int,
    score_batch_size: int = 32,
) -> List[BeamState]:
    """
    Beam search to find the best ICD sequences for a query.

    At each step ALL (beam × candidate) combinations are batched into groups of
    `score_batch_size` and scored in a single Qwen-VL forward pass each, then
    the top `beam_size` beams are retained.

    Args:
        scorer:            QwenVLScorer instance.
        candidate_pool:    Full pool list.
        candidate_indices: Indices (into pool) available for this query.
        query_item:        Dict with "image", "question", "answer".
        shot_num:          Number of ICDs to select.
        beam_size:         Number of parallel beams.
        score_batch_size:  Max sequences per Qwen-VL forward pass.

    Returns:
        List of (selected_indices, cumulative_log_prob) sorted by score desc.
    """
    beams: List[BeamState] = [([], 0.0)]

    for step in range(shot_num):
        # Collect every (partial_seq, candidate) combination to score this step
        all_icd_lists: List[List[Dict]] = []
        all_meta: List[Tuple[List[int], float, int]] = []  # (partial_seq, partial_score, cand_idx)

        for partial_seq, partial_score in beams:
            used = set(partial_seq)
            for cand_idx in candidate_indices:
                if cand_idx in used:
                    continue
                icd_list = [candidate_pool[i] for i in partial_seq] + [candidate_pool[cand_idx]]
                all_icd_lists.append(icd_list)
                all_meta.append((partial_seq, partial_score, cand_idx))

        # Score in batches of score_batch_size (single forward pass per batch)
        all_scores: List[float] = []
        for i in range(0, len(all_icd_lists), score_batch_size):
            batch = all_icd_lists[i : i + score_batch_size]
            batch_scores = scorer.score_batch_tf(
                icd_lists=batch,
                query_image=query_item["image"],
                query_text=query_item["question"],
                gt_answer=query_item["answer"],
            )
            all_scores.extend(batch_scores)

        # Reconstruct new beams
        new_beams: List[BeamState] = [
            (partial_seq + [cand_idx], partial_score + score)
            for (partial_seq, partial_score, cand_idx), score
            in zip(all_meta, all_scores)
        ]
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    return beams  # sorted best-first


# --------------------------------------------------------------------------- #
#  Main generation loop
# --------------------------------------------------------------------------- #

def _save_pool(pool_items, pool_embs, images_dir, pool_out):
    """Persist the candidate pool (id/image/question/answer/embedding) for
    inference. Images are saved to disk as JPEG."""
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for it, emb in zip(pool_items, pool_embs):
        img = it.get("image")
        image_ref = None
        if img is not None:                       # text-only tasks (SST-2) have no image
            p = images_dir / f"{it['id']}.jpg"
            if not p.exists():
                if isinstance(img, str):
                    img = Image.open(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(p, format="JPEG", quality=85)
            image_ref = str(p.resolve())
        records.append({
            "id": it["id"],
            "image": image_ref,
            "question": it["question"],
            "answer": it["answer"],
            "embedding": emb.tolist(),
        })
    Path(pool_out).parent.mkdir(parents=True, exist_ok=True)
    with open(pool_out, "w") as f:
        json.dump(records, f)
    print(f"Saved candidate pool ({len(records)}) → {pool_out}")


def _atomic_dump(obj, out_path: Path) -> None:
    """Write JSON to a temp file in the same dir, then atomically replace the
    target. Guarantees the on-disk checkpoint is never a half-written file."""
    out_path = Path(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, out_path)


def _load_pool_from_json(pool_in: str) -> Tuple[List[Dict], torch.Tensor, List[str]]:
    """Reload a previously-saved candidate pool (id/image-path/q/a/embedding).
    Lets us resume a crashed run WITHOUT re-embedding the whole pool."""
    print(f"[resume] Loading existing pool from {pool_in} …")
    with open(pool_in) as f:
        recs = json.load(f)
    pool_items = [
        {"id": r["id"], "image": r["image"], "question": r["question"], "answer": r["answer"]}
        for r in recs
    ]
    pool_embs = torch.tensor([r["embedding"] for r in recs], dtype=torch.float32)
    pool_ids = [r["id"] for r in recs]
    print(f"[resume] Pool reloaded: {len(pool_items)} items, embeddings {tuple(pool_embs.shape)}")
    return pool_items, pool_embs, pool_ids


def _load_pool(pool_in: str) -> Tuple[List[Dict], torch.Tensor, List[str]]:
    """Load pool from legacy .json or {base}.jsonl + {base}.pt."""
    path = Path(pool_in)
    jsonl = path if path.suffix == ".jsonl" else path.with_suffix(".jsonl")
    pt = jsonl.with_suffix(".pt")
    if jsonl.exists() and pt.exists():
        from lever_lm.dataset import VQAv2CandidatePool
        print(f"[resume] Loading pool from {jsonl} + {pt.name} …")
        pool = VQAv2CandidatePool(str(jsonl))
        pool_items = pool.items
        pool_embs = pool.get_embeddings()
        pool_ids = [it["id"] for it in pool_items]
        print(f"[resume] Pool reloaded: {len(pool_items)} items, embeddings {tuple(pool_embs.shape)}")
        return pool_items, pool_embs, pool_ids
    return _load_pool_from_json(pool_in)


def generate(args: argparse.Namespace):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ----- Resume support: reload finished records + pool, skip done anchors --
    done_records: List[Dict] = []
    if args.resume and out_path.exists():
        with open(out_path) as f:
            done_records = json.load(f)
        print(f"[resume] Found {len(done_records)} finished anchor records in {out_path}")

    # ----- Anchors (queries to generate data for) and candidate pool -----
    task = get_task(args.task)
    if args.anchor_random:
        # P1: random anchors + a *decoupled* (disjoint) larger pool, both
        # sampled from the full train set (avoids "anchor == pool" self-ref
        # and the "first-N" sampling bias). Task-agnostic (vqa/caption/sst2).
        total = task.total("train")
        print(f"[{task.name}] Full set size: {total}")
        rng = random.Random(args.seed)
        perm = list(range(total))
        rng.shuffle(perm)
        pool_idx = sorted(perm[: args.pool_size])
        anchor_idx = sorted(perm[args.pool_size: args.pool_size + args.max_samples])
        print(f"Sampling {len(pool_idx)} pool + {len(anchor_idx)} anchors (disjoint, random).")
        # On resume we only need the anchors that are NOT yet done.
        remaining_anchor_idx = anchor_idx[len(done_records):]
        if args.resume:
            print(f"[resume] Skipping first {len(done_records)} anchors; "
                  f"{len(remaining_anchor_idx)} remaining.")
        split_data = (
            task.load("train", indices=remaining_anchor_idx)
            if remaining_anchor_idx else []
        )
        # Pool: reuse saved embeddings when --pool_in set; else sample + embed fresh.
        if args.pool_in and (
            os.path.exists(args.pool_in)
            or Path(args.pool_in).with_suffix(".jsonl").exists()
        ):
            pool_data = None  # loaded below from pool file
        else:
            pool_data = task.load("train", indices=pool_idx)
    else:
        split_data = load_vqav2(args.split, args.vqav2_path, max_samples=args.max_samples)
        if args.resume and len(done_records):
            split_data = split_data[len(done_records):]
        pool_data = load_vqav2("train", args.vqav2_path, max_samples=args.pool_size)

    # Initialise Qwen-VL scorer
    scorer = QwenVLScorer(
        model_name=args.qwen_model,
        device=args.device,
    )

    # Build (or reload) the candidate pool
    if args.pool_in and (
        os.path.exists(args.pool_in)
        or Path(args.pool_in).with_suffix(".jsonl").exists()
    ):
        pool_items, pool_embs, pool_ids = _load_pool(args.pool_in)
    else:
        pool_items, pool_embs = build_candidate_pool(
            pool_data, scorer, embed_batch=args.embed_batch
        )
        pool_ids = [s["id"] for s in pool_items]
        if args.pool_out:
            _save_pool(pool_items, pool_embs, args.pool_images_dir, args.pool_out)

    print(f"Anchors to process: {len(split_data)}, Pool: {len(pool_items)}")

    # Start from finished records (resume) so checkpoints stay complete.
    out_records = list(done_records)
    base_n = len(done_records)
    for j, item in enumerate(tqdm(split_data, desc="Generating beams")):
        n = base_n + j
        # Embed query
        query_emb = scorer.embed(item["image"], item["question"])  # [D]

        # Retrieve top-K candidates
        cand_indices = retrieve_top_k(
            query_emb, pool_embs, args.K,
            exclude_id=item["id"], pool_ids=pool_ids
        )
        cand_embs = pool_embs[cand_indices]  # [K, D]

        # P2: optional InfoScore baseline  log P(y* | x)  (no ICD)
        baseline = 0.0
        if args.reward_mode == "info":
            baseline = scorer.score_tf(
                [], item["image"], item["question"], item["answer"]
            )
            if math.isinf(baseline) or math.isnan(baseline):
                baseline = 0.0

        # Beam search (teacher-forcing scoring inside)
        beams = beam_search_icd(
            scorer=scorer,
            candidate_pool=pool_items,
            candidate_indices=cand_indices,
            query_item=item,
            shot_num=args.shot_num,
            beam_size=args.beam_size,
            score_batch_size=args.score_batch_size,
        )

        # Map global pool indices → local K indices
        global_to_local = {g: l for l, g in enumerate(cand_indices)}
        beam_labels = []
        beam_rewards = []
        for seq, reward in beams:
            local_seq = [global_to_local[g] for g in seq]
            beam_labels.append(local_seq)
            beam_rewards.append(reward - baseline if args.reward_mode == "info" else reward)

        out_records.append({
            "query_id": item["id"],
            "query_emb": query_emb.tolist(),
            "cand_embs": cand_embs.tolist(),
            "beam_labels": beam_labels,
            "beam_rewards": beam_rewards,
            "baseline": baseline,
        })

        # Periodic checkpoint (so a long run survives crashes). Atomic write:
        # dump to a temp file then os.replace, so a power-cut mid-write can
        # never corrupt the existing checkpoint.
        if args.save_every and (n + 1) % args.save_every == 0:
            _atomic_dump(out_records, out_path)

    _atomic_dump(out_records, out_path)
    print(f"Saved {len(out_records)} records → {out_path}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Lever-LM beam ICD data")
    p.add_argument("--task", default="vqa", choices=["vqa", "caption", "sst2"],
                   help="Which task to generate ICD beam data for")
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--vqav2_path", default=None,
                   help="Local VQAv2 root directory; omit to use HF Hub")
    p.add_argument("--pool_size", type=int, default=None,
                   help="Subsample pool to this size (debug/small-scale)")
    p.add_argument("--qwen_model", default="/home/jiyi/.cache/modelscope/qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--shot_num", type=int, default=2)
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--output_file", default="data/vqav2_train_beams.json")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--embed_batch", type=int, default=16)
    p.add_argument("--score_batch_size", type=int, default=32,
                   help="Max ICD sequences per Qwen-VL forward pass during beam scoring")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    # P1: random anchors + decoupled pool
    p.add_argument("--anchor_random", action="store_true",
                   help="Randomly sample anchors + a disjoint pool from full train")
    # P2: reward mode
    p.add_argument("--reward_mode", default="info", choices=["abs", "info"],
                   help="'info' = logP(y|c,x) - logP(y|x) baseline; 'abs' = raw logP")
    # Pool persistence (for inference)
    p.add_argument("--pool_out", default=None,
                   help="If set, save candidate pool json (id/image/question/answer/embedding)")
    p.add_argument("--pool_images_dir", default="data/pool_images")
    p.add_argument("--save_every", type=int, default=200,
                   help="Checkpoint output every N anchors (0 = only at end)")
    # Resume a crashed run: keep finished records in --output_file and reuse the
    # already-embedded pool from --pool_in (skips re-embedding the whole pool).
    p.add_argument("--resume", action="store_true",
                   help="Resume: skip anchors already in --output_file, reuse --pool_in")
    p.add_argument("--pool_in", default=None,
                   help="Existing pool json (from a previous --pool_out) to reuse on resume")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args)
