"""
icl_inference.py  –  Evaluate Lever-LM + Qwen-VL on VQAv2 ICL.

Pipeline:
  1. Load a trained PointerSelector checkpoint.
  2. For each query in the evaluation set:
       a. Embed the query with Qwen-VL.
       b. Retrieve K candidates (from pre-computed pool embeddings).
       c. Select `shot_num` ICDs using PointerSelector.greedy select().
       d. Construct an ICL prompt  [ICD_1, …, ICD_n, query]  for Qwen-VL.
       e. Generate the answer and compare to ground truth.
  3. Report VQA accuracy (exact-match after normalisation).

Usage:
    # Full evaluation
    python icl_inference.py \
        --ckpt_path checkpoints/lever_lm_pointer_rce/last.ckpt \
        --pool_file data/vqav2_pool.json \
        --split     validation \
        --shot_num  2 \
        --K         64 \
        [--vqav2_path /path/to/vqav2]

    # Quick smoke test (first 200 samples, CPU)
    python icl_inference.py \
        --ckpt_path checkpoints/lever_lm_pointer_rce/last.ckpt \
        --pool_file data/vqav2_pool.json \
        --max_samples 200 \
        --device cpu

    # Random-retrieval baseline (no PointerSelector)
    python icl_inference.py \
        --baseline random \
        --pool_file data/vqav2_pool.json
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from lever_lm.lever_lm_module import LeverLMModule
from lever_lm.qwen_vl_scorer import QwenVLScorer, _load_image
from lever_lm.dataset import VQAv2CandidatePool, load_vqav2_hf


# --------------------------------------------------------------------------- #
#  VQA answer normalisation  (standard VQA eval protocol)
# --------------------------------------------------------------------------- #

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]")


def _normalize(ans: str) -> str:
    ans = ans.lower().strip()
    ans = _ARTICLES.sub("", ans)
    ans = _PUNCT.sub(" ", ans)
    ans = " ".join(ans.split())
    return ans


_SHORT_ANSWER_SUFFIX = (
    "\nAnswer the question using a single word or a very short phrase only."
)


def _extract_short_pred(raw: str) -> str:
    """For 0-shot: Qwen often outputs a full sentence; keep the first phrase."""
    s = raw.strip().split("\n")[0].strip()
    # First sentence / clause
    for sep in (".", "?", "!", ","):
        if sep in s:
            s = s.split(sep)[0].strip()
    words = s.split()
    if len(words) > 5:
        s = " ".join(words[:5])
    return s


def vqa_accuracy(pred: str, gt: str, relaxed: bool = False) -> float:
    """VQA exact match after normalization; optional relaxed rules for 0-shot."""
    np, ng = _normalize(pred), _normalize(gt)
    if not ng:
        return 0.0
    if np == ng:
        return 1.0
    if not relaxed:
        return 0.0
    # yes / no prefix (common VQA answers)
    if ng in ("yes", "no"):
        first = np.split()[0] if np.split() else np
        if first == ng or np.startswith(ng + " "):
            return 1.0
    # first word match
    if np.split() and np.split()[0] == ng.split()[0]:
        return 1.0
    # short GT contained in prediction (≤3 words)
    if len(ng.split()) <= 3 and ng in np:
        return 1.0
    return 0.0


def _prompt_with_protocol(question: str, protocol: str) -> str:
    if protocol in ("unified", "legacy_zeroshot_relaxed"):
        return question + _SHORT_ANSWER_SUFFIX
    return question


def _score_prediction(pred: str, gt: str, protocol: str) -> Tuple[str, float]:
    if protocol in ("unified", "legacy_zeroshot_relaxed"):
        pred = _extract_short_pred(pred)
    relaxed = protocol == "legacy_zeroshot_relaxed"
    return pred, vqa_accuracy(pred, gt, relaxed=relaxed)


# --------------------------------------------------------------------------- #
#  Candidate retrieval (same as generate_data.py)
# --------------------------------------------------------------------------- #

def retrieve_top_k(
    query_emb: torch.Tensor,
    pool_embs: torch.Tensor,
    K: int,
    exclude_id: Optional[str] = None,
    pool_ids: Optional[List[str]] = None,
) -> torch.Tensor:
    """Return [K] tensor of indices into the pool."""
    q = F.normalize(query_emb.unsqueeze(0), dim=-1)
    p = F.normalize(pool_embs, dim=-1)
    sims = (q @ p.T).squeeze(0)

    if exclude_id is not None and pool_ids is not None:
        for i, pid in enumerate(pool_ids):
            if pid == exclude_id:
                sims[i] = float("-inf")

    _, top_k = sims.topk(K)
    return top_k  # [K]


# --------------------------------------------------------------------------- #
#  Qwen-VL answer generation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate_answer(
    scorer: QwenVLScorer,
    icd_list: List[Dict],
    query_image: Union[str, Image.Image],
    query_text: str,
    max_new_tokens: int = 10,
) -> str:
    """Generate an answer for a query given an ICD list using Qwen-VL."""
    messages = scorer._build_messages(icd_list, query_image, query_text)
    inputs, _ = scorer._prepare_inputs(messages)

    gen_ids = scorer.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    # Decode only newly generated tokens
    new_ids = gen_ids[0][inputs["input_ids"].shape[1]:]
    answer = scorer.processor.decode(new_ids, skip_special_tokens=True).strip()
    return answer


# --------------------------------------------------------------------------- #
#  Baselines
# --------------------------------------------------------------------------- #

def random_select(cand_indices: List[int], shot_num: int) -> List[int]:
    return random.sample(cand_indices, min(shot_num, len(cand_indices)))


# --------------------------------------------------------------------------- #
#  Main evaluation loop
# --------------------------------------------------------------------------- #

def evaluate(args: argparse.Namespace):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto"
        else "cpu"
    )

    # ------------------------------------------------------------------ #
    # Load candidate pool
    # ------------------------------------------------------------------ #
    pool = VQAv2CandidatePool(args.pool_file)
    pool_embs = pool.get_embeddings().to(device)   # [N, D]
    pool_ids = [pool.get_item(i)["id"] for i in range(len(pool))]
    print(f"Pool: {len(pool)} candidates, dim={pool_embs.shape[1]}")

    # ------------------------------------------------------------------ #
    # Load PointerSelector (unless baseline)
    # ------------------------------------------------------------------ #
    selector: Optional[LeverLMModule] = None
    if args.baseline is None:
        if args.ckpt_path is None:
            raise ValueError("--ckpt_path is required unless --baseline is set")
        selector = LeverLMModule.load_from_checkpoint(
            args.ckpt_path, map_location=device
        )
        selector.eval()
        selector.to(device)
        print(f"Loaded PointerSelector from {args.ckpt_path}")

    # ------------------------------------------------------------------ #
    # Load Qwen-VL scorer / generator
    # ------------------------------------------------------------------ #
    scorer = QwenVLScorer(
        model_name=args.qwen_model,
        device=args.device,
    )

    # ------------------------------------------------------------------ #
    # Load evaluation set
    # ------------------------------------------------------------------ #
    from generate_data import load_vqav2
    eval_data = load_vqav2(args.split, args.vqav2_path, max_samples=args.max_samples)

    print(f"Evaluating on {len(eval_data)} samples.")

    # ------------------------------------------------------------------ #
    # Evaluation loop
    # ------------------------------------------------------------------ #
    results = []
    total_acc = 0.0

    for item in tqdm(eval_data, desc="ICL eval"):
        # Embed query
        query_emb = scorer.embed(item["image"], item["question"]).to(device)

        # Retrieve top-K candidates
        cand_global = retrieve_top_k(
            query_emb, pool_embs, args.K,
            exclude_id=item["id"], pool_ids=pool_ids
        )  # [K]

        if args.baseline == "random":
            selected_local = random_select(list(range(args.K)), args.shot_num)
            selected_global = [cand_global[i].item() for i in selected_local]
        elif args.baseline == "zeroshot":
            selected_global = []
        else:
            # PointerSelector greedy selection
            cand_embs = pool_embs[cand_global]          # [K, D]
            sel_local = selector.select_icds(
                query_emb.unsqueeze(0),
                cand_embs.unsqueeze(0),
                shot_num=args.shot_num,
            )[0]                                        # [shot_num]
            selected_global = [cand_global[i.item()].item() for i in sel_local]

        icd_list = pool.get_items(selected_global)

        prompt_text = _prompt_with_protocol(item["question"], args.eval_protocol)
        pred = generate_answer(
            scorer, icd_list, item["image"], prompt_text,
            max_new_tokens=args.max_new_tokens,
        )
        pred, acc = _score_prediction(pred, item["answer"], args.eval_protocol)
        total_acc += acc

        results.append({
            "id": item["id"],
            "question": item["question"],
            "gt_answer": item["answer"],
            "pred_answer": pred,
            "correct": bool(acc),
            "selected_icd_ids": [pool.get_item(g)["id"] for g in selected_global],
        })

    final_acc = total_acc / len(eval_data)
    print(f"\nVQA Accuracy: {final_acc:.4f}  ({total_acc:.0f}/{len(eval_data)})")

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    if args.output_file:
        out = Path(args.output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "accuracy": final_acc,
                "eval_protocol": args.eval_protocol,
                "results": results,
            }, f, indent=2)
        print(f"Results saved → {out}")

    return final_acc


# --------------------------------------------------------------------------- #
#  Multi-shot sweep  (reproduce Table 1 style Avg:1~8 metric)
# --------------------------------------------------------------------------- #

def sweep_shot_nums(args: argparse.Namespace):
    """Evaluate for each shot_num in [1..8] and report average accuracy.

    Loads QwenVL, pool, and PointerSelector once and reuses across all shot_nums.
    """
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto"
        else "cpu"
    )

    # Load pool once
    pool = VQAv2CandidatePool(args.pool_file)
    pool_embs = pool.get_embeddings().to(device)
    pool_ids = [pool.get_item(i)["id"] for i in range(len(pool))]
    print(f"Pool: {len(pool)} candidates, dim={pool_embs.shape[1]}")

    # Load PointerSelector once
    selector: Optional[LeverLMModule] = None
    if args.baseline is None:
        if args.ckpt_path is None:
            raise ValueError("--ckpt_path is required unless --baseline is set")
        selector = LeverLMModule.load_from_checkpoint(
            args.ckpt_path, map_location=device
        )
        selector.eval()
        selector.to(device)
        print(f"Loaded PointerSelector from {args.ckpt_path}")

    # Load Qwen-VL once
    scorer = QwenVLScorer(model_name=args.qwen_model, device=args.device)

    # Load eval data once
    from generate_data import load_vqav2
    eval_data = load_vqav2(args.split, args.vqav2_path, max_samples=args.max_samples)
    print(f"Evaluating on {len(eval_data)} samples.")

    shot_accs = {}
    for s in range(1, 9):
        print(f"\n{'='*40}  shot_num={s}  {'='*40}")
        total_acc = 0.0
        results = []

        for item in tqdm(eval_data, desc=f"shot={s}"):
            query_emb = scorer.embed(item["image"], item["question"]).to(device)
            cand_global = retrieve_top_k(
                query_emb, pool_embs, args.K,
                exclude_id=item["id"], pool_ids=pool_ids
            )

            if args.baseline == "random":
                selected_local = random_select(list(range(args.K)), s)
                selected_global = [cand_global[i].item() for i in selected_local]
            elif args.baseline == "zeroshot":
                selected_global = []
            else:
                cand_embs = pool_embs[cand_global]
                sel_local = selector.select_icds(
                    query_emb.unsqueeze(0),
                    cand_embs.unsqueeze(0),
                    shot_num=s,
                )[0]
                selected_global = [cand_global[i.item()].item() for i in sel_local]

            icd_list = pool.get_items(selected_global)
            prompt_text = _prompt_with_protocol(item["question"], args.eval_protocol)
            pred = generate_answer(
                scorer, icd_list, item["image"], prompt_text,
                max_new_tokens=args.max_new_tokens,
            )
            pred, acc = _score_prediction(pred, item["answer"], args.eval_protocol)
            total_acc += acc
            results.append({
                "id": item["id"],
                "question": item["question"],
                "gt_answer": item["answer"],
                "pred_answer": pred,
                "correct": bool(acc),
            })

        acc_val = total_acc / len(eval_data)
        shot_accs[s] = acc_val
        print(f"shot_num={s}  VQA Accuracy: {acc_val:.4f}  ({total_acc:.0f}/{len(eval_data)})")

        if args.output_file:
            out = Path(args.output_file).with_suffix(f".shot{s}.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump({"shot_num": s, "accuracy": acc_val, "results": results}, f, indent=2)

    avg = sum(shot_accs.values()) / len(shot_accs)
    print("\n" + "="*50)
    print(f"{'shot':>6}  {'acc':>8}")
    for s, a in shot_accs.items():
        print(f"{s:>6}  {a:>8.4f}")
    print(f"{'Avg':>6}  {avg:>8.4f}")

    if args.output_file:
        out_summary = Path(args.output_file).with_suffix(".sweep_summary.json")
        with open(out_summary, "w") as f:
            json.dump({"shot_accs": shot_accs, "avg_1_8": avg}, f, indent=2)
        print(f"Summary saved → {out_summary}")

    return shot_accs, avg


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lever-LM ICL inference on VQAv2")

    # Required
    p.add_argument("--pool_file", required=True,
                   help="Pool: legacy .json with inline embeddings, or .jsonl + sibling .pt")

    # Model
    p.add_argument("--ckpt_path",   default=None,
                   help="PointerSelector checkpoint (omit for baselines)")
    p.add_argument("--qwen_model",  default="/home/jiyi/.cache/modelscope/qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--baseline",    default=None, choices=[None, "random", "zeroshot"],
                   help="Use a retrieval baseline instead of PointerSelector")

    # Data
    p.add_argument("--vqav2_path",  default=None)
    p.add_argument("--split",       default="validation")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--shot_num",    type=int, default=2)
    p.add_argument("--K",           type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=10)
    p.add_argument(
        "--eval_protocol",
        default="unified",
        choices=["unified", "legacy_lever_strict", "legacy_zeroshot_relaxed"],
        help="unified: short-answer suffix + extract + strict match for all methods",
    )

    # Sweep
    p.add_argument("--sweep",       action="store_true",
                   help="Sweep shot_num 1..8 and report Avg:1~8")

    # Output
    p.add_argument("--output_file", default=None)
    p.add_argument("--device",      default="auto")
    p.add_argument("--seed",        type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sweep:
        sweep_shot_nums(args)
    else:
        evaluate(args)
