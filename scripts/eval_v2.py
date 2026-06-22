"""
Evaluate the trained autoregressive PointerSelector on the held-out v2 val set.

Focus: does the model actually USE order (the [3,9] vs [9,3] goal)?

Metrics:
  - Top-1 beam match: model's argmax-logP beam == reward-argmax beam (order-sensitive).
  - Pairwise ORDER accuracy: among beam pairs that are exact permutations of each
    other (same ICD set, different order), how often does the model's log-prob
    ranking agree with the reward ranking? (An order-blind model is stuck at 50%.)
  - Kendall-tau between model log-prob ranking and reward ranking over beams.
  - select() agreement: greedy sequence vs the top-reward beam (set + exact order).
"""

import argparse
import json
import torch
import torch.nn.functional as F
from itertools import combinations

from lever_lm.lever_lm_module import LeverLMModule
from lever_lm.dataset import VQAv2BeamDataset, collate_beam_batch
from torch.utils.data import DataLoader


def kendall_tau(a, b):
    """Simple Kendall tau over small lists (no ties handling beyond sign)."""
    n = len(a)
    if n < 2:
        return 0.0
    conc = disc = 0
    for i, j in combinations(range(n), 2):
        s = (a[i] - a[j]) * (b[i] - b[j])
        if s > 0:
            conc += 1
        elif s < 0:
            disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_file", default="data/vqav2_train_beams_v2_val.json")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--shot_num", type=int, default=2)
    ap.add_argument("--max_beams", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    module = LeverLMModule.load_from_checkpoint(args.ckpt, map_location=device)
    model = module.model.to(device).eval()

    ds = VQAv2BeamDataset(args.val_file, max_beams=args.max_beams,
                          shot_num=args.shot_num, K=args.K)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_beam_batch)

    top1 = 0
    n_samples = 0
    taus = []
    perm_pairs = 0
    perm_correct = 0
    sel_set_match = 0
    sel_order_match = 0
    sel_in_any_beam = 0
    sel_cand_recall = 0.0

    with torch.no_grad():
        for batch in dl:
            q = batch["query_emb"].to(device)
            c = batch["cand_embs"].to(device)
            bl = batch["beam_labels"].to(device)
            br = batch["beam_rewards"].to(device)
            bm = batch["beam_mask"].to(device)
            B = q.size(0)

            lp = model.compute_log_probs_per_beam(q, c, bl)  # [B, nb]
            sel = model.select(q, c, shot_num=args.shot_num)  # [B, shot]

            for b in range(B):
                valid = bm[b].nonzero(as_tuple=True)[0].tolist()
                if not valid:
                    continue
                n_samples += 1
                rewards = {i: br[b, i].item() for i in valid}
                logps = {i: lp[b, i].item() for i in valid}

                # top-1 (order-sensitive sequence match)
                best_r = max(valid, key=lambda i: rewards[i])
                best_lp = max(valid, key=lambda i: logps[i])
                if best_lp == best_r:
                    top1 += 1

                # kendall tau between model logp and reward over valid beams
                rv = [rewards[i] for i in valid]
                lv = [logps[i] for i in valid]
                taus.append(kendall_tau(lv, rv))

                # pairwise ORDER accuracy on permutation pairs
                for i, j in combinations(valid, 2):
                    si = sorted(bl[b, i].tolist())
                    sj = sorted(bl[b, j].tolist())
                    if si == sj and bl[b, i].tolist() != bl[b, j].tolist():
                        # exact permutation of each other
                        perm_pairs += 1
                        if (logps[i] - logps[j]) * (rewards[i] - rewards[j]) > 0:
                            perm_correct += 1

                # select() agreement vs top-reward beam
                tgt = bl[b, best_r].tolist()
                got = sel[b].tolist()
                if set(got) == set(tgt):
                    sel_set_match += 1
                    if got == tgt:
                        sel_order_match += 1

                # fairer: does select() match ANY beam, and how many of its
                # picks fall inside the union of all beam candidates?
                beam_sets = [set(bl[b, i].tolist()) for i in valid]
                if any(set(got) == s for s in beam_sets):
                    sel_in_any_beam += 1
                union = set().union(*beam_sets)
                sel_cand_recall += sum(1 for g in got if g in union) / len(got)

    print(f"val samples evaluated:        {n_samples}")
    print(f"Top-1 beam match (ordered):   {top1/n_samples:.3f}")
    print(f"Kendall-tau (logP vs reward): {sum(taus)/len(taus):.3f}")
    if perm_pairs:
        print(f"Permutation pairs found:      {perm_pairs}")
        print(f"ORDER accuracy on perms:      {perm_correct/perm_pairs:.3f}  "
              f"(order-blind model = 0.500 by construction)")
    else:
        print("Permutation pairs found:      0 (beams rarely share the same set; "
              "order still matters within the autoregressive scoring)")
    print(f"select() set match vs best:   {sel_set_match/n_samples:.3f}")
    print(f"select() exact-order match:   {sel_order_match/n_samples:.3f}")
    print(f"select() matches ANY beam:    {sel_in_any_beam/n_samples:.3f}")
    print(f"select() picks in beam-union: {sel_cand_recall/n_samples:.3f}  "
          f"(random ≈ {2*( (3*2) )/32/2:.3f})")


if __name__ == "__main__":
    main()
