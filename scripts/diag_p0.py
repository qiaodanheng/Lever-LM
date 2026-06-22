"""
P0 诊断脚本（GPU-free）：定位两个核心问题，给 P1/P2 改造提供依据。

Part A 召回质量：从候选池 embedding + answer 出发，统计 answer-overlap@K
        （top-K 候选里是否存在与 query 同答案的样本），对比 K=16/32/64，
        论证"扩池 + 加大 K"的收益。
Part B  −∞ 成因：从 beam 数据读取 beam_rewards，区分 all-inf / valid 记录，
        join 答案文本，比较两组的答案长度 / 类型 / token 数，验证
        "答案长/罕见 → span 匹配失败 → −∞"的假设。

输出：终端报告 + data/diag_p0_summary.json
"""
import json
import math
import re
import statistics
from collections import Counter

import numpy as np

POOL_FILE = "data/vqav2_pool.json"
BEAMS_FILE = "data/vqav2_train_beams.json"
QWEN_PATH = "/home/jiyi/.cache/modelscope/qwen/Qwen2-VL-2B-Instruct"

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]")


def normalize(ans: str) -> str:
    ans = str(ans).lower().strip()
    ans = _ARTICLES.sub("", ans)
    ans = _PUNCT.sub(" ", ans)
    return " ".join(ans.split())


def is_yesno(a: str) -> bool:
    return normalize(a) in {"yes", "no"}


def is_number(a: str) -> bool:
    return normalize(a).replace(" ", "").isdigit()


# ===========================================================
# 载入池子
# ===========================================================
print("载入候选池 …")
pool = json.load(open(POOL_FILE))
N = len(pool)
ids = [p["id"] for p in pool]
answers = [p["answer"] for p in pool]
id2ans = {p["id"]: p["answer"] for p in pool}
norm_ans = [normalize(a) for a in answers]
emb = np.array([p["embedding"] for p in pool], dtype=np.float32)  # [N, D]
emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
print(f"  池子 N={N}, dim={emb.shape[1]}")

# ===========================================================
# Part A: 召回质量 answer-overlap@K
# ===========================================================
print("\n" + "=" * 60)
print("Part A  召回质量（answer-overlap@K）")
print("=" * 60)

Ks = [16, 32, 64]
maxK = max(Ks)
sims_full = emb @ emb.T  # [N,N]（5000x5000 ≈ 100MB float32，OK）
np.fill_diagonal(sims_full, -1.0)  # 排除自己

# 每个 query 取 top-maxK
topk_idx = np.argpartition(-sims_full, maxK, axis=1)[:, :maxK]  # [N, maxK] 未排序
# 对每行按相似度排序
row_sims = np.take_along_axis(sims_full, topk_idx, axis=1)
order = np.argsort(-row_sims, axis=1)
topk_idx = np.take_along_axis(topk_idx, order, axis=1)  # [N, maxK] 已排序

partA = {}
for K in Ks:
    hit = 0           # ≥1 个同答案候选
    same_counts = []  # top-K 中同答案候选数
    for q in range(N):
        qa = norm_ans[q]
        cand = topk_idx[q, :K]
        same = sum(1 for c in cand if norm_ans[c] == qa)
        same_counts.append(same)
        if same > 0:
            hit += 1
    partA[K] = {
        "answer_overlap@K": hit / N,
        "avg_same_answer_in_topK": statistics.mean(same_counts),
    }
    print(f"  K={K:>2}:  answer-overlap@K = {hit/N:6.1%}   "
          f"平均同答案候选数 = {statistics.mean(same_counts):.2f}")

# top-1 邻居相似度分布（看召回到的最近邻有多像）
top1 = row_sims[np.arange(N), 0] if False else np.take_along_axis(sims_full, topk_idx[:, :1], axis=1).ravel()
print(f"  最近邻 cosine: 均值={top1.mean():.3f}  中位={np.median(top1):.3f}  "
      f"min={top1.min():.3f}  max={top1.max():.3f}")
print("  解读: answer-overlap@K 越高 → 越可能召回到直接有用的示范；"
      "若随 K 增大明显上升，说明加大 K / 扩池有收益。")

# ===========================================================
# Part B: −∞ 成因
# ===========================================================
print("\n" + "=" * 60)
print("Part B  −∞ 成因分析")
print("=" * 60)
print("载入 beam 数据（1.4GB，稍等）…")
beams = json.load(open(BEAMS_FILE))
print(f"  beam 记录数: {len(beams)}")

all_inf_ids, valid_ids = [], []
finite_per_record = []
best_reward_valid = []
for r in beams:
    rewards = r.get("beam_rewards", [])
    finite = [x for x in rewards if not (x is None or math.isinf(x) or math.isnan(x))]
    finite_per_record.append(len(finite))
    if len(finite) == 0:
        all_inf_ids.append(r["query_id"])
    else:
        valid_ids.append(r["query_id"])
        best_reward_valid.append(max(finite))

n_all_inf = len(all_inf_ids)
n_valid = len(valid_ids)
print(f"  全部 −∞ 记录: {n_all_inf}  ({n_all_inf/len(beams):.1%})")
print(f"  含有效 reward 记录: {n_valid}  ({n_valid/len(beams):.1%})")
print(f"  每记录有效 beam 数分布: {dict(Counter(finite_per_record))}")
if best_reward_valid:
    print(f"  有效记录 best reward: 均值={statistics.mean(best_reward_valid):.2f}  "
          f"中位={statistics.median(best_reward_valid):.2f}  "
          f"min={min(best_reward_valid):.2f}  max={max(best_reward_valid):.2f}")


def ans_stats(id_list, label):
    al = [id2ans[i] for i in id_list if i in id2ans]
    if not al:
        print(f"  [{label}] 无法 join 到答案（id 不在池中）"); return {}
    wlen = [len(str(a).split()) for a in al]
    clen = [len(str(a)) for a in al]
    yn = sum(is_yesno(a) for a in al) / len(al)
    num = sum(is_number(a) for a in al) / len(al)
    top = Counter(normalize(a) for a in al).most_common(12)
    print(f"\n  [{label}]  (join 到 {len(al)} 条)")
    print(f"    答案词数: 均值={statistics.mean(wlen):.2f}  中位={statistics.median(wlen)}  max={max(wlen)}")
    print(f"    答案字符数: 均值={statistics.mean(clen):.2f}")
    print(f"    yes/no 占比: {yn:.1%}   纯数字占比: {num:.1%}")
    print(f"    高频答案: {top}")
    return {"n": len(al), "avg_words": statistics.mean(wlen), "median_words": statistics.median(wlen),
            "max_words": max(wlen), "avg_chars": statistics.mean(clen),
            "yesno_ratio": yn, "number_ratio": num, "top_answers": top}

stat_inf = ans_stats(all_inf_ids, "全部 −∞ 的 query 答案")
stat_val = ans_stats(valid_ids, "有效 query 答案")

# 多词答案在两组的占比对比（验证"长答案 → −∞"）
def multiword_ratio(id_list):
    al = [id2ans[i] for i in id_list if i in id2ans]
    return sum(1 for a in al if len(str(a).split()) >= 2) / max(1, len(al))

print(f"\n  多词(≥2词)答案占比:  −∞组 = {multiword_ratio(all_inf_ids):.1%}   "
      f"有效组 = {multiword_ratio(valid_ids):.1%}")
print("  解读: 若 −∞ 组多词/长答案占比明显更高 → 印证'绝对 logP + span 匹配'对"
      "长/罕见答案脆弱，P2 改 teacher-forcing 算答案概率可大幅减少 −∞。")

# 可选: Qwen tokenizer 看答案 token 数
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    def avg_tok(id_list):
        al = [id2ans[i] for i in id_list if i in id2ans]
        tl = [len(tok(str(a), add_special_tokens=False).input_ids) for a in al[:2000]]
        return statistics.mean(tl)
    print(f"\n  答案 token 数(前2000):  −∞组={avg_tok(all_inf_ids):.2f}   有效组={avg_tok(valid_ids):.2f}")
except Exception as e:
    print(f"  (tokenizer 统计跳过: {e})")

# ===========================================================
# 保存摘要
# ===========================================================
summary = {
    "pool_size": N,
    "partA_recall_quality": partA,
    "nearest_neighbor_cosine_mean": float(top1.mean()),
    "partB_inf": {
        "total_records": len(beams),
        "all_inf": n_all_inf,
        "all_inf_ratio": n_all_inf / len(beams),
        "valid": n_valid,
        "finite_per_record_dist": dict(Counter(finite_per_record)),
        "inf_group": stat_inf,
        "valid_group": stat_val,
        "multiword_ratio_inf": multiword_ratio(all_inf_ids),
        "multiword_ratio_valid": multiword_ratio(valid_ids),
    },
}
json.dump(summary, open("data/diag_p0_summary.json", "w"), ensure_ascii=False, indent=2,
          default=lambda o: list(o) if isinstance(o, tuple) else str(o))
print("\n摘要已保存 → data/diag_p0_summary.json")
