import math
from typing import Dict, List

# 召回率
def compute_recall(retrieved: List[str], relevant: List[str], top_k: int) -> float:
    retrieved_k = retrieved[:top_k]
    hits = [doc for doc in retrieved_k if any(ans in doc for ans in relevant)]
    return len(set(hits)) / len(set(relevant)) if relevant else 0

# 精确率
def compute_precision(retrieved: List[str], relevant: List[str], top_k: int) -> float:
    retrieved_k = retrieved[:top_k]
    hits = [doc for doc in retrieved_k if any(ans in doc for ans in relevant)]
    return len(set(hits)) / top_k

# 命中率
def compute_hit(retrieved: List[str], relevant: List[str], top_k: int) -> float:
    retrieved_k = retrieved[:top_k]
    hits = [doc for doc in retrieved_k if any(ans in doc for ans in relevant)]
    return 1 if hits else 0

# 平均倒数排名（MRR）
def compute_mrr(retrieved: List[str], relevant: List[str], top_k: int) -> float:
    retrieved_k = retrieved[:top_k]
    for i, doc_id in enumerate(retrieved_k):
        if doc_id in relevant:
            return 1 / (i + 1)
    return 0.0

# 归一化折损累积增益（NDCG）
def compute_ndcg(retrieved: List[str], relevant: List[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k):
        if doc_id in relevant:
            dcg += 1 / math.log2(i + 2)
    ideal_dcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def evaluate(
        results: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        top_k: int = 5,
        metric_ids: List[int] = [1, 2, 3, 4, 5]
) -> Dict[str, float]:
    metric_functions = [
        compute_recall,
        compute_precision,
        compute_hit,
        compute_mrr,
        compute_ndcg
    ]
    metric_names = [
        "recall",
        "precision",
        "hit",
        "mrr",
        "ndcg"
    ]
    total = {metric_names[i - 1]: 0 for i in metric_ids}
    num_queries = len(results)

    for query, retrieved_docs in results.items():
        gt = ground_truth.get(query, [])
        for metric_id in metric_ids:
            func = metric_functions[metric_id - 1]
            name = metric_names[metric_id - 1]
            total[name] += func(retrieved_docs, gt, top_k)

    return {k: round(v / num_queries, 4) for k, v in total.items()}
