import math
from typing import Dict, List

# 平均倒数排名（MRR）
def compute_mrr(retrieved: List[str], relevant: List[str]) -> float:
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1 / (i + 1)
    return 0.0

# 归一化折损累积增益（NDCG）
def compute_ndcg(retrieved: List[str], relevant: List[str], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += 1 / math.log2(i + 2)
    ideal_dcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def evaluate(
        results: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        top_k: int = 5
) -> Dict[str, float]:
    total = {"recall": 0, "precision": 0, "hit": 0, "mrr": 0, "ndcg": 0}
    num_queries = len(results)

    for query, retrieved_docs in results.items():
        gt = ground_truth.get(query, [])
        retrieved_k = retrieved_docs[:top_k]

        # 输出详细信息
        print(f"Query: {query}")
        print(f"Retrieved docs: {retrieved_k}")
        print(f"Ground truth: {gt}")

        # 精确匹配
        #hits = [doc for doc in retrieved_k if doc in gt]
        # 模糊匹配
        hits = [doc for doc in retrieved_k if any(ans in doc for ans in gt)]
        print(f"Hits: {hits}")

        total["recall"] += len(set(hits)) / len(set(gt)) if gt else 0
        total["precision"] += len(set(hits)) / top_k
        total["hit"] += 1 if hits else 0
        # 计算MRR时，需找到第一个匹配的位置
        mrr_score = 0.0
        for i, doc in enumerate(retrieved_k):
            if doc in gt:
                mrr_score = 1 / (i + 1)
                break
        total["mrr"] += mrr_score
        # NDCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_k):
            if doc in gt:
                dcg += 1 / math.log2(i + 2)
        ideal_dcg = sum(1 / math.log2(i + 2) for i in range(min(len(gt), top_k)))
        total["ndcg"] += dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    return {k: round(v / num_queries, 4) for k, v in total.items()}