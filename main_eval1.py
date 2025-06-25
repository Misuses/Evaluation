import json
from retrieval import retrieve_grouding_document, retrieve_grouding_document_whoosh, retrieve_grouding_document_vector
import matplotlib.pyplot as plt
import numpy as np
from evaluation import evaluate

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# === 加载测试数据集 ===
results_static = {}
results_dynamic = {}
results_keyword = {}
results_vector = {}
ground_truth = {}

with open("D:\\XM\\Evaluation\\NQ-open.dev.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        query = item["question"]
        gt = item["answer"]
        ground_truth[query] = gt

        # === 静态权重 α = 0.5 ===
        docs_static = retrieve_grouding_document(query, use_vector_search=True, use_keyword_search=True)
        ids_static = [doc.metadata["source"] for doc, _ in docs_static]
        results_static[query] = ids_static

        # === 动态权重 ===
        docs_dynamic = retrieve_grouding_document(query)
        ids_dynamic = [doc.metadata["source"] for doc, _ in docs_dynamic]
        results_dynamic[query] = ids_dynamic

        # === 只使用关键词检索 ===
        docs_keyword = retrieve_grouding_document_whoosh(query)
        ids_keyword = [doc.metadata["source"] for doc, _ in docs_keyword]
        results_keyword[query] = ids_keyword

        # === 只使用向量检索 ===
        docs_vector = retrieve_grouding_document_vector(query)
        ids_vector = [doc.metadata["source"] for doc, _ in docs_vector]
        results_vector[query] = ids_vector

# === 输出评估指标 ===
# 例如，只评估召回率（1）和 MRR（4）
metric_ids = [1, 4]
static_metrics = evaluate(results_static, ground_truth, top_k=5, metric_ids=metric_ids)
dynamic_metrics = evaluate(results_dynamic, ground_truth, top_k=5, metric_ids=metric_ids)
keyword_metrics = evaluate(results_keyword, ground_truth, top_k=5, metric_ids=metric_ids)
vector_metrics = evaluate(results_vector, ground_truth, top_k=5, metric_ids=metric_ids)

print("\n📊 静态权重混合检索：")
print(static_metrics)

print("\n📊 动态权重混合检索：")
print(dynamic_metrics)

print("\n📊 只使用关键词检索：")
print(keyword_metrics)

print("\n📊 只使用向量检索：")
print(vector_metrics)

# === 可视化评估结果 ===
filtered_metrics = [
    "recall" if 1 in metric_ids else None,
    "precision" if 2 in metric_ids else None,
    "hit" if 3 in metric_ids else None,
    "mrr" if 4 in metric_ids else None,
    "ndcg" if 5 in metric_ids else None
]
filtered_metrics = [m for m in filtered_metrics if m is not None]

static_values = [static_metrics[m] for m in filtered_metrics]
dynamic_values = [dynamic_metrics[m] for m in filtered_metrics]
keyword_values = [keyword_metrics[m] for m in filtered_metrics]
vector_values = [vector_metrics[m] for m in filtered_metrics]

x = np.arange(len(filtered_metrics))  # 标签位置
width = 0.2  # 柱状图宽度

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - 1.5 * width, static_values, width, label='静态权重')
rects2 = ax.bar(x - 0.5 * width, dynamic_values, width, label='动态权重')
rects3 = ax.bar(x + 0.5 * width, keyword_values, width, label='只使用关键词检索')
rects4 = ax.bar(x + 1.5 * width, vector_values, width, label='只使用向量检索')

# 添加文本标签和标题
ax.set_ylabel('指标值')
ax.set_title('不同检索方式评估指标对比', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(filtered_metrics)

# 将图例放在底部中间，避免遮挡柱状图
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)

def autolabel(rects):
    """在每个柱子上添加文本标签"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# 添加数值标签
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
