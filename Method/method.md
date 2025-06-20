# 《Hybrid Retrieval for HallucinationMitigation in Large Language Models: AComparative Analysis》 MAP与 NDCG 实现方法

在信息检索领域，MAP 和 NDCG 是评估检索系统性能的核心指标。以下结合论文实验场景，详细介绍其实现逻辑与代码示例。


## 一、MAP（Mean Average Precision）实现

### 1. 核心概念
- **AP（Average Precision）**：单个查询的平均准确率，衡量相关文档在检索结果中的排序质量  
- **MAP**：所有查询 AP 的均值，反映检索系统的整体性能  

### 2. 数学公式
**单个查询 AP 计算**：  
\[
$$AP = \frac{1}{|Rel_q|} \sum_{i=1}^{n} Precision@i \cdot \mathbb{1}[rel_i=1]$$
\]  
- \($$|Rel_q|$$\)：查询 \($$q$$\) 的相关文档总数  
- \($$Precision@i$$\)：前 \($$i$$\) 个结果的准确率  
- \(\$$mathbb{1}[rel_i=1]$$\)：指示第 \($$i$$\) 个结果是否相关  

**整体 MAP 计算**：  
\[
$$MAP = \frac{1}{|Q|} \sum_{q \in Q} AP(q)$$
\]  

### 3. 代码实现
```python
def calculate_ap(relevant_indices, k=3):
    """计算单个查询的AP值（k=3为论文实验设定的截断长度）"""
    ap = 0
    relevant_count = 0
    for i in range(min(k, len(relevant_indices))):
        if relevant_indices[i] == 1:
            relevant_count += 1
            ap += relevant_count / (i + 1)  # 累加位置i的准确率
    return ap / max(1, sum(relevant_indices[:k]))  # 归一化处理

def calculate_map(query_results):
    """计算所有查询的MAP值
    query_results格式: {query_id: list(0/1), ...}，0表示不相关，1表示相关
    """
    ap_values = [calculate_ap(rels) for rels in query_results.values()]
    return sum(ap_values) / len(ap_values)
```

### 4. 实现要点
- **截断参数**：论文实验中使用 \$$(k=3$$\)，即仅评估前3个检索结果 
- **边界处理**：当无相关结果时，AP 默认为0  
- **应用场景**：适用于评估检索结果中相关文档的排序精度  


## 二、NDCG（Normalized Discounted Cumulative Gain）实现

### 1. 核心概念
- **DCG（Discounted Cumulative Gain）**：考虑排名折损的累积增益，靠前的相关文档权重更高  
- **IDCG（Ideal DCG）**：理想情况下（相关文档按顺序排列）的 DCG  
- **NDCG**：DCG 与 IDCG 的比值，范围 [0,1]，值越高排名越优  

### 2. 数学公式
**DCG@k 计算**：  
\[
$$DCG@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$
\]  
**NDCG@k 计算**：  
\[
$$NDCG@k = \frac{DCG@k}{IDCG@k}$$
\]  
- \($$rel_i$$\)：第 \($$i$$\) 个结果的相关度（二进制时为0或1）  
- 分母对数函数用于折损，使靠前结果权重更高 

### 3. 代码实现
```python
import math

def calculate_dcg(relevant_scores, k=3):
    """计算DCG值（k=3为论文实验设定的截断长度）"""
    dcg = 0
    for i in range(min(k, len(relevant_scores))):
        rel = relevant_scores[i]
        dcg += rel / math.log2(i + 2)  # i从0开始，对应log2(i+1)的分母为i+2
    return dcg

def calculate_ndcg(relevant_scores, k=3):
    """计算NDCG值"""
    dcg = calculate_dcg(relevant_scores, k)
    ideal_relevant = sorted(relevant_scores, reverse=True)  # 理想排序
    idcg = calculate_dcg(ideal_relevant, k)
    return dcg / max(1e-9, idcg)  # 避免除零错误

def calculate_average_ndcg(query_results, k=3):
    """计算所有查询的平均NDCG值"""
    ndcg_values = [calculate_ndcg(rels, k) for rels in query_results.values()]
    return sum(ndcg_values) / len(ndcg_values)
```

### 4. 实现要点
- **相关度处理**：论文实验中使用二进制相关度（相关/不相关） 
- **理想排序**：IDCG 计算时需将相关文档按相关度降序排列  
- **敏感度**：NDCG 对靠前结果的相关性更敏感，适合评估排序质量  


## 三、论文实验中的指标应用
### 1. 实验设置
- **数据集**：HaluBench 数据集，评估前3个检索结果（\($$k=3$$\)） 
- **对比方法**：稀疏检索（BM25）、密集检索（Sentence Transformers）、混合检索 

### 2. 关键结果
| 指标       | 稀疏检索 | 密集检索 | 混合检索 |
|------------|----------|----------|----------|
| MAP@3      | 0.724    | 0.768    | 0.897    |
| NDCG@3     | 0.732    | 0.783    | 0.915    |

- 混合检索在两项指标上均显著优于单一检索方法，验证了其在提升检索相关性上的有效性 
- NDCG 的提升幅度更大，表明混合检索在优化结果排序上效果更突出 

### 3. 代码应用示例
```python
# 模拟论文实验数据
query_results = {
    "q1": [1, 0, 1],  # 前3个结果中第1、3个相关
    "q2": [1, 1, 0],
    "q3": [0, 1, 1]
}

# 计算指标
map_score = calculate_map(query_results)
ndcg_score = calculate_average_ndcg(query_results)

print(f"MAP@3: {map_score:.3f}")
print(f"NDCG@3: {ndcg_score:.3f}")
```


## 四、指标对比与适用场景

| 指标   | 核心特点                          | 优势场景                     |
|--------|-----------------------------------|------------------------------|
| MAP    | 关注相关文档的平均准确率          | 评估检索系统的整体查准率     |
| NDCG   | 强调排名顺序的重要性              | 优化检索结果的排序质量       |

- 在论文中，NDCG 更能体现混合检索在排序上的优势，因其对靠前结果的相关性更敏感 
- MAP 则从整体上衡量检索系统召回相关文档的能力，适合跨方法对比 
```
