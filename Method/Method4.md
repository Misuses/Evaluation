## DH-RAG — Dynamic Historical Context-Powered Retrieval-Augmented Generation

### 1. 背景与动机
在多轮对话和知识密集型问答任务中，传统的 Retrieval-Augmented Generation（RAG）通常依赖**静态知识库**（Static Knowledge Base, KB）进行信息检索。这种方式虽然可以提升知识覆盖率，但存在两个突出问题：
1. **缺乏对动态对话历史的有效利用**  
   多轮对话中用户的意图会随时间变化，单纯依靠静态知识无法反映上下文中的最新信息。
2. **历史信息利用粒度过粗**  
   传统方法若考虑历史信息，往往是简单拼接最近若干轮对话，导致噪音增加、信息冗余，同时难以根据语义相关性动态筛选历史内容。

**DH-RAG** 针对以上问题，设计了两个核心模块：
- **历史学习驱动的查询重构模块**（History-Learning Based Query Reconstruction Module, HLQRM）
- **动态历史信息更新模块**（Dynamic History Information Updating Module, DHIUM）

其目标是在生成响应时同时利用长期静态知识与短期动态历史，从而**增强多轮对话的连贯性与知识准确性**。

---

### 2. 历史学习驱动的查询重构模块（History-Learning Based Query Reconstruction）

#### 2.1 核心目标
- 输入：当前用户查询 \(q_t\)、静态知识库 \(K\)、动态历史信息库 \(H\)
- 输出：重构后的查询 \(q'_t\) 与融合上下文 \(C\)，用于驱动 LLM 生成最终响应 \(r_t\)
- 关键思想：从 **静态 KB** 检索长期知识，从 **动态历史库 H** 检索与当前任务相关的短期上下文，并通过注意力机制融合。

#### 2.2 数学描述
1. **静态知识检索**
$$\[
D_k = \text{Retrieve}(q_t, K)
\]$$
2. **动态历史信息检索**  
   使用两种策略获得动态历史信息集合：
\$$[
D_h^{HM} = \text{HierarchicalMatching}(q_t, H)
\]$$
\$$[
D_h^{CoT} = \text{ChainOfThoughtTracking}(q_t, H)
\]$$
\
综合动态历史信息：
\$$[
D_h = D_h^{HM} \cup D_h^{CoT}
\]$$
4. **信息融合（Attention 加权）**
\
\$$[w_i = \text{softmax}(q_t^T W d_i)\]$$
   
\$$[
C = \sum_i w_i \cdot d_i
\]$$
4. **LLM 生成**
\$$[
r_t = \text{LLM}(q'_t, C)
\]$$

#### 2.3 代码实现示例
```python
def history_learning_query_reconstruction(qt, K, H):
    # 静态知识检索
    Dk = retrieve(qt, K)
    
    # 动态历史信息检索（两种策略）
    Dh_HM = hierarchical_matching(qt, H)  # 分层匹配
    Dh_CoT = chain_of_thought_tracking(qt, H)  # 推理链追踪
    Dh = Dh_HM + Dh_CoT
    
    # 信息融合（注意力机制）
    C = integrate_with_attention(qt, Dk + Dh)
    
    # LLM 生成
    rt = LLM_generate(qt, C)
    
    return rt, C


#### 2.4 实现要点

* 检索策略需结合三大方法以保证历史信息的高相关性与低冗余。
* 注意力融合可引入可学习参数矩阵 $W$，动态分配不同来源信息的权重。
* 生成的上下文 $C$ 不仅用于响应生成，还用于后续的动态历史信息更新。

---

### 3. 动态历史信息更新模块（Dynamic History Information Updating）

#### 3.1 核心目标

维护一个可动态调整的历史信息库 $H$，保证其中保留的历史信息在**相关性**和**时效性**上都最优。

#### 3.2 更新过程

1. **插入新三元组**

$$
(q_t, p_t, r_t) \quad \text{其中} \ p_t \ \text{为检索到的证据片段}
$$

2. **计算综合权重**

$$
w_i = \alpha \cdot \text{Relevance}(q_i, q_t) + (1 - \alpha) \cdot \text{Recency}(t_i)
$$

3. **容量控制与淘汰**
   当历史信息数超过设定阈值 $N$ 时，淘汰权重最低的条目。

#### 3.3 代码实现示例

```python
def update_history_database(H, qt, pt, rt, alpha=0.5, N=200):
    H.append((qt, pt, rt))
    weights = []
    for qi, pi, ri in H:
        rel = relevance(qi, qt)      # 语义相关性
        rec = recency_score(qi)      # 新鲜度得分
        weights.append(alpha * rel + (1 - alpha) * rec)
    
    # 按权重排序并保留Top-N
    H = [entry for _, entry in sorted(zip(weights, H), reverse=True)[:N]]
    return H
```

#### 3.4 实现要点

* **Relevance**：可通过嵌入向量的余弦相似度计算。
* **Recency**：根据时间戳进行归一化，近期对话赋予更高分数。
* 参数 $\alpha$ 控制相关性与新鲜度的平衡。

---

### 4. 动态历史信息库的三大策略

#### 4.1 历史查询聚类（Historical Query Clustering）

* 使用聚类方法（如 K-Means 或层次聚类）将历史查询按语义相似性分组：

$$
\{c_1, c_2, ..., c_k\} = \text{Cluster}(Q)
$$

* 检索时仅匹配与当前查询所属簇的历史信息，减少干扰。

#### 4.2 分层匹配（Hierarchical Matching, HM）

* **三层结构**：

  1. **类别层**：根据主题类别初筛
  2. **摘要层**：匹配简短摘要
  3. **历史条目层**：检索完整历史条目
* 优势：逐层过滤，降低搜索空间。

#### 4.3 推理链追踪（Chain of Thought Tracking, CoT）

* 通过链路化历史三元组形成推理链 $T$：

$$
T^* = \arg\max_T \text{sim}(q_t, T)
$$

* 在链内检索最相关的推理路径，保证逻辑连贯。

---

### 5. 实验应用与效果

#### 5.1 数据集

* MobileCS2（移动客服场景多轮对话）
* PopQA-Mod、TriviaQA-Mod（知识问答）
* CoQA（对话式阅读理解）
* TopiOCQA（主题连续对话）

#### 5.2 对比方法

* 传统 BM25
* Self-RAG
* 标准 LLM（无外部检索）

#### 5.3 结果与分析

* 在所有数据集上，DH-RAG 在 BLEU、F1、EM 等指标均优于对比方法。
* 多轮对话场景提升幅度尤为显著（上下文一致性增强）。
* 消融实验表明：

  * 去掉动态更新模块 → 性能下降明显
  * 去掉结果融合（Attention） → 上下文利用效率下降

---

### 6. 适用场景

| 场景         | 优势                    |
| ---------- | --------------------- |
| 多轮对话系统     | 持续保持上下文一致性，减少语义漂移     |
| 客服问答       | 快速利用近期交互信息响应用户意图变化    |
| 知识密集型 QA   | 结合最新上下文与长期知识，提高答案准确性  |
| 法律/医疗等垂直领域 | 动态吸收新案例/新指南，保证生成结果时效性 |

---


