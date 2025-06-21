# 《Blended rag: Improving rag (retriever-augmented generation) accuracy with semantic search and hybrid query-based retrievers》 NDCG@k，P@k与 F1 实现方法


## 一、核心概念  
### 1. Blended RAG的技术定位  
Blended RAG是一种通过融合**语义搜索技术**（密集向量索引、稀疏编码器索引）与**混合查询策略**来提升检索增强生成（RAG）系统准确性的方法。其核心在于解决传统RAG中单一检索器在文档规模扩大时的相关性衰减问题，通过多模态检索融合实现更精准的上下文提取。  

### 2. 三元检索器架构  
- **BM25索引**：基于关键词频率（TF）、逆文档频率（IDF）和文档长度归一化的稀疏检索模型，擅长精确术语匹配。  
- **密集向量索引（KNN）**：利用Sentence Transformers将文本映射为高维语义向量，通过余弦相似度计算语义距离，捕捉深层语义关联。  
- **稀疏编码器索引（ELSER）**：将文档和查询映射为高维稀疏向量，通过预训练语料扩展术语关联（如“AI”→“人工智能”），兼顾语义理解与检索效率。  

### 3. 混合查询策略  
- **多字段匹配模式**：  
  - **Cross Fields**：跨字段联合检索（如标题+正文），适用于查询文本位置不确定场景。  
  - **Best Fields**：聚焦单一字段（如正文）的最优匹配，优先提取关键术语。  
  - **Phrase Prefix**：基于短语前缀匹配（如“machine learning”→“machine learning models”），强化语义连续性。  


## 二、评价指标体系  
### 1. 核心指标定义与公式  
#### （1）NDCG@k（归一化折损累积增益）  
$$
\text{NDCG@k} = \frac{\sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}}{\text{IDCG@k}}
$$  
-$$  rel_i：第 i 个检索结果的相关性分数（如TREC-COVID中-1=无关，2=高度相关）。  $$
-$$   \text{IDCG@k} ：理想排序下的折损累积增益，用于归一化处理。   $$

#### （2）P@k（前k检索准确率）  
$$
\text{P@k} = \frac{\text{前k个结果中的相关文档数}}{k}
$$  
- 例：NQDataset基准P@20=0.633，Blended RAG提升至0.67。  

#### （3）F1分数与Exact Match（EM）  
$$
\text{F1} = 2 \times \frac{\text{精确率} \times \text{召回率}}{\text{精确率} + \text{召回率}}
$$  

### 2. 指标对比表  
| 指标         | 用途                          | 基准值（NQDataset） | Blended RAG结果 |
|--------------|-------------------------------|---------------------|-----------------|
| NDCG@10      | 检索结果排序质量              | 0.633               | 0.67（+5.8%）   |
| P@20         | 前20结果准确率                | 0.633               | 0.67            |
| F1           | 答案生成精确性                | -                   | 68.4%           |



## 三、数学公式推导  
### 1. 检索模型核心公式  
#### （1）BM25评分函数  
$$
\text{BM25}(Q,D) = \sum_{t \in Q} \text{IDF}(t) \times \frac{\text{TF}(t,D) \times (k_1 + 1)}{\text{TF}(t,D) + k_1 \times (1 - b + b \times \frac{|D|}{\text{avgdl}})}
$$  
 $$ \text{TF}(t,D) ：词 t 在文档 D 中的词频； \text{IDF}(t) = \log(\frac{N - n(t) + 0.5}{n(t) + 0.5}) ，N为文档总数，n(t)为包含t的文档数。  $$
 $$ k_1（默认1.2）和b（默认0.75）为调节参数，控制词频饱和效应和文档长度归一化。  $$

#### （2）余弦相似度（密集向量检索）  
$$
\text{Sim}(v_q, v_d) = \frac{v_q \cdot v_d}{\|v_q\| \cdot \|v_d\|}
$$
$$
 v_q为查询向量，v_d为文档向量，通过Sentence Transformers模型（如all-MiniLM-L6-v2）生成。
$$
#### （3）稀疏编码器向量表示  
$$
\mathbf{s} = \text{ELSER}(text) = \sum_{t \in \text{expand}(text)} w_t \cdot \mathbf{e}_t
$$  
$$ \text{expand}(text)通过预训练语料扩展术语（如“AI”→“人工智能”“机器学习”），\mathbf{e}_t为术语t的嵌入向量，w_t为权重。 $$ 


## 四、代码实现框架  
### 1. BlendedRetriever类集成  
```python
import numpy as np
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class BlendedRetriever:
    def __init__(self, bm25_host, dense_model_path, sparse_encoder_host):
        """初始化三元检索器"""
        # BM25索引（Elasticsearch）
        self.bm25 = Elasticsearch([bm25_host])
        # 密集向量模型
        self.dense_encoder = SentenceTransformer(dense_model_path)
        # 稀疏编码器（ELSER）
        self.sparse_encoder = Elasticsearch([sparse_encoder_host])
    
    def _bm25_query(self, query, field="content", k=10):
        """BM25关键词检索"""
        body = {
            "query": {
                "match": {field: query}  # 基础匹配查询
            },
            "size": k
        }
        return self.bm25.search(index="documents", body=body)
    
    def _dense_retrieve(self, query, docs_embeddings, doc_ids, k=10):
        """密集向量检索"""
        query_emb = self.dense_encoder.encode([query])[0]
        sim_scores = cosine_similarity([query_emb], docs_embeddings)[0]
        top_indices = np.argsort(sim_scores)[-k:][::-1]
        return [(doc_ids[i], sim_scores[i]) for i in top_indices]
    
    def _sparse_encoder_query(self, query, fields=["title", "content"], 
                             query_type="best_fields", k=10):
        """稀疏编码器混合查询"""
        if query_type == "best_fields":
            query_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "type": "best_fields",
                        "fields": fields
                    }
                },
                "size": k
            }
        elif query_type == "cross_fields":
            # 跨字段联合检索
            query_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "type": "cross_fields",
                        "fields": fields
                    }
                },
                "size": k
            }
        return self.sparse_encoder.search(index="elser_index", body=query_body)
    
    def blend_retrieve(self, query, k=10, metadata=True):
        """混合检索主流程"""
        # 1. BM25基础匹配
        bm25_results = self._bm25_query(query, k=20)
        bm25_docs = [{"id": hit["_id"], "score": hit["_score"], "text": hit["_source"]["content"]} 
                    for hit in bm25_results["hits"]["hits"]]
        
        # 2. 稀疏编码器混合查询（Best Fields）
        sparse_results = self._sparse_encoder_query(query, query_type="best_fields", k=20)
        sparse_docs = [{"id": hit["_id"], "score": hit["_score"], "text": hit["_source"]["content"]} 
                     for hit in sparse_results["hits"]["hits"]]
        
        # 3. 密集向量检索
        # docs_embeddings和doc_ids需提前通过self.dense_encoder.encode生成
        # dense_results = self._dense_retrieve(query, docs_embeddings, doc_ids, k=20)
        
        # 4. 结果融合（加权合并分数）
        blended_results = self._fuse_results(bm25_docs, sparse_docs, dense_results=None)
        return blended_results[:k]
    
    def _fuse_results(self, bm25_docs, sparse_docs, dense_results=None):
        """多检索器结果融合"""
        # 统一ID索引
        doc_dict = {}
        for doc in bm25_docs:
            doc_dict[doc["id"]] = {"bm25_score": doc["score"], "text": doc["text"]}
        for doc in sparse_docs:
            if doc["id"] in doc_dict:
                doc_dict[doc["id"]]["sparse_score"] = doc["score"]
            else:
                doc_dict[doc["id"]] = {"sparse_score": doc["score"], "text": doc["text"]}
        if dense_results:
            for doc_id, sim in dense_results:
                if doc_id in doc_dict:
                    doc_dict[doc_id]["dense_score"] = sim
                else:
                    doc_dict[doc_id] = {"dense_score": sim, "text": ""}
        
        # 加权融合分数（示例权重，实际需根据数据集调优）
        fused_results = []
        for doc_id, info in doc_dict.items():
            score = 0.3 * info.get("bm25_score", 0) + 0.4 * info.get("sparse_score", 0)
            if "dense_score" in info:
                score += 0.3 * info["dense_score"]
            fused_results.append({"id": doc_id, "score": score, "text": info["text"]})
        
        # 按分数排序
        return sorted(fused_results, key=lambda x: x["score"], reverse=True)
```  



### 2. 评价指标计算实现  
```python
def calculate_ndcg(scores, k=10):
    """计算NDCG@k（假设scores为相关性分数列表）"""
    # 理想排序的IDCG
    ideal_ranking = sorted(scores, reverse=True)
    idcg = sum([rel / np.log2(i+2) for i, rel in enumerate(ideal_ranking[:k])])
    
    # 实际排序的DCG
    dcg = sum([rel / np.log2(i+2) for i, rel in enumerate(scores[:k])])
    
    return dcg / idcg if idcg > 0 else 0

def calculate_p_at_k(relevant_docs, total_docs, k=20):
    """计算P@k"""
    return len(relevant_docs[:k]) / k

def calculate_f1(pred, gold):
    """计算F1分数"""
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    common = set(pred_tokens) & set(gold_tokens)
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(gold_tokens) if gold_tokens else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
```
