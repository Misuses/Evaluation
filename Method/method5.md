# 《ImpRAG: Retrieval-Augmented Generation with Implicit Queries》
-  Zhang W, Lin X V, Stratos K, et al. ImpRAG: Retrieval-Augmented Generation with Implicit Queries[J]. arXiv preprint arXiv:2506.02279, 2025.
  
在检索增强生成（RAG）领域，ImpRAG通过**统一解码器语言模型（LLM）的检索与生成能力**，其核心在于基于LLM底层层组的检索嵌入生成方法——无需显式查询构建，直接从模型注意力状态中提取隐式查询与段落嵌入。以下结合论文《ImpRAG: Retrieval-Augmented Generation with Implicit Queries》的实验场景，详细介绍该方法的实现逻辑、代码示例与实验验证。


## 一、核心概念
ImpRAG的检索嵌入生成方法突破了传统RAG“检索-生成分离”的范式，核心依赖于对预训练decoder-only LLM的**层组划分**与**注意力状态复用**，关键概念如下：
- **层组划分**：将LLM垂直划分为三层组，其中**底层组（\(L_B\)， layers 0~b）** 专门负责检索嵌入生成，复用LLM原有参数（无需额外训练独立检索模型）；
- **注意力状态提取**：从底层组最后一层（b层）的**查询注意力头（Query Head）** 与**键注意力头（Key Head）** 中提取状态，而非使用输出隐藏层，利用Transformer注意力机制的原生相关性建模能力；
- **Last-token Pooling**：对查询/段落的最后一个token对应的注意力状态进行池化，聚合全局语义信息（论文验证该方式优于其他池化策略）；
- **分组平均（Grouped Averaging）**：针对Grouped-Query Attention（GQA，LLaMA 3系列默认机制），对每组内的注意力头状态取平均，平衡计算效率与嵌入表征质量；
- **隐式查询**：嵌入生成过程无需人工设计显式文本查询，直接从输入prompt的注意力状态中提炼信息需求，提升跨任务泛化性。


## 二、数学公式
ImpRAG的检索嵌入生成包含**注意力状态提取→池化→分组平均→相似度计算**四个核心步骤，数学定义如下：

### 1. 注意力状态提取
设底层组最后一层（b层）的注意力配置为：
- \$$(h_k\)：键注意力头数量（Key Heads）；$$
- \$$(g\)：查询注意力组数（Grouped-Query Attention中的组数量）$$；
- \$$(d_h\)：单个注意力头的维度（Head Dimension）$$；
- \$$(X_q \in \mathbb{R}^{T_q \times D}\)：输入查询的token序列（\(T_q\)为查询长度，\(D = h_k \times d_h\)为模型总维度）$$；
- \$$(X_p \in \mathbb{R}^{T_p \times D}\)：知识 corpus 中段落的token序列（\(T_p\)为段落长度）$$。

$$通过底层组\(L_B\)的因果注意力计算后，提取最后一个token（\(T_q\)位置）的**查询注意力状态**与段落最后一个token（\(T_p\)位置）的**键注意力状态**：$$
- 查询注意力状态：\ $$(S_q \in \mathbb{R}^{(h_k \times g) \times d_h}\)$$ $$（GQA下查询头按组划分，共\(h_k \times g\)个查询头）$$；
- 段落注意力状态：\ $$(S_p \in \mathbb{R}^{h_k \times d_h}\)（键头无分组，共\(h_k\)个键头）。$$


### 2. 分组平均池化（查询嵌入）
对查询注意力状态按“组”取平均，得到最终查询嵌入\ $$(E_q\)（消除GQA分组冗余，保留组内语义一致性）$$：
$$E_q^g = \text{LastTokenPooling}(S_q) \quad \Rightarrow \quad E_q^g \in \mathbb{R}^{(h_k \times g) \times d_h}$$

$$E_q = \frac{1}{g} \sum_{i=1}^g E_q^g[(i-1)h_k:i \cdot h_k, :] \quad \Rightarrow \quad E_q \in \mathbb{R}^{h_k \times d_h}$$

其中\ $$(E_q^g[(i-1)h_k:i \cdot h_k, :]\)表示第\(i\)组查询头的状态矩阵。$$


### 3. 段落嵌入生成
对段落的键注意力状态直接进行Last-token Pooling（无需分组，因键头无GQA分组）：
$$E_p = \text{LastTokenPooling}(S_p) \quad \Rightarrow \quad E_p \in \mathbb{R}^{h_k \times d_h}$$


### 4. 嵌入相似度计算
采用**点积相似度**衡量查询与段落的语义相关性（利用Transformer注意力机制的原生匹配逻辑）：
$$s(q, p) = E_q \cdot E_p^T$$

其中\(s(q, p)\)为查询\(q\)与段落\(p\)的相似度得分，得分越高表示相关性越强。


## 三、代码实现
基于LLaMA 3系列模型（论文中Llama-3.2 3B/3.1 8B），实现ImpRAG检索嵌入生成的核心代码如下，需依赖PyTorch与Transformers库，并自定义注意力状态提取逻辑：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def init_imprag_model(model_name: str = "meta-llama/Llama-3.2-3B", layer_boundary_b: int = 7):
    """
    初始化ImpRAG模型：加载LLaMA模型并指定底层组边界b（参考论文4.1，b=7表示底层组为layers 0~7）
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # LLaMA默认无pad_token，需手动指定
    
    # 加载模型并启用注意力状态保存（需自定义LLaMA模型的forward函数以输出注意力状态）
    class LlamaWithAttentionOutput(AutoModelForCausalLM.from_pretrained(model_name).__class__):
        def forward(self, **kwargs):
            outputs = super().forward(** kwargs)
            # 提取底层组最后一层（b层）的查询注意力状态（query_states）和键注意力状态（key_states）
            b_layer = self.model.layers[layer_boundary_b]  # 底层组最后一层
            query_states = b_layer.self_attn.q_proj(outputs.hidden_states[layer_boundary_b])  # (batch, seq_len, h_k*g*d_h)
            key_states = b_layer.self_attn.k_proj(outputs.hidden_states[layer_boundary_b])    # (batch, seq_len, h_k*d_h)
            return outputs, query_states, key_states
    
    model = LlamaWithAttentionOutput.from_pretrained(model_name, output_hidden_states=True)
    return model, tokenizer

def extract_last_token_state(state: torch.Tensor) -> torch.Tensor:
    """
    Last-token Pooling：提取序列最后一个非pad token的注意力状态
    state: (batch, seq_len, hidden_dim) -> 输出: (batch, hidden_dim)
    """
    # 找到每个样本的最后一个非pad token位置（假设pad_token_id=2）
    pad_mask = (state != 2).any(dim=-1)  # (batch, seq_len)，True表示非pad
    last_token_idx = pad_mask.sum(dim=1) - 1  # (batch,)，最后一个非pad位置
    # 提取最后一个token的状态
    return state[torch.arange(state.shape[0]), last_token_idx]

def calculate_imprag_embeddings(model, tokenizer, texts: list, is_query: bool = True, layer_boundary_b: int = 7):
    """
    计算ImpRAG的查询嵌入或段落嵌入
    texts: 输入文本列表（查询或段落）
    is_query: True=计算查询嵌入，False=计算段落嵌入
    返回: 嵌入矩阵 (num_texts, h_k*d_h)
    """
    device = next(model.parameters()).device
    # 文本编码（max_length=512，参考论文实验设置）
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    # 前向传播，获取底层组b层的注意力状态
    with torch.no_grad():
        outputs, query_states, key_states = model(** inputs)
    
    # 1. Last-token Pooling：提取最后一个非pad token的状态
    if is_query:
        # 查询：使用查询注意力状态（query_states）
        pooled_state = extract_last_token_state(query_states)  # (batch, h_k*g*d_h)
        # 2. 分组平均：按GQA的组划分，每组内取平均（LLaMA 3的g=8，h_k=32，参考官方配置）
        g = 8  # 查询注意力组数
        h_k = 32  # 键注意力头数
        d_h = pooled_state.shape[-1] // (h_k * g)  # 单个头维度
        # 重塑为(batch, g, h_k, d_h)，按组平均后重塑为(batch, h_k*d_h)
        pooled_state_reshaped = pooled_state.view(-1, g, h_k, d_h)  # (batch, g, h_k, d_h)
        query_emb = pooled_state_reshaped.mean(dim=1).view(-1, h_k * d_h)  # (batch, h_k*d_h)
        return query_emb
    else:
        # 段落：使用键注意力状态（key_states），无需分组
        para_emb = extract_last_token_state(key_states)  # (batch, h_k*d_h)
        return para_emb

def compute_similarity(query_emb: torch.Tensor, para_emb: torch.Tensor) -> torch.Tensor:
    """
    计算查询嵌入与段落嵌入的点积相似度
    query_emb: (num_queries, embed_dim)
    para_emb: (num_paras, embed_dim)
    返回: 相似度矩阵 (num_queries, num_paras)
    """
    return torch.matmul(query_emb, para_emb.T)  # 点积计算相似度

# ------------------- 示例：生成查询与段落嵌入并计算相似度 -------------------
if __name__ == "__main__":
    # 1. 初始化模型（层边界b=7，参考论文4.1设置）
    model, tokenizer = init_imprag_model(model_name="meta-llama/Llama-3.2-3B", layer_boundary_b=7)
    model.eval()
    
    # 2. 输入文本（查询来自NQ数据集，段落来自Wikipedia corpus）
    queries = ["What is the capital of France?"]  # 示例查询
    paragraphs = [
        "Paris is the capital and most populous city of France.",  # 相关段落
        "London is the capital and largest city of the United Kingdom."  # 无关段落
    ]
    
    # 3. 计算嵌入
    query_emb = calculate_imprag_embeddings(model, tokenizer, queries, is_query=True)
    para_emb = calculate_imprag_embeddings(model, tokenizer, paragraphs, is_query=False)
    
    # 4. 计算相似度
    similarity_scores = compute_similarity(query_emb, para_emb)
    print("相似度得分（查询-段落）：")
    for i, query in enumerate(queries):
        for j, para in enumerate(paragraphs):
            print(f"查询: {query} \n段落{j+1}: {para} \n得分: {similarity_scores[i][j].item():.4f}\n")
```


## 四、实现要点
1. **层边界\(b\)的选择**  
   层边界\(b\)（底层组的最后一层）直接影响检索性能：论文5.1节通过ablation实验表明，当\(b=7\)（即底层组包含layers 0~7）时，Llama-3.2 3B/Llama-3.1 8B的检索召回率达到最优（如NQ数据集召回率78.4%）；若\(b<7\)，底层组参数不足导致嵌入表征能力弱；若\(b>7\)，则会挤占生成层参数，降低自蒸馏的训练信号精度（见图1左半部分）。

2. **注意力机制的保留**  
   论文3.1节明确：底层组需**保留因果注意力（Causal Attention）**，而非改为双向注意力。因LLM预训练的因果注意力已适配文本序列的语义建模，双向注意力反而会破坏查询与段落的时序逻辑，导致相似度计算偏差（实验验证无性能提升）。

3. **池化方式的合理性**  
   对比实验表明，Last-token Pooling优于平均池化（Average Pooling）或首token池化：最后一个token的注意力状态已聚合整个输入序列的语义信息，更适合作为全局嵌入；而平均池化会稀释关键信息，首token池化则无法捕捉长序列的尾部语义。

4. **计算效率优化**  
   该方法复用LLM底层参数，无需训练独立检索模型（如Contriever），减少30%+的参数量；同时，段落嵌入可预计算并存储（论文附录C提到用Product Quantization压缩KV状态，仅损失0.4%检索精度），推理时直接加载，降低实时计算成本。


## 五、论文实验中的方法应用
### 1. 实验设置
- **数据集**：检索嵌入生成的训练依赖需检索的数据集（NQ、HotpotQA）与无检索的指令微调数据集（如oasst1、SQuAD），知识corpus为2021年12月Wikipedia快照；
- **基线对比**：与传统检索嵌入方法对比，包括RA-IT（用Contriever生成嵌入）、RA-DIT（用微调后的Contriever生成嵌入）、RA-DIT-Llama（用LLaMA前8层生成嵌入）；
- **评估指标**：检索性能用**检索召回率（Retrieval Recall）** 衡量（即top-k检索结果中包含答案的比例，论文中\(k=10\)）。

### 2. 关键实验结果
表1展示了Llama-3.2 3B模型在8个知识密集型任务上的检索召回率对比，ImpRAG的检索嵌入生成方法显著优于基线：

| 任务       | RA-IT（Contriever） | RA-DIT（微调Contriever） | RA-DIT-Llama（LLaMA前8层） | ImpRAG（本文方法） |
|------------|---------------------|--------------------------|----------------------------|--------------------|
| NQ（基础QA）| 77.0%               | 77.5%                    | 78.0%                      | 78.4%              |
| HotpotQA（多跳）| 48.8%            | 49.3%                    | 49.8%                      | 50.2%              |
| T-Rex（关系抽取）| 84.3%          | 85.0%                    | 85.9%                      | 90.2%              |
| AIDA（实体链接）| 38.1%          | 38.2%                    | 38.4%                      | 58.3%              |

*注：数据来自论文Table 2，括号内为检索嵌入生成方法*

关键结论：
- 在AIDA（实体链接）任务中，ImpRAG的检索召回率提升20个百分点，因传统方法需用显式查询（如“British”），而ImpRAG的隐式嵌入能捕捉上下文语义（如“EU rejects German call to boycott British lamb”）；
- 多跳任务（HotpotQA）中，ImpRAG的嵌入能建模段落间依赖（论文附录A验证Full Attention Concatenation策略提升2.5%精度），优于基线的独立段落嵌入。

### 3. 层边界Ablation实验
论文Figure 2（左）展示了不同层边界\(b\)对NQ数据集检索召回率与生成Exact Match（EM）的影响：
- 当\(b=7\)时，检索召回率达78.4%，EM达44.1%，为最优值；
- 当\(b<7\)（如\(b=5\)），召回率降至74.3%，EM降至42.1%（底层组参数不足）；
- 当\(b>7\)（如\(b=9\)），召回率仅提升至78.3%，但EM降至43.5%（生成层参数被挤占）。


## 六、与传统检索嵌入方法的对比
ImpRAG的检索嵌入生成方法与传统方法（如Contriever、RA-DIT）的核心差异体现在“模型统一性”与“查询方式”，具体对比见表2：

| 对比维度       | 传统方法（Contriever/RA-DIT） | ImpRAG（本文方法）          |
|----------------|--------------------------------|-----------------------------|
| 模型依赖       | 独立编码器（如BERT-based）     | 复用LLM底层参数（无额外模型）|
| 注意力机制     | 双向注意力（破坏时序逻辑）     | 因果注意力（适配LLM预训练） |
| 查询方式       | 显式文本查询（需人工设计模板） | 隐式注意力状态（无人工干预） |
| 跨任务泛化性   | 差（模板不匹配时召回率降30%+） | 优（AIDA任务召回率提升20%） |
| 检索召回率（NQ）| 77.0%~77.5%                   | 78.4%                       |
| 参数量         | 额外1.2B参数（Contriever）     | 0额外参数（复用LLM）        |

核心优势：ImpRAG通过“统一模型+隐式嵌入”解决了传统RAG的“检索-生成鸿沟”——传统方法的检索模型与生成模型独立训练，嵌入空间不一致；而ImpRAG的嵌入与生成共享同一LLM语义空间，显著提升跨任务泛化性。


## 七、总结
ImpRAG的检索嵌入生成方法是其实现“查询无关RAG”的核心：通过层组划分复用LLM底层参数，从注意力状态中提取隐式嵌入，既避免了显式查询的人工成本，又解决了传统检索与生成的模型割裂问题。实验表明，该方法在8个知识密集型任务上平均提升5.0~23.2个检索召回率百分点，尤其在格式多样的未见过任务（如实体链接、关系抽取）中表现突出，为通用RAG系统的构建提供了新范式。
