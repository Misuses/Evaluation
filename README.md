# 多模式信息检索评估

## 项目概述  
本项目构建多模式信息检索系统，融合**关键词检索**与**向量检索**能力，支持静态、动态权重融合策略。通过召回率、精确率等核心指标量化检索效果，结合可视化图表对比不同方案性能，助力高效信息检索与结果评估。  

### 数据集（Dataset）
本项目使用了多个数据集来评估检索系统的性能：
- **HaluEval 数据集**：聚焦大语言模型幻觉缓解相关的混合检索研究场景，专门用于验证检索策略在纠正模型幻觉答案、提升回答准确性方面的效果。其覆盖 19 世纪美国文学期刊及现代女性杂志等内容，通过构建包含知识文本、问题、正确答案与幻觉答案的样本，为评估检索系统对模型幻觉的干预能力提供真实且针对性强的数据支撑。
- **Natural Questions 数据集**：是一个用于评估检索增强生成（RAG）系统性能的重要数据集。它涵盖了广泛的自然语言问题，问题来源多样，涉及文学、语言、历史、科学等多个领域 ，旨在模拟真实世界中用户的信息查询需求。该数据集通过收集大量的自然语言问题，并人工标注相应的答案，为研究人员提供了丰富且具有代表性的测试样本。
- **SQuAD（Stanford Question Answering Dataset）数据集**：是斯坦福大学构建的问答数据集，是评估阅读理解和问答系统能力的经典基准之一。数据来自维基百科页面，要求模型在给定的上下文中定位答案片段。SQuAD 通过提供明确上下文和问题的对应关系，测试检索增强生成（RAG）系统在理解文本、提取信息方面的能力，并为构建高质量问答模型提供了标准测试环境。
- **PopQA数据集** 是一个大规模开放领域问答（QA）数据集，包含14,000个实体中心问答对。每个问题通过使用模板从Wikidata检索的知识元组转换而来。每个问题附带原始的subject_entity、object_entity和relationship_type注释，以及Wikipedia的月度页面浏览量。
- 详细内容可以查看[dataset.md](https://github.com/Misuses/Evaluation/blob/main/Dataset/dataset.md "访问数据集描述")
### 方法（Method）
本项目涉及多种检索和评估方法：
- **多模式检索**：
    - **关键词检索**：基于 Whoosh 引擎实现关键词匹配，快速定位含目标词的文档，适配明确词项查询场景。
    - **向量检索**：依托 OllamaEmbeddings 与 Milvus 向量库，挖掘语义关联，解决模糊、长尾查询需求。
- **融合策略**：
    - **静态融合**：通过 `static_alpha` 参数固定向量检索、关键词检索的权重占比，简单直接。
    - **动态融合**：依据查询“具体性”（如关键词密度、语义复杂度）自适应调整权重，贴合多样化检索意图。
- **评估体系**：使用召回率、精确率、命中数、平均倒数排名（MRR）、归一化折损累积增益（NDCG）等指标来评估检索系统的性能。
    - 详细内容可以查看[method.md](https://github.com/Misuses/Evaluation/blob/main/Method/method.md "访问方法描述")，[method1.md](https://github.com/Misuses/Evaluation/blob/main/Method/method1.md "访问方法描述")，[method3.md](https://github.com/Misuses/Evaluation/blob/main/Method/method3.md "访问方法描述")，[Method4.md](https://github.com/Misuses/Evaluation/blob/main/Method/Method4.md "访问方法描述")，[method5.md](https://github.com/Misuses/Evaluation/blob/main/Method/method5.md "访问方法描述")
  
## 核心功能  
### 1. 多模式检索  
- **关键词检索**：基于 Whoosh 引擎实现关键词匹配，快速定位含目标词的文档，适配明确词项查询场景。  
- **向量检索**：依托 OllamaEmbeddings 与 Milvus 向量库，挖掘语义关联，解决模糊、长尾查询需求。  

### 2. 融合策略  
- **静态融合**：通过 `static_alpha` 参数固定向量检索、关键词检索的权重占比，简单直接。  
- **动态融合**：依据查询“具体性”（如关键词密度、语义复杂度）自适应调整权重，贴合多样化检索意图。  

### 3. 评估体系 
#### （1）召回率（Recall）  
- **定义**：检索结果覆盖的相关文档，占“所有相关文档”的比例。  
- **公式**：  
  ![Recall公式](Formula/Recall.png)  
- **逻辑**：召回率越高，系统“找全相关文档”能力越强；值为 1 时，无相关文档遗漏。  

#### （2）精确率（Precision）  
- **定义**：检索结果里，真正相关文档的占比。  
- **公式**：  
  ![Precision公式](Formula/Precision.png)  
- **逻辑**：精确率越高，结果“噪声越少”；值为 1 时，所有输出均为相关文档。  

#### （3）命中数（Hit）  
- **定义**：判断检索结果是否包含**至少 1 条相关文档**。  
- **公式**：  
  ![Hit公式](Formula/Hit.png)  
- **逻辑**：二值指标，快速验证“是否命中相关内容”，简化初步效果判断。  

#### （4）平均倒数排名（Mean Reciprocal Rank, MRR）  
- **定义**：衡量“首个相关文档”在结果中的排名（排名越前，值越高）。  
- **公式**：  
  ![MRR公式](Formula/MRR.png)    
- **逻辑**：关注“首条相关结果的位置”，排名第 1 时 MRR=1，体现结果“首条相关性”优劣。  

#### （5）归一化折损累积增益（Normalized Discounted Cumulative Gain, NDCG）  
- **定义**：综合“相关性得分”与“排名顺序”，评估结果质量（得分越高，质量越好）。 
  - 归一化折损累积增益（NDCG）：  
    ![NDCG公式](Formula/NDCG.png#subset=NDCG)  
- **逻辑**：兼顾“相关性强度”与“排序合理性”，NDCG=1 时，结果与“理想排序”完全一致，深度度量检索质量。  

### 4. 可视化分析  
通过 Matplotlib 生成柱状图，直观对比不同检索方式/融合策略的指标表现，快速定位最优方案。  


## 代码结构  
```
.
├── config/       # 配置文件（数据库、模型参数等）
│   └── config.py 
├── retrieval.py  # 检索核心逻辑（关键词、向量、融合策略）
├── evaluation.py # 评估指标计算（召回率、精确率、MRR、NDCG 等）
└── main_eval1.py # 主脚本（加载数据、执行检索+评估+可视化）
```  


## 依赖说明  
需安装以下库（建议通过 `pip` 或 `conda` 配置环境）：  
- 基础工具：`os` `json` `logging`（Python 内置）  
- 分词与检索：`jieba`（中文分词）、`whoosh`（关键词检索）  
- 向量与语义：`langchain_ollama`（OllamaEmbeddings）、`langchain`（向量存储）  
- 数据科学：`sklearn`（TF-IDF 计算）、`matplotlib`（可视化）、`numpy`（数值计算）  


## 使用流程  
1. **环境准备**：安装依赖库，确保 Milvus、Ollama 服务正常运行。  
2. **参数配置**：修改 `config/config.py`，设置数据库连接、检索阈值等参数。  
3. **执行评估**：运行 `main_eval1.py`，自动完成**数据加载→检索→指标计算→可视化输出**。  


## 贡献与反馈  
- 欢迎提交 PR 优化功能（如扩展检索模型、新增评估指标）。  
- 发现问题或需求，可在 GitHub Issues 反馈，会优先响应处理。  


## 协议说明  
项目采用 **MIT 许可证**，允许自由使用、修改、分发，需保留原版权声明。  

 
