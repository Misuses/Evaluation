# HaluEval 数据集介绍
## 一、数据集简介
HaluEval 数据集聚焦大语言模型幻觉缓解相关的混合检索研究场景，专门用于验证检索策略在纠正模型幻觉答案、提升回答准确性方面的效果。其覆盖 19 世纪美国文学期刊及现代女性杂志等内容，通过构建包含知识文本、问题、正确答案与幻觉答案的样本，为评估检索系统对模型幻觉的干预能力提供真实且针对性强的数据支撑，助力探究混合检索在大语言模型应用中的价值 。

## 二、数据集格式结构
### （一）数据存储形式  
以 JSON 格式组织单条数据，结构如下：  
```json
{
    "knowledge": "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.First for Women is a woman's magazine published by Bauer Media Group in the USA.", 
    "question": "Which magazine was started first Arthur's Magazine or First for Women?", 
    "right_answer": "Arthur's Magazine", 
    "hallucinated_answer": "First for Women was started first."
}
```  

### （二）字段逻辑说明  
- **`knowledge`**：存储与问题相关的知识文本，涵盖杂志的历史背景、出版信息等，是检索系统获取正确答案依据的核心内容，为检索策略提供事实素材 。  
- **`question`**：提出对比类问题，聚焦不同杂志的创办时间先后，明确检索与评估的任务目标，模拟实际应用中用户对特定知识的查询需求 。  
- **`right_answer`**：标注基于 `knowledge` 可推导的正确答案，作为评估检索系统输出、验证模型回答是否被纠正的标准参照 。  
- **`hallucinated_answer`**：呈现大语言模型易产生的幻觉答案，用于测试检索策略对这类错误回答的“修正能力”，即验证检索结果能否辅助模型摒弃幻觉答案、输出正确内容 。  


## 三、选用此数据集的文献  
**文献标题**：《Hybrid Retrieval for Hallucination Mitigation in Large Language Models: A Comparative Analysis》  
**作者**：Mala C S, Gezici G, Giannotti F  
**发表平台**：arXiv preprint arXiv:2504.05324  
**文献关联说明**：该文献利用 HaluEval 数据集，对比不同混合检索策略在缓解大语言模型幻觉问题中的表现。通过分析数据集里知识文本、问题、幻觉答案与正确答案的关联，验证混合检索对纠正模型错误回答、提升输出可靠性的作用，为大语言模型结合检索优化幻觉问题提供实证依据 。 

# Natural Questions数据集介绍
## 一、数据集简介
Natural Questions 数据集是一个用于评估检索增强生成（RAG）系统性能的重要数据集。它涵盖了广泛的自然语言问题，问题来源多样，涉及文学、语言、历史、科学等多个领域 ，旨在模拟真实世界中用户的信息查询需求。该数据集通过收集大量的自然语言问题，并人工标注相应的答案，为研究人员提供了丰富且具有代表性的测试样本，以便于评估不同检索和生成策略在回答自然语言问题方面的准确性和有效性，推动 RAG 系统在提高回答质量、增强语义理解能力等方面的发展。

## 二、数据集格式结构
### （一）数据存储形式
Natural Questions 数据集以 JSON 格式存储数据，单条数据呈现以下结构：
```json
{
    "question": "how did long john silver lose his leg in treasure island", 
    "answer": ["in the Royal Navy"]
}
```

### （二）字段说明
- **`question`**：该字段记录自然语言形式的问题，是用户实际提出的信息查询需求，这些问题表述自然、多样，涵盖了各种不同的语法结构和语义类型，能够充分测试检索和生成模型对不同类型问题的理解和处理能力。
- **`answer`**：以列表形式存储针对 `question` 的答案。答案是经过人工标注确认的，具有较高的准确性和权威性，用于作为评估模型回答正确性的参考标准。

## 三、选用此数据集的文献
**文献标题**：《Blended rag: Improving rag (retriever-augmented generation) accuracy with semantic search and hybrid query-based retrievers》
**作者**：Sawarkar K, Mangal A, Solanki S R
**发表平台**：2024 IEEE 7th International Conference on Multimedia Information Processing and Retrieval (MIPR)
**文献关联说明**：在这篇文献中，作者团队使用 Natural Questions 数据集来评估他们所提出的混合检索增强生成（Blended RAG）方法。通过在该数据集上进行实验，对比不同检索策略（包括语义搜索和基于混合查询的检索器）在回答问题时的准确性，验证了 Blended RAG 方法能够有效提升 RAG 系统的性能，展示了该方法在处理自然语言问题、获取准确答案方面的优势，为 RAG 系统的优化提供了重要的实证依据。 
