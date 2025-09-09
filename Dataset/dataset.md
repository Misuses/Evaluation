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

---

# SQuAD 数据集介绍

### 一、数据集简介

SQuAD（Stanford Question Answering Dataset）是斯坦福大学构建的问答数据集，是评估阅读理解和问答系统能力的经典基准之一。数据来自维基百科页面，要求模型在给定的上下文中定位答案片段。SQuAD 通过提供明确上下文和问题的对应关系，测试检索增强生成（RAG）系统在理解文本、提取信息方面的能力，并为构建高质量问答模型提供了标准测试环境。

该数据集包含两个版本：

* **SQuAD v1.1**：所有问题在上下文中都可以找到答案；
* **SQuAD v2.0**：引入了无法回答的问题，增强模型应对“不可回答问题”的能力。

### 二、数据集格式结构

#### （一）数据存储形式

以 JSON 格式组织单条数据，结构如下（以 SQuAD v1.1 为例）：

```json
{
    "context": "Architecturally, the school has a Catholic character...",
    "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
    "answers": [
        {
            "text": "Saint Bernadette Soubirous",
            "answer_start": 515
        }
    ]
}
```

对于 SQuAD v2.0，可能包含以下字段：

```json
"is_impossible": true
```

用于标注该问题在 `context` 中没有可用答案。

#### （二）字段逻辑说明

* **`context`**：问题所依赖的背景文本，模型需从中抽取或判断答案。
* **`question`**：自然语言问题，覆盖事实类、时间类、人物类等不同类型。
* **`answers`**：存储答案片段，包含 `text` 和其在 `context` 中的起始位置。
* **`is_impossible`**（可选）：用于标注无法从上下文中回答的问题（仅 SQuAD v2.0）。

### 三、选用此数据集的文献

**文献标题**：《DAT: Dynamic Alpha Tuning for Hybrid Retrieval in Retrieval-Augmented Generation》

**作者**：Wang R, Chen S, Zhao K, et al.

**发表平台**：arXiv:2503.23013 

**文献关联说明**：该文献提出了一种动态调整稀疏-稠密检索比例的混合检索方法（DAT），并在 SQuAD 数据集上进行评估。实验结果表明，DAT 方法在准确性和稳定性方面均优于固定加权策略，有效提升了模型在不同问题类型下的检索增强生成表现，验证了 SQuAD 在检索比例自适应调控研究中的基准价值。


---

#  PopQA 数据集简介

* **全称**：**PopQA** (Popular Culture Question Answering Dataset)

* **提出时间**：2022 年，由 Facebook AI (Meta AI Research) 团队提出。

* **用途**：专门用于评估 **开放域问答（Open-domain QA）系统**在 **流行文化（Popular Culture）知识**上的表现。

* **动机**：

  * 传统 QA 数据集（如 **Natural Questions、TriviaQA**）多集中在百科知识，而缺乏对 **长尾知识和大众文化领域**的覆盖。
  * PopQA 补充了这一空缺，强调评估模型是否掌握 **不在维基百科等知识库主流内容中的知识**。

* **数据规模**：

  * 约 **14 万条问答对（Q-A pairs）**。
  * 覆盖 **电影、电视剧、音乐、电子游戏、体育、明星等领域**。

* **任务形式**：

  * 输入：一个自然语言问题。
  * 输出：答案（一般是 **实体名称**，如人物、作品、地名）。

---

## 数据集格式

PopQA 的数据文件通常是 **JSONL (JSON Lines)** 格式，每一行是一条问答样本，基本字段如下：

```json
{
  "id": "popqa_000001",
  "question": "Who played Jack in the movie Titanic?",
  "answers": ["Leonardo DiCaprio"],
  "category": "movies",
  "source": "imdb"
}
```

### 字段说明

* **id**: 样本编号（唯一标识）。
* **question**: 提出的问题（自然语言形式）。
* **answers**: 答案列表（通常只有 1 个正确答案，但也可能有多个同义答案）。
* **category**: 问题所属的流行文化类别（如 *movies*, *music*, *sports*, *tv* 等）。
* **source**: 数据来源（如 IMDb、MusicBrainz、Sports DB 等）。

---

### 三、选用此数据集的文献

**文献标题**：《DH-RAG: A Dynamic Historical Context-Powered Retrieval-Augmented Generation Method for Multi-Turn Dialogue》

**作者**：Zhang F, Zhu D, Ming J, et al.

**发表平台**：arXiv preprint arXiv:2502.13847, 2025

**文献关联说明**：该文献提出了一种面向多轮对话的动态历史上下文增强检索生成方法（DH-RAG），通过引入历史对话语境的动态建模机制，有效缓解了多轮交互中信息丢失与语境漂移的问题。在实验中，研究者选用了 PopQA 数据集作为评测基准，验证了该方法在流行文化知识问答场景下的有效性。PopQA 数据集具有以下特点：其问题覆盖电影、音乐、体育、电视剧等流行文化领域，知识分布呈现明显的长尾特性，并且部分问答需要依赖上下文信息才能得到正确答案。这些特性与多轮对话场景高度契合，使得 PopQA 成为检验 DH-RAG 在处理 **长尾知识、上下文依赖及对话连贯性** 方面表现的理想数据集。实验结果表明，DH-RAG 在保持对话一致性的同时显著提升了答案的准确率和相关性，凸显了 PopQA 在多轮对话检索增强生成方法研究中的重要基准价值。

---






