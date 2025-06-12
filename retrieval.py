import os
import json
import logging
from typing import List, Tuple, Dict, Optional
import jieba
from whoosh import index
from whoosh.fields import Schema, ID, TEXT
from whoosh.analysis import Tokenizer, Token
from whoosh.query import Or, Term
from whoosh import scoring
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Milvus
from langchain.schema import Document

from config.config import DATABASE_CONFIG, MODEL_CONFIG

# === 日志配置 ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === 1. Whoosh 索引配置 ===
class JiebaTokenizer(Tokenizer):
    def __call__(self, text, **kwargs):
        pos = 0
        offset = 0
        for word in jieba.lcut(text):
            start = text.find(word, offset)
            end = start + len(word)
            t = Token(text=word, pos=pos, startchar=start, endchar=end)
            yield t
            pos += 1
            offset = end

jieba_analyzer = JiebaTokenizer()
schema = Schema(
    doc_id=ID(stored=True, unique=True),
    source=ID(stored=True),
    content=TEXT(stored=True, analyzer=jieba_analyzer)
)

INDEX_DIR = "whoosh_index1"

if os.path.exists(INDEX_DIR):
    try:
        ix = index.open_dir(INDEX_DIR)
        print(f"✅ Whoosh 索引已打开，文档数: {ix.doc_count_all()}")
    except Exception as e:
        print(f"打开Whoosh索引失败: {e}")
else:
    print("索引目录不存在，请先创建索引。")
def compute_specificity_score(query_terms: List[str], corpus: List[str]) -> float:
    if not corpus or not query_terms:
        return 0.0
    try:
        tfidf = TfidfVectorizer()
        tfidf.fit(corpus)
        query = " ".join(query_terms)
        vec = tfidf.transform([query])
        return vec.sum() / max(1, len(query_terms))
    except Exception:
        return 0.5

def rrf_fusion(dense_docs, sparse_docs, w_dense: float = 0.5, w_sparse: float = 0.5, epsilon: float = 1) -> List[Tuple[Document, float]]:
    scores = {}
    seen_docs = {}
    for rank, (doc, _) in enumerate(dense_docs):
        src = doc.metadata.get("source", doc.page_content[:30])
        scores[src] = scores.get(src, 0) + w_dense / (epsilon + rank + 1)
        seen_docs[src] = doc
    for rank, (doc, _) in enumerate(sparse_docs):
        src = doc.metadata.get("source", doc.page_content[:30])
        scores[src] = scores.get(src, 0) + w_sparse / (epsilon + rank + 1)
        seen_docs[src] = doc
    ranked = sorted(
        [(seen_docs[src], score) for src, score in scores.items()],
        key=lambda x: x[1],
        reverse=True
    )
    return ranked


def vector_search(query: str, top_k: int = DATABASE_CONFIG["milvus_query_top_k"]) -> List[Tuple[Document, float]]:
    if not ix:
        logging.info("Whoosh 索引未初始化，无法进行向量检索")
        return []
    try:
        embeddings = OllamaEmbeddings(model=MODEL_CONFIG["ollama_embedding_model_name"])
        vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": DATABASE_CONFIG["milvus_docker_url"]},
            collection_name=DATABASE_CONFIG["milvus_collection_name"],
            text_field="question",
        )
        logging.info(f"开始进行向量检索，查询: {query}")

        if not query or not isinstance(query, str):
            logging.error("查询内容无效")
            return []

        results = vector_store.similarity_search_with_score(query=query, k=top_k)

        if not results:
            logging.warning("向量检索返回空结果")
            return []

        processed_results = []
        for i, (doc, dist) in enumerate(results):
            metadata = doc.metadata or {}
            question = metadata.get("question", "")
            answer = metadata.get("answer", "")
            page_content = f"{question}\n{answer}" if question or answer else doc.page_content

            metadata["source"] = answer if answer else f"milvus_doc_{i}"

            processed_results.append((
                Document(page_content=page_content, metadata=metadata),
                1.0 / (1.0 + dist)
            ))

        logging.info(f"向量检索成功返回 {len(processed_results)} 个结果")
        return processed_results
    except KeyError as e:
        logging.error(f"Milvus 检索失败 - 键错误: {e}")
        return []
    except ValueError as e:
        logging.error(f"Milvus 检索失败 - 值错误: {e}")
        return []
    except Exception as e:
        import traceback
        logging.error(f"Milvus 检索失败: {str(e)}\n{traceback.format_exc()}")
        return []

def keyword_search(query: str, top_k: int = DATABASE_CONFIG["bm25_query_top_k"]) -> List[Tuple[Document, float]]:
    if not ix:
        return []
    tokens = [tok for tok in jieba.lcut(query) if tok.strip()]
    if not tokens:
        return []
    whoosh_query = Or([Term("content", tok) for tok in tokens])
    whoosh_hits = []
    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        results = searcher.search(whoosh_query, limit=top_k)
        max_score = max((hit.score for hit in results), default=1.0)
        for hit in results:
            doc = Document(
                page_content=hit["content"],
                metadata={"source": hit["source"], "id": hit["doc_id"]}
            )
            score = hit.score / max_score if max_score > 0 else 0.0
            whoosh_hits.append((doc, score))
    return whoosh_hits

def retrieve_grouding_document(
    query: str,
    corpus: Optional[List[str]] = None,
    use_vector_search: bool = True,
    use_keyword_search: bool = True,
    static_alpha: Optional[float] = None
) -> List[Tuple[Document, float]]:
    if corpus is None and ix:
        with ix.searcher() as s:
            corpus = [s.stored_fields(docnum)["content"] for docnum in range(s.doc_count_all())]
    if not corpus:
        return []
    if static_alpha is not None:
        w_dense = static_alpha
        w_sparse = 1.0 - static_alpha
        logging.info(f"[静态融合] α={static_alpha:.2f} -> 向量权重: {w_dense:.2f}, 关键词权重: {w_sparse:.2f}")
    else:
        query_terms = [tok for tok in jieba.lcut(query) if tok.strip()]
        specificity = compute_specificity_score(query_terms, corpus)
        w_sparse = max(min(specificity, 1.0), 0.0)
        w_dense = 1.0 - w_sparse
        logging.info(f"[动态融合] 查询具体性: {specificity:.2f}, 向量权重: {w_dense:.2f}, 关键词权重: {w_sparse:.2f}")
    dense_docs = vector_search(query) if use_vector_search else []
    sparse_docs = keyword_search(query) if use_keyword_search else []
    if dense_docs and sparse_docs:
        return rrf_fusion(dense_docs, sparse_docs, w_dense, w_sparse)
    elif dense_docs:
        return dense_docs
    elif sparse_docs:
        return sparse_docs
    else:
        return []

def retrieve_grouding_document_whoosh(query: str, top_k: int = DATABASE_CONFIG["bm25_query_top_k"]) -> List[Tuple[Document, float]]:
    return keyword_search(query, top_k)

def retrieve_grouding_document_vector(
        query: str,
        top_k: int = DATABASE_CONFIG["milvus_query_top_k"]
) -> List[Tuple[Document, float]]:
    return vector_search(query, top_k)