import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma

from utils.document_loader import load_document
from utils.text_splitter import split_documents

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_community.retrievers import BM25Retriever
from FlagEmbedding import FlagReranker

# Rerank 模型全局缓存
_reranker = None

# 1. 初始化嵌入模型（和Ollama联动）
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# 2. 向量库持久化路径
# 获取当前文件（rag_base.py）所在的绝对路径
CURRENT_FILE_PATH = os.path.abspath(__file__)
# 获取 utils 目录
UTILS_DIR = os.path.dirname(CURRENT_FILE_PATH)
# 获取项目根目录（utils 的上一级，也就是 LangChainTest 目录）
PROJECT_ROOT = os.path.dirname(UTILS_DIR)
# 向量库统一放在项目根目录，路径永远固定
CHROMA_PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

# 初始化大模型
llm = OllamaLLM(model="qwen2.5:7b", temperature=0.3)

# BM25 检索器全局缓存（混合检索用）
_bm25_retriever = None

# 替换原有模板，强化防幻觉 + 要求标注来源，生成侧优化核心
RAG_PROMPT = """
你是专业的知识库问答助手，必须严格遵守以下所有规则。

【核心原则】
所有回答必须100%基于下方【参考资料】生成，禁止使用任何自身知识、常识、推理补充资料以外的内容。

【回答规则】
1.  参考资料中有明确答案的，分点输出核心内容，语言简洁准确，不做额外扩展。
2.  参考资料中没有相关信息的，直接回复「参考资料中未提及相关内容」，禁止猜测、推断、强行回答。
3.  每个核心要点结尾标注对应资料编号，格式：[资料X]。
4.  答案必须和原文表述一致，不得改写含义、不得替换专业术语、不得增减信息。
5.  如果参考资料内容有冲突，按资料编号靠前的内容为准，不得自行整合。

【参考资料】
{context}

【用户问题】
{question}
"""


# ========== 1. 初始化 BM25 检索器 ==========
def init_bm25_retriever(chunks, top_k: int = 5):
    """基于分块文档初始化 BM25 关键词检索器"""
    global _bm25_retriever
    _bm25_retriever = BM25Retriever.from_documents(chunks)
    _bm25_retriever.k = top_k
    return _bm25_retriever


# ========== 2. 核心：RRF 倒数秩融合算法 ==========
def rrf_fuse(rank_lists, weights=None, k=60):
    """
    倒数秩融合（Reciprocal Rank Fusion）：合并多路检索结果并重新排序
    rank_lists: 多路检索的文档列表，按相关性从高到低排序
    k: 平滑系数，默认60是工业界通用值
    weights: 对应每一路的权重，默认等权重
    """
    if weights is None:
        weights = [1.0] * len(rank_lists)

    scores = {}
    doc_map = {}

    for list_idx, rank_list in enumerate(rank_lists):
        weight = weights[list_idx]
        for rank, doc in enumerate(rank_list, 1):
            # 用文档内容作为唯一标识
            doc_id = doc.page_content
            doc_map[doc_id] = doc
            # 核心公式：分数 *= 对应路的权重
            scores[doc_id] = scores.get(doc_id, 0) + weight * (1 / (k + rank))

    # 按融合分数从高到低排序
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in sorted_docs]


# ========== 3. 获取混合检索结果 ==========
def ensemble_retrieve(query: str, collection_name: str = "default", top_k: int = 10):
    """向量检索 + BM25 关键词检索 混合召回 + RRF 重排序"""
    global _bm25_retriever

    # 自动懒加载：如果BM25未初始化，从向量库读取文档并构建
    if _bm25_retriever is None:
        db = get_vector_db(collection_name)
        # 从向量库取出所有文档内容与元数据
        all_data = db.get()
        docs = [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(all_data["documents"], all_data["metadatas"])
        ]
        _bm25_retriever = BM25Retriever.from_documents(docs)
        print("✅ BM25 检索器已自动从向量库初始化完成")

    # 1. 向量语义检索
    db = get_vector_db(collection_name)
    vector_retriever = db.as_retriever(search_kwargs={"k": top_k})
    vector_docs = vector_retriever.invoke(query)

    # 2. BM25 关键词检索
    _bm25_retriever.k = top_k
    bm25_docs = _bm25_retriever.invoke(query)

    # 3. 融合重排序并返回 Top N，向量为主，关键词为辅
    fused_docs = rrf_fuse([vector_docs, bm25_docs], weights=[0.8, 0.2])
    return fused_docs[:top_k]


def build_vector_db(file_path: str, collection_name: str = "default"):
    """
    加载文档→分块→向量化→存入ChromaDB并持久化
    """
    # 加载并分块文档
    docs = load_document(file_path)
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)

    # 新增：同步初始化 BM25 检索器
    init_bm25_retriever(chunks)

    print(f"\n正在向量化并存入向量库，共 {len(chunks)} 个文本块...")

    # 创建向量库并持久化
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR
    )

    print(f"向量库构建完成，已持久化到 {CHROMA_PERSIST_DIR}")
    return vector_db


def get_vector_db(collection_name: str = "default") -> Chroma:
    """加载已持久化的向量库"""
    # print(f"当前向量库路径：{os.path.abspath(CHROMA_PERSIST_DIR)}")
    vector_db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    return vector_db


def similarity_search(query: str, collection_name: str = "default", top_k: int = 3):
    """相似度检索，返回Top K个相关文本块"""
    db = get_vector_db(collection_name)
    results = db.similarity_search(query, k=top_k)

    print(f"\n=== 检索结果（Top{top_k}） ===")
    for i, doc in enumerate(results):
        print(f"\n[{i + 1}] 相似度相关文档：")
        print(f"内容：{doc.page_content}")
        print(f"来源：{doc.metadata}")
    return results


# ===================== 查询改写函数（多轮对话专用） =====================
def rewrite_query(query: str, history: list = None) -> str:
    """
    结合对话历史，将用户当前问题改写为完整独立的检索问句
    解决：指代消解、上下文依赖、问题不完整 等多轮场景问题
    history: 历史对话列表，格式 [{"role": "user", "content": "xxx"}, {"role": "assistant", "content": "xxx"}]
    """
    # 没有历史直接返回原问题
    if not history or len(history) == 0:
        return query

    # 只保留最近3轮，避免历史过长引入噪声
    recent_history = history[-6:]
    history_text = "\n".join([f"{item['role']}: {item['content']}" for item in recent_history])

    rewrite_prompt = f"""你是一个查询改写助手。
你的任务：结合对话历史，把用户当前的问题，改写为一个完整、独立、适合向量检索的问句。

规则：
1.  补全省略的主语、指代（它、这个、那个等），消除歧义
2.  保留原问题的核心语义和关键词，不要添加额外信息
3.  绝对不要回答问题，只输出改写后的问句
4.  直接输出结果，不要加前缀、解释、引号

对话历史：
{history_text}

当前问题：
{query}

改写后的问题："""

    result = llm.invoke(rewrite_prompt)
    return result.strip()


def rag_answer(
    query: str,
    collection_name: str = "default",
    top_k: int = 3,
    mode: str = "vector",          # 粗排模式选择
    enable_rerank: bool = False,   # 是否开启精排
    enable_rewrite: bool = False,  # 是否开启查询改写
    history: list = None           # 对话历史，用于改写
) -> str:
    """
    完整RAG问答：可选查询改写 → 检索 → 可选精排 → 生成答案
    """
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    output_parser = StrOutputParser()

    # ---------- 新增：查询改写（仅检索用，最终回答仍用原问题） ----------
    if enable_rewrite:
        search_query = rewrite_query(query, history)
    else:
        search_query = query

    # ---------- 第一步：执行粗排检索 ----------
    if mode == "vector":
        db = get_vector_db(collection_name)
        retriever_k = top_k * 6 if enable_rerank else top_k
        retriever = db.as_retriever(search_kwargs={"k": retriever_k})
        docs = retriever.invoke(search_query)
    elif mode == "ensemble":
        retriever_k = top_k * 6 if enable_rerank else top_k
        docs = ensemble_retrieve(search_query, collection_name, top_k=retriever_k)

    # ---------- 第二步：可选二阶段精排 ----------
    if enable_rerank:
        docs = rerank_documents(query, docs, top_n=top_k)

    # ---------- 第三步：拼接上下文 + 生成答案 ----------
    # 给资料编号，对应提示词里的[资料X]标注
    context_segments = []
    for idx, doc in enumerate(docs, 1):
        context_segments.append(f"[资料{idx}]\n{doc.page_content}")
    context = "\n\n".join(context_segments)

    chain = prompt | llm | output_parser
    answer = chain.invoke({"context": context, "question": query})
    return answer


def rag_answer_with_source(query: str, collection_name: str = "default", top_k: int = 3):
    """带原文溯源的RAG问答"""
    # 先检索获取原文
    db = get_vector_db(collection_name)
    docs = db.similarity_search(query, k=top_k)

    # 拼接上下文
    context = "\n\n".join([f"资料{i + 1}：{doc.page_content}" for i, doc in enumerate(docs)])

    # 生成答案
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    # 组装结果
    result = {
        "answer": answer,
        "sources": [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]
    }
    return result


def compare_answer(query: str, collection_name: str = "default"):
    """对比纯大模型回答和RAG增强回答的差异"""
    # 纯模型回答
    pure_answer = llm.invoke(query)

    # RAG增强回答
    rag_answer_result = rag_answer(query, collection_name)

    print("=" * 60)
    print(f"问题：{query}")
    print("=" * 60)
    print("\n【纯大模型回答】")
    print(pure_answer)
    print("\n【RAG增强回答】")
    print(rag_answer_result)
    print("\n" + "=" * 60)


def debug_search(query: str, collection_name: str = "test_kb", top_k: int = 10):
    """调试专用：打印Top N检索结果，查看目标内容排名"""
    db = get_vector_db(collection_name)
    docs_with_score = db.similarity_search_with_score(query, k=top_k)

    print(f"\n=== 检索调试：问题「{query}」 ===")
    for i, (doc, score) in enumerate(docs_with_score):
        print(f"\n[{i + 1}] 相似度得分：{round(score, 4)}")
        print(f"内容预览：{doc.page_content[:120]}...")


def debug_search_rerank(query: str, collection_name: str = "test_kb", top_k: int = 20, top_n: int = 5):
    """调试专用：对比粗排 vs 精排的排序差异"""
    db = get_vector_db(collection_name)
    raw_docs = db.similarity_search(query, k=top_k)

    print(f"\n{'=' * 60}")
    print(f"问题：{query}")
    print(f"【粗排 Top {top_k} → 精排 Top {top_n}】")
    print('=' * 60)

    print("\n--- 粗排前 5 名 ---")
    for i, doc in enumerate(raw_docs[:5], 1):
        preview = doc.page_content[:80].replace('\n', ' ')
        print(f"[{i}] {preview}...")

    reranked_docs = rerank_documents(query, raw_docs, top_n=top_n)
    print("\n--- 精排后结果 ---")
    for i, doc in enumerate(reranked_docs, 1):
        preview = doc.page_content[:80].replace('\n', ' ')
        print(f"[{i}] {preview}...")


def check_db_status(collection_name: str = "test_kb"):
    """检查向量库状态：文档数量、集合名、路径"""
    db = get_vector_db(collection_name)
    # 获取集合内的文档总数
    doc_count = db._collection.count()
    print(f"\n=== 向量库状态检查 ===")
    print(f"集合名：{collection_name}")
    print(f"持久化路径：{CHROMA_PERSIST_DIR}")
    print(f"集合内文档块数量：{doc_count}")

    # 取第一条样本看看内容
    if doc_count > 0:
        sample = db._collection.get(limit=1)
        print(f"样本内容预览：{sample['documents'][0][:80]}...")
    else:
        print("❌ 当前集合为空，没有任何文档数据")


def compare_retrieve_rank(query: str, target_anchor: str, top_k: int = 10):
    """
    对比测试：纯向量检索 vs 混合检索，目标段落的排名位置
    target_anchor: 目标段落的唯一原文锚点，用于匹配命中
    """
    print(f"\n{'=' * 70}")
    print(f"测试问题：{query}")
    print(f"目标锚点：{target_anchor}")
    print('=' * 70)

    # ========== 1. 纯向量检索 ==========
    db = get_vector_db(collection_name="test_kb")
    vector_docs = db.similarity_search(query, k=top_k)

    vector_rank = -1
    for idx, doc in enumerate(vector_docs, 1):
        if target_anchor in doc.page_content:
            vector_rank = idx
            break

    # ========== 2. 混合检索 ==========
    ensemble_docs = ensemble_retrieve(query, collection_name="test_kb", top_k=top_k)

    ensemble_rank = -1
    for idx, doc in enumerate(ensemble_docs, 1):
        if target_anchor in doc.page_content:
            ensemble_rank = idx
            break

    # ========== 3. 打印对比结果 ==========
    print(f"{'模式':<12} {'目标排名':<10} {'是否进Top3':<12} {'是否进Top5':<12}")
    print("-" * 70)

    def format_rank(rank):
        return f"第 {rank} 名" if rank != -1 else "未命中"

    print(
        f"{'纯向量检索':<12} {format_rank(vector_rank):<10} {'✅' if 1 <= vector_rank <= 3 else '❌':<12} {'✅' if 1 <= vector_rank <= 5 else '❌':<12}")
    print(
        f"{'混合检索':<12} {format_rank(ensemble_rank):<10} {'✅' if 1 <= ensemble_rank <= 3 else '❌':<12} {'✅' if 1 <= ensemble_rank <= 5 else '❌':<12}")

    # 打印结论
    print("\n📊 对比结论：")
    if vector_rank == -1 and ensemble_rank == -1:
        print("   ❌ 两种方式都未命中目标，优先排查分块与向量库问题")
    elif vector_rank == -1:
        print("   ✅ 混合检索优化有效：纯向量未命中，混合检索成功召回")
    elif ensemble_rank == -1:
        print("   ⚠️  混合检索负向优化：纯向量能命中，混合检索反而丢失了")
    elif ensemble_rank < vector_rank:
        print(f"   ✅ 混合检索优化有效：排名提升 {vector_rank - ensemble_rank} 位")
    elif ensemble_rank == vector_rank:
        print("   ➖ 效果持平：两种方式排名一致")
    else:
        print(f"   ⚠️  混合检索负向：排名下降 {ensemble_rank - vector_rank} 位，建议调低BM25权重")

    return {
        "vector_rank": vector_rank,
        "ensemble_rank": ensemble_rank
    }


def _get_reranker():
    """懒加载 bge-reranker-base 模型，全局复用实例"""
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(
            "BAAI/bge-reranker-base",
            use_fp16=True   # 开启半精度，省显存，速度更快
        )
    return _reranker


def rerank_documents(query: str, docs, top_n: int = 3):
    """
    二阶段精排：对候选文档片段做相关性打分重排序
    query: 用户问题
    docs: 粗排返回的候选文档列表
    top_n: 精排后返回的数量
    """
    if not docs:
        return []

    reranker = _get_reranker()

    # 构造「问题-文档」对
    pairs = [[query, doc.page_content] for doc in docs]

    # 批量计算相关性分数
    scores = reranker.compute_score(pairs)

    # 按分数从高到低排序
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # 返回 Top N
    return [doc for doc, score in scored_docs[:top_n]]


def get_collection_stats(collection_name: str = "test_kb"):
    """查看集合统计信息：文档数、向量维度、样本预览"""
    db = get_vector_db(collection_name)
    collection = db._collection
    doc_count = collection.count()

    print(f"\n=== 向量库状态检查 ===")
    print(f"集合名：{collection_name}")
    print(f"持久化路径：{CHROMA_PERSIST_DIR}")
    print(f"集合内文档块数量：{doc_count}")

    if doc_count == 0:
        print("⚠️ 当前集合为空，无向量数据")
        return

    # 显式指定返回向量
    sample = collection.get(limit=1, include=["embeddings", "documents", "metadatas"])

    # 安全判断：先判空，再用 len() 检查长度，完全避开数组布尔判断
    embeddings = sample.get("embeddings")
    if embeddings is not None and len(embeddings) > 0:
        # 取第一条向量的长度即为维度，numpy数组和列表都支持 len()
        dim = len(embeddings[0])
        print(f"向量维度：{dim}")
    else:
        print("⚠️ 未获取到向量数据，请检查嵌入模型配置")

    # 打印第一条样本预览
    docs = sample.get("documents", [])
    if docs:
        preview = docs[0][:80].replace("\n", " ")
        print(f"样本内容预览：{preview}...")


if __name__ == "__main__":
    # 构建测试文档的向量库
    doc_path = "../data/test.md"
    # db = build_vector_db(doc_path, collection_name="test_kb")

    # 执行检查
    # check_db_status()
    # get_collection_stats(collection_name="test_kb")

    test_query = "RAG的核心定义是什么"

    # ========== 新增：多轮对话+查询改写 测试 ==========
    # 模拟两轮对话历史
    chat_history = [
        {"role": "user", "content": "RAG相比微调有哪些优势？"},
        {"role": "assistant", "content": "RAG具备落地成本低、更新速度快、可解释性强三大优势。"}
    ]

    # 用户第二轮用指代提问
    follow_up_query = "它的落地成本具体是多少？"

    print("\n>>> 不开改写，直接检索回答：")
    print(rag_answer(follow_up_query, collection_name="test_kb", mode="ensemble"))

    print("\n>>> 开启查询改写后回答：")
    print(rag_answer(follow_up_query, collection_name="test_kb", mode="ensemble",
                     enable_rewrite=True, history=chat_history))

    # =============================三模式问答层对比测试==============================
    # test_questions = [
    #     "RAG 的全称是什么？核心定义是什么？",
    #     "中文场景下分块推荐的 chunk_size 是多少？",
    #     "相比微调，RAG 有哪些优势？"
    # ]
    #
    # for q in test_questions:
    #     print(f"\n>>> 问题：{q}")
    #     print("【纯向量回答：】\n", rag_answer(q, collection_name="test_kb"))
    #     print("【混合检索回答：】\n", rag_answer(q, collection_name="test_kb", mode="ensemble"))
    #     print("【混合检索 + Rerank 精排回答：】\n", rag_answer(q, collection_name="test_kb",
    #           mode="ensemble", enable_rerank=True))
    # =============================三模式问答层对比测试==============================

    # =============================三道基准测试题 + 对应锚点=============================
    # test_cases = [
    #     {
    #         "query": "RAG 的全称是什么？核心定义是什么？",
    #         "anchor": "检索增强生成（Retrieval-Augmented Generation，简称RAG）"
    #     },
    #     {
    #         "query": "中文场景下分块推荐的 chunk_size 是多少？",
    #         "anchor": "基础块大小（chunk_size）：推荐500-800字符"
    #     },
    #     {
    #         "query": "相比微调，RAG 有哪些优势？",
    #         "anchor": "相比模型微调方案，RAG具备三大显著优势"
    #     }
    # ]

    # 混合检索
    #  compare_retrieve_rank(case["query"], case["anchor"], top_k=10)

    # 混合检索 + Rerank 精排
    # debug_search_rerank(test_query, top_k=20, top_n=5)
    # =============================三道基准测试题 + 对应锚点=============================
