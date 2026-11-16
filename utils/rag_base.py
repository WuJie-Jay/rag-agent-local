import os
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

from langchain_chroma import Chroma
from utils.document_loader import load_document
from utils.text_splitter import split_documents

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# 1. 初始化嵌入模型（和Ollama联动）
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# 2. 向量库持久化路径
# CHROMA_PERSIST_DIR = "../chroma_db"

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

# RAG专属提示词模板
RAG_PROMPT = """
你是一个专业的知识库问答助手，请严格基于以下【参考资料】回答用户的问题。
回答要求：
1. 只使用参考资料中的信息，资料里没有的内容就回答"参考资料中未提及相关内容"
2. 答案要准确、简洁，分点表述关键信息
3. 如果参考资料有多条，整合后再回答，不要逐条罗列
4. 如果资料里有答案但你不确定，直接摘抄原文对应句子。

【参考资料】
{context}

【用户问题】
{question}
"""


def build_vector_db(file_path: str, collection_name: str = "default"):
    """
    加载文档→分块→向量化→存入ChromaDB并持久化
    """
    # 加载并分块文档
    docs = load_document(file_path)
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)

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


def rag_answer(query: str, collection_name: str = "default", top_k: int = 3):
    """完整RAG问答：检索→拼接提示词→生成答案"""
    # 1. 获取检索器
    db = get_vector_db(collection_name)
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    # 2. 构建提示词
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    # 3. 构建RAG链路：检索→填充上下文→提示词→模型→解析结果
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # 4. 执行问答
    answer = rag_chain.invoke(query)
    return answer


def debug_search(query: str, collection_name: str = "test_kb", top_k: int = 10):
    """调试专用：打印Top N检索结果，查看目标内容排名"""
    db = get_vector_db(collection_name)
    docs_with_score = db.similarity_search_with_score(query, k=top_k)

    print(f"\n=== 检索调试：问题「{query}」 ===")
    for i, (doc, score) in enumerate(docs_with_score):
        print(f"\n[{i + 1}] 相似度得分：{round(score, 4)}")
        print(f"内容预览：{doc.page_content[:120]}...")


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


if __name__ == "__main__":
    # 构建测试文档的向量库
    # doc_path = "../data/test.md"
    # db = build_vector_db(doc_path, collection_name="test_kb")

    # # 测试检索
    # query = "文档里提到的核心技术是什么？"
    # similarity_search(query, collection_name="test_kb", top_k=3)

    # 测试问答
    test_query = "中文场景下，文本分块推荐的 chunk_size 范围是多少？重叠区间通常设为块大小的多少比例？"
    print("=== RAG增强回答 ===")
    result = rag_answer(test_query, collection_name="test_kb")
    print(result)

    # test_query = "文档里提到的核心技术是什么？有什么优势？"
    # res = rag_answer_with_source(test_query, collection_name="test_kb")
    # print("=== 最终回答 ===")
    # print(res["answer"])
    # print("\n=== 引用来源 ===")
    # for i, src in enumerate(res["sources"]):
    #     print(f"\n[{i + 1}] 来源信息：{src['metadata']}")
    #     print(f"原文片段：{src['content'][:150]}...")

    # 选一个只有文档里才有的细节问题测试
    # test_query = "中文场景下，文本分块推荐的 chunk_size 范围是多少？重叠区间通常设为块大小的多少比例？"
    # compare_answer(test_query, collection_name="test_kb")

    # debug_search("RAG 的全称是什么？它的核心定义是什么？")
    # debug_search("相比模型微调方案，RAG 技术有哪些核心优势？")
