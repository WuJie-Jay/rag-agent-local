from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

llm = Ollama(model="qwen2.5:7b", temperature=0.2)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="test_kb")
retriever = db.as_retriever(search_kwargs={"k": 3})


# 1. 问题改写提示词
REWRITE_PROMPT = """
给定对话历史和用户当前的问题，请将用户问题改写为一句完整、独立、语义明确的问题。
要求：
- 补全省略的主语、指代内容
- 保留原始问题的核心意图
- 只输出改写后的问题本身，不要解释，不要多余内容

【对话历史】
{history}

【当前问题】
{question}

改写后的问题：
"""

# 2. RAG回答提示词（沿用V3优化版）
RAG_PROMPT = """
你是一名严谨的知识库问答助手。
回答规则：
1. 先判断下方【参考资料】是否与用户问题直接相关
2. 若参考资料完全无法解答问题，请直接回复「参考资料中未提及相关内容」，禁止强行基于无关资料生成答案
3. 若资料相关，所有回答必须严格基于参考资料内容，禁止编造、补充资料以外的信息
4. 答案分点输出，条理清晰，聚焦核心信息

【参考资料】
{context}

【用户问题】
{question}
"""


def rewrite_question(history: str, question: str) -> str:
    """改写多轮问题为完整独立问题"""
    prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"history": history, "question": question})
    return result.strip()


def rag_answer(question: str) -> str:
    """标准RAG单轮回答"""
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def rag_answer_with_threshold(query: str, threshold: float = 0.5, top_k: int = 3):
    """
    带相似度阈值过滤的RAG问答
    低于阈值直接返回无相关内容，不调用大模型
    """
    # 带相似度得分的检索
    docs_with_score = db.similarity_search_with_score(query, k=top_k)

    # 过滤低于阈值的结果
    valid_docs = [doc for doc, score in docs_with_score if score >= threshold]

    # 无有效结果，直接兜底
    if not valid_docs:
        return "参考资料中未提及相关内容"

    # 有有效结果，再拼接提示词调用大模型
    context = "\n\n".join([doc.page_content for doc in valid_docs])
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    return answer


if __name__ == "__main__":
    print("=== 多轮RAG对话测试（输入exit退出） ===")
    history_text = ""

    while True:
        user_q = input("\n你：")
        if user_q.lower() == "exit":
            break

        # 有历史对话时，先改写问题
        if history_text:
            rewritten = rewrite_question(history_text, user_q)
            print(f"（改写后问题：{rewritten}）")
            final_q = rewritten
        else:
            final_q = user_q

        # RAG回答
        # answer = rag_answer(final_q)
        answer = rag_answer_with_threshold(final_q)
        print(f"\nAI：{answer}")

        # 更新对话历史，只保留最近3轮，避免过长
        history_text += f"用户：{user_q}\n助手：{answer}\n"
        # 简单截断：保留最近300字符
        if len(history_text) > 300:
            history_text = history_text[-300:]
