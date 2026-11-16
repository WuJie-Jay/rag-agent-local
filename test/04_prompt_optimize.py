from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = Ollama(model="qwen2.5:7b", temperature=0.3)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
CHROMA_DIR = "../chroma_db"
COLLECTION = "test_kb"

TEST_QUESTIONS = [
    "RAG的全称是什么？",
    "RAG系统完整的七个核心环节是什么？",
    "加入重排序优化后，RAG系统的召回率是多少？",
    "相比模型微调，RAG有哪些优势？",
    "GPT-5的参数量是多少？"  # 超范围题，用于验证防幻觉
]


# ========== 版本1：基础版（基线） ==========
PROMPT_V1 = """
参考以下资料回答问题。
资料：{context}
问题：{question}
"""

# ========== 版本2：加角色+防幻觉约束 ==========
PROMPT_V2 = """
你是一名严谨的知识库问答助手。
回答规则：
1. 只能使用下方【参考资料】中的信息进行回答
2. 参考资料中没有的内容，必须回答"参考资料中未提及相关内容"
3. 禁止编造、推测资料以外的任何信息

【参考资料】
{context}

【用户问题】
{question}
"""

# ========== 版本3：加输出格式+分点要求 ==========
PROMPT_V3 = """
你是一名严谨的知识库问答助手。
回答规则：
1. 所有回答必须严格基于下方【参考资料】，资料未提及的内容直接回复"参考资料中未提及相关内容"
2. 禁止编造、补充、推测任何资料外的信息
3. 答案分点输出，条理清晰，每点聚焦一个核心信息

【参考资料】
{context}

【用户问题】
{question}
"""


def run_rag_with_prompt(prompt_template, question):
    """用指定提示词执行RAG问答"""
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings, collection_name=COLLECTION)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return chain.invoke(question)


def run_experiment():
    """跑完整轮对比实验"""
    versions = [
        ("V1 基础版", PROMPT_V1),
        ("V2 加防幻觉", PROMPT_V2),
        ("V3 加格式规范", PROMPT_V3),
    ]

    for q in TEST_QUESTIONS:
        print("\n" + "=" * 70)
        print(f"问题：{q}")
        print("=" * 70)
        for name, prompt in versions:
            answer = run_rag_with_prompt(prompt, q)
            print(f"\n【{name}】")
            print(answer)
        print("\n")


if __name__ == "__main__":
    run_experiment()
