"""
=== 本地化RAG智能知识库Agent ===


#### 1. 基础事实题（测试基础检索准确率）

**问题**：RAG 的全称是什么？它的核心定义是什么？
**测试目标**：验证基础概念的检索命中能力，确认文档核心内容可正常召回。

#### 2. 细节数字题（测试精准检索与防幻觉）

**问题**：中文场景下，文本分块推荐的 chunk_size 范围是多少？重叠区间通常设为块大小的多少比例？
**测试目标**：验证细粒度参数信息的检索精度，同时检验模型是否会编造数值。

#### 3. 归纳总结题（测试跨块信息整合）

**问题**：相比模型微调方案，RAG 技术有哪些核心优势？
**测试目标**：验证多文本块的信息整合能力，检验答案的完整性与准确性。


验证查询改写是否生效：
1. 第一轮提问：`RAG相比模型微调有哪些核心优势？`
2. 第二轮提问：`它的落地成本大概是多少？`
3. 第三轮提问：`那分块参数一般怎么设置？`
"""

import os
import sys

# ========== 导入各模块工具（对应你的项目目录结构） ==========
from utils.document_loader import load_document
from utils.text_splitter import split_documents
from utils.rag_base import build_vector_db, rag_answer
from core.react_agent import run_react_agent

# ========== 全局默认配置 ==========
DEFAULT_COLLECTION = "test_kb"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


def print_title():
    """打印项目标题"""
    print("=" * 60)
    print("        本地化 RAG 智能知识库 Agent 系统")
    print("=" * 60)
    print()


def build_knowledge_base():
    """功能1：构建知识库（上传文档→分块→向量化入库）"""
    print("\n--- 构建知识库模式 ---")
    print("支持格式：.pdf / .txt / .md")

    file_path = input("请输入文档路径（相对/绝对路径均可）：").strip()
    if not os.path.exists(file_path):
        print(f"❌ 错误：文件不存在 → {file_path}")
        return

    collection_name = input(f"请输入知识库集合名（默认 {DEFAULT_COLLECTION}）：").strip()
    if not collection_name:
        collection_name = DEFAULT_COLLECTION

    try:
        print("\n正在加载文档...")
        docs = load_document(file_path)
        print(f"✅ 文档加载完成，共 {len(docs)} 段/页")

        print("正在文本分块...")
        chunks = split_documents(docs, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
        print(f"✅ 分块完成，共 {len(chunks)} 个文本块")

        print("正在向量化并存入向量库...")
        build_vector_db(file_path, collection_name=collection_name)
        print(f"✅ 知识库构建成功！集合名：{collection_name}")
    except Exception as e:
        print(f"❌ 构建失败：{str(e)}")


def rag_chat_mode():
    """功能2：基础 RAG 单轮问答模式"""
    print("\n--- 基础 RAG 问答模式 ---")
    print("输入问题即可获取答案，输入 exit 返回主菜单")

    collection_name = input(f"请选择知识库集合名（默认 {DEFAULT_COLLECTION}）：").strip()
    if not collection_name:
        collection_name = DEFAULT_COLLECTION
    print(f"✅ 已连接知识库：{collection_name}\n")

    while True:
        question = input("你：").strip()
        if question.lower() == "exit":
            print("退出问答模式，返回主菜单\n")
            break
        if not question:
            continue

        try:
            answer = rag_answer(question, collection_name=collection_name, mode="ensemble")
            print(f"AI：{answer}\n")
        except Exception as e:
            print(f"❌ 回答出错：{str(e)}\n")


def agent_chat_mode():
    """功能3：Agent 智能对话模式（支持多工具调用）"""
    print("\n--- Agent 智能对话模式 ---")
    print("支持自主调用知识库、计算器等工具，处理复杂多步任务")
    print("输入问题即可对话，输入 q 返回主菜单\n")

    # 维护多轮对话历史
    chat_history = []

    while True:
        question = input("你：").strip()
        if question.lower() == "q":
            print("退出Agent模式，返回主菜单\n")
            break
        if not question:
            continue

        try:
            # 传入对话历史，开启查询改写
            answer = run_react_agent(question, max_steps=5, chat_history=chat_history)
            print(f"\nAI：{answer}\n")

            # 追加本轮对话到历史
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": answer})
        except Exception as e:
            print(f"❌ 执行出错：{str(e)}\n")


def main_menu():
    """主菜单循环"""
    print_title()

    while True:
        print("-" * 60)
        print("功能菜单：")
        print("  1. 构建知识库（上传文档并向量化）")
        print("  2. 基础 RAG 问答")
        print("  3. Agent 智能对话（支持工具调用）")
        print("  4. 退出系统")
        print("-" * 60)

        choice = input("请输入功能编号：").strip()

        if choice == "1":
            build_knowledge_base()
        elif choice == "2":
            rag_chat_mode()
        elif choice == "3":
            agent_chat_mode()
        elif choice == "4":
            print("\n👋 感谢使用，再见！")
            sys.exit(0)
        else:
            print("❌ 无效输入，请输入 1-4 之间的数字\n")


if __name__ == "__main__":
    main_menu()
