from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.document_loader import load_document


# def split_documents(
#         documents: list[Document],
#         chunk_size: int = 500,
#         chunk_overlap: int = 50
# ) -> list[Document]:
#     """
#     对文档列表进行递归字符分块
#     """
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", "。", "！", "？", " ", ""],  # 中文优化分隔符
#         length_function=len  # 按字符数计算长度
#     )
#
#     chunks = splitter.split_documents(documents)
#     print(f"分块完成：原始 {len(documents)} 段 → 分块后 {len(chunks)} 块")
#     return chunks


def split_documents(docs, chunk_size=800, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", "。", "，", " ", ""],
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(docs)


def run_split_experiment(documents: list[Document]):
    """分块参数对比实验"""
    # 实验组配置
    experiments = [
        {"name": "小组块", "chunk_size": 200, "chunk_overlap": 20},
        {"name": "中组块", "chunk_size": 500, "chunk_overlap": 50},
        {"name": "大组块", "chunk_size": 1000, "chunk_overlap": 100},
    ]

    results = []
    for exp in experiments:
        chunks = split_documents(documents, exp["chunk_size"], exp["chunk_overlap"])
        avg_len = sum(len(c.page_content) for c in chunks) / len(chunks)

        result = {
            "实验组": exp["name"],
            "chunk_size": exp["chunk_size"],
            "块数量": len(chunks),
            "平均块长度": round(avg_len, 1),
            "语义完整性": "人工评估：差/一般/好"  # 自己看完内容后填写
        }
        results.append(result)

    # 打印对比表格
    print("\n" + "=" * 60)
    print("分块参数对比实验结果")
    print("=" * 60)
    for res in results:
        print(f"{res['实验组']}: 块数={res['块数量']}, 平均长度={res['平均块长度']}")

    return results


if __name__ == "__main__":
    # 加载测试文档
    # docs = load_document("../data/test.pdf")
    docs = load_document("../data/test.txt")

    # run_split_experiment(docs)

    # 基础分块测试
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    # 打印前3块的详细信息
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n=== 第{i + 1}块（长度{len(chunk.page_content)}字符） ===")
        print(chunk.page_content)
