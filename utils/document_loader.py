# from langchain_community.document_loaders import PyPDFLoader
# 把原来的 PyPDFLoader 换成 PyMuPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import os
from langchain_community.document_loaders import TextLoader


def load_text(file_path: str) -> list[Document]:
    """加载TXT/Markdown文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"成功加载文本文件，共 {len(documents)} 段")
    return documents


# def load_pdf(file_path: str) -> list[Document]:
#     """
#     加载单个PDF文件，返回文档列表
#     每个Document包含page_content（文本内容）和metadata（元信息，如页码）
#     """
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"文件不存在: {file_path}")
#
#     loader = PyPDFLoader(file_path)
#     documents = loader.load()
#     print(f"成功加载PDF，共 {len(documents)} 页")
#     return documents


def load_pdf(file_path: str):
    """加载PDF文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents


def load_document(file_path: str) -> list[Document]:
    """
    通用文档加载函数，根据文件后缀自动选择加载方式
    支持：.pdf, .txt, .md
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext in [".txt", ".md"]:
        return load_text(file_path)
    else:
        raise ValueError(f"暂不支持的文件格式: {ext}")


if __name__ == "__main__":
    # 测试：替换成你自己的PDF文件路径
    pdf_path = "../data/test.pdf"
    docs_pdf = load_document(pdf_path)

    md_path = "../data/test.md"
    docs_md = load_document(md_path)

    # 打印前2页的内容预览
    for i, doc in enumerate(docs_pdf[:2]):
        print(f"\n=== 第{i + 1}页 ===")
        print(f"内容预览：{doc.page_content[:200]}...")
        print(f"元信息：{doc.metadata}")

    for i, doc in enumerate(docs_md[:1]):
        print(f"内容预览：{doc.page_content[:200]}...")
        print(f"元信息：{doc.metadata}")


