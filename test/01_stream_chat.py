from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = Ollama(model="qwen2.5:7b", temperature=0.3)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的技术文档助手，回答详细有条理。"),
    ("user", "{input}")
])

# 实例化输出解析器
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if __name__ == "__main__":
    question = "详细介绍大模型幻觉的成因和解决办法"
    print("回答：", end="", flush=True)

    # 流式调用，逐块打印
    for chunk in chain.stream({"input": question}):
        print(chunk, end="", flush=True)

    print("\n\n回答完成")
