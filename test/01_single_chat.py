from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 初始化本地大模型
llm = Ollama(
    model="qwen2.5:7b",
    temperature=0.3,
    base_url="http://localhost:11434"  # Ollama默认服务地址
)

# 2. 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的技术助手，回答精准简洁，分点输出。"),
    ("user", "{input}")
])

# 3. 构建输出解析器
output_parser = StrOutputParser()

# 4. 串联成链路：提示词 -> 模型 -> 解析结果
chain = prompt | llm | output_parser

# 5. 调用并获取结果
if __name__ == "__main__":
    question = "请用3句话解释什么是RAG技术"
    result = chain.invoke({"input": question})
    print("回答：")
    print(result)

