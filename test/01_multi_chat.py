from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

llm = Ollama(model="qwen2.5:7b", temperature=0.3)
output_parser = StrOutputParser()

# 对话历史列表：按顺序存放所有消息
chat_history = [
    SystemMessage(content="你是一个耐心的技术老师，用通俗的语言解答问题。")
]

def chat(user_input):
    # 把用户新问题加入历史
    chat_history.append(HumanMessage(content=user_input))
    # 把完整历史传给模型
    response = llm.invoke(chat_history)
    # 把模型回答加入历史
    chat_history.append(AIMessage(content=response))
    return response

if __name__ == "__main__":
    print("=== 多轮对话测试（输入exit退出） ===")
    while True:
        user_input = input("你：")
        if user_input.lower() == "exit":
            break
        answer = chat(user_input)
        print(f"AI：{answer}\n")
