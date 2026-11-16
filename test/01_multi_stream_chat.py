import time
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 初始化模型
llm = Ollama(model="qwen2.5:7b", temperature=0.3)

# 对话历史
chat_history = [
    SystemMessage(content="你是一个耐心的技术老师，用通俗的语言解答问题。")
]


def typewriter_print(text, delay=0.05):
    """
    逐字打印函数：模拟打字机效果
    :param text: 要打印的文本
    :param delay: 每个字符的间隔时间，单位秒，数值越小速度越快
    """
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)


def chat(user_input):
    # 1. 用户提问加入对话历史
    chat_history.append(HumanMessage(content=user_input))

    # 2. 流式接收模型输出，同时逐字打印
    full_answer = ""
    print("AI：", end="", flush=True)

    for chunk in llm.stream(chat_history):
        # 把当前块拆成单个字符，逐字打印
        for char in chunk:
            print(char, end="", flush=True)
            time.sleep(0.05)  # 每个字的间隔，可调节速度
        # 同步拼接完整回答，用于对话历史
        full_answer += chunk

    print("\n")

    # 3. 完整回答加入对话历史，保证多轮记忆正确
    chat_history.append(AIMessage(content=full_answer))

    return full_answer


if __name__ == "__main__":
    print("=== 多轮对话·逐字打字机模式（输入q退出） ===")
    while True:
        user_input = input("你：")
        if user_input.lower() == "q":
            break
        chat(user_input)
