import json
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

llm = Ollama(model="qwen2.5:7b", temperature=0.1)


# ========== 1. 定义真实的工具函数 ==========
def calculator(expression: str) -> str:
    """计算器工具：输入数学表达式字符串，返回计算结果"""
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算失败：{str(e)}"


# ========== 2. 定义工具描述（给模型看的） ==========
TOOLS_DESCRIPTION = """
你可以调用以下工具来完成任务：
工具名称：calculator
工具描述：当用户需要进行数学计算、数值运算时，调用此工具。输入为合法的数学表达式字符串。
参数：expression（字符串类型，待计算的数学表达式）

调用工具时，请严格按照以下JSON格式输出，不要有任何多余内容：
{"name": "工具名称", "parameters": {"参数名": "参数值"}}
"""

SYSTEM_PROMPT = f"""
你是一个智能助手，可以回答问题，也可以调用工具完成任务。
{TOOLS_DESCRIPTION}

规则：
1. 如果需要调用工具，只输出JSON格式的调用指令，不要有其他文字
2. 如果不需要调用工具，直接用自然语言回答用户问题
3. 一次只调用一个工具
"""


def run_agent(user_input: str):
    # 第一步：把问题传给模型，让它决策是否调用工具
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    response = llm.invoke(messages)
    response = response.strip()
    print(f"模型原始输出：{response}\n")

    # 第二步：判断是不是工具调用
    try:
        tool_call = json.loads(response)
        if "name" in tool_call and "parameters" in tool_call:
            # 是工具调用，执行函数
            tool_name = tool_call["name"]
            params = tool_call["parameters"]
            print(f"识别到工具调用：工具={tool_name}, 参数={params}\n")

            # 第三步：执行对应工具
            if tool_name == "calculator":
                tool_result = calculator(params["expression"])
            else:
                tool_result = f"未知工具：{tool_name}"

            print(f"工具执行结果：{tool_result}\n")

            # 第四步：把结果回传给模型，生成最终答案
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"工具执行结果：{tool_result}"})

            final_answer = llm.invoke(messages)
            print(f"最终回答：{final_answer}")
            return final_answer
    except json.JSONDecodeError:
        # 不是JSON，说明是直接回答
        print(f"最终回答：{response}")
        return response


if __name__ == "__main__":
    # 测试1：需要调用工具的问题
    print("=== 测试1：计算类问题 ===")
    run_agent("12345 + 67890 等于多少？")

    print("\n" + "=" * 50 + "\n")

    # 测试2：不需要调用工具的问题
    print("=== 测试2：常识类问题 ===")
    run_agent("水的化学式是什么？")
