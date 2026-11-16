from langchain_core.tools import tool
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage
import json


# 用 @tool 装饰器定义工具，自动生成描述和参数Schema
@tool
def calculator(expression: str) -> str:
    """
    数学计算器工具。
    当用户需要进行数学计算、算术运算、数值求解时，调用此工具。
    参数 expression 为合法的数学表达式字符串，例如 "12 + 34 * 5"。
    """
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算出错：{str(e)}"


@tool
def get_current_time() -> str:
    """获取当前系统时间，当用户问现在几点、当前时间时调用此工具。"""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"当前时间：{now}"


# 工具列表
tools = [calculator, get_current_time]

# 构建工具名称->工具对象的映射
tool_map = {tool.name: tool for tool in tools}


# ========== 2. 自动生成工具描述文本 ==========
def build_tools_prompt(tools: list) -> str:
    """把工具列表转换成模型能看懂的文字描述"""
    desc_list = []
    for t in tools:
        desc_list.append(f"""
工具名称：{t.name}
工具描述：{t.description}
参数说明：{t.args_schema.schema()['properties']}
""")
    return "\n".join(desc_list)


# ========== 3. 系统提示词（含工具说明 + 格式约束） ==========
SYSTEM_PROMPT = f"""
你是一个智能助手，可以回答问题，也可以调用工具完成任务。

你可以使用以下工具：
{build_tools_prompt(tools)}

调用规则：
1. 如果需要调用工具，请严格按照以下JSON格式输出，不要有任何多余文字、解释、markdown标记：
{{"name": "工具名称", "parameters": {{"参数名": "参数值"}}}}
2. 如果不需要调用工具，直接用自然语言回答用户问题
3. 一次只能调用一个工具
"""

# ========== 4. 初始化模型 ==========
llm = Ollama(model="qwen2.5:7b", temperature=0.1)


# ========== 5. 完整 Agent 执行逻辑 ==========
def run_agent(user_input: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input)
    ]

    # 第一步：模型决策是否调用工具
    response = llm.invoke(messages).strip()
    # print(f"模型原始输出：\n{response}\n")

    # 第二步：尝试解析为工具调用
    try:
        tool_call = json.loads(response)
        if "name" in tool_call and "parameters" in tool_call:
            tool_name = tool_call["name"]
            tool_params = tool_call["parameters"]
            print(f"识别到工具调用：工具={tool_name}, 参数={tool_params}")

            # 第三步：执行对应工具
            if tool_name not in tool_map:
                tool_result = f"错误：不存在工具 {tool_name}"
            else:
                tool_result = tool_map[tool_name].invoke(tool_params)

            print(f"工具执行结果：{tool_result}\n")

            # 第四步：把工具结果回传给模型，生成最终答案
            messages.append(SystemMessage(content=response))
            messages.append(HumanMessage(content=f"工具执行结果：{tool_result}"))

            final_answer = llm.invoke(messages)
            return final_answer

    except json.JSONDecodeError:
        # 不是JSON，说明是直接回答
        return response


if __name__ == "__main__":
    print("=== 测试1：计算类问题 ===")
    print(run_agent("234 乘以 567 等于多少？"))

    print("\n" + "="*50 + "\n")

    print("=== 测试2：时间类问题 ===")
    print(run_agent("现在几点了？"))

    print("\n" + "="*50 + "\n")

    print("=== 测试3：常识类问题 ===")
    print(run_agent("介绍一下Python语言"))
