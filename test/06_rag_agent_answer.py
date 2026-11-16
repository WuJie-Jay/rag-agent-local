from langchain_core.tools import tool
import json
import re
from langchain_community.llms import Ollama
from utils.rag_base import rag_answer


# ========== 1. 定义基础工具 ==========
@tool
def calculator(expression: str) -> str:
    """数学计算器工具。当需要进行数学计算、算术运算时调用，参数为合法的数学表达式字符串。"""
    try:
        result = eval(expression)
        return f"计算成功，结果：{result}"
    except Exception as e:
        return f"计算失败：{str(e)}"


@tool
def get_current_time() -> str:
    """获取当前系统时间，询问现在几点、当前时间时调用。"""
    from datetime import datetime
    return f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def knowledge_base_qa(question: str) -> str:
    """
    RAG技术专属知识库问答工具。
    仅当用户询问 RAG 技术原理、系统架构、优化方法、技术栈选型等 RAG 专业问题时调用。
    常识问题、通用知识、无关领域问题禁止调用此工具。
    参数 question 为用户的问题字符串。
    """
    print(f"\n[工具调试] 收到问题：{question}")
    try:
        answer = rag_answer(question, collection_name="test_kb")
        if not answer or "未提及相关内容" in answer:
            return "【知识库无结果】参考资料中未提及相关内容"
        return f"【知识库查询成功】{answer}"
    except Exception as e:
        return f"【知识库查询失败】{str(e)}"


# 把新工具加入工具列表
tools = [calculator, get_current_time, knowledge_base_qa]
tool_map = {t.name: t for t in tools}

# ========== 2. ReAct 系统提示词 ==========
REACT_SYSTEM_PROMPT = """
你是一个会思考、会调用工具的智能助手。解决问题时，请严格按照以下格式进行思考和行动：

Thought: 你对当前问题的分析，下一步打算做什么
Action: 要调用的工具名称，只能从 [{tool_names}] 中选择
Action Input: 调用工具的参数，严格按照JSON格式输出，必须包含所有必填参数
Observation: 工具执行后的结果（这一步由系统返回，你不用写）

...（可以重复多轮 Thought/Action/Action Input/Observation）

当你认为信息足够、可以给出最终答案时，请输出：
Thought: 我已经收集到足够信息，可以给出最终答案
Final Answer: 你的最终答案

=== 调用示例 ===
问题：计算 12 乘以 34 等于多少
Thought: 这是一个数学计算问题，需要调用计算器工具
Action: calculator
Action Input: {{"expression": "12 * 34"}}
Observation: 计算成功，结果：408
Thought: 已经得到计算结果，可以给出最终答案
Final Answer: 12 乘以 34 的结果是 408
=== 示例结束 ===

规则：
1. 必须严格按照上面的格式输出，不要多余内容
2. 常识类问题直接回答，不要调用知识库工具。
3. 一次只能调用一个工具，必须传入正确的参数名和参数值
4. 不知道的信息就调用工具，所有答案必须严格基于工具返回的结果，禁止编造工具中没有的内容、数据、来源。
5. 工具调用失败就根据错误信息调整，或者告知用户
6. 如果工具返回包含【知识库无结果】【参考资料中未提及相关内容】，说明知识库没有答案，必须直接回复"知识库中未查询到相关内容"，绝对禁止使用自身知识补充、推测、扩展任何信息。
7. 所有答案必须 100% 基于工具返回的内容，禁止超出工具范围自行发挥、添加额外知识。
8. RAG 技术相关的专业问题，必须调用 knowledge_base_qa 工具查询，禁止直接回答。
9. 绝对禁止你自行生成 Observation 内容，Observation 只能由系统返回给你。
10. 一次只能输出一步：要么输出 Thought+Action+Action Input，要么输出 Thought+Final Answer，禁止一次性输出多步。


可用工具列表：
{tool_descriptions}
"""


def build_react_prompt(tools):
    """生成带工具信息的系统提示词"""
    tool_names = ", ".join([t.name for t in tools])
    tool_descs = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    return REACT_SYSTEM_PROMPT.format(tool_names=tool_names, tool_descriptions=tool_descs)


# ========== 3. 初始化模型 ==========
llm = Ollama(model="qwen2.5:7b", temperature=0.1)
system_prompt = build_react_prompt(tools)


# ========== 4. 解析模型输出，提取Action ==========
def parse_action(response: str):
    # ========== 新增：强制截断，Observation及之后的幻觉内容全部丢弃 ==========
    obs_index = response.find("Observation:")
    if obs_index != -1:
        response = response[:obs_index].strip()

    # 优先判断有没有Action：有Action就必须先执行工具，不提前返回最终答案
    action_match = re.search(r"Action:\s*(\w+)", response)
    input_match = re.search(r"Action Input:\s*(\{[\s\S]*?\})", response)

    if action_match and input_match:
        try:
            params = json.loads(input_match.group(1).strip())
            return {
                "type": "action",
                "tool_name": action_match.group(1).strip(),
                "params": params
            }
        except json.JSONDecodeError:
            return {"type": "error", "msg": "参数JSON解析失败"}

    # 没有Action时，再判断是不是最终答案
    final_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
    if final_match:
        return {"type": "final", "content": final_match.group(1).strip()}

    return {"type": "error", "msg": "无法解析模型输出"}


# ========== 5. ReAct 主循环 ==========
def run_react_agent(user_input: str, max_steps: int = 5):
    """
    ReAct Agent 主函数
    max_steps: 最大循环步数，防止死循环
    """
    # 初始化对话历史
    history = f"System: {system_prompt}\n\nUser: {user_input}\n"
    step = 0

    while step < max_steps:
        step += 1
        print(f"\n--- 第 {step} 步思考 ---")

        # 模型思考
        response = llm.invoke(history).strip()
        print(response)
        history += f"{response}\n"

        # 解析输出
        result = parse_action(response)

        if result["type"] == "final":
            print("\n=== 最终答案 ===")
            return result["content"]

        elif result["type"] == "action":
            tool_name = result["tool_name"]
            params = result["params"]

            if tool_name not in tool_map:
                obs = f"错误：不存在工具 {tool_name}"
            else:
                # 执行工具
                obs = tool_map[tool_name].invoke(params)

            print(f"Observation: {obs}")
            history += f"Observation: {obs}\n"

        else:
            # 解析失败，让模型重新输出
            obs = f"格式错误，请严格按照指定格式输出。错误信息：{result['msg']}"
            print(f"Observation: {obs}")
            history += f"Observation: {obs}\n"

    return "任务执行超时，已达到最大步数限制"


# ========== 测试 ==========
if __name__ == "__main__":
    print("=== 场景1：常识问题（直接回答，不调用工具） ===")
    print(run_react_agent("水的化学式是什么？"))

    print("\n=== 场景2：知识库问题（调用RAG工具） ===")
    print(run_react_agent("RAG 的全称是什么？它的核心定义是什么？"))

    print("\n=== 场景3：混合任务（先查知识库再计算） ===")
    print(run_react_agent("RAG相比微调有几个优势？把这个数量乘以10是多少？"))


