import os
# 关闭所有遥测与警告
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TELEMETRY_DISABLED"] = "1"

import json
import re
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 导入你的RAG函数
from utils.rag_base import rag_answer


# ===================== 1. 工具定义 =====================
@tool
def calculator(expression: str) -> str:
    """
    数学计算器工具。
    当用户需要进行数学计算、算术运算、数值求解时调用此工具。
    参数 expression 为纯数学表达式字符串。
    """
    try:
        expression = expression.strip().strip('"').strip("'").strip("`")
        result = eval(expression)
        return f"计算成功，结果：{result}"
    except Exception as e:
        return f"计算失败：{str(e)}"


@tool
def knowledge_base_qa(question: str) -> str:
    """
    RAG技术专属知识库问答工具。
    仅当用户询问 RAG 技术原理、系统架构、优化方法等专业问题时调用。
    常识问题、通用知识禁止调用。
    """
    try:
        answer = rag_answer(question, collection_name="test_kb")
        if not answer or "未提及" in answer:
            return "【知识库无结果】参考资料中未提及相关内容"
        return f"【知识库查询成功】{answer}"
    except Exception as e:
        return f"【知识库查询失败】{str(e)}"


tools = [calculator, knowledge_base_qa]
tool_map = {t.name: t for t in tools}


# ===================== 2. 系统提示词（强化格式约束） =====================
SYSTEM_PROMPT = """
你是一个严格遵守规则的智能助手，通过调用工具解决问题。

执行规则：
1. 严格按照格式输出：先写Thought，再写Action和Action Input，一次只输出一步
2. 绝对禁止自己生成 Observation 内容，Observation 只能由系统返回
3. Action Input 必须是合法 JSON 对象，键名必须和工具参数完全一致
4. 工具返回【知识库无结果】时，直接告知用户查询失败，禁止用自身知识补充答案
5. 常识问题可直接回答，RAG专业问题必须调用知识库工具
6. 信息足够后输出 Final Answer 结束

可用工具：
{tool_descs}

=== 格式示例 ===
Thought: 分析问题，说明下一步动作
Action: calculator
Action Input: {{"expression": "12 * 34"}}
Observation: 计算成功，结果：408
Thought: 信息足够，可以给出答案
Final Answer: 12 乘以 34 的结果是 408
=== 示例结束 ===
"""

# 生成工具描述文本
tool_descs = "\n".join([f"- {t.name}：{t.description}" for t in tools])
system_prompt_text = SYSTEM_PROMPT.format(tool_descs=tool_descs)


# ===================== 3. 初始化模型（新版官方Ollama集成） =====================
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.0,
    base_url="http://localhost:11434"
)


# ===================== 4. 解析函数（强制单步，防幻觉） =====================
def parse_model_output(text: str):
    # 截断Observation及之后的内容，防止模型自行编造
    obs_pos = text.find("Observation:")
    if obs_pos != -1:
        text = text[:obs_pos].strip()

    # 优先匹配最终答案
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        return {"type": "final", "content": final_match.group(1).strip()}

    # 匹配工具调用
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(\{[\s\S]*?\})", text)
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

    return {"type": "error", "msg": "未识别到有效输出格式"}


# ===================== 5. ReAct 主循环 =====================
def run_react_agent(user_input: str, max_steps: int = 5):
    messages = [
        SystemMessage(content=system_prompt_text),
        HumanMessage(content=user_input)
    ]
    step = 0

    while step < max_steps:
        step += 1
        print(f"\n--- 第 {step} 步思考 ---")

        response = llm.invoke(messages)
        response_text = response.content.strip()
        print(response_text)
        messages.append(AIMessage(content=response_text))

        result = parse_model_output(response_text)

        if result["type"] == "final":
            print("\n=== 最终答案 ===")
            return result["content"]

        elif result["type"] == "action":
            tool_name = result["tool_name"]
            params = result["params"]

            if tool_name not in tool_map:
                obs = f"错误：不存在工具 {tool_name}"
            else:
                obs = tool_map[tool_name].invoke(params)

            print(f"Observation: {obs}")
            messages.append(HumanMessage(content=f"Observation: {obs}"))

        else:
            obs = f"格式错误：{result['msg']}，请严格按照指定格式输出"
            print(f"Observation: {obs}")
            messages.append(HumanMessage(content=f"Observation: {obs}"))

    return "任务执行超时，已达到最大步数限制"


# ===================== 测试运行 =====================
if __name__ == "__main__":
    def test_case(name, question):
        print(f"\n{'='*60}")
        print(f"【{name}】{question}")
        print('='*60)
        ans = run_react_agent(question)
        print(f"\n✅ 最终结果：{ans}")

    test_case("场景1 计算类", "123 加 456 再乘以 2 等于多少？")
    test_case("场景2 知识库类", "RAG系统有哪几个核心环节？")
    test_case("场景3 混合任务", "RAG相比微调有几个优势？把这个数量乘以10是多少？")
