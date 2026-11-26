"""
本地化 RAG 智能知识库后端 API 服务
基于 FastAPI 封装，提供单轮问答、多轮会话、流式输出、知识库管理等能力
"""
import os
import sys
import json
from typing import Generic, TypeVar, Optional, Dict, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# 把项目根目录加入路径，保证能导入 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rag_base import rag_answer, build_vector_db, similarity_search


# ===================== 1. 应用初始化 =====================
app = FastAPI(
    title="本地化 RAG 智能知识库 API",
    description="基于 LangChain + Chroma + Ollama 的 RAG 后端服务",
    version="1.0.0"
)

# 配置跨域，前端本地调试可直接调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== 2. 统一响应格式 =====================
T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    """全局统一响应体，所有接口返回格式一致"""
    code: int = 200
    message: str = "success"
    data: Optional[T] = None

    @classmethod
    def success(cls, data: T = None, message: str = "success"):
        return cls(code=200, message=message, data=data)

    @classmethod
    def error(cls, code: int = 500, message: str = "系统错误"):
        return cls(code=code, message=message, data=None)


# ===================== 3. 请求参数模型 =====================
class ChatRequest(BaseModel):
    """单轮问答请求参数"""
    question: str
    collection_name: str = "test_kb"
    mode: str = "ensemble"       # vector / ensemble
    enable_rerank: bool = True
    top_k: int = 3


class BuildKBRequest(BaseModel):
    """构建知识库请求参数"""
    file_path: str
    collection_name: str = "test_kb"


class SearchRequest(BaseModel):
    """相似度检索请求参数"""
    query: str
    collection_name: str = "test_kb"
    top_k: int = 5


class ChatSessionRequest(BaseModel):
    """多轮会话问答请求参数"""
    session_id: str
    question: str
    collection_name: str = "test_kb"
    mode: str = "ensemble"
    enable_rerank: bool = True
    enable_rewrite: bool = True
    top_k: int = 3


# ===================== 4. 会话管理（内存版） =====================
# 全局会话存储：session_id -> 对话历史列表
session_store: Dict[str, List[dict]] = {}

def get_session_history(session_id: str) -> List[dict]:
    """获取会话历史，不存在则新建空会话"""
    if session_id not in session_store:
        session_store[session_id] = []
    return session_store[session_id]

def append_session_history(session_id: str, user_msg: str, assistant_msg: str):
    """追加一轮对话，自动截断只保留最近3轮"""
    history = get_session_history(session_id)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    # 保留最近6条消息 = 3轮对话，避免历史过长引入噪声
    session_store[session_id] = history[-6:]


# ===================== 5. 核心接口实现 =====================

@app.get("/health", summary="服务健康检查")
def health_check():
    return ApiResponse.success(data={"status": "online"})


@app.post("/api/kb/build", summary="构建知识库")
def build_knowledge_base(req: BuildKBRequest):
    try:
        build_vector_db(req.file_path, collection_name=req.collection_name)
        return ApiResponse.success(message="知识库构建成功")
    except Exception as e:
        return ApiResponse.error(message=f"构建失败：{str(e)}")


@app.post("/api/kb/search", summary="知识库相似度检索")
def knowledge_search(req: SearchRequest):
    try:
        docs = similarity_search(req.query, req.collection_name, top_k=req.top_k)
        data = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]
        return ApiResponse.success(data=data)
    except Exception as e:
        return ApiResponse.error(message=f"检索失败：{str(e)}")


@app.post("/api/chat/qa", summary="单轮RAG问答（非流式）")
def rag_single_chat(req: ChatRequest):
    try:
        answer = rag_answer(
            query=req.question,
            collection_name=req.collection_name,
            top_k=req.top_k,
            mode=req.mode,
            enable_rerank=req.enable_rerank
        )
        return ApiResponse.success(data={"answer": answer})
    except Exception as e:
        return ApiResponse.error(message=f"问答失败：{str(e)}")


@app.post("/api/chat/stream", summary="流式RAG问答（SSE协议）")
def rag_stream_chat(req: ChatRequest):
    """
    流式输出接口，逐字返回答案，对齐主流大模型交互体验
    前端使用 EventSource 接收数据
    """
    def generate():
        try:
            # 先拿到完整答案，模拟逐字推送
            full_answer = rag_answer(
                query=req.question,
                collection_name=req.collection_name,
                top_k=req.top_k,
                mode=req.mode,
                enable_rerank=req.enable_rerank
            )
            # 按字符逐段推送，模拟打字效果
            for char in full_answer:
                yield f"data: {json.dumps({'content': char}, ensure_ascii=False)}\n\n"
            # 结束标记
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/chat/session", summary="多轮会话问答（带查询改写）")
def rag_session_chat(req: ChatSessionRequest):
    try:
        history = get_session_history(req.session_id)

        answer = rag_answer(
            query=req.question,
            collection_name=req.collection_name,
            top_k=req.top_k,
            mode=req.mode,
            enable_rerank=req.enable_rerank,
            enable_rewrite=req.enable_rewrite,
            history=history
        )

        # 保存本轮对话到会话
        append_session_history(req.session_id, req.question, answer)

        return ApiResponse.success(data={
            "answer": answer,
            "session_id": req.session_id,
            "round": len(history) // 2 + 1
        })
    except Exception as e:
        return ApiResponse.error(message=f"对话失败：{str(e)}")


# ===================== 6. 全局异常处理器 =====================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """统一拦截所有未捕获的异常，返回标准错误格式"""
    return JSONResponse(
        status_code=200,
        content={
            "code": 500,
            "message": f"系统内部错误：{str(exc)}",
            "data": None
        }
    )


# ===================== 本地启动入口 =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
