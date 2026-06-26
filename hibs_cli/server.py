"""
FastAPI 服务: Hibs-LMT hibs 0.16 (基于 hibs 0.16) HTTP API
==========================================

端点:
  POST /generate   - 文本生成
  POST /chat       - 多轮对话
  GET  /info       - 模型信息
  GET  /health     - 健康检查

启动:
    hibs serve --ckpt model.pt --port 8000
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="输入 prompt")
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=0, le=200)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    prompt: str
    generated: str
    tokens_generated: int


class ChatMessage(BaseModel):
    role: str = Field(..., description="user / assistant / system")
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)


class ChatResponse(BaseModel):
    reply: str
    history: List[ChatMessage]


def create_app(ckpt_path: str, device: str = None):
    """创建 FastAPI app"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    # 延迟导入, 避免无 torch 时报错
    from hibs_cli.inference import InferenceEngine

    app = FastAPI(
        title="Hibs-LMT hibs 0.16 (基于 hibs 0.16) API",
        description="基于复数 SSM + 扭量纤维丛的语言模型",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 加载模型 (启动时一次性加载)
    engine_state = {"engine": None, "history": []}

    @app.on_event("startup")
    async def load_engine():
        print(f"加载模型: {ckpt_path}")
        engine_state["engine"] = InferenceEngine(ckpt_path, device=device)

    @app.get("/health")
    async def health():
        return {"status": "ok", "engine_loaded": engine_state["engine"] is not None}

    @app.get("/info")
    async def info():
        if engine_state["engine"] is None:
            raise HTTPException(503, "模型未加载")
        return engine_state["engine"].info()

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        if engine_state["engine"] is None:
            raise HTTPException(503, "模型未加载")

        result = engine_state["engine"].generate(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        # 提取生成的部分 (去除 prompt)
        if result.startswith(req.prompt):
            generated = result[len(req.prompt):]
        else:
            generated = result

        return GenerateResponse(
            prompt=req.prompt,
            generated=generated,
            tokens_generated=len(generated),  # 字符数, 不精确
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        """多轮对话 (简单实现: 拼接 history + 当前输入)"""
        if engine_state["engine"] is None:
            raise HTTPException(503, "模型未加载")

        # 拼接对话历史
        prompt_parts = []
        for msg in req.messages:
            if msg.role == "system":
                prompt_parts.append(f"[系统] {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"[用户] {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"[助手] {msg.content}")
        prompt_parts.append("[助手] ")
        prompt = "\n".join(prompt_parts)

        result = engine_state["engine"].generate(
            prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )

        # 提取生成部分
        if result.startswith(prompt):
            reply = result[len(prompt):].split("\n")[0]
        else:
            reply = result.split("\n")[0]

        # 更新历史
        new_history = list(req.messages) + [ChatMessage(role="assistant", content=reply)]

        return ChatResponse(reply=reply, history=new_history)

    return app