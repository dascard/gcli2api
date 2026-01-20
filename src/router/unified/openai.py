"""
Unified OpenAI Router - 聚合网关
处理 /v1 路径下的请求，聚合 Gemini, Codex, Antigravity 并进行请求分发
"""

from fastapi import APIRouter, Depends, HTTPException

from log import log
from src.utils import (
    authenticate_flexible,
    get_available_models,
    CODEX_MODELS_LIST
)
from src.router.base_router import create_openai_model_list
from src.models import model_to_dict, OpenAIChatCompletionRequest

# 路由器初始化
router = APIRouter()

# ==================== 聚合模型列表 ====================

@router.get("/v1/models")
async def list_all_models(token: str = Depends(authenticate_flexible)):
    """
    [聚合版] 返回所有模型列表 (Gemini + Codex + Antigravity)
    """
    # 1. 获取 Gemini 模型
    models = get_available_models("gemini")
    
    # 2. Codex 模型列表 (已导入)
    
    # 3. 获取 Antigravity 模型 (异步获取)
    try:
        from src.router.antigravity.model_list import get_antigravity_models_with_features
        antigravity_models = await get_antigravity_models_with_features()
    except Exception as e:
        log.warning(f"无法获取 Antigravity 模型: {e}")
        antigravity_models = []

    log.info(f"[Unified Gateway] 聚合模型: Gemini({len(models)}) + Codex({len(CODEX_MODELS_LIST)}) + Antigravity({len(antigravity_models)})")
    
    # 创建基础列表 (Gemini)
    model_list = create_openai_model_list(models, owned_by="google")
    data = [model_to_dict(model) for model in model_list.data]
    
    # 已有 ID 集合，用于去重
    existing_ids = {m["id"] for m in data}
    
    # 合并 Codex 模型 (包含变体)
    codex_names = get_available_models("codex")
    codex_list_obj = create_openai_model_list(codex_names, owned_by="openai")
    
    for codex_model in codex_list_obj.data:
        m_dict = model_to_dict(codex_model)
        if m_dict["id"] not in existing_ids:
            data.append(m_dict)
            existing_ids.add(m_dict["id"])
            
    # 合并 Antigravity 模型
    if antigravity_models:
        antigravity_list_obj = create_openai_model_list(antigravity_models, owned_by="antigravity")
        for ag_model in antigravity_list_obj.data:
            ag_dict = model_to_dict(ag_model)
            if ag_dict["id"] not in existing_ids:
                data.append(ag_dict)
                existing_ids.add(ag_dict["id"])

    return {"object": "list", "data": data}


# ==================== 请求路由分发 ====================

@router.post("/v1/chat/completions")
async def chat_completions(
    request: OpenAIChatCompletionRequest,
    token: str = Depends(authenticate_flexible)
):
    """
    智能路由分发：根据模型名称将请求转发到对应的后端
    """
    from src.utils import get_base_model_from_feature_model
    
    # 获取基础模型名（剥离功能前缀）
    model_id = get_base_model_from_feature_model(request.model)
    
    # 1. 检查是否为 Codex 模型
    codex_ids = {m["id"] for m in CODEX_MODELS_LIST}
    if model_id in codex_ids:
        log.info(f"[Unified Gateway] Routing to Codex: {request.model} (Base: {model_id})")
        from src.router.codex.openai import chat_completions as handler
        return await handler(request, token)
        
    # 2. 检查是否为 Antigravity 模型 (简单判断：如果不是Gemini的基础模型，且不是Codex，尝试Antigravity?)
    # 但由于Antigravity模型很多，且可能是动态的，这里先不强行路由到Antigravity，除非我们确定。
    # 由于 Geminicli 的 router 是默认处理者，如果它处理不了会报错。
    # 暂时默认所有非Codex请求都给 Geminicli 处理 (Antigravity 也有自己的独立端点，用户若用Unified，可能期望它能Work)
    # 如果要支持 Antigravity，需要 import src.router.antigravity.openai and call it.
    
    log.info(f"[Unified Gateway] Routing to Geminicli (Default): {model_id}")
    from src.router.geminicli.openai import chat_completions as handler
    return await handler(request, token)
