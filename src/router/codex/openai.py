"""
Codex OpenAI Router - 处理 OpenAI 格式 API 请求并转发到 OpenAI API
"""

import json
import uuid
import httpx
import time

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from log import log
from src.utils import authenticate_bearer
from src.models import OpenAIChatCompletionRequest, model_to_dict
from src.router.hi_check import is_health_check_request, create_health_check_response
from src.credential_manager import CredentialManager

# ==================== 常量 ====================

OPENAI_API_URL = "https://api.openai.com/v1"

# Codex 支持的模型列表 (从 utils 导入)
from src.utils import CODEX_MODELS_LIST as CODEX_MODELS

# ==================== 路由器初始化 ====================

router = APIRouter()

# 凭证管理器
credential_manager = CredentialManager()


# ==================== 辅助函数 ====================

async def get_codex_credential() -> Optional[tuple]:
    """获取可用的 Codex 凭证"""
    await credential_manager._ensure_initialized()
    return await credential_manager.get_valid_credential(mode="codex")


async def refresh_codex_token_if_needed(filename: str, credential_data: dict) -> dict:
    """如果需要，刷新 Codex 令牌"""
    from src.codex_oauth import CodexCredentials, CodexOAuth

    creds = CodexCredentials.from_dict(credential_data)
    if creds and creds.is_expired():
        log.info(f"Codex token expired, refreshing: {filename}")
        oauth = CodexOAuth()
        try:
            new_creds = await oauth.refresh_tokens(creds.refresh_token)
            if new_creds:
                # 更新存储
                from src.storage_adapter import get_storage_adapter
                storage = await get_storage_adapter()
                await storage.store_credential(filename, new_creds.to_dict(), mode="codex")
                return new_creds.to_dict()
        except Exception as e:
            log.error(f"Failed to refresh Codex token: {e}")
        finally:
            await oauth.close()

    return credential_data


# ==================== API 路由 ====================

@router.get("/codex/v1/models")
async def list_models(token: str = Depends(authenticate_bearer)):
    """从 OpenAI API 动态获取可用的模型列表"""
    import httpx

    # 获取 Codex 凭证
    credential_result = await get_codex_credential()
    if not credential_result:
        raise HTTPException(
            status_code=503,
            detail="No available Codex credentials. Please add credentials first."
        )

    filename, credential_data = credential_result

    # 刷新令牌（如果需要）
    credential_data = await refresh_codex_token_if_needed(filename, credential_data)
    access_token = credential_data.get("access_token")

    if not access_token:
        raise HTTPException(
            status_code=503,
            detail="Codex credential has no access token."
        )

    # 从 OpenAI API 获取模型列表
    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    # ---------------- 动态模型列表处理 ----------------
    from src.utils import get_available_models
    from src.router.base_router import create_openai_model_list
    
    # 生成本地所有变体 (包含假流式等)
    local_variant_names = get_available_models("codex")
    local_variants_obj = create_openai_model_list(local_variant_names, owned_by="openai")
    local_variants = [model_to_dict(m) for m in local_variants_obj.data]

    # 直接返回本地模型列表，不包含 OpenAI 原生模型 (如 gpt-3.5 等)
    return JSONResponse(content={
        "object": "list",
        "data": local_variants
    })


@router.post("/codex/v1/chat/completions")
async def chat_completions(
    openai_request: OpenAIChatCompletionRequest,
    token: str = Depends(authenticate_bearer)
):
    """
    处理 OpenAI 格式的聊天完成请求

    Args:
        openai_request: OpenAI 格式的请求体
        token: Bearer 认证令牌
    """
    log.debug(f"[CODEX] Request for model: {openai_request.model}")

    # 转换为字典
    request_dict = model_to_dict(openai_request)

    # 健康检查
    if is_health_check_request(request_dict, format="openai"):
        response = create_health_check_response(format="openai")
        return JSONResponse(content=response)

    # 获取 Codex 凭证
    credential_result = await get_codex_credential()
    if not credential_result:
        raise HTTPException(
            status_code=503,
            detail="No available Codex credentials. Please add credentials first."
        )

    filename, credential_data = credential_result

    # 刷新令牌（如果需要）
    credential_data = await refresh_codex_token_if_needed(filename, credential_data)
    access_token = credential_data.get("access_token")

    if not access_token:
        raise HTTPException(
            status_code=503,
            detail="Codex credential has no access token."
        )

    # CliproxyAPI 默认 Base URL
    OPENAI_API_URL = "https://chatgpt.com/backend-api/codex"

    original_model = model_to_dict(openai_request).get("model", "")
    # Check for fake streaming
    from src.utils import get_base_model_from_feature_model, is_fake_streaming_model
    use_fake_streaming = is_fake_streaming_model(original_model)
    real_model = get_base_model_from_feature_model(original_model)

    # 准备请求
    # CliproxyAPI 默认 Base URL
    # [PROTOCOL IMPLEMENTATION] Matches CliproxyAPI internal/translator/codex/openai/responses/codex_openai-responses_request.go
    
    # Import Official Codex Prompt to satisfy backend validation
    from src.router.codex.default_prompt import GPT_5_2_CODEX_PROMPT
    
    # 1. 基础字段设置
    request_dict = {
        "model": real_model,
        "stream": True,
        "store": False,
        "parallel_tool_calls": True,
        "include": ["reasoning.encrypted_content"]
    }

    # 2. 构造 input 数组
    # 结构: [{"type": "message", "role": "...", "content": [{"type": "input_text", "text": "..."}]}]
    messages = model_to_dict(openai_request).get("messages", [])
    input_list = []
    
    # 始终使用官方 Prompt 作为 instructions 以通过校验
    instructions = GPT_5_2_CODEX_PROMPT
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # [FIX] Backend Error: "System messages are not allowed"
        # Since we are forcing Official Instructions, we cannot use System role for user context.
        # Workaround: Convert System messages to User messages.
        if role == "system":
            role = "user"
            # Optional: Add a prefix to distinguish, but usually raw content is fine for GPT models
            # content = f"[System Context]\n{content}" 
            
        # [FIX] Backend Error: "Invalid value: 'input_text'" for assistant messages
        # Assistant messages must us 'output_text' type.
        content_type = "input_text"
        if role == "assistant":
            content_type = "output_text"

        # 转换复杂结构
        input_item = {
            "type": "message",
            "role": role,
            "content": [
                {
                    "type": content_type,
                    "text": content
                }
            ]
        }
        input_list.append(input_item)

    request_dict["instructions"] = instructions
    request_dict["input"] = input_list
    
    # 3. 移除不支持的字段 (Temperature 等) - 已经在最上面通过重新构造 request_dict 实现了
    
    # log.info(f"[CODEX] Final Request Body: {json.dumps(request_dict)}")
    # log.debug(f"[CODEX] Real model: {real_model}, FakeStream: {use_fake_streaming}")

    # 获取 account_id (组织 ID)
    account_id = credential_data.get("account_id")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "codex_cli_rs/0.50.0 (Mac OS 26.0.1; arm64) Apple_Terminal/464",
        "Version": "0.21.0",
        "Openai-Beta": "responses=experimental",
        "Session_id": str(uuid.uuid4()),
        "Accept": "text/event-stream",
        "Connection": "Keep-Alive",
        "Originator": "codex_cli_rs",
    }
    
    if account_id:
        headers["Chatgpt-Account-Id"] = account_id
        # 为了兼容性保留 OpenAI-Organization，但 Cliproxy 主要使用 Chatgpt-Account-Id
        headers["OpenAI-Organization"] = account_id

    is_streaming = openai_request.stream
    
    # ========== 假流式处理逻辑 ==========
    if use_fake_streaming and is_streaming:
        async def fake_stream_generator():
            # 强制非流式请求
            request_dict["stream"] = False
            
            import httpx # Moved here
            async with httpx.AsyncClient(timeout=120.0) as client:
                try:
                    # 获取完整响应
                    response = await client.post(
                        f"{OPENAI_API_URL}/responses", # Updated URL path
                        headers=headers,
                        json=request_dict
                    )
                    
                    if response.status_code != 200:
                        # 错误透传
                        error_data = response.json()
                        yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
                        yield "data: [DONE]\n\n".encode('utf-8')
                        return

                    full_data = response.json()
                    choices = full_data.get("choices", [])
                    if not choices:
                        yield "data: [DONE]\n\n".encode('utf-8')
                        return
                        
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    role = message.get("role", "assistant")
                    
                    import time
                    created_time = int(time.time())
                    chunk_id = f"chatcmpl-{uuid.uuid4()}"
                    
                    # 1. 发送角色信息
                    role_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": real_model,
                        "choices": [{"index": 0, "delta": {"role": role}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(role_chunk)}\n\n".encode('utf-8')
                    
                    # 2. 分块发送内容 (模拟打字机)
                    # 简单按字符切分，每块 4 个字符
                    chunk_size = 4
                    for i in range(0, len(content), chunk_size):
                        sub_content = content[i:i+chunk_size]
                        content_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": real_model,
                            "choices": [{"index": 0, "delta": {"content": sub_content}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(content_chunk)}\n\n".encode('utf-8')
                        await asyncio.sleep(0.01) # 微小延迟模拟
                        
                    # 3. 发送结束标记
                    end_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": real_model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(end_chunk)}\n\n".encode('utf-8')
                    yield "data: [DONE]\n\n".encode('utf-8')

                except Exception as e:
                    log.error(f"Codex fake stream failed: {e}")
                    err_chunk = {"error": {"message": str(e), "type": "server_error"}}
                    yield f"data: {json.dumps(err_chunk)}\n\n".encode('utf-8')
                    yield "data: [DONE]\n\n".encode('utf-8')

        return StreamingResponse(fake_stream_generator(), media_type="text/event-stream")

    # ========== 正常非流式请求 ==========
    if not is_streaming:
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{OPENAI_API_URL}/responses",
                    headers=headers,
                    json=request_dict
                )

                # 记录调用结果
                if response.status_code == 200:
                    await credential_manager.record_api_call_result(
                        filename, success=True, mode="codex"
                    )
                else:
                    await credential_manager.record_api_call_result(
                        filename, success=False, error_code=response.status_code, mode="codex"
                    )

                return JSONResponse(
                    content=response.json(),
                    status_code=response.status_code
                )

            except Exception as e:
                log.error(f"Codex API request failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    # ========== 正常流式请求 (包含流式抗截断) ==========
    # 注意：Codex/OpenAI 原生支持流式，且通常没有截断问题，这里直接透传
    # ========== 流式转发与协议转换逻辑 ==========
    async def stream_generator():
        # Cliproxy 协议转换状态
        response_id = f"chatcmpl-{uuid.uuid4()}"
        created_at = int(time.time())
        model_name = real_model
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                # async with client.stream(
                #     "POST", 
                #     f"{OPENAI_API_URL}/responses", 
                #     headers=headers, 
                #     json=request_dict
                # ) as response:
                
                async with client.stream(
                    "POST", 
                    f"{OPENAI_API_URL}/responses", 
                    headers=headers, 
                    json=request_dict
                ) as response:
                    
                    if response.status_code != 200:
                        error_content = await response.aread()
                        log.error(f"[CODEX] Error response body: {error_content.decode('utf-8')}")
                        error_chunk = {'error': {'message': f'Codex API Error: {response.status_code}', 'type': 'upstream_error', 'code': response.status_code}}
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # log.info(f"[CODEX] Streaming started for {original_model}")
                    
                    async for line_bytes in response.aiter_lines():
                        line = line_bytes.strip()
                        if not line or not line.startswith("data: "):
                            continue
                            
                        data_str = line[6:] # Remove "data: "
                        if data_str == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                            
                        try:
                            codex_data = json.loads(data_str)
                            event_type = codex_data.get("type", "")
                            
                            # ---------- 转换逻辑 ----------
                            # 参考 Cliproxy: internal/translator/codex/openai/chat-completions/codex_openai_response.go
                            
                            chunk = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created_at,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": None
                                    }
                                ]
                            }
                            
                            should_yield = False

                            if event_type == "response.created":
                                # 提取元数据
                                response_obj = codex_data.get("response", {})
                                response_id = response_obj.get("id", response_id)
                                created_at = response_obj.get("created_at", created_at)
                                model_name = response_obj.get("model", model_name)
                                # created 事件通常不包含 content，但在 OpenAI 协议中可以发一个空帧或者是只包含 role 的帧
                                chunk["id"] = response_id
                                chunk["created"] = created_at
                                chunk["model"] = model_name
                                chunk["choices"][0]["delta"]["role"] = "assistant"
                                should_yield = True
                                
                            elif event_type == "response.output_text.delta":
                                delta = codex_data.get("delta", "")
                                if delta:
                                    chunk["choices"][0]["delta"]["content"] = delta
                                    should_yield = True
                                    
                            elif event_type == "response.reasoning_summary_text.delta":
                                # 处理思考过程 (Reasoning)
                                delta = codex_data.get("delta", "")
                                if delta:
                                    # 注意：OpenAI 标准字段不支持 reasoning_content，通常放在 extending 字段或 content
                                    # 这里我们遵循常见社区标准，或者如果客户端支持 reasoning_content
                                    chunk["choices"][0]["delta"]["reasoning_content"] = delta 
                                    should_yield = True
                                    
                            elif event_type == "response.completed":
                                chunk["choices"][0]["finish_reason"] = "stop"
                                chunk["choices"][0]["delta"] = {} # Empty delta
                                should_yield = True
                                
                            # TODO: Handle tool calls (response.output_item.done) if needed using Cliproxy logic
                                
                            if should_yield:
                                yield f"data: {json.dumps(chunk)}\n\n"
                                
                        except json.JSONDecodeError:
                            log.warning(f"[CODEX] Failed to decode JSON line: {line}")
                            continue
                        except Exception as e:
                            log.error(f"[CODEX] Error processing line: {e}")
                            continue
                            
                    yield "data: [DONE]\n\n"
                    log.info(f"[CODEX] Streaming finished.")

            except Exception as e:
                log.error(f"[CODEX] Stream error: {e}")
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": 500
                    }
                }
                # 注意：这里 yield bytes 还是 str 取决于 Response 类。StreamingResponse通常接受 str (会被编码) 或 bytes
                # 为了安全，上面代码一直用的 f-string (str)，FastAPI 会处理
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
