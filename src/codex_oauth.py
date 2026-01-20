"""
Codex OAuth 认证模块
实现 OpenAI Codex CLI 的 OAuth PKCE 认证流程
"""

import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx

from log import log

# ====================== Codex OAuth 常量 ======================

CODEX_AUTH_URL = "https://auth.openai.com/oauth/authorize"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_CALLBACK_PORT = 1455
CODEX_REDIRECT_URI = f"http://localhost:{CODEX_CALLBACK_PORT}/auth/callback"
CODEX_SCOPES = "openid email profile offline_access"


# ====================== 数据类 ======================

@dataclass
class CodexCredentials:
    """Codex OAuth 凭证数据"""
    id_token: str
    access_token: str
    refresh_token: str
    account_id: str
    email: str
    expiry: str  # ISO 8601 格式，统一使用 "expiry" 键以匹配 CredentialManager
    type: str = "codex"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["CodexCredentials"]:
        """从字典创建"""
        try:
            return cls(
                id_token=data.get("id_token", ""),
                access_token=data.get("access_token", ""),
                refresh_token=data.get("refresh_token", ""),
                account_id=data.get("account_id", ""),
                email=data.get("email", ""),
                # 兼容旧字段 "expire"
                expiry=data.get("expiry") or data.get("expire", ""),
                type=data.get("type", "codex"),
            )
        except Exception as e:
            log.error(f"Failed to create CodexCredentials from dict: {e}")
            return None

    def is_expired(self) -> bool:
        """检查访问令牌是否过期"""
        if not self.expiry:
            return True
        try:
            expire_dt = datetime.fromisoformat(self.expiry.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            # 提前 5 分钟认为过期
            return now >= (expire_dt - timedelta(minutes=5))
        except Exception:
            return True


@dataclass
class PKCECodes:
    """PKCE 验证码"""
    code_verifier: str
    code_challenge: str


# ====================== PKCE 实现 ======================

def generate_pkce_codes() -> PKCECodes:
    """
    生成 PKCE 验证码对
    - code_verifier: 随机字符串 (43-128 字符)
    - code_challenge: code_verifier 的 SHA-256 哈希的 Base64URL 编码
    """
    # 生成 32 字节随机数据，Base64URL 编码后约 43 字符
    code_verifier = secrets.token_urlsafe(32)

    # SHA-256 哈希
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()

    # Base64URL 编码 (无填充)
    code_challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    return PKCECodes(
        code_verifier=code_verifier,
        code_challenge=code_challenge
    )


# ====================== JWT 解析 ======================

def parse_jwt_claims(id_token: str) -> Optional[Dict[str, Any]]:
    """
    解析 JWT ID Token 获取用户信息
    注意：这是简化实现，仅解析 payload，不验证签名
    """
    try:
        # JWT 格式: header.payload.signature
        parts = id_token.split(".")
        if len(parts) != 3:
            return None

        # 解码 payload (第二部分)
        payload = parts[1]
        # 添加填充
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding

        decoded = base64.urlsafe_b64decode(payload)
        claims = json.loads(decoded)
        return claims

    except Exception as e:
        log.warning(f"Failed to parse JWT token: {e}")
        return None


def get_account_id_from_claims(claims: Dict[str, Any]) -> str:
    """从 JWT claims 中提取账户 ID"""
    # OpenAI 使用 organization 或 org_id
    if "org_id" in claims:
        return claims["org_id"]
    if "organizations" in claims and claims["organizations"]:
        orgs = claims["organizations"]
        if isinstance(orgs, list) and len(orgs) > 0:
            org = orgs[0]
            if isinstance(org, dict):
                return org.get("id", "")
            return str(org)
    return claims.get("sub", "")


def get_email_from_claims(claims: Dict[str, Any]) -> str:
    """从 JWT claims 中提取邮箱"""
    return claims.get("email", "")


# ====================== OAuth 认证类 ======================

class CodexOAuth:
    """Codex OAuth 认证处理器"""

    def __init__(self, proxy_url: Optional[str] = None):
        """
        初始化

        Args:
            proxy_url: 代理 URL (可选)
        """
        self.proxy_url = proxy_url
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端"""
        if self._http_client is None:
            proxies = self.proxy_url if self.proxy_url else None
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                proxy=proxies,
                follow_redirects=True
            )
        return self._http_client

    async def close(self):
        """关闭 HTTP 客户端"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def generate_auth_url(self, state: str, pkce_codes: PKCECodes) -> str:
        """
        生成 OAuth 授权 URL

        Args:
            state: 状态参数 (用于防止 CSRF)
            pkce_codes: PKCE 验证码

        Returns:
            授权 URL
        """
        params = {
            "client_id": CODEX_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": CODEX_REDIRECT_URI,
            "scope": CODEX_SCOPES,
            "state": state,
            "code_challenge": pkce_codes.code_challenge,
            "code_challenge_method": "S256",
            "prompt": "login",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
        }
        return f"{CODEX_AUTH_URL}?{urlencode(params)}"

    async def exchange_code_for_tokens(
        self,
        code: str,
        pkce_codes: PKCECodes
    ) -> Optional[CodexCredentials]:
        """
        用授权码换取访问令牌

        Args:
            code: 授权码
            pkce_codes: PKCE 验证码

        Returns:
            CodexCredentials 或 None
        """
        try:
            client = await self._get_http_client()

            data = {
                "grant_type": "authorization_code",
                "client_id": CODEX_CLIENT_ID,
                "code": code,
                "redirect_uri": CODEX_REDIRECT_URI,
                "code_verifier": pkce_codes.code_verifier,
            }

            response = await client.post(
                CODEX_TOKEN_URL,
                data=data,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                }
            )

            if response.status_code != 200:
                log.error(f"Token exchange failed: {response.status_code} - {response.text}")
                return None

            token_data = response.json()

            # 解析 ID Token 获取用户信息
            id_token = token_data.get("id_token", "")
            claims = parse_jwt_claims(id_token) if id_token else {}

            account_id = get_account_id_from_claims(claims) if claims else ""
            email = get_email_from_claims(claims) if claims else ""

            # 计算过期时间
            expires_in = token_data.get("expires_in", 3600)
            expire_dt = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

            return CodexCredentials(
                id_token=id_token,
                access_token=token_data.get("access_token", ""),
                refresh_token=token_data.get("refresh_token", ""),
                account_id=account_id,
                email=email,
                expiry=expire_dt.isoformat(),
            )

        except Exception as e:
            log.error(f"Token exchange error: {e}")
            return None

    async def refresh_tokens(
        self,
        refresh_token: str
    ) -> Optional[CodexCredentials]:
        """
        使用刷新令牌获取新的访问令牌

        Args:
            refresh_token: 刷新令牌

        Returns:
            新的 CodexCredentials 或 None
        """
        try:
            client = await self._get_http_client()

            data = {
                "client_id": CODEX_CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "scope": "openid profile email",
            }

            response = await client.post(
                CODEX_TOKEN_URL,
                data=data,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                }
            )

            if response.status_code != 200:
                log.error(f"Token refresh failed: {response.status_code} - {response.text}")
                return None

            token_data = response.json()

            # 解析 ID Token
            id_token = token_data.get("id_token", "")
            claims = parse_jwt_claims(id_token) if id_token else {}

            account_id = get_account_id_from_claims(claims) if claims else ""
            email = get_email_from_claims(claims) if claims else ""

            # 计算过期时间
            expires_in = token_data.get("expires_in", 3600)
            expire_dt = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

            return CodexCredentials(
                id_token=id_token,
                access_token=token_data.get("access_token", ""),
                refresh_token=token_data.get("refresh_token", refresh_token),  # 可能不返回新的 refresh_token
                account_id=account_id,
                email=email,
                expiry=expire_dt.isoformat(),
            )

        except Exception as e:
            log.error(f"Token refresh error: {e}")
            return None


# ====================== 全局状态管理 ======================

# 存储进行中的 Codex 认证流程
codex_auth_flows: Dict[str, Dict[str, Any]] = {}
MAX_CODEX_AUTH_FLOWS = 20


def create_codex_auth_flow() -> Dict[str, Any]:
    """
    创建新的 Codex 认证流程

    Returns:
        包含 state, auth_url, pkce_codes 的字典
    """
    # 清理过期流程
    cleanup_codex_auth_flows()

    # 检查流程数量限制
    if len(codex_auth_flows) >= MAX_CODEX_AUTH_FLOWS:
        # 删除最旧的流程
        oldest_state = min(codex_auth_flows.keys(), key=lambda k: codex_auth_flows[k]["created_at"])
        del codex_auth_flows[oldest_state]

    # 生成状态和 PKCE
    state = secrets.token_urlsafe(32)
    pkce_codes = generate_pkce_codes()

    # 生成授权 URL
    oauth = CodexOAuth()
    auth_url = oauth.generate_auth_url(state, pkce_codes)

    # 保存流程状态
    flow_data = {
        "state": state,
        "pkce_codes": pkce_codes,
        "auth_url": auth_url,
        "created_at": time.time(),
        "code": None,  # 回调后填充
        "completed": False,
    }
    codex_auth_flows[state] = flow_data

    return {
        "state": state,
        "auth_url": auth_url,
    }


def cleanup_codex_auth_flows():
    """清理过期的认证流程 (超过 10 分钟)"""
    now = time.time()
    expired_states = [
        state for state, data in codex_auth_flows.items()
        if now - data["created_at"] > 600  # 10 分钟
    ]
    for state in expired_states:
        del codex_auth_flows[state]


async def complete_codex_auth_flow(state: str, code: str) -> Optional[CodexCredentials]:
    """
    完成 Codex 认证流程

    Args:
        state: 状态参数
        code: 授权码

    Returns:
        CodexCredentials 或 None
    """
    if state not in codex_auth_flows:
        log.error(f"Codex auth flow not found for state: {state}")
        return None

    flow_data = codex_auth_flows[state]
    pkce_codes = flow_data["pkce_codes"]

    # 换取令牌
    oauth = CodexOAuth()
    try:
        credentials = await oauth.exchange_code_for_tokens(code, pkce_codes)
        if credentials:
            # 标记流程完成
            flow_data["completed"] = True
            # 清理流程数据
            del codex_auth_flows[state]
            return credentials
        return None
    finally:
        await oauth.close()
