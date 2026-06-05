"""
LLM 服务模块
使用 OpenAI 兼容 API 调用通义千问 Qwen3-max
"""
import json
import time
import logging
from typing import List, Dict, Optional, Generator
from openai import OpenAI

from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from utils.logger import get_logger
from utils.helpers import estimate_tokens

logger = get_logger(__name__)

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
)

# 重试配置
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒


def chat_completion(
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict]] = None,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> Dict:
    """
    调用 LLM 进行对话补全（支持 function calling，含自动重试）
    """
    kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    # 估算 token 数
    total_text = " ".join([m.get("content", "") or "" for m in messages])
    est_tokens = estimate_tokens(total_text)
    has_tools = "携带 tools" if tools else "无 tools"

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            logger.debug("LLM 调用 (第 %d/%d 次): model=%s, msgs=%d, est_tokens~%d, %s",
                        attempt, MAX_RETRIES, LLM_MODEL, len(messages), est_tokens, has_tools)
            response = client.chat.completions.create(**kwargs)
            elapsed = time.time() - t0

            resp_dict = response.model_dump()
            choice = resp_dict.get("choices", [{}])[0]
            finish_reason = choice.get("finish_reason", "?")
            content_len = len(choice.get("message", {}).get("content") or "")
            tool_calls = choice.get("message", {}).get("tool_calls") or []
            tool_calls_count = len(tool_calls)

            logger.info("LLM 响应: finish=%s, content_len=%d, tool_calls=%d, 耗时 %.1fs",
                       finish_reason, content_len, tool_calls_count, elapsed)
            return resp_dict

        except Exception as e:
            last_error = e
            logger.warning("LLM 调用失败 (第 %d/%d 次): %s", attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                logger.info("准备重试 (%d/%d), 等待 %ds...", attempt + 1, MAX_RETRIES, RETRY_DELAY)
                time.sleep(RETRY_DELAY)

    logger.error("LLM 调用最终失败，已重试 %d 次: %s", MAX_RETRIES, last_error, exc_info=True)
    raise last_error


def stream_chat(
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict]] = None,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> Generator[str, None, None]:
    """流式调用 LLM"""
    kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    try:
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
    except Exception as e:
        logger.error("LLM 流式调用失败: %s", e, exc_info=True)
        yield f"\n[错误: {e}]"


def extract_tool_calls(response: Dict) -> List[Dict]:
    """从 LLM 响应中提取 tool calls"""
    tool_calls = []
    try:
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append({
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "arguments": args,
            })
    except Exception as e:
        logger.error("解析 tool_calls 失败: %s", e, exc_info=True)
    return tool_calls


def extract_content(response: Dict) -> str:
    """从 LLM 响应中提取文本内容"""
    try:
        choice = response.get("choices", [{}])[0]
        return choice.get("message", {}).get("content", "") or ""
    except Exception:
        return ""
