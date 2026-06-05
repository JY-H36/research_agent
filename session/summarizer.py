"""
自动摘要模块
当会话上下文超过阈值时，自动生成对话摘要
"""
import logging
from typing import List
from sqlalchemy.orm import Session as DBSession

from database.models import Message, MessageRole
from session.session_manager import SessionManager
from config import MAX_CONTEXT_MESSAGES
from utils.logger import get_logger

logger = get_logger(__name__)


SUMMARY_PROMPT = """请对以下对话内容生成一份结构化的摘要，要求包含以下四个部分：

1. **讨论主题**：一句话概括对话的核心主题
2. **关键讨论点**：按时间线梳理的主要讨论内容和重要观点
3. **涉及论文/知识库**：对话中提及或检索到的论文和知识点
4. **待解决事项**：对话中提到但尚未解决的问题或后续需要跟进的事项

要求：
- 使用中文
- 保持简洁，每个部分 2-5 句话
- 不要遗漏关键技术细节和决策

对话内容：
---
{conversation}
---

请生成摘要："""


def check_and_summarize(session_id: int, db: DBSession):
    """检查会话消息数是否超过阈值，若是则自动生成摘要"""
    session_mgr = SessionManager(db)
    msg_count = session_mgr.count_messages(session_id)

    if msg_count < MAX_CONTEXT_MESSAGES:
        return None

    all_msgs = session_mgr.get_messages(session_id, limit=500, exclude_system=True)
    if len(all_msgs) < MAX_CONTEXT_MESSAGES:
        return None

    latest_summary = session_mgr.get_latest_summary(session_id)
    if latest_summary and latest_summary.end_message_id:
        latest_msg_id = all_msgs[-1].id if all_msgs else 0
        if latest_msg_id <= latest_summary.end_message_id:
            logger.debug("摘要跳过: 最近消息已覆盖 (last_summarized=%d, latest_msg=%d)",
                        latest_summary.end_message_id, latest_msg_id)
            return None

    logger.info("触发自动摘要: session=%d, msg_count=%d", session_id, msg_count)
    summary_content = generate_summary(all_msgs)

    if summary_content:
        start_id = all_msgs[0].id if all_msgs else None
        end_id = all_msgs[-1].id if all_msgs else None
        session_mgr.add_summary(session_id, summary_content, start_id, end_id)
        logger.info("自动摘要完成: session=%d, 覆盖消息 #%d~#%d",
                   session_id, start_id or 0, end_id or 0)

    return summary_content


def generate_summary(messages: List[Message]) -> str:
    """调用 LLM 生成对话摘要"""
    if not messages:
        return ""

    conversation_parts = []
    for msg in messages:
        role_label = "用户" if msg.role == MessageRole.USER else "助手"
        content = msg.content
        if len(content) > 2000:
            content = content[:2000] + "...(内容过长已截断)"
        conversation_parts.append(f"[{role_label}]: {content}")

    conversation_text = '\n\n'.join(conversation_parts)

    if len(conversation_text) > 8000:
        head = conversation_text[:3000]
        tail = conversation_text[-4000:]
        conversation_text = head + "\n\n...(中间对话已省略)...\n\n" + tail
        logger.debug("摘要: 对话文本截断 (原始 %d 字符)", len(conversation_text))

    prompt = SUMMARY_PROMPT.format(conversation=conversation_text)
    logger.debug("摘要生成开始: %d 条消息, prompt_len=%d", len(messages), len(prompt))

    try:
        from agent.llm_service import chat_completion, extract_content
        response = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            temperature=0.3,
            max_tokens=1024,
        )
        return extract_content(response)
    except Exception as e:
        logger.error("摘要生成失败: %s", e, exc_info=True)
        return f"[自动摘要生成失败: {e}]"
