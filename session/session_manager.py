"""
会话管理模块
负责会话的 CRUD、消息管理、上下文构建
"""
import logging
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import desc

from database.models import Session, Message, MessageRole, Summary
from config import MAX_CONTEXT_MESSAGES
from utils.logger import get_logger

logger = get_logger(__name__)


class SessionManager:
    """会话管理器"""

    def __init__(self, db: DBSession):
        self.db = db

    # ============================================================
    # 会话 CRUD
    # ============================================================
    def create_session(self, title: str = "新会话") -> int:
        session = Session(title=title)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        logger.info("会话创建: id=%d, title='%s'", session.id, title)
        return session.id

    def get_session(self, session_id: int) -> Optional[Session]:
        return self.db.query(Session).filter(Session.id == session_id).first()

    def list_sessions(self) -> List[dict]:
        sessions = (
            self.db.query(Session)
            .order_by(desc(Session.updated_at))
            .all()
        )
        return [
            {
                "id": s.id,
                "title": s.title,
                "message_count": len(s.messages),
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            }
            for s in sessions
        ]

    def delete_session(self, session_id: int):
        session = self.get_session(session_id)
        if session:
            self.db.delete(session)
            self.db.commit()
            logger.info("会话删除: id=%d", session_id)

    def update_session_title(self, session_id: int, title: str):
        session = self.get_session(session_id)
        if session:
            session.title = title
            self.db.commit()

    # ============================================================
    # 消息管理
    # ============================================================
    def add_message(self, session_id: int, role: str, content: str) -> int:
        msg = Message(
            session_id=session_id,
            role=MessageRole(role),
            content=content,
        )
        self.db.add(msg)

        session = self.get_session(session_id)
        if session:
            session.updated_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(msg)
        logger.debug("消息写入: session=%d, role=%s, len=%d, msg_id=%d",
                    session_id, role, len(content), msg.id)
        return msg.id

    def get_messages(
        self, session_id: int, limit: int = 100,
        offset: int = 0, exclude_system: bool = False,
    ) -> List[Message]:
        query = self.db.query(Message).filter(Message.session_id == session_id)
        if exclude_system:
            query = query.filter(Message.role != MessageRole.SYSTEM)
        return (
            query.order_by(Message.created_at)
            .offset(offset).limit(limit).all()
        )

    def count_messages(self, session_id: int) -> int:
        return (
            self.db.query(Message)
            .filter(Message.session_id == session_id)
            .count()
        )

    # ============================================================
    # 上下文构建
    # ============================================================
    def build_context(self, session_id: int, max_recent: int = MAX_CONTEXT_MESSAGES) -> List[dict]:
        messages = []

        sys_msg = (
            self.db.query(Message)
            .filter(Message.session_id == session_id, Message.role == MessageRole.SYSTEM)
            .order_by(Message.created_at).first()
        )
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg.content})

        latest_summary = (
            self.db.query(Summary)
            .filter(Summary.session_id == session_id)
            .order_by(desc(Summary.created_at)).first()
        )
        if latest_summary:
            summary_text = (
                f"[以下是之前对话的摘要，请基于这些背景信息继续对话]\n"
                f"{latest_summary.content}\n"
                f"[摘要结束]"
            )
            if messages:
                messages[0]["content"] += f"\n\n{summary_text}"
            logger.debug("上下文注入摘要: summary_id=%d", latest_summary.id)

        recent_msgs = (
            self.db.query(Message)
            .filter(Message.session_id == session_id, Message.role != MessageRole.SYSTEM)
            .order_by(desc(Message.created_at))
            .limit(max_recent).all()
        )
        recent_msgs = list(reversed(recent_msgs))

        for msg in recent_msgs:
            messages.append({"role": msg.role.value, "content": msg.content})

        return messages

    # ============================================================
    # 摘要管理
    # ============================================================
    def add_summary(self, session_id: int, content: str,
                    start_msg_id: int = None, end_msg_id: int = None) -> int:
        summary = Summary(
            session_id=session_id,
            content=content,
            start_message_id=start_msg_id,
            end_message_id=end_msg_id,
        )
        self.db.add(summary)
        self.db.commit()
        self.db.refresh(summary)
        logger.info("摘要写入: session=%d, summary_id=%d, coverage=[%s:%s]",
                   session_id, summary.id, start_msg_id, end_msg_id)
        return summary.id

    def get_latest_summary(self, session_id: int) -> Optional[Summary]:
        return (
            self.db.query(Summary)
            .filter(Summary.session_id == session_id)
            .order_by(desc(Summary.created_at)).first()
        )
