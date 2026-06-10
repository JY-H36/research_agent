"""
SQLAlchemy ORM 数据模型定义
5 张核心表：sessions, messages, summaries, documents, chunks
"""
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from database.connection import Base


# ============================================================
# 枚举类型
# ============================================================
class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ============================================================
# 会话表
# ============================================================
class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), default="新会话")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    messages = relationship(
        "Message", back_populates="session",
        order_by="Message.created_at",
        cascade="all, delete-orphan"
    )
    summaries = relationship(
        "Summary", back_populates="session",
        order_by="Summary.created_at",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Session(id={self.id}, title='{self.title}')>"


# ============================================================
# 消息表
# ============================================================
class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(SQLEnum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="messages")

    def __repr__(self):
        return f"<Message(id={self.id}, role='{self.role.value}', session_id={self.session_id})>"


# ============================================================
# 摘要表
# ============================================================
class Summary(Base):
    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    start_message_id = Column(Integer, nullable=True)   # 摘要覆盖的起始消息 ID
    end_message_id = Column(Integer, nullable=True)      # 摘要覆盖的结束消息 ID
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="summaries")

    def __repr__(self):
        return f"<Summary(id={self.id}, session_id={self.session_id})>"


# ============================================================
# 文档表（知识库）
# ============================================================
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False)
    md5_hash = Column(String(32), nullable=False, unique=True, index=True)
    file_path = Column(String(1000), nullable=True)
    file_size = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}')>"


# ============================================================
# 分块表
# ============================================================
class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    section_name = Column(Text, default="")
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Chunk(id={self.id}, doc_id={self.document_id}, section='{self.section_name}')>"


# ============================================================
# 确保知识图谱模型也被 Base.metadata 注册
# ============================================================
import knowledge_graph.models  # noqa: E402, F401 — 将 kg_* 表注册到 Base.metadata
