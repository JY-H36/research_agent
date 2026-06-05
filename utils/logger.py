"""
统一日志系统
- 控制台彩色输出（开发调试）
- 按天滚动的文件日志（INFO+）
- 按天滚动的错误日志（ERROR+WARNING）
- 内存环形缓冲区（前端展示，最近 500 条）
- trace_id 全链路追踪（contextvars）
"""
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from collections import deque
from contextvars import ContextVar
from typing import List, Dict, Optional

from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# ============================================================
# trace_id 上下文变量（跨调用栈传递，无需显式传参）
# ============================================================
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")

def set_trace_id(tid: str):
    _trace_id.set(tid)

def get_trace_id() -> str:
    return _trace_id.get()

# ============================================================
# 内存环形缓冲区（供 Streamlit 前端消费）
# ============================================================
_memory_buffer: deque = deque(maxlen=500)


def get_memory_logs() -> List[Dict]:
    """获取内存中的日志列表，供 Streamlit 展示"""
    return list(_memory_buffer)


def clear_memory_logs():
    _memory_buffer.clear()


# ============================================================
# 日志标签（模块标识）
# ============================================================
TAG_MAP = {
    "agent.agent_core":              "AGENT",
    "agent.tools":                   "TOOL",
    "agent.middleware":              "MIDDLEWARE",
    "agent.llm_service":             "LLM",
    "knowledge_base.document_processor": "KB",
    "knowledge_base.embedding_service":  "EMBED",
    "knowledge_base.vector_store":       "VEC",
    "knowledge_base.retriever":          "RETRIEVER",
    "session.session_manager":       "SESSION",
    "session.summarizer":            "SUMMARY",
    "database.connection":           "DB",
    "app":                           "APP",
}


def _get_tag(logger_name: str) -> str:
    """根据 logger 名称获取短标签"""
    for prefix, tag in TAG_MAP.items():
        if logger_name.startswith(prefix):
            return tag
    # 取最后一个点后的部分
    return logger_name.rsplit(".", 1)[-1].upper()[:12]


# ============================================================
# 彩色格式化器（控制台）
# ============================================================
LEVEL_COLORS = {
    logging.DEBUG:    Fore.LIGHTBLACK_EX,
    logging.INFO:     Fore.CYAN,
    logging.WARNING:  Fore.YELLOW,
    logging.ERROR:    Fore.RED + Style.BRIGHT,
    logging.CRITICAL: Fore.WHITE + Style.BRIGHT,
}

LEVEL_ICONS = {
    logging.DEBUG:    "·",
    logging.INFO:     "✔",
    logging.WARNING:  "⚠",
    logging.ERROR:    "✖",
    logging.CRITICAL: "☠",
}


class ColoredFormatter(logging.Formatter):
    """控制台彩色格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        # 注入额外字段
        if not hasattr(record, 'tag'):
            record.tag = _get_tag(record.name)
        tid = get_trace_id()
        record.trace_id = f"[{tid}]" if tid else ""

        color = LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        icon = LEVEL_ICONS.get(record.levelno, " ")
        record.level_color = color
        record.level_icon = icon

        ts = datetime.now().strftime("%H:%M:%S")
        tag_colored = f"{Fore.GREEN}[{record.tag}]{Style.RESET_ALL}"
        level_colored = f"{color}[{record.levelname}]{Style.RESET_ALL}"
        tid_colored = f"{Fore.LIGHTYELLOW_EX}{record.trace_id}{Style.RESET_ALL}" if record.trace_id else ""

        header = f"{Fore.LIGHTBLACK_EX}{ts}{Style.RESET_ALL} {tag_colored} {level_colored} {tid_colored}"
        msg = f"{color}{record.getMessage()}{Style.RESET_ALL}"

        # 异常信息
        exc = ""
        if record.exc_info and record.exc_info[1]:
            import traceback
            exc = f"\n{Fore.RED}{''.join(traceback.format_exception(*record.exc_info))}{Style.RESET_ALL}"

        return f"{header} {msg}{exc}"


class PlainFormatter(logging.Formatter):
    """文件日志格式化器（无颜色）"""

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, 'tag'):
            record.tag = _get_tag(record.name)
        tid = get_trace_id()
        record.trace_id = f"[{tid}]" if tid else ""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"{ts} [{record.tag}] [{record.levelname}] {record.trace_id}"
        msg = record.getMessage()

        result = f"{header} {msg}"
        if record.exc_info and record.exc_info[1]:
            import traceback
            result += "\n" + "".join(traceback.format_exception(*record.exc_info))
        return result


# ============================================================
# 内存 Handler
# ============================================================
class MemoryRingHandler(logging.Handler):
    """将日志存入内存环形缓冲区"""

    def emit(self, record: logging.LogRecord):
        try:
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tag": getattr(record, 'tag', _get_tag(record.name)),
                "level": record.levelname,
                "trace_id": get_trace_id(),
                "message": self.format(record),
            }
            _memory_buffer.append(entry)
        except Exception:
            self.handleError(record)


# ============================================================
# 初始化函数
# ============================================================
_logging_initialized = False


def setup_logging(
    log_dir: str = None,
    console_level: int = logging.DEBUG,
    file_level: int = logging.INFO,
):
    """
    初始化日志系统（全局调用一次）
    参数:
        log_dir: 日志文件目录，默认在项目根目录下的 logs/
        console_level: 控制台最低输出级别
        file_level: 文件最低输出级别
    """
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True

    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 最低级别，由各 handler 自行过滤

    # 去掉已有 handler（避免重复）
    root_logger.handlers.clear()

    # ---- 控制台 Handler ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(console_handler)

    # ---- 按天滚动文件 Handler（INFO+） ----
    info_path = os.path.join(log_dir, "agent_")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "agent.log"),  # 当前日志
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setLevel(file_level)
    file_handler.setFormatter(PlainFormatter())
    root_logger.addHandler(file_handler)

    # ---- 按天滚动错误文件 Handler（ERROR+WARNING） ----
    error_handler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "error.log"),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    error_handler.suffix = "%Y-%m-%d"
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(PlainFormatter())
    root_logger.addHandler(error_handler)

    # ---- 内存环形缓冲区 Handler ----
    memory_handler = MemoryRingHandler()
    memory_handler.setLevel(logging.DEBUG)
    memory_handler.setFormatter(PlainFormatter())
    root_logger.addHandler(memory_handler)

    # 启动日志
    root_logger.info("══════════════════════════════════════")
    root_logger.info("日志系统初始化完成")
    root_logger.info(f"  日志目录: {log_dir}")
    root_logger.info(f"  控制台级别: {logging.getLevelName(console_level)}")
    root_logger.info(f"  文件级别: {logging.getLevelName(file_level)}")
    root_logger.info(f"  保留天数: 30")
    root_logger.info("══════════════════════════════════════")


def get_logger(name: str) -> logging.Logger:
    """获取模块日志器（自动注入 tag）"""
    logger = logging.getLogger(name)
    return logger
