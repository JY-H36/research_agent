"""
工具函数模块
提供 MD5 计算、token 估算、时间格式化等通用功能
"""
import hashlib
import os
from datetime import datetime


def compute_md5_from_bytes(data: bytes) -> str:
    """对字节数据计算 MD5 哈希值"""
    return hashlib.md5(data).hexdigest()


def compute_md5_from_file(file_path: str) -> str:
    """对文件内容计算 MD5 哈希值"""
    with open(file_path, "rb") as f:
        return compute_md5_from_bytes(f.read())


def estimate_tokens(text: str) -> int:
    """粗略估算文本的 token 数量（中文按字数，英文按空格分词）"""
    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    english_words = len([w for w in text.split() if w.isascii() and w.isalpha()])
    return chinese_chars + english_words


def format_timestamp(dt: datetime = None) -> str:
    """格式化时间戳为可读字符串"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(dir_path: str):
    """确保目录存在，不存在则创建"""
    os.makedirs(dir_path, exist_ok=True)


def generate_trace_id() -> str:
    """生成唯一 trace_id"""
    import uuid
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"req_{ts}_{short_uuid}"
