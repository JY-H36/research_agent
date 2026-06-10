"""
文档处理模块
- PDF → Markdown 转换（Docling）
- 按 Markdown 标题层级分块（## / ###）
- MD5 校验
"""
import os
import re
import logging
from typing import List, Dict, Tuple
from docling.document_converter import DocumentConverter

from config import MAX_CHUNK_SIZE, CHUNK_OVERLAP
from utils.helpers import compute_md5_from_file
from utils.logger import get_logger

logger = get_logger(__name__)

# Docling 转换器（全局复用，内部有缓存）
_docling_converter = DocumentConverter()


# ============================================================
# MD5
# ============================================================
def compute_md5(file_path: str) -> str:
    return compute_md5_from_file(file_path)


# ============================================================
# PDF → Markdown（Docling）
# ============================================================
def convert_pdf_to_markdown(pdf_path: str) -> str:
    """
    使用 Docling 将 PDF 转为 Markdown
    返回完整的 MD 文本；失败返回空字符串
    """
    try:
        logger.info("Docling 开始转换: %s", os.path.basename(pdf_path))
        result = _docling_converter.convert(pdf_path)
        md_text = result.document.export_to_markdown()

        if not md_text or not md_text.strip():
            logger.warning("Docling 转换结果为空: %s", os.path.basename(pdf_path))
            return ""

        logger.info("Docling 转换完成: %s, MD 长度 %d 字符",
                   os.path.basename(pdf_path), len(md_text))
        return md_text

    except Exception as e:
        logger.error("Docling 转换失败: %s — %s", os.path.basename(pdf_path), e, exc_info=True)
        return ""


# ============================================================
# Markdown 标题检测（替代旧的正则猜章节）
# ============================================================
HEADING_PATTERN = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)


def detect_md_sections(md_text: str) -> List[Dict]:
    """
    按 Markdown 标题（# ~ ####）检测章节边界
    返回: [
        {"title": "Introduction", "level": 2, "start": 0, "end": 1500},
        {"title": "Method", "level": 2, "start": 1500, "end": 4500},
        ...
    ]
    """
    headings = []
    for m in HEADING_PATTERN.finditer(md_text):
        level = len(m.group(1))         # # = 1, ## = 2, ### = 3
        title = m.group(2).strip()
        pos = m.start()
        headings.append({"title": title, "level": level, "pos": pos})

    if not headings:
        logger.debug("未检测到 Markdown 标题，整文单章节处理")
        return [{"title": "全文", "level": 1, "start": 0, "end": len(md_text)}]

    # 过滤掉过长行（非标题）和过高层级（#### 以上太细）
    headings = [h for h in headings if len(h["title"]) <= 120 and 1 <= h["level"] <= 3]

    # 计算每个章节的字符范围
    sections = []
    for i, h in enumerate(headings):
        start = h["pos"]
        if i + 1 < len(headings):
            end = headings[i + 1]["pos"]
        else:
            end = len(md_text)
        sections.append({
            "title": h["title"],
            "level": h["level"],
            "start": start,
            "end": end,
        })

    logger.debug("MD 标题检测: %d 个章节 (h1~h3)", len(sections))
    return sections


# ============================================================
# 按 MD 标题分块
# ============================================================
def chunk_by_md_sections(
    md_text: str,
    max_size: int = MAX_CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    按 Markdown 标题层级分块
    - 每个 ## 或 ### 作为一个逻辑单元
    - 超过 max_size 的章节进一步切分
    返回: [{"section_name": str, "level": int, "chunk_index": int, "content": str}, ...]
    """
    sections = detect_md_sections(md_text)
    chunks = []

    for sec in sections:
        section_text = md_text[sec["start"]:sec["end"]].strip()
        if not section_text:
            continue

        # 用标题行作为 section_name（取第一行）
        first_line = section_text.split('\n', 1)[0].strip()
        section_name = first_line.lstrip('#').strip()[:200]

        if len(section_text) <= max_size:
            chunks.append({
                "section_name": section_name,
                "level": sec["level"],
                "chunk_index": 0,
                "content": section_text,
            })
        else:
            sub_chunks = _split_long_text(section_text, max_size, overlap)
            for sub_idx, sub_text in enumerate(sub_chunks):
                chunks.append({
                    "section_name": section_name,
                    "level": sec["level"],
                    "chunk_index": sub_idx,
                    "content": sub_text,
                })

    logger.info("MD 分块完成: %d 个 chunk (max_size=%d, overlap=%d)",
               len(chunks), max_size, overlap)
    return chunks


# ============================================================
# 长文本二次切分
# ============================================================
def _split_long_text(text: str, max_size: int, overlap: int) -> List[str]:
    """将长文本按固定大小切分为多个块，块之间有 overlap"""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_size, text_len)

        if end < text_len:
            search_start = max(start, end - 100)
            natural_break = _find_natural_break(text, search_start, end + 100)
            if natural_break != -1 and natural_break > start:
                end = natural_break + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end >= text_len:
            break

        new_start = end - overlap
        if new_start <= start:
            new_start = end
        start = new_start

    return chunks


def _find_natural_break(text: str, search_start: int, search_end: int) -> int:
    search_end = min(search_end, len(text))
    for i in range(search_end - 1, search_start, -1):
        if text[i] in '.。！？\n' and (i + 1 >= len(text) or text[i + 1] in ' \n\t'):
            return i
    return -1


# ============================================================
# TXT 文件处理（保持兼容）
# ============================================================
def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk') as f:
            return f.read()


# ============================================================
# 完整处理流程
# ============================================================
def process_document(file_path: str) -> Tuple[str, List[Dict], str]:
    """
    处理文档的完整流程
    PDF → Docling 转 MD → 按 ##/### 标题分块
    TXT → 直接按 MD 标题分块

    返回: (file_md5, chunks_list, md_text)
      chunks_list: [{"section_name": str, "level": int, "chunk_index": int, "content": str}, ...]
      md_text: Docling 转换后的 Markdown 全文（供知识图谱实体提取使用）
    """
    logger.info("开始处理文档: %s", os.path.basename(file_path))

    # 1. MD5
    file_md5 = compute_md5(file_path)
    logger.debug("MD5: %s", file_md5)

    # 2. 提取/转换文本
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        md_text = convert_pdf_to_markdown(file_path)
        if not md_text:
            raise ValueError(
                f"Docling 无法转换此 PDF。"
                f"如果 PDF 为扫描版/纯图片型，建议先 OCR 处理后再上传。"
            )
    elif ext in ('.txt', '.md'):
        md_text = extract_text_from_txt(file_path)
        if not md_text:
            raise ValueError(f"文件内容为空: {file_path}")
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    # 3. 按 MD 标题分块
    chunks = chunk_by_md_sections(md_text)

    if not chunks:
        raise ValueError(f"文档分块结果为空（文本长度 {len(md_text)} 字符），请检查文件内容")

    logger.info("文档处理完成: %s, %d chunks", os.path.basename(file_path), len(chunks))
    return file_md5, chunks, md_text
