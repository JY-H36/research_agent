"""
科研灵感助手 Agent — Streamlit 前端入口
"""
import streamlit as st
import os
from datetime import datetime

from config import UPLOAD_DIR, ALLOWED_EXTENSIONS
from utils.helpers import ensure_dir, compute_md5_from_bytes
from utils.logger import setup_logging, get_memory_logs, clear_memory_logs, get_logger
from database.connection import init_db, SessionLocal
from database.models import Document, Chunk
from knowledge_base.document_processor import process_document
from knowledge_base.embedding_service import embed_texts
from knowledge_base.vector_store import add_chunks, get_chunk_count
from knowledge_base.retriever import rebuild_retriever
from agent.agent_core import ResearchAgent

logger = get_logger(__name__)

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="科研灵感助手",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 初始化
# ============================================================
@st.cache_resource
def initialize_system():
    """系统初始化（仅执行一次）"""
    ensure_dir(UPLOAD_DIR)
    setup_logging()
    init_db()
    logger.info("系统初始化完成: DB 就绪, 日志系统就绪")


def init_session_state():
    """初始化 Streamlit session state"""
    defaults = {
        "current_session_id": None,
        "agent": None,
        "messages": [],
        "tool_logs": [],
        "kb_stats": {"doc_count": 0, "chunk_count": 0},
        "upload_status": None,
        "log_filter_level": "全部",
        "log_filter_tag": "全部",
        "log_auto_scroll": True,
        "log_errors_only": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# 知识库操作
# ============================================================
def get_kb_stats() -> dict:
    db = SessionLocal()
    try:
        doc_count = db.query(Document).count()
        chunk_count = get_chunk_count()
        return {"doc_count": doc_count, "chunk_count": chunk_count}
    finally:
        db.close()


def add_document_to_kb(uploaded_file, status_container=None) -> tuple:
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name

    def update_status(msg):
        logger.info("[KB] %s", msg)
        if status_container is not None:
            status_container.text(msg)

    update_status(f"步骤 1/6: 检查文件格式 + 计算 MD5...")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"不支持的文件格式: {ext}，仅支持 {', '.join(ALLOWED_EXTENSIONS)}"

    file_md5 = compute_md5_from_bytes(file_bytes)

    db = SessionLocal()
    try:
        existing = db.query(Document).filter(Document.md5_hash == file_md5).first()
        if existing:
            return False, f"该文档已存在于知识库中（文件名: {existing.filename}，入库时间: {existing.created_at}）"

        update_status(f"步骤 2/6: 保存文件到磁盘...")
        ensure_dir(UPLOAD_DIR)
        file_path = os.path.join(UPLOAD_DIR, f"{file_md5}_{filename}")
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        update_status(f"步骤 3/6: Docling 转换 PDF → Markdown...")
        file_md5_check, chunks = process_document(file_path)

        if not chunks:
            os.remove(file_path)
            return False, "文档内容为空或无法解析。PDF 可能是扫描版/纯图片型，建议先 OCR 处理。"

        # 保存 MD 文件（供预览和后续分析）
        md_path = os.path.join(UPLOAD_DIR, f"{file_md5}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            # 先写文件名作标题
            f.write(f"# {os.path.splitext(filename)[0]}\n\n")
            for c in chunks:
                f.write(c["content"])
                f.write("\n\n---\n\n")
        logger.debug("MD 文件已保存: %s", md_path)

        update_status(f"步骤 4/6: 存入 MySQL ({len(chunks)} 个片段)...")
        doc = Document(
            filename=filename, md5_hash=file_md5,
            file_path=file_path,  # 存 PDF 路径
            file_size=len(file_bytes), chunk_count=len(chunks),
        )
        db.add(doc)
        db.flush()

        for i, chunk in enumerate(chunks):
            # section_name 包含标题层级前缀，如 "[h2] Method"
            level_prefix = f"[h{chunk.get('level', 2)}]"
            full_section = f"{level_prefix} {chunk.get('section_name', '')}".strip()[:500]
            chunk_obj = Chunk(
                document_id=doc.id, chunk_index=i,
                section_name=full_section,
                content=chunk["content"],
            )
            db.add(chunk_obj)
        db.commit()

        update_status(f"步骤 5/6: 向量化 + 存入 Chroma ({len(chunks)} 条)...")
        chunk_texts = [c["content"] for c in chunks]
        embeddings = embed_texts(chunk_texts)

        chunk_ids = [f"doc{doc.id}_chunk{c['chunk_index']}" for c in chunks]
        metadatas = [
            {
                "document_id": doc.id,
                "section_name": chunk.get("section_name", ""),
                "level": chunk.get("level", 2),
                "chunk_index": chunk["chunk_index"],
                "filename": filename,
            }
            for chunk in chunks
        ]

        add_chunks(chunk_ids, chunk_texts, embeddings, metadatas)

        update_status(f"步骤 6/6: 重建 BM25 索引...")
        rebuild_retriever()

        update_status(f"✅ 入库完成！")
        section_count = len(set(c["section_name"] for c in chunks))
        return True, f"入库成功！「{filename}」→ {section_count} 个章节, {len(chunks)} 个片段"

    except Exception as e:
        db.rollback()
        logger.error("文档入库失败: %s — %s", filename, e, exc_info=True)
        return False, f"入库失败: {str(e)}"
    finally:
        db.close()


# ============================================================
# 日志面板
# ============================================================
LEVEL_COLOR_MAP = {
    "DEBUG":    "#888888",
    "INFO":     "#4FC3F7",
    "WARNING":  "#FFB74D",
    "ERROR":    "#EF5350",
    "CRITICAL": "#FF1744",
}


def render_log_panel():
    """渲染侧边栏日志面板"""
    st.divider()
    st.header("📋 系统日志")

    # 筛选控件
    col1, col2 = st.columns(2)
    with col1:
        level_filter = st.selectbox(
            "级别", ["全部", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            key="log_filter_level_select",
            label_visibility="collapsed",
        )
        st.session_state.log_filter_level = level_filter
    with col2:
        tag_filter = st.selectbox(
            "标签",
            ["全部", "AGENT", "TOOL", "LLM", "KB", "RETRIEVER", "EMBED", "VEC", "SESSION", "SUMMARY", "DB", "APP"],
            key="log_filter_tag_select",
            label_visibility="collapsed",
        )
        st.session_state.log_filter_tag = tag_filter

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        st.session_state.log_auto_scroll = st.checkbox("自动滚动", value=True, key="log_auto_scroll_cb")
    with col_b:
        st.session_state.log_errors_only = st.checkbox("仅错误", value=False, key="log_errors_only_cb")
    with col_c:
        if st.button("🔄 刷新", use_container_width=True, key="log_refresh_btn"):
            pass  # st.rerun 自然刷新

    # 日志列表
    logs = get_memory_logs()

    # 筛选
    if st.session_state.log_filter_level != "全部":
        logs = [l for l in logs if l["level"] == st.session_state.log_filter_level]
    if st.session_state.log_filter_tag != "全部":
        logs = [l for l in logs if l["tag"] == st.session_state.log_filter_tag]
    if st.session_state.log_errors_only:
        logs = [l for l in logs if l["level"] in ("ERROR", "CRITICAL", "WARNING")]

    # 渲染
    log_container = st.container(height=350)

    if not logs:
        log_container.caption("暂无日志")

    log_html_parts = []
    for entry in logs[-200:]:  # 最多显示最近 200 条
        color = LEVEL_COLOR_MAP.get(entry["level"], "#FFFFFF")
        tid_str = f"<span style='color:#FFD54F'>[{entry['trace_id']}]</span>" if entry['trace_id'] else ""
        log_html_parts.append(
            f"<div style='font-family:monospace;font-size:12px;line-height:1.5;"
            f"border-bottom:1px solid #333;padding:2px 0'>"
            f"<span style='color:#888'>{entry['timestamp']}</span> "
            f"<span style='color:#81C784'>[{entry['tag']}]</span> "
            f"<span style='color:{color};font-weight:bold'>[{entry['level']}]</span> "
            f"{tid_str} "
            f"<span style='color:#DDD'>{entry['message']}</span>"
            f"</div>"
        )

    log_container.markdown(
        "<div style='max-height:350px;overflow-y:auto'>" + "\n".join(log_html_parts) + "</div>",
        unsafe_allow_html=True,
    )

    # 导出按钮
    if logs:
        log_text = "\n".join([
            f"{e['timestamp']} [{e['tag']}] [{e['level']}] [{e['trace_id']}] {e['message']}"
            for e in logs
        ])
        st.download_button(
            "📥 导出日志", log_text,
            file_name=f"agent_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
            key="log_export_btn",
        )


# ============================================================
# 侧边栏
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.title("🔬 科研灵感助手")

        # ---- 知识库管理 ----
        st.header("📚 知识库管理")

        uploaded_file = st.file_uploader(
            "上传论文（PDF/TXT）",
            type=["pdf", "txt"],
            key="kb_uploader",
            help="支持上传 PDF 或 TXT 格式的学术论文",
        )
        if uploaded_file is not None:
            file_key = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.file_id}"
            if st.session_state.get("_last_processed_file") != file_key:
                st.session_state["_last_processed_file"] = file_key
                logger.info("收到上传文件: %s (%.1f KB)", uploaded_file.name, len(uploaded_file.getvalue() or b'') / 1024)
                with st.status("正在处理文档...", expanded=True) as status:
                    success, msg = add_document_to_kb(uploaded_file, status_container=status)
                    if success:
                        status.update(label="处理完成！", state="complete")
                    else:
                        status.update(label="处理失败", state="error")
                st.session_state.upload_status = (success, msg)
                st.session_state.kb_stats = get_kb_stats()
                st.rerun()

        if st.session_state.upload_status:
            success, msg = st.session_state.upload_status
            if success:
                st.success(msg)
            else:
                st.warning(msg)
            st.session_state.upload_status = None

        kb_stats = get_kb_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("文档数", kb_stats["doc_count"])
        with col2:
            st.metric("片段数", kb_stats["chunk_count"])

        st.divider()

        # ---- 会话管理 ----
        st.header("💬 会话管理")

        if st.button("➕ 新建会话", use_container_width=True, key="new_session_btn"):
            new_id = _create_new_session()
            st.session_state.current_session_id = new_id
            st.session_state.messages = []
            st.session_state.tool_logs = []
            st.rerun()

        if st.session_state.agent:
            sessions = st.session_state.agent.get_session_list()
            for s in sessions:
                col_title, col_del = st.columns([4, 1])
                with col_title:
                    is_active = s["id"] == st.session_state.current_session_id
                    label = f"{'🔵 ' if is_active else ''}{s['title']} ({s['message_count']}条)"
                    if st.button(label, key=f"session_{s['id']}", use_container_width=True):
                        _switch_to_session(s["id"])
                        st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"del_{s['id']}", help="删除会话"):
                        if st.session_state.agent:
                            st.session_state.agent.delete_session(s["id"])
                        if s["id"] == st.session_state.current_session_id:
                            st.session_state.current_session_id = None
                            st.session_state.messages = []
                        st.rerun()

        st.divider()

        # ---- 工具调用日志（精简版） ----
        st.header("🔧 工具调用")
        if st.session_state.tool_logs:
            for log in st.session_state.tool_logs[-5:]:
                with st.expander(
                    f"{'✅' if log['success'] else '❌'} {log['tool_name']} ({log['elapsed_seconds']}s)",
                    expanded=False,
                ):
                    st.caption(f"时间: {log['timestamp']}")
                    st.caption(f"输入: {log['input_args']}")
                    st.caption(f"输出: {log['output_summary']}")
        else:
            st.caption("暂无工具调用")

        # ---- 系统日志 ----
        render_log_panel()


def _create_new_session() -> int:
    if st.session_state.agent:
        st.session_state.agent.close()
    agent = ResearchAgent(session_id=None)
    st.session_state.agent = agent
    return agent.session_id


def _switch_to_session(session_id: int):
    if st.session_state.agent:
        st.session_state.agent.switch_session(session_id)
    else:
        agent = ResearchAgent(session_id=session_id)
        st.session_state.agent = agent
    st.session_state.current_session_id = session_id
    st.session_state.tool_logs = []
    if st.session_state.agent:
        st.session_state.messages = st.session_state.agent.get_session_messages()


# ============================================================
# 主聊天区域
# ============================================================
def render_chat():
    if st.session_state.current_session_id:
        sessions = st.session_state.agent.get_session_list() if st.session_state.agent else []
        current_title = "新会话"
        for s in sessions:
            if s["id"] == st.session_state.current_session_id:
                current_title = s["title"]
                break
        st.caption(f"当前会话: {current_title}")

    if not st.session_state.messages:
        st.markdown("""
        ### 👋 欢迎使用科研灵感助手！

        我可以帮助你：
        - 🔍 **检索知识库**：上传论文后，向我提问，我会从论文中检索相关知识
        - 💡 **分析科研想法**：告诉我你的研究方向或想法，我帮你分析可行性和创新性
        - 📝 **讨论技术方案**：基于知识库中的论文，和你一起讨论具体的技术方案

        **开始使用：**
        1. 先在左侧上传一些相关论文到知识库
        2. 然后在这里向我提问吧！
        """)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("输入你的科研问题或想法..."):
        if st.session_state.agent is None:
            agent = ResearchAgent(session_id=st.session_state.current_session_id)
            st.session_state.agent = agent
            st.session_state.current_session_id = agent.session_id

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    result = st.session_state.agent.chat(prompt)
                except Exception as e:
                    logger.error("Agent.chat 异常: %s", e, exc_info=True)
                    result = {
                        "response": f"抱歉，处理请求时出错: {str(e)}",
                        "tool_calls": [],
                        "retrieved_chunks": [],
                        "middleware_logs": [],
                    }

            response_text = result.get("response", "")
            retrieved_chunks = result.get("retrieved_chunks", [])
            tool_calls = result.get("tool_calls", [])

            st.markdown(response_text)

            if retrieved_chunks:
                with st.expander(f"📚 检索片段 ({len(retrieved_chunks)} 条)", expanded=False):
                    for chunk in retrieved_chunks:
                        st.markdown(f"""
                        > **片段 {chunk.get('index', '?')}** | "
                        > 文档 {chunk.get('document_id', '?')} | "
                        > 章节: {chunk.get('section', '未知')} | "
                        > 相关度: {chunk.get('relevance_score', 'N/A')}
                        >
                        > {chunk.get('content', '')[:500]}{'...' if len(chunk.get('content', '')) > 500 else ''}
                        """)
                        st.divider()

            if tool_calls:
                with st.expander(f"🔧 工具调用 ({len(tool_calls)} 次)", expanded=False):
                    for tc in tool_calls:
                        st.caption(f"工具: {tc.get('name', '?')}")
                        st.caption(f"参数: {tc.get('arguments', {})}")

        st.session_state.messages.append({"role": "assistant", "content": response_text})

        if st.session_state.agent:
            st.session_state.tool_logs = st.session_state.agent.middleware.get_logs()

        st.rerun()


# ============================================================
# 主入口
# ============================================================
def main():
    initialize_system()
    init_session_state()
    render_sidebar()

    st.title("🔬 科研灵感助手 Agent")
    st.caption("基于通义千问 Qwen3-max · 混合检索（语义 + BM25）· 自动摘要 · 全链路日志追踪")

    render_chat()


if __name__ == "__main__":
    main()
