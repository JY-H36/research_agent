"""
科研灵感助手 Agent — Streamlit 前端入口
"""
import streamlit as st
import os
from datetime import datetime

from config import UPLOAD_DIR, ALLOWED_EXTENSIONS
import config
from utils.helpers import ensure_dir, compute_md5_from_bytes
from utils.logger import setup_logging, get_memory_logs, clear_memory_logs, get_logger
from database.connection import init_db, SessionLocal
from database.models import Document, Chunk
from knowledge_base.document_processor import process_document
from knowledge_base.embedding_service import embed_texts
from knowledge_base.vector_store import add_chunks, get_chunk_count
from knowledge_base.retriever import rebuild_retriever
from knowledge_graph.entity_extractor import extract_and_ingest
from knowledge_graph.paper_extraction_agent import format_extraction_for_display, format_extraction_summary
from knowledge_graph.graph_store import get_graph_stats as get_kg_stats, clear_graph, get_all_papers, get_paper_detail, delete_paper, export_papers_to_excel
from knowledge_graph.visualization import build_pyvis_graph, build_stats_html
from knowledge_graph.entity_normalizer import normalize_all_entities
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
    extraction_display = None  # 前端展示的提取结果

    def update_status(msg):
        logger.info("[KB] %s", msg)
        if status_container is not None:
            status_container.text(msg)

    update_status(f"步骤 1/6: 检查文件格式 + 计算 MD5...")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"不支持的文件格式: {ext}，仅支持 {', '.join(ALLOWED_EXTENSIONS)}", None

    file_md5 = compute_md5_from_bytes(file_bytes)

    db = SessionLocal()
    try:
        existing = db.query(Document).filter(Document.md5_hash == file_md5).first()
        if existing:
            return False, f"该文档已存在于知识库中（文件名: {existing.filename}，入库时间: {existing.created_at}）", None

        update_status(f"步骤 2/6: 保存文件到磁盘...")
        ensure_dir(UPLOAD_DIR)
        file_path = os.path.join(UPLOAD_DIR, f"{file_md5}_{filename}")
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        update_status(f"步骤 3/6: Docling 转换 PDF → Markdown...")
        file_md5_check, chunks, md_text = process_document(file_path)

        if not chunks:
            os.remove(file_path)
            return False, "文档内容为空或无法解析。PDF 可能是扫描版/纯图片型，建议先 OCR 处理。", None

        # 保存 MD 文件（供预览和后续分析）
        md_path = os.path.join(UPLOAD_DIR, f"{file_md5}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {os.path.splitext(filename)[0]}\n\n")
            for c in chunks:
                f.write(c["content"])
                f.write("\n\n---\n\n")
        logger.debug("MD 文件已保存: %s", md_path)

        update_status(f"步骤 4/6: 存入 MySQL ({len(chunks)} 个片段)...")
        doc = Document(
            filename=filename, md5_hash=file_md5,
            file_path=file_path,
            file_size=len(file_bytes), chunk_count=len(chunks),
        )
        db.add(doc)
        db.flush()

        for i, chunk in enumerate(chunks):
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

        chunk_ids = [f"doc{doc.id}_chunk{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": doc.id,
                "section_name": chunk.get("section_name", ""),
                "level": chunk.get("level", 2),
                "chunk_index": i,
                "filename": filename,
            }
            for i, chunk in enumerate(chunks)
        ]

        add_chunks(chunk_ids, chunk_texts, embeddings, metadatas)

        update_status(f"步骤 6/7: 重建 BM25 索引...")
        rebuild_retriever()

        update_status(f"步骤 7/7: Agent 提取论文信息...")
        try:
            paper_id, extraction_result = extract_and_ingest(
                md_text,
                document_id=doc.id,
                filename=filename,
            )
            logger.info("论文实体已入图: paper_id=%s, document_id=%d", paper_id, doc.id)
            # 生成前端展示用的提取结果
            extraction_display = format_extraction_for_display(extraction_result)
            extraction_summary = format_extraction_summary(extraction_result)
            update_status(f"✅ 入库完成！{extraction_summary}")
        except Exception as e:
            logger.warning("实体入图失败（不影响入库）: %s", e)
            extraction_display = None

        section_count = len(set(c["section_name"] for c in chunks))
        return True, f"入库成功！「{filename}」→ {section_count} 个章节, {len(chunks)} 个片段", extraction_display

    except Exception as e:
        db.rollback()
        logger.error("文档入库失败: %s — %s", filename, e, exc_info=True)
        return False, f"入库失败: {str(e)}", None
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
                    success, msg, extraction_display = add_document_to_kb(uploaded_file, status_container=status)
                    if success:
                        status.update(label="处理完成！", state="complete")
                    else:
                        status.update(label="处理失败", state="error")
                st.session_state.upload_status = (success, msg)
                st.session_state.extraction_display = extraction_display  # 保存提取结果
                st.session_state.kb_stats = get_kb_stats()
                st.rerun()

        if st.session_state.upload_status:
            success, msg = st.session_state.upload_status
            if success:
                st.success(msg)
                # 展示论文信息提取结果
                if st.session_state.get("extraction_display"):
                    with st.expander("📋 查看提取的论文信息", expanded=True):
                        st.markdown(st.session_state.extraction_display)
                    st.session_state.extraction_display = None  # 只展示一次
            else:
                st.warning(msg)
            st.session_state.upload_status = None

        kb_stats = get_kb_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("文档数", kb_stats["doc_count"])
        with col2:
            st.metric("片段数", kb_stats["chunk_count"])

        # ---- 知识图谱统计 ----
        try:
            kg_stats = get_kg_stats()
        except Exception:
            kg_stats = None
        if kg_stats and kg_stats.get("papers", 0) > 0:
            st.divider()
            st.header("🕸️ 知识图谱")
            kg_col1, kg_col2, kg_col3, kg_col4 = st.columns(4)
            with kg_col1:
                st.metric("论文", kg_stats.get("papers", 0))
            with kg_col2:
                methods = kg_stats.get("methods", {})
                fe_count = methods.get("feature_extractor", 0) if isinstance(methods, dict) else 0
                st.metric("特征提取器", fe_count)
            with kg_col3:
                net_count = methods.get("network_architecture", 0) if isinstance(methods, dict) else 0
                st.metric("网络框架", net_count)
            with kg_col4:
                loss_count = methods.get("loss_function", 0) if isinstance(methods, dict) else 0
                st.metric("损失函数", loss_count)
            kg_col5, kg_col6, kg_col7 = st.columns(3)
            with kg_col5:
                st.metric("数据集", kg_stats.get("datasets", 0))
            with kg_col6:
                total_rels = sum(kg_stats.get("relationships", {}).values())
                st.metric("关系", total_rels)

            # ---- 查看知识图谱 ----
            # ---- 实体归一化 ----
            col_norm1, col_norm2 = st.columns(2)
            with col_norm1:
                llm_label = "🤖 LLM归一化" if config.KG_NORMALIZATION_USE_LLM else "🔗 归一化实体"
                llm_help = ("LLM Agent 理解语义 + 描述 + 引用文献综合判断，合并相似实体"
                           if config.KG_NORMALIZATION_USE_LLM
                           else "合并相似名称的方法/数据集/任务（如 wav2vec2→wav2vec 2.0）")
                if st.button(llm_label, use_container_width=True, key="normalize_kg_btn",
                             help=llm_help):
                    with st.spinner("正在归一化..."):
                        norm_result = normalize_all_entities(dry_run=False)
                        total = (norm_result.get("methods", {}).get("merged", 0) +
                                 norm_result.get("datasets", {}).get("merged", 0) +
                                 norm_result.get("tasks", {}).get("merged", 0))
                        if total > 0:
                            # 检查是否使用了 LLM
                            llm_verified = any(
                                norm_result.get(t, {}).get("llm_verified")
                                for t in ["methods", "datasets", "tasks"]
                            )
                            extra = " (LLM验证)" if llm_verified else ""
                            st.toast(f"✅ 已合并 {total} 个重复实体{extra}", icon="🔗")
                        else:
                            st.toast("未发现可合并的重复实体", icon="✅")
            with col_norm2:
                if st.button("🔍 分析重复", use_container_width=True, key="analyze_dup_btn",
                             help="仅分析不合并，查看哪些实体名称相似"):
                    with st.spinner("正在分析..."):
                        analysis = normalize_all_entities(dry_run=True)
                        groups = analysis.get("details", [])
                        if groups:
                            st.toast(f"发现 {len(groups)} 组可合并实体，详见日志", icon="🔍")
                        else:
                            st.toast("未发现重复实体", icon="✅")

            with st.expander("🕸️ 查看知识图谱", expanded=False):
                st.caption("交互式网络图，可拖拽/缩放节点")
                # 实体类型过滤
                show_types = st.multiselect(
                    "显示实体类型",
                    options=["paper", "author", "method", "dataset", "task", "metric", "venue"],
                    default=["paper", "method", "dataset"],
                    format_func=lambda x: {"paper": "📄 论文", "author": "👤 作者",
                                           "method": "🔧 方法", "dataset": "📊 数据集",
                                           "task": "🎯 任务", "metric": "📏 指标",
                                           "venue": "🏛️ 发表地"}.get(x, x),
                    key="kg_viz_types",
                )
                html = build_pyvis_graph(height="420px", filter_types=show_types or None)
                if html:
                    st.components.v1.html(html, height=450, scrolling=False)
                else:
                    st.caption("图谱为空，请先上传论文")

        # ---- 清空知识库 ----
        st.divider()
        with st.expander("⚠️ 清空知识库", expanded=False):
            st.caption("此操作将删除所有已上传的论文、向量索引和知识图谱数据，不可恢复。")
            if st.button("🗑️ 确认清空全部知识库", type="secondary", use_container_width=True,
                         key="clear_kb_btn"):
                _clear_knowledge_base()

        st.divider()

        # ---- 会话管理 ----
        st.header("💬 会话管理")

        # 始终从 DB 查询会话列表（不依赖 agent 是否已初始化）
        sessions = _list_sessions_from_db()

        if st.button("➕ 新建会话", use_container_width=True, key="new_session_btn"):
            new_id = _create_new_session()
            st.session_state.current_session_id = new_id
            st.session_state.messages = []
            st.session_state.tool_logs = []
            st.rerun()

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
                    _delete_session(s["id"])
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


def _list_sessions_from_db() -> list:
    """直接从 DB 查询会话列表，不依赖 agent"""
    from database.connection import SessionLocal
    from database.models import Session
    from sqlalchemy import desc
    db = SessionLocal()
    try:
        sessions = db.query(Session).order_by(desc(Session.updated_at)).all()
        return [{"id": s.id, "title": s.title, "message_count": len(s.messages),
                 "created_at": s.created_at, "updated_at": s.updated_at} for s in sessions]
    finally:
        db.close()


def _clear_knowledge_base():
    """清空全部知识库：MySQL 文档/分块表 + Chroma 向量库 + 知识图谱 + BM25 索引 + 上传文件"""
    import shutil
    from database.connection import SessionLocal
    from database.models import Document as DBDocument, Chunk as DBChunk
    from knowledge_base.vector_store import clear_collection
    from knowledge_base.retriever import rebuild_retriever
    from knowledge_graph.graph_store import clear_graph

    db = SessionLocal()
    try:
        # 1. 清空 Chroma 向量库
        clear_collection()
        logger.info("Chroma 向量库已清空")

        # 2. 清空 MySQL documents + chunks 表
        db.query(DBChunk).delete()
        db.query(DBDocument).delete()
        db.commit()
        logger.info("MySQL documents/chunks 表已清空")

        # 3. 清空知识图谱
        clear_graph()
        logger.info("知识图谱已清空")

        # 4. 清空上传文件目录
        if os.path.exists(UPLOAD_DIR):
            for f in os.listdir(UPLOAD_DIR):
                fpath = os.path.join(UPLOAD_DIR, f)
                if os.path.isfile(fpath):
                    os.remove(fpath)
            logger.info("上传文件目录已清空")

        # 5. 重建空 BM25 索引
        rebuild_retriever()
        logger.info("BM25 索引已重建（空）")

        st.session_state.kb_stats = {"doc_count": 0, "chunk_count": 0}
        st.toast("✅ 知识库已清空", icon="🗑️")
    except Exception as e:
        db.rollback()
        logger.error("清空知识库失败: %s", e, exc_info=True)
        st.toast(f"❌ 清空失败: {e}", icon="❌")
    finally:
        db.close()


def _delete_session(session_id: int):
    """删除会话"""
    from database.connection import SessionLocal
    from database.models import Session as DBSession
    db = SessionLocal()
    try:
        s = db.query(DBSession).filter(DBSession.id == session_id).first()
        if s:
            db.delete(s)
            db.commit()
    finally:
        db.close()
    if session_id == st.session_state.current_session_id:
        st.session_state.current_session_id = None
        st.session_state.messages = []


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
            for p in msg.get("papers", []):
                _render_paper_card(p)

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
                        "papers": [],
                        "middleware_logs": [],
                    }

            response_text = result.get("response", "")
            retrieved_chunks = result.get("retrieved_chunks", [])
            tool_calls = result.get("tool_calls", [])
            papers = result.get("papers", [])

            if response_text:
                st.markdown(response_text)

            # 论文卡片
            if papers:
                st.divider()
                st.markdown(f"### 📄 检索到的论文 ({len(papers)} 篇)")
                for p in papers:
                    _render_paper_card(p)

            if retrieved_chunks:
                with st.expander(f"📚 知识库检索片段 ({len(retrieved_chunks)} 条)", expanded=False):
                    for chunk in retrieved_chunks:
                        st.markdown(f"""
                        > **片段 {chunk.get('index', '?')}** |
                        > 文档 {chunk.get('document_id', '?')} |
                        > 章节: {chunk.get('section', '未知')} |
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

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "papers": papers,
        })

        if st.session_state.agent:
            st.session_state.tool_logs = st.session_state.agent.middleware.get_logs()

        st.rerun()


# ============================================================
# 主入口
# ============================================================
def _render_paper_card(paper: dict):
    """渲染单篇论文的资料卡片"""
    title = paper.get("title", "Unknown Title")
    authors = ", ".join(paper.get("authors", [])[:8])
    if len(paper.get("authors", [])) > 8:
        authors += f" 等 ({len(paper['authors'])} 人)"
    year = paper.get("year", "")
    venue = paper.get("venue", "")
    citations = paper.get("citation_count", 0)
    abstract = paper.get("abstract", "")
    abstract_cn = paper.get("abstract_cn", "")
    pdf_url = paper.get("pdf_url", "")
    paper_url = paper.get("url", "")
    arxiv_id = paper.get("arxiv_id", "")
    source_label = paper.get("source", "")

    expander_title = f"**{title}**"
    if year:
        expander_title += f" ({year})"
    if venue:
        expander_title += f" — *{venue}*"
    if source_label:
        expander_title += f" `{source_label}`"

    with st.expander(expander_title, expanded=False):
        st.caption(f"👤 {authors}")
        if citations:
            st.caption(f"📊 引用量: {citations}")

        links = []
        if paper_url:
            links.append(f"[链接]({paper_url})")
        if arxiv_id:
            links.append(f"[arXiv:{arxiv_id}](https://arxiv.org/abs/{arxiv_id})")
        if pdf_url:
            links.append(f"[PDF]({pdf_url})")
        if links:
            st.caption(" | ".join(links))

        if abstract:
            with st.container():
                st.caption("📝 摘要 (EN)")
                st.markdown(f"> {abstract[:800]}{'...' if len(abstract) > 800 else ''}")

        if abstract_cn:
            with st.container():
                st.caption("📝 摘要 (CN)")
                st.markdown(f"> {abstract_cn[:800]}")

        if pdf_url:
            st.session_state.setdefault("_dl_btn_counter", 0)
            st.session_state["_dl_btn_counter"] += 1
            dl_key = f"dl_{st.session_state['_dl_btn_counter']}"
            if st.button("⬇ 下载 PDF 并加入知识库", key=dl_key, use_container_width=True):
                _download_and_ingest_paper(paper)


def _download_and_ingest_paper(paper: dict):
    """下载论文 PDF 并加入知识库"""
    pdf_url = paper.get("pdf_url", "")
    title = paper.get("title", "unknown")[:80].replace("/", "_").replace(":", "_")
    if not pdf_url:
        st.error("该论文无可下载的 PDF 链接")
        return

    status_placeholder = st.empty()
    try:
        from knowledge_base.paper_search import download_paper_pdf
        from utils.helpers import compute_md5_from_file

        status_placeholder.info("正在下载 PDF...")
        ensure_dir(UPLOAD_DIR)
        save_path = os.path.join(UPLOAD_DIR, f"_download_{title}.pdf")
        if not download_paper_pdf(pdf_url, save_path):
            status_placeholder.error("PDF 下载失败")
            return

        file_md5 = compute_md5_from_file(save_path)
        db = SessionLocal()
        existing = db.query(Document).filter(Document.md5_hash == file_md5).first()
        db.close()
        if existing:
            os.remove(save_path)
            status_placeholder.warning(f"该论文已在知识库中（{existing.filename}）")
            return

        status_placeholder.info("正在 Docling 解析 + 向量化...")
        _, chunks, _ = process_document(save_path)
        if not chunks:
            os.remove(save_path)
            status_placeholder.error("文档解析失败")
            return

        db = SessionLocal()
        try:
            doc = Document(filename=f"{title}.pdf", md5_hash=file_md5,
                           file_path=save_path, file_size=os.path.getsize(save_path),
                           chunk_count=len(chunks))
            db.add(doc)
            db.flush()
            for i, chunk in enumerate(chunks):
                level_prefix = f"[h{chunk.get('level', 2)}]"
                full_section = f"{level_prefix} {chunk.get('section_name', '')}".strip()[:500]
                db.add(Chunk(document_id=doc.id, chunk_index=i,
                             section_name=full_section, content=chunk["content"]))
            db.commit()
        finally:
            db.close()

        chunk_texts = [c["content"] for c in chunks]
        embeddings = embed_texts(chunk_texts)
        chunk_ids = [f"doc{doc.id}_chunk{i}" for i in range(len(chunks))]
        metadatas = [{"document_id": doc.id, "section_name": c.get("section_name", ""),
                       "level": c.get("level", 2), "chunk_index": i,
                       "filename": f"{title}.pdf"} for i, c in enumerate(chunks)]
        add_chunks(chunk_ids, chunk_texts, embeddings, metadatas)
        rebuild_retriever()
        st.session_state.kb_stats = get_kb_stats()
        status_placeholder.success(f"✅ 论文已加入知识库：「{title}」({len(chunks)} 个片段)")
    except Exception as e:
        status_placeholder.error(f"下载/入库失败: {str(e)}")
        logger.error("论文下载入库异常: %s", e, exc_info=True)


def main():
    initialize_system()
    init_session_state()
    render_sidebar()

    tab_chat, tab_kb = st.tabs(["💬 对话", "📚 知识库管理"])

    with tab_chat:
        st.title("🔬 科研灵感助手 Agent")
        st.caption("基于通义千问 Qwen3-max · 混合检索（语义 + BM25）· 自动摘要 · 全链路日志追踪")
        render_chat()

    with tab_kb:
        render_kb_management()


# ============================================================
# 知识库管理页面
# ============================================================
def render_kb_management():
    """知识库管理页面：浏览论文、查看详情、删除论文"""
    st.title("📚 知识库管理")
    st.caption("查看、浏览和删除知识库中的论文")

    # ---- 操作按钮 ----
    col_refresh, col_export, col_count = st.columns([1, 1, 4])
    with col_refresh:
        if st.button("🔄 刷新", key="kb_mgmt_refresh", use_container_width=True):
            st.rerun()
    with col_export:
        if st.button("📥 导出Excel", key="kb_export_excel", use_container_width=True,
                     help="一键导出所有论文信息到Excel文件（含标题、作者、关键词、方法、数据集等）"):
            with st.spinner("正在生成Excel..."):
                excel_bytes, result = export_papers_to_excel()
            if excel_bytes:
                st.download_button(
                    label="💾 点击下载Excel",
                    data=excel_bytes,
                    file_name=result,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="kb_download_excel",
                )
                st.success(f"✅ 已生成: {result}")
            else:
                st.warning(result)

    # ---- 获取论文列表 ----
    try:
        papers = get_all_papers()
    except Exception as e:
        st.error(f"获取论文列表失败: {e}")
        return

    if not papers:
        st.info("知识库中暂无论文。请先在侧边栏上传论文 PDF。")
        return

    st.caption(f"共 **{len(papers)}** 篇论文")

    # ---- 搜索过滤 ----
    search_query = st.text_input(
        "🔍 搜索论文（按标题、关键词、年份）",
        placeholder="输入关键词筛选...",
        key="kb_search",
    )

    if search_query:
        query_lower = search_query.lower()
        papers = [
            p for p in papers
            if query_lower in p.get("title", "").lower()
            or query_lower in p.get("venue_name", "").lower()
            or any(query_lower in kw.lower() for kw in p.get("keywords", []))
            or (p.get("year") and query_lower in str(p["year"]))
        ]

    st.divider()

    # ---- 论文列表 ----
    if not papers:
        st.info("没有匹配的论文")
        return

    for i, p in enumerate(papers):
        paper_id = p["paper_id"]
        title = p.get("title", "Untitled")[:200]
        year = p.get("year", "—")
        venue = p.get("venue_name", "")
        keywords = p.get("keywords", [])
        method_count = p.get("method_count", 0)
        dataset_count = p.get("dataset_count", 0)
        task_count = p.get("task_count", 0)
        author_count = p.get("author_count", 0)
        created_at = p.get("created_at", "")

        # 实体统计标签
        stat_parts = []
        if author_count:
            stat_parts.append(f"👤{author_count}")
        if method_count:
            stat_parts.append(f"🔧{method_count}")
        if dataset_count:
            stat_parts.append(f"📊{dataset_count}")
        if task_count:
            stat_parts.append(f"🎯{task_count}")

        stat_str = " · ".join(stat_parts) if stat_parts else ""

        # 标题行
        expander_title = f"**{i+1}. {title}**"
        if year and year != "—":
            expander_title += f" ({year})"
        if venue:
            expander_title += f" — *{venue[:60]}*"

        with st.expander(expander_title, expanded=False):
            # 操作按钮行
            col_info, col_del = st.columns([5, 1])
            with col_info:
                if stat_str:
                    st.caption(stat_str)
                if keywords:
                    kw_str = " · ".join([f"`{kw}`" for kw in keywords[:10]])
                    st.caption(f"🏷️ {kw_str}")
                if created_at:
                    st.caption(f"📅 入库时间: {created_at}")

            with col_del:
                delete_key = f"delete_paper_{paper_id[:12]}"
                if st.button("🗑️ 删除", key=delete_key, type="secondary",
                             help="删除该论文及其所有关联数据", use_container_width=True):
                    _handle_delete_paper(paper_id, title)

            # 查看详情按钮
            detail_key = f"detail_btn_{paper_id[:12]}"
            if st.button("📋 查看完整信息", key=detail_key):
                with st.spinner("加载论文详情..."):
                    _show_paper_detail(paper_id)


def _show_paper_detail(paper_id: str):
    """展示单篇论文的完整详情"""
    try:
        detail = get_paper_detail(paper_id)
    except Exception as e:
        st.error(f"获取论文详情失败: {e}")
        return

    if not detail:
        st.warning("论文详情加载失败")
        return

    # 基本信息
    st.markdown(f"### {detail.get('title', 'Untitled')}")
    year = detail.get("year", "")
    venue = detail.get("venue_name", "")
    if year or venue:
        st.caption(f"{year or ''} {'—' if year and venue else ''} *{venue or ''}*")

    doi = detail.get("doi", "")
    arxiv_id = detail.get("arxiv_id", "")
    if doi:
        st.caption(f"DOI: [{doi}](https://doi.org/{doi})")
    if arxiv_id:
        st.caption(f"arXiv: [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})")
    st.caption(f"📅 入库: {detail.get('created_at', '')}")

    # 关键词
    keywords = detail.get("keywords", [])
    if keywords:
        kw_str = " · ".join([f"`{kw}`" for kw in keywords])
        st.caption(f"🏷️ 关键词: {kw_str}")

    # 摘要
    abstract = detail.get("abstract", "")
    abstract_cn = detail.get("abstract_cn", "")
    if abstract:
        with st.expander("📝 英文摘要", expanded=False):
            st.markdown(abstract[:2000])
    if abstract_cn:
        with st.expander("📝 中文摘要", expanded=False):
            st.markdown(abstract_cn[:2000])

    # 作者
    authors = detail.get("authors", [])
    if authors:
        st.markdown("#### 👤 作者")
        if isinstance(authors[0], dict):
            author_lines = []
            for a in authors:
                parts = [f"**{a.get('name', '?')}**"]
                if a.get("affiliation"):
                    parts.append(f"({a['affiliation'][:100]})")
                if a.get("is_corresponding"):
                    parts.append("✉️ *通讯作者*")
                author_lines.append(" · ".join(parts))
            st.markdown("\n".join(author_lines))
        else:
            st.markdown(", ".join(authors))

    # 提出的方法（存在Paper属性上）
    method_name = detail.get("method_name", "")
    method_summary = detail.get("method_summary", "")
    if method_name or method_summary:
        st.markdown("#### 🏗️ 提出的方法")
        if method_name:
            st.markdown(f"**{method_name}**")
        if method_summary:
            st.markdown(f"> {method_summary}")
    # 方法组件（按特征提取器 / 网络框架 / 损失函数 三组显示）
    methods = detail.get("methods", [])
    if methods:
        fes = [m for m in methods if m.get("category") == "feature_extractor" or m.get("role") == "feature_extractor"]
        nets = [m for m in methods if m.get("category") == "network_architecture" or m.get("role") == "network_architecture"]
        losses = [m for m in methods if m.get("category") == "loss_function" or m.get("role") == "loss_function"]

        if fes:
            st.markdown("#### 🎛️ 特征提取器")
            for m in fes:
                v = f" *({m['variant']})*" if m.get("variant") else ""
                desc = m.get("description", "")
                st.markdown(f"- **{m['name']}**{v}")
                if desc: st.caption(f"  > {desc}")

        if nets:
            st.markdown("#### 🧠 网络框架")
            for m in nets:
                desc = m.get("description", "")
                st.markdown(f"- **{m['name']}**")
                if desc: st.caption(f"  > {desc}")

        if losses:
            st.markdown("#### 📉 损失函数")
            for m in losses:
                desc = m.get("description", "")
                st.markdown(f"- **{m['name']}**")
                if desc: st.caption(f"  > {desc}")

    # 数据集
    datasets = detail.get("datasets", [])
    if datasets:
        st.markdown("#### 📊 数据集")
        eval_ds = [d for d in datasets if d["role"] == "eval"]
        train_ds = [d for d in datasets if d["role"] == "train"]
        for d in eval_ds:
            desc = d.get("description", "")
            task = d.get("task", "")
            split = d.get("split", "")
            parts = [f"`评测`"]
            if task:
                parts.append(task)
            if split:
                parts.append(f"split={split}")
            st.markdown(f"- **{d['name']}** {' | '.join(parts)}")
            if desc:
                st.caption(f"  > {desc}")
        for d in train_ds:
            split = d.get("split", "")
            parts = [f"`训练`"]
            if split:
                parts.append(f"split={split}")
            st.markdown(f"- **{d['name']}** {' | '.join(parts)}")

    # 任务
    tasks = detail.get("tasks", [])
    if tasks:
        st.markdown("#### 🎯 研究任务")
        for t in tasks:
            primary = "⭐ " if t.get("is_primary") else ""
            desc = t.get("description", "")
            st.markdown(f"- {primary}**{t['name']}**")
            if desc:
                st.caption(f"  > {desc}")

    # 指标
    metrics = detail.get("metrics", [])
    if metrics:
        st.markdown("#### 📏 评价指标")
        for m in metrics:
            value = m.get("value")
            val_str = f"**{value}**" if value is not None else "—"
            condition = f" ({m['condition']})" if m.get("condition") else ""
            notes = m.get("notes", "")
            st.markdown(f"- {m['name']}: {val_str}{condition}")
            if notes:
                st.caption(f"  > {notes}")

    # 引用统计
    cites = detail.get("cites_count", 0)
    cited_by = detail.get("cited_by_count", 0)
    if cites or cited_by:
        st.caption(f"📖 引用其他论文: {cites} · 被引用: {cited_by}")


def _handle_delete_paper(paper_id: str, title: str):
    """处理论文删除（含二次确认）"""
    # 使用 session_state 管理确认状态
    confirm_key = f"_confirm_delete_{paper_id[:12]}"
    if confirm_key not in st.session_state:
        st.session_state[confirm_key] = False

    if not st.session_state[confirm_key]:
        st.warning(f"⚠️ 确认删除论文「**{title[:120]}**」？")
        st.caption("此操作将删除论文本身、关联的图谱数据、向量索引和上传文件，**不可恢复**。")
        col_y, col_n = st.columns([1, 3])
        with col_y:
            if st.button("✅ 确认删除", key=f"confirm_{paper_id[:12]}",
                        type="primary", use_container_width=True):
                st.session_state[confirm_key] = True
                st.rerun()
        with col_n:
            if st.button("❌ 取消", key=f"cancel_{paper_id[:12]}",
                        use_container_width=True):
                st.session_state.pop(confirm_key, None)
                st.rerun()
    else:
        with st.spinner("正在删除论文..."):
            success, msg = delete_paper(paper_id)
        if success:
            st.success(msg)
            st.session_state.pop(confirm_key, None)
            # 刷新统计
            st.session_state.kb_stats = get_kb_stats()
            # 延迟刷新
            import time
            time.sleep(1)
            st.rerun()
        else:
            st.error(msg)
            st.session_state.pop(confirm_key, None)


if __name__ == "__main__":
    main()
