"""
Agent 核心编排模块
ReAct 模式：用户消息 → LLM 决策（是否调用工具）→ 执行工具 → 生成回复
"""
import time
from typing import Dict, List

from agent.llm_service import chat_completion, extract_tool_calls, extract_content
from agent.tools import TOOL_DEFINITIONS, execute_tool
from agent.middleware import ToolMiddleware
from agent.paper_search_mcp import call_tool as mcp_call_tool
from agent.paper_search_mcp import encode_papers_in_message, decode_papers_from_message
from session.session_manager import SessionManager
from session.summarizer import check_and_summarize
from database.connection import SessionLocal
from utils.helpers import generate_trace_id, estimate_tokens
from utils.logger import get_logger, set_trace_id

logger = get_logger(__name__)


SYSTEM_PROMPT = """你是一个科研灵感助手 Agent，核心能力是**基于知识库中已上传的论文**，帮助研究人员进行深度分析和方案讨论。

## 核心业务（必须严格遵循）

### 1. 知识库论文检索（首要工具）
每次收到科研问题，**第一时间**调用 search_knowledge_base 检索知识库。知识库中的论文是分析的基础。

### 2. 文献综述 / 研究方向分析（核心业务）
当用户要求分析某个研究方向、生成综述、或探讨研究空白时，按以下结构组织回复：

**组织方式**（根据论文内容灵活选择其一或组合）：
- **按细分领域**：将该研究方向拆解为多个子领域，分别介绍每个子领域的研究现状。例如：
  - 部分伪造音频检测 → 伪造内容检测 vs 伪造边界定位
  - 说话人识别 → 近场(near-field) vs 远场(far-field)
  - 音频深度伪造检测 → 全伪造检测 vs 部分伪造检测 vs 跨数据集泛化
- **按研究维度**：从不同角度梳理论文，如常见数据集、主流模型架构、评价指标、损失函数设计、数据增强策略等
- **按技术路线演进**：梳理该领域方法的演变脉络，从经典方法到最新 SOTA

**综述结尾**必须包含：
- **研究空白与趋势**：基于知识库论文的 limitations 和 future work，归纳当前未被充分探索的方向
- **潜在创新点**：指出可以从哪些角度做出差异化贡献

### 3. 引用来源规范（强制执行）
**每次引用知识库中的论文内容时，必须注明来源**，格式如下：
- 论文名称 + 章节：如「*Are audio DeepFake detection models polyglots?*（Introduction 章节）」
- 论文名称 + 作者：如「Zhang et al. 的 *SafeEar: Content Privacy-Preserving Audio Deepfake Detection*」
- **禁止**使用"文档 ID: 5"或"片段 3"这类无意义的标识

知识库检索结果中已包含每篇论文的文件名（即论文标题）和章节名，直接使用即可。

### 4. 联网搜索（辅助工具）
当知识库中确实找不到相关内容、或用户明确要求查找最新发表/未收录的论文时，使用 search_papers_online 联网搜索。联网搜索结果作为知识库的补充，优先级低于知识库已有论文。

### 5. 科研想法评审
当用户提出具体的科研想法时：
1. 先从知识库检索相关工作
2. 分析该想法与现有工作的异同
3. 评估新颖性和可行性
4. 基于知识库中的方法提出改进建议

## 行为准则
- 使用中文回复，专业术语保留英文
- 保持严谨学术态度，所有论断必须有知识库依据
- 知识库没有的信息要诚实说明，不要编造
- 分析要有结构层次（用 Markdown 标题组织），不要平铺文字
"""



class ResearchAgent:
    """科研助手 Agent 主类"""

    def __init__(self, session_id: int = None):
        self.db = SessionLocal()
        self.session_manager = SessionManager(self.db)
        self.middleware = ToolMiddleware()

        if session_id is None:
            self.session_id = self.session_manager.create_session("新会话")
        else:
            self.session_id = session_id

        self._ensure_system_prompt()
        logger.info("Agent 初始化完成 (会话 ID: %d)", self.session_id)

    def _ensure_system_prompt(self):
        msgs = self.session_manager.get_messages(self.session_id, limit=1)
        if not msgs or msgs[0].role.value != "system":
            from database.models import Message, MessageRole
            sys_msg = Message(
                session_id=self.session_id,
                role=MessageRole.SYSTEM,
                content=SYSTEM_PROMPT,
            )
            self.db.add(sys_msg)
            self.db.commit()

    # ============================================================
    # 主对话接口
    # ============================================================
    def chat(self, user_message: str) -> Dict:
        """
        处理用户消息，返回 Agent 回复
        """
        # 生成 trace_id
        trace_id = generate_trace_id()
        set_trace_id(trace_id)
        t_start = time.time()

        logger.info("═══ 新请求开始 ═══")
        logger.debug("会话 ID: %d, 用户消息: %s", self.session_id, user_message[:200])

        # 1. 保存用户消息
        self.session_manager.add_message(self.session_id, "user", user_message)

        # 2. 构建上下文
        context_messages = self.session_manager.build_context(self.session_id)
        msg_count = self.session_manager.count_messages(self.session_id)
        total_est_tokens = estimate_tokens(" ".join([m.get("content", "") or "" for m in context_messages]))
        logger.debug("上下文: %d 条消息, est_tokens~%d, 消息总数=%d",
                    len(context_messages), total_est_tokens, msg_count)

        # 3. ReAct 循环（最多 3 轮 tool call）
        final_response = ""
        all_tool_calls = []
        all_retrieved_chunks = []
        all_papers = []

        try:
            # 第 1 轮：LLM 决策 + 执行工具
            response = chat_completion(messages=list(context_messages), tools=TOOL_DEFINITIONS)
            tool_calls = extract_tool_calls(response)

            if tool_calls:
                logger.info("LLM 决定调用 %d 个工具", len(tool_calls))

                # 执行工具
                tool_results_text = []
                for tc in tool_calls:
                    if tc["name"] == "search_papers_online":
                        executor = mcp_call_tool
                    else:
                        executor = execute_tool

                    result = self.middleware.execute_with_tracking(
                        tool_name=tc["name"],
                        tool_args=tc["arguments"],
                        executor=executor,
                    )
                    all_tool_calls.append(tc)

                    if tc["name"] == "search_knowledge_base" and result.get("chunks"):
                        all_retrieved_chunks.extend(result["chunks"])
                    if tc["name"] == "search_papers_online" and result.get("papers"):
                        all_papers.extend(result["papers"])

                    # 收集工具结果文本
                    result_text = result.get("result", "")
                    if result_text:
                        tool_results_text.append(result_text)

                # 第 2 轮：重建干净上下文（不含 tool_calls 历史），让 LLM 专注生成回复
                # Qwen 看到 tool_calls 历史会继续尝试调用工具 → 死循环
                clean_messages = [
                    m for m in context_messages
                    if m.get("role") != "tool"  # 去除之前的 tool 消息
                ]
                # 将工具结果包装为用户消息，LLM 看到后直接基于它回复
                combined_results = "\n\n".join(tool_results_text)
                clean_messages.append({
                    "role": "user",
                    "content": (
                        f"以下是根据你的问题检索到的信息：\n\n{combined_results}\n\n"
                        f"请基于以上信息回答用户最初的问题。如果信息不足以回答，请如实说明。"
                        f"回复中引用论文时请使用论文名称和章节。"
                    ),
                })

                logger.debug("第 2 轮 LLM 调用 (干净上下文, %d 条消息)", len(clean_messages))
                final_response_obj = chat_completion(messages=clean_messages, tools=None)
                final_response = extract_content(final_response_obj)
                logger.debug("第 2 轮 LLM: len=%d", len(final_response) if final_response else 0)
            else:
                # 无需工具，直接用第一轮回复
                final_response = extract_content(response)
                logger.debug("LLM 直接回复, len=%d", len(final_response) if final_response else 0)

            # 兜底
            if not final_response:
                logger.warning("LLM 未生成文本回复")
                final_response = "抱歉，处理您的请求时遇到问题，请尝试重新表述您的问题。"

        except Exception as e:
            logger.error("Agent 处理异常: %s", e, exc_info=True)
            final_response = f"抱歉，处理您的请求时出现了错误: {str(e)}"

        # 4. 保存助手回复（论文元数据嵌入消息内容，实现持久化）
        if final_response:
            persistent_content = encode_papers_in_message(final_response, all_papers)
            self.session_manager.add_message(self.session_id, "assistant", persistent_content)

        # 5. 检查自动摘要
        try:
            if msg_count >= 18:  # 接近阈值时检查
                logger.debug("消息数 %d, 检查摘要触发条件", msg_count)
            check_and_summarize(self.session_id, self.db)
        except Exception as e:
            logger.error("摘要检查异常: %s", e, exc_info=True)

        total_elapsed = time.time() - t_start
        logger.info("═══ 请求完成, 总耗时 %.1fs, 回复长度 %d 字符, 工具调用 %d 次 ═══",
                   total_elapsed, len(final_response), len(all_tool_calls))

        return {
            "response": final_response,
            "tool_calls": all_tool_calls,
            "retrieved_chunks": all_retrieved_chunks,
            "papers": all_papers,
            "middleware_logs": self.middleware.get_logs(),
        }

    # ============================================================
    # 会话管理
    # ============================================================
    def get_session_messages(self) -> List[Dict]:
        msgs = self.session_manager.get_messages(self.session_id, limit=100)
        result = []
        for m in msgs:
            if m.role.value == "system":
                continue
            clean_content, papers = decode_papers_from_message(m.content)
            entry = {"role": m.role.value, "content": clean_content, "timestamp": m.created_at}
            if papers:
                entry["papers"] = papers
            result.append(entry)
        return result

    def get_session_list(self) -> List[Dict]:
        return self.session_manager.list_sessions()

    def switch_session(self, session_id: int):
        logger.info("切换会话: %d → %d", self.session_id, session_id)
        self.session_id = session_id
        self.middleware.clear_logs()

    def delete_session(self, session_id: int):
        logger.info("删除会话: %d", session_id)
        self.session_manager.delete_session(session_id)

    def close(self):
        if self.db:
            self.db.close()
