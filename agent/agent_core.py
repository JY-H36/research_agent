"""
Agent 核心编排模块
ReAct 模式：用户消息 → LLM 决策（是否调用工具）→ 执行工具 → 生成回复
"""
import json
import time
from typing import Dict, List

from agent.llm_service import chat_completion, extract_tool_calls, extract_content
from agent.tools import TOOL_DEFINITIONS, execute_tool
from agent.middleware import ToolMiddleware
from session.session_manager import SessionManager
from session.summarizer import check_and_summarize
from database.connection import SessionLocal
from utils.helpers import generate_trace_id, estimate_tokens
from utils.logger import get_logger, set_trace_id

logger = get_logger(__name__)


SYSTEM_PROMPT = """你是一个科研灵感助手 Agent，专门帮助研究人员进行学术论文相关的咨询和科研方案讨论。

你的核心能力：
1. **知识库检索**：使用 search_knowledge_base 工具从知识库中检索相关论文片段。
2. **联网论文搜索**：使用 search_papers_online 工具从 arXiv、Semantic Scholar、OpenAlex 检索最新论文。当用户想查找最新文献、了解研究现状、或问"有没有某某方向的论文"时使用此工具。
3. **论文总结分析**：当 search_papers_online 返回论文列表后，对检索到的论文进行简要的总结分析——归纳共同主题、对比不同方法、指出研究趋势。
4. **科研分析**：基于论文知识，分析用户的科研想法，讨论可行性和创新性。
5. **方案建议**：根据现有论文中的方法和技术，提出可行的科研方案和改进方向。

行为准则：
- 当用户提出科研问题时，主动调用 search_knowledge_base 检索知识库。
- 当用户想要找最新论文或知识库中无相关内容时，使用 search_papers_online 联网搜索。
- 回答时注明信息来源。
- 使用中文回复，专业术语可保留英文。
- 保持严谨的学术态度，不编造不存在的论文或数据。
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

        # 3. ReAct 循环
        final_response = ""
        all_tool_calls = []
        all_retrieved_chunks = []
        all_papers = []

        try:
            response = chat_completion(messages=context_messages, tools=TOOL_DEFINITIONS)
            tool_calls = extract_tool_calls(response)

            if tool_calls:
                logger.info("LLM 决定调用 %d 个工具", len(tool_calls))

                tool_results = []
                for tc in tool_calls:
                    result = self.middleware.execute_with_tracking(
                        tool_name=tc["name"],
                        tool_args=tc["arguments"],
                        executor=execute_tool,
                    )
                    tool_results.append({
                        "tool_call_id": tc["id"],
                        "tool_name": tc["name"],
                        "result": result,
                    })
                    all_tool_calls.append(tc)

                    if tc["name"] == "search_knowledge_base" and result.get("chunks"):
                        all_retrieved_chunks.extend(result["chunks"])

                    if tc["name"] == "search_papers_online" and result.get("papers"):
                        all_papers.extend(result["papers"])

                # 构建包含工具调用结果的消息
                assistant_tool_msg = response["choices"][0]["message"]
                context_messages.append({
                    "role": "assistant",
                    "content": assistant_tool_msg.get("content") or "",
                    "tool_calls": assistant_tool_msg.get("tool_calls", []),
                })

                for tr in tool_results:
                    context_messages.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_call_id"],
                        "content": json.dumps(tr["result"].get("result", ""), ensure_ascii=False),
                    })

                logger.debug("第 2 次 LLM 调用 (携带工具结果, %d 条消息)", len(context_messages))
                final_response_obj = chat_completion(messages=context_messages, tools=None)
                final_response = extract_content(final_response_obj)
            else:
                logger.debug("LLM 无需工具, 直接生成回复")
                final_response = extract_content(response)

        except Exception as e:
            logger.error("Agent 处理异常: %s", e, exc_info=True)
            final_response = f"抱歉，处理您的请求时出现了错误: {str(e)}"

        # 4. 保存助手回复（论文元数据嵌入消息内容，实现持久化）
        if final_response:
            from agent.tools import encode_papers_in_message
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
        from agent.tools import decode_papers_from_message
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
