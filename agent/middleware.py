"""
Agent Middleware 模块
追踪工具调用和思考过程，通过统一日志系统输出
"""
import time
import logging
from typing import Dict, List, Callable

from utils.logger import get_logger

logger = get_logger(__name__)


class ToolCallRecord:
    """单次工具调用记录"""
    def __init__(self):
        self.tool_name: str = ""
        self.input_args: Dict = {}
        self.output_summary: str = ""
        self.elapsed_seconds: float = 0.0
        self.success: bool = False
        self.timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "tool_name": self.tool_name,
            "input_args": self.input_args,
            "output_summary": self.output_summary,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "success": self.success,
            "timestamp": self.timestamp,
        }


class ToolMiddleware:
    """
    工具调用中间件
    - 通过统一日志系统输出格式化日志
    - 维护 tool_call_logs 列表供 Streamlit 展示
    - 支持自定义 before/after hook
    """

    def __init__(self):
        self.call_logs: List[Dict] = []
        self._before_hooks: List[Callable] = []
        self._after_hooks: List[Callable] = []

    def add_before_hook(self, hook: Callable):
        self._before_hooks.append(hook)

    def add_after_hook(self, hook: Callable):
        self._after_hooks.append(hook)

    def execute_with_tracking(
        self,
        tool_name: str,
        tool_args: Dict,
        executor: Callable[[str, Dict], Dict],
    ) -> Dict:
        """
        包裹工具执行，自动记录日志
        """
        from datetime import datetime
        from agent.tools import get_tool_result_summary

        record = ToolCallRecord()
        record.tool_name = tool_name
        record.input_args = tool_args
        record.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- before hooks ---
        logger.info("🔧 工具调用开始: %s | 参数: %s", tool_name, tool_args)
        for hook in self._before_hooks:
            try:
                hook(tool_name, tool_args)
            except Exception as e:
                logger.error("before hook 执行失败: %s", e)

        # --- 执行 ---
        start_time = time.time()
        try:
            result = executor(tool_name, tool_args)
            record.success = result.get("success", False)
            record.output_summary = get_tool_result_summary(tool_name, result)
        except Exception as e:
            record.success = False
            record.output_summary = f"异常: {str(e)}"
            result = {"success": False, "result": None, "error": str(e)}
            logger.error("工具执行异常: %s — %s", tool_name, e, exc_info=True)

        record.elapsed_seconds = time.time() - start_time

        # --- after hooks ---
        if record.success:
            logger.info("✅ 工具调用完成: %s | 耗时 %.2fs | %s",
                        tool_name, record.elapsed_seconds, record.output_summary)
        else:
            logger.error("❌ 工具调用失败: %s | 耗时 %.2fs | %s",
                         tool_name, record.elapsed_seconds, record.output_summary)

        self.call_logs.append(record.to_dict())

        for hook in self._after_hooks:
            try:
                hook(tool_name, tool_args, result, record.elapsed_seconds)
            except Exception as e:
                logger.error("after hook 执行失败: %s", e)

        return result

    def get_logs(self) -> List[Dict]:
        return self.call_logs

    def clear_logs(self):
        self.call_logs = []
