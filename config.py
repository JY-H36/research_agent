"""
全局配置模块
所有可配置参数集中管理，支持环境变量覆盖
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Hugging Face 镜像（Docling 下载模型用）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ============================================================
# DashScope API 配置
# ============================================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
LLM_MODEL = "qwen3-max"
EMBEDDING_MODEL = "text-embedding-v4"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ============================================================
# MySQL 数据库配置
# ============================================================
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "020306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "research_assistant")

# ============================================================
# Chroma 向量库配置
# ============================================================
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
CHROMA_COLLECTION_NAME = "research_papers"

# ============================================================
# 文档分块配置
# ============================================================
MAX_CHUNK_SIZE = 1000      # 每个 chunk 最大字符数
CHUNK_OVERLAP = 100        # chunk 之间的重叠字符数

# ============================================================
# 检索配置
# ============================================================
RETRIEVAL_TOP_K = 5                # 最终返回的 top-k 结果数
RRF_K = 60                         # Reciprocal Rank Fusion 常数
QUERY_VARIANTS = 5                 # 查询改写生成的变体数量
CANDIDATE_POOL_SIZE = 15           # 每个变体检索的候选数（扩大候选池）
RERANKER_MODEL = "BAAI/bge-reranker-large"  # Cross-Encoder 重排序模型

# ============================================================
# 会话配置
# ============================================================
MAX_CONTEXT_ROUNDS = 10     # 触发自动摘要的对话轮数
MAX_CONTEXT_MESSAGES = 20   # 触发自动摘要的消息条数 (轮数 × 2)

# ============================================================
# 上传文件配置
# ============================================================
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

# ============================================================
# LLM 参数
# ============================================================
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2048
