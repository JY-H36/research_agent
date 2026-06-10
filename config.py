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
# 知识图谱配置
# ============================================================
KG_ENABLE_LLM_EXTRACTION = True   # 是否启用 LLM 实体提取（上传论文后自动提取方法/数据集/指标）
KG_LLM_EXTRACTION_ASYNC = False   # LLM 提取是否异步（v2.0 先同步）
KG_PAPER_EXTRACTION_MAX_CHARS = 50000  # 论文提取 Agent 最大文本长度（利用 Qwen3-max 长上下文）
KG_NORMALIZATION_USE_LLM = True        # 是否使用 LLM 做实体归一化（替代纯模糊匹配）
KG_RESOLVER_USE_LLM = True             # 是否使用 LLM 做入库时实体消歧（L3 语义判断）

# ============================================================
# LLM 参数
# ============================================================
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2048
