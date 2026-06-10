"""
预下载 Cross-Encoder 重排序模型到本地
运行一次后，reranker 会直接从本地缓存加载，不再联网下载
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RERANKER_MODEL

# 确保使用 HF 镜像（国内加速）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print(f"正在下载模型: {RERANKER_MODEL}")
print(f"HF 镜像: {os.environ.get('HF_ENDPOINT', '(默认)')}")
print(f"模型将缓存到: {os.path.expanduser('~/.cache/huggingface/hub/')}")
print("下载约 1.3GB，请耐心等待...")
print()

from sentence_transformers import CrossEncoder

model = CrossEncoder(RERANKER_MODEL, max_length=512)
print()
print(f"✅ 模型下载完成: {RERANKER_MODEL}")
print("此后 reranker 将直接从本地缓存加载，无需联网。")
