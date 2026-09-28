"""完成版の設定。資料・モデル・索引はすべてMac内に置く。"""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
STATE = Path(os.environ.get("RAG_STATE_DIR", PROJECT / ".rag"))
DEFAULTS = {
    "folder": str(Path.home() / "Documents"),
    "embedding_model": "qwen3-embedding:4b",
    "chat_model": "qwen3.5:4b",
    "read_images": True,
    "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
    "candidate_limit": 100,
    "rerank_limit": 20,
    "min_similarity": 0.45,
}
