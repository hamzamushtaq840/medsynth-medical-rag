# predownload.py
from huggingface_hub import snapshot_download
import os

HF_CACHE = "./hf_cache"
EMBED_CACHE = os.path.join(HF_CACHE, "embeddings")
RERANK_CACHE = os.path.join(HF_CACHE, "rerankers")
VECTOR_DB_DIR = "./vector_db"

os.makedirs(EMBED_CACHE, exist_ok=True)
os.makedirs(RERANK_CACHE, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# ---- Embedding model ----
snapshot_download(
    repo_id="BAAI/bge-small-en-v1.5",
    cache_dir=EMBED_CACHE,
)
print("Embedding model downloaded")

# ---- Cross-encoder reranker ----
snapshot_download(
    repo_id="BAAI/bge-reranker-base",
    cache_dir=RERANK_CACHE,
)
print("Reranker model downloaded")

# ---- Vector DB (your repo) ----
snapshot_download(
    repo_id="hamzamushtaq840/vector_db_bge_small",
    local_dir=VECTOR_DB_DIR,
    allow_patterns=[
        "28cf8486-17a5-4699-9eee-8cdde885bdd1/*",
        "chroma.sqlite3"
    ]
)
print("Vector DB downloaded")
