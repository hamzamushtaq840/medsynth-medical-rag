# predownload.py
from huggingface_hub import snapshot_download
import os

# Directories
VECTOR_DB_DIR = "./vector_db"
EMBEDDINGS_DIR = os.path.join(VECTOR_DB_DIR, "models")

# Ensure directories exist
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Download only the specific folder and file from your vector DB repo
snapshot_download(
    repo_id="hamzamushtaq840/vector_db",
    cache_dir=VECTOR_DB_DIR,
    allow_patterns=[
        "28cf8486-17a5-4699-9eee-8cdde885bdd1/*",
        "chroma.sqlite3"
    ]
)
print("Vector DB downloaded successfully.")

# Pre-download sentence-transformers embeddings model
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir=EMBEDDINGS_DIR
)
print("Embeddings model downloaded successfully.")
