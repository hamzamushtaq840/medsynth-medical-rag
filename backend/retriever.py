# retriever.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# Config
VECTOR_DB_DIR = "./vector_db"
EMBEDDINGS_DIR = os.path.join(VECTOR_DB_DIR, "models")

# Ensure embeddings folder exists
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Load embeddings from pre-downloaded model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    cache_folder=EMBEDDINGS_DIR
)

# Load Chroma vectorstore from local folder
vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


def retrieve_text(query: str):
    docs = retriever.invoke(query)
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    return texts, metas
