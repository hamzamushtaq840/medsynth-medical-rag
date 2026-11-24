from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import os

VECTOR_DB_DIR = "./vector_db"
HF_CACHE = "./hf_cache"

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
)

# Vector store
vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Cross-encoder reranker
reranker = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-base",
    model_kwargs={"device": "cpu"},
)


def retrieve_text(query: str, top_k: int = 5):
    docs = retriever.invoke(query)
    if not docs:
        return [], []

    # Prepare pairs for scoring
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.score(pairs)  # returns a list of scores

    # Sort docs by score descending
    ranked_docs = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]

    return [d.page_content for d in ranked_docs[:top_k]], [d.metadata for d in ranked_docs[:top_k]]
