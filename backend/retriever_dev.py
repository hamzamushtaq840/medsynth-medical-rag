# retriever.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Config via env (fallbacks can be set here or in .env)
import os
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./vector_db")

# create embeddings and load Chroma
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
# Initialize Chroma vectorstore
vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)

# retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


def retrieve_text(query: str):
    docs = retriever.invoke(query)
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    return texts, metas
