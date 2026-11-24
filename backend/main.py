from utils.history_helpers import append_message, get_history, get_recent_messages, clear_history
from retriever import retrieve_text
import os
import json
import uuid
import sqlite3
import typing as t
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# config from env
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Updated System Prompt for MedSynth
MEDSYNTH_SYSTEM_PROMPT = """You are MedSynth, an expert AI medical consultant and diagnostic assistant. 
Your goal is to provide accurate, clinical, and helpful medical information.

GUIDELINES:
1. Use the 'Relevant medical context' provided to ground your answer in clinical data.
2. If no context is provided, rely on your extensive medical training to answer the user's query accurately.
3. Maintain a professional, empathetic, and clinical tone.
4. Structure your response clearly (e.g., using bullet points for symptoms or recommendations).
5. Always include a standard medical disclaimer: 'This information is for educational purposes and does not replace professional medical advice.'
6. If the user's query is vague, ask clarifying clinical questions."""

SYSTEM_PROMPT = MEDSYNTH_SYSTEM_PROMPT
HISTORY_WINDOW = int(os.getenv("HISTORY_WINDOW", "6"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

app = FastAPI(title="MedSynth RAG Chat (Medical Consultant)")


@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
@app.head("/health")
def health():
    return {"status": "ok"}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chat_history.db")

if not os.path.exists(DB_PATH):
    open(DB_PATH, "a").close()

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row


def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


_db = get_db_conn()
_db.execute(
    """CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        ts REAL NOT NULL
    )"""
)
_db.commit()


class ChatRequest(BaseModel):
    message: str
    session_id: t.Optional[str] = None
    stream: t.Optional[bool] = True


def build_prompt(session_id: str, user_message: str, retrieved_texts: t.List[str]):
    # Get recent conversation history
    recent = get_recent_messages(session_id, _db, n=HISTORY_WINDOW)

    history_block = ""
    if recent:
        history_lines = []
        for m in recent:
            prefix = "Patient" if m["role"] == "user" else "MedSynth"
            history_lines.append(f"{prefix}: {m['content']}")
        history_block = "\n".join(history_lines)

    # Format retrieved medical context
    context_block = ""
    if retrieved_texts:
        context_block = "\n".join([f"[Medical Knowledge {i + 1}]: {text}" for i, text in enumerate(retrieved_texts)])

    # Constructing a more robust prompt for medical reasoning
    prompt_parts = [
        "### ROLE",
        SYSTEM_PROMPT,
        "",
        "### PREVIOUS CLINICAL CONVERSATION" if history_block else "",
        history_block if history_block else "",
        "",
        "### RELEVANT MEDICAL KNOWLEDGE (RAG)" if context_block else "",
        context_block if context_block else "No specific database extracts found. Proceed with general medical expertise.",
        "",
        "### CURRENT PATIENT QUERY",
        user_message,
        "",
        "### MEDSYNTH RESPONSE (Clinical & Thorough):"
    ]

    return "\n".join([p for p in prompt_parts if p])


def call_llm_stream(prompt: str):
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    # We pass the persona in the system message and the full context in the user message
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    stream = client.chat.completions.create(model=LLM_MODEL, messages=messages, stream=True)
    for chunk in stream:
        blocks = chunk.choices[0].delta
        if blocks and hasattr(blocks, "content") and blocks.content:
            yield blocks.content


def sse_event(data: str):
    return f"data: {data}\n\n"


@app.post("/chat")
def chat(req: ChatRequest, background: BackgroundTasks):
    session_id = req.session_id or str(uuid.uuid4())
    user_msg = req.message.strip()
    stream = req.stream if req.stream is not None else True

    append_message(session_id, "user", user_msg, _db)

    try:
        # Utilizing the MedQA vector DB
        retrieved_texts, retrieved_meta = retrieve_text(user_msg)
        retrieved_texts = retrieved_texts[:RETRIEVAL_K]
    except Exception as e:
        # Log error but don't crash; MedSynth should try to answer anyway
        print(f"Retriever error: {e}")
        retrieved_texts = []

    prompt = build_prompt(session_id, user_msg, retrieved_texts)

    def stream_generator():
        assistant_accum = ""
        try:
            for chunk in call_llm_stream(prompt):
                assistant_accum += chunk
                yield sse_event(json.dumps({"delta": chunk}))
            yield sse_event(json.dumps({"done": True}))
        except Exception as exc:
            yield sse_event(json.dumps({"error": str(exc)}))
        finally:
            if assistant_accum.strip():
                append_message(session_id, "assistant", assistant_accum, _db)

    if stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        full = []
        for chunk in call_llm_stream(prompt):
            full.append(chunk)
        answer = "".join(full)
        append_message(session_id, "assistant", answer, _db)
        return JSONResponse({"session_id": session_id, "answer": answer})


@app.get("/history/{session_id}")
def get_session_history(session_id: str):
    return {"session_id": session_id, "messages": get_history(session_id, _db)}


@app.post("/clear/{session_id}")
def clear_session(session_id: str):
    clear_history(session_id, _db)
    return {"session_id": session_id, "cleared": True}
