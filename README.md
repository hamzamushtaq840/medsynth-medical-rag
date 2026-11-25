# MedSynth — Clinical RAG Medical Assistant

MedSynth is an end-to-end Retrieval-Augmented Generation (RAG) system designed for clinical question answering. It grounds **Google Gemini 2.5 Flash** responses in 10,178 USMLE-style clinical cases from the [MedQA (`medical_meadow_medqa`)](https://huggingface.co/datasets/medalpaca/medical_meadow_medqa) dataset using a two-stage hybrid retrieval and cross-encoder reranking pipeline. The full evaluation notebook is available in the `bge_ranker/rag_embeddings.ipynb`.

---

## Architecture & Key Features

- **Two-Stage Hybrid Retrieval (`bge_ranker`)**:
  - **Stage 1 (Candidate Generation)**: Combines dense semantic vector search (`BAAI/bge-small-en-v1.5` in ChromaDB with asymmetric query prefixing) and sparse lexical retrieval (`BM25`) via LangChain's `EnsembleRetriever` (`0.6 / 0.4` weighting) to capture both clinical semantics and exact medical terminology.
  - **Stage 2 (Cross-Encoder Reranking)**: Refines the top candidates down to the top 5 most relevant clinical cases using `BAAI/bge-reranker-large`.
- **Clinical Preprocessing**: Strips multiple-choice option distractors (`A`–`E`) and option letter prefixes prior to indexing so retrieval matches natural clinical vignettes rather than exam formatting.
- **Real-Time Streaming Inference**: Streams grounded responses from Gemini 2.5 Flash over Server-Sent Events (SSE) with persistent SQLite session history.
- **Full-Stack Web Interface**: Built with a FastAPI backend and a Next.js 16 (App Router) + Tailwind CSS frontend.

---

## Tech Stack

### Backend (Python)

- **API Framework**: FastAPI
- **Orchestration**: LangChain (`EnsembleRetriever`, `ContextualCompressionRetriever`)
- **LLM**: Google Gemini 2.5 Flash (via OpenAI-compatible endpoint)
- **Vector Store**: ChromaDB
- **Retrieval & Reranking Models**: `BAAI/bge-small-en-v1.5`, `BAAI/bge-reranker-large`, `rank_bm25`
- **Session Storage**: SQLite

### Frontend (TypeScript / React)

- **Framework**: Next.js 16 (App Router)
- **Styling**: Tailwind CSS
- **State & Streaming**: React Hooks + native Fetch SSE stream reader

---

## Prerequisites

- Python 3.12+
- Node.js 18+
- A valid Gemini API key
- Optional: CUDA-capable GPU for faster embedding and cross-encoder inference (defaults to CPU in Docker)

---

## Getting Started

### 1. Backend Setup

Clone the repository and navigate to the backend directory:

```bash
git clone <your-repo-url>
cd langchain-medical-assistant/backend
```

Install Python dependencies using `uv`:

```bash
uv sync
```

Create a `.env` file in `backend/` with the following configuration:

```env
LLM_API_KEY=your-google-api-key-here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-2.5-flash
SYSTEM_PROMPT=You are a helpful medical assistant. Use the retrieved clinical context to answer concisely and accurately.
CHAT_DB=/app/chat_history.db
HISTORY_WINDOW=6
RETRIEVAL_K=5
EMBED_MODEL=BAAI/bge-small-en-v1.5
EMBED_DEVICE=cpu
```

Launch the backend service via Docker Compose:

```bash
docker compose up
```

The FastAPI server will be available at `http://localhost:7860`.

### 2. Frontend Setup

In a separate terminal, navigate to the frontend directory, install dependencies, and start the development server:

```bash
cd ../frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## API Reference

| Method | Endpoint                     | Description                                                          |
| :----- | :--------------------------- | :------------------------------------------------------------------- |
| `POST` | `/`                          | Health check returning backend service status                        |
| `POST` | `/chat`                      | Generates a standard JSON response grounded in retrieved MedQA cases |
| `POST` | `/chat` (`{"stream": true}`) | Streams the generated response token-by-token via SSE                |
| `POST` | `/history/{session_id}`      | Retrieves conversation history for a given session                   |
| `POST` | `/clear/{session_id}`        | Clears stored conversation history for a given session               |

---

## Retrieval Evaluation

The two-stage retrieval pipeline (`BAAI/bge-small-en-v1.5` + `BM25` + `BAAI/bge-reranker-large`) was evaluated on a benchmark of 20 paraphrased clinical vignettes against the 10,178-document MedQA knowledge base (`k = 5`):

| Metric       |   Score    | Interpretation                                                                                              |
| :----------- | :--------: | :---------------------------------------------------------------------------------------------------------- |
| **Recall@5** | **0.7500** | 75% of paraphrased clinical queries retrieve the exact matching ground-truth case within the top 5 results. |
| **MRR@5**    | **0.6417** | Mean Reciprocal Rank; relevant cases typically appear at rank 1 or rank 2.                                  |
| **nDCG@5**   | **0.6197** | Normalized Discounted Cumulative Gain measuring ranking quality across the top 5 positions.                 |

---

## Disclaimer

This software is developed strictly for academic research and educational purposes. It is not a certified medical device and should not be used for clinical diagnosis, treatment decisions, or as a substitute for professional medical advice.
