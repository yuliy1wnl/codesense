# CodeSense

> Understand any codebase instantly. Paste a GitHub URL, ask questions in plain English, get answers backed by real source code.


## What It Does

CodeSense is a production-grade RAG (Retrieval-Augmented Generation) system that lets you chat with any GitHub repository. Instead of blindly splitting code into chunks, it parses the actual AST (Abstract Syntax Tree) of each file — so every chunk is a meaningful function or class, not a random slice of text.

Ask questions like:
- *"How does routing work in this project?"*
- *"Where is authentication handled?"*
- *"What does the APIRouter class do?"*
- *"How do I run this project?"*
- *"Find potential bugs in the upload handler"*

Every answer comes with exact source citations — file name, function name, and line numbers.

---

## Architecture

```
GitHub URL
    ↓
Repo Cloner (GitPython)
    ↓
File Walker (language detection)
    ↓
AST Chunker (tree-sitter — splits by functions & classes)
    ↓
Dual Indexing
├── Code Index (Qdrant)
└── Docs Index (Qdrant)
    ↓
Query Router (LangGraph — detects intent)
    ↓
Hybrid Retriever (vector search + BM25 keyword search)
    ↓
Re-ranker (cross-encoder/ms-marco-MiniLM-L-6-v2)
    ↓
LLM (Llama 3.2 via Ollama)
    ↓
Answer + Source Citations
```

### Key Design Decisions

**AST-based chunking over fixed-size splitting**
Most RAG tutorials split code every 500 characters. CodeSense uses tree-sitter to parse the actual structure of each file — every chunk is a complete function or class. This means retrieved chunks are always syntactically meaningful and self-contained.

**Hybrid retrieval**
Vector search alone misses exact keyword matches (function names, variable names). BM25 alone misses semantic similarity. Combining both gives significantly better retrieval across different question types.

**Cross-encoder reranking**
Vector similarity scores aren't reliable enough for final ranking. A cross-encoder reads the question and each chunk together — producing much more accurate relevance scores. We retrieve 20 candidates and rerank to the top 5.

**Query intent routing**
Different questions need different retrieval strategies. "How do I run this?" should search docs, not source code. "Where is auth handled?" should search code only. The router detects intent and routes accordingly — reducing noise in retrieved chunks.

**Dual indexes**
Code files and documentation are indexed separately. This lets the router search only the relevant index for each question type, improving both speed and accuracy.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Repo ingestion | GitPython |
| Code parsing | tree-sitter (Python, JS, TS, Java, Go) |
| Vector database | Qdrant |
| Embeddings | nomic-embed-text via Ollama |
| Orchestration | LangGraph |
| Hybrid search | Qdrant + rank-bm25 |
| Reranking | sentence-transformers cross-encoder |
| LLM | Llama 3.2 via Ollama |
| Evaluation | RAGAS |
| Backend | FastAPI |
| Frontend | Next.js + Tailwind CSS |

**Fully local — no API keys required. Runs entirely on your machine.**

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker Desktop
- Ollama
- 16GB RAM recommended (8GB minimum)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/codesense
cd codesense
```

### 2. Install Ollama and pull models

```bash
brew install ollama
ollama serve &
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. Start Qdrant

```bash
docker-compose up -d
```

Verify at `http://localhost:6333/dashboard`

### 4. Set up Python environment

```bash
python3 -m venv venv
source venv/bin/activate
echo 'export PYTHONPATH=$PYTHONPATH:.' >> venv/bin/activate
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Start the backend

```bash
uvicorn backend.api.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`

---

## Usage

1. Paste any public GitHub URL into the index field
2. Wait for indexing to complete (2-10 minutes depending on repo size)
3. Ask questions in the chat

### API Endpoints

```bash
# Index a repository
POST /ingest
{"github_url": "https://github.com/owner/repo"}

# Ask a question
POST /query
{"repo_id": "repo", "question": "How does routing work?"}

# Stream a response
POST /query/stream
{"repo_id": "repo", "question": "Explain the architecture"}

# Run evaluation
POST /evaluate
{"repo_id": "repo", "questions": ["..."]}

# List indexed repos
GET /repos
```

---

## Evaluation

CodeSense includes a RAGAS evaluation pipeline to measure retrieval and answer quality:

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "fastapi",
    "questions": [
      "How does routing work?",
      "How do I run this project?",
      "What does APIRouter do?"
    ]
  }'
```

Metrics measured:
- **Faithfulness** — is the answer grounded in the retrieved code?
- **Answer relevancy** — does the answer address the question?
- **Context precision** — were the right chunks retrieved?
- **Context recall** — was all necessary context retrieved?

---

## Project Structure

```
codesense/
├── backend/
│   ├── ingestion/
│   │   ├── cloner.py       # GitHub repo cloner
│   │   ├── walker.py       # File walker + language detection
│   │   ├── chunker.py      # AST-based chunker (tree-sitter)
│   │   └── embedder.py     # Embed + store in Qdrant
│   ├── retrieval/
│   │   ├── router.py       # LangGraph query intent router
│   │   ├── retriever.py    # Hybrid search (vector + BM25)
│   │   └── reranker.py     # Cross-encoder reranker
│   ├── generation/
│   │   ├── generator.py    # LLM answer generation
│   │   └── memory.py       # Multi-turn chat memory
│   ├── evaluation/
│   │   └── evaluator.py    # RAGAS evaluation
│   └── api/
│       └── main.py         # FastAPI app
├── frontend/               # Next.js chat UI
├── tests/                  # Integration tests
├── docker-compose.yml      # Qdrant
└── requirements.txt
```

---

## Roadmap

- [x] Phase 1 — Ingestion pipeline (AST chunking, dual indexing)
- [x] Phase 2 — Retrieval engine (hybrid search, reranking, query routing)
- [x] Phase 3 — Generation & memory (streaming, multi-turn chat)
- [x] Phase 4 — Evaluation dashboard (RAGAS metrics)
- [x] Phase 5 — Next.js frontend
- [ ] Deployment (Docker + cloud)
- [ ] Support for private repositories
- [ ] VS Code extension

---

## License

MIT
