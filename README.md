# CodeSense

> Understand any codebase instantly. Paste a GitHub URL, ask questions in plain English, get answers backed by real source code.


## What It Does

CodeSense is a production-grade RAG (Retrieval-Augmented Generation) system that lets you chat with any GitHub repository. Instead of blindly splitting code into chunks, it parses the actual AST (Abstract Syntax Tree) of each file so every chunk is a meaningful function or class, not a random slice of text.

Ask questions like:
- *"How does routing work in this project?"*
- *"Where is authentication handled?"*
- *"What does the APIRouter class do?"*
- *"How do I run this project?"*
- *"Find potential bugs in the upload handler"*

Every answer comes with exact source citations: file name, function name, and line numbers.

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
Query Router (detects intent — code vs docs vs both)
    ↓
Hybrid Retriever (vector search + BM25 keyword search)
    ↓
Re-ranker (cross-encoder/ms-marco-MiniLM-L-6-v2)
    ↓
LLM (Llama 3.1 via Ollama — fully local)
    ↓
Answer + Source Citations
```

### Key Design Decisions

**AST-based chunking over fixed-size splitting**
Most RAG tutorials split code every 500 characters. CodeSense uses tree-sitter to parse the actual structure of each file every chunk is a complete function or class. This means retrieved chunks are always syntactically meaningful and self-contained. Benchmarked against naive text splitting on the FastAPI repository: 97.7% of AST chunks are semantically named (functions/classes) vs 0% with naive splitting, with average chunk size of 8.9 lines vs 33.7 lines.

**Hybrid retrieval with name-boosting**
Vector search alone misses exact keyword matches (function names, class names). BM25 alone misses semantic similarity. Combining both gives significantly better retrieval across different question types. BM25 uses proper tokenization via regex word extraction naive `.split()` causes punctuation to attach to tokens (e.g. `depends:` ≠ `depends`), breaking exact keyword matching. Additionally, chunks whose name exactly matches a query token receive a 3x BM25 score boost, surfacing definition chunks over usage chunks.

**Full corpus BM25 with paginated scroll**
BM25 requires scoring against the full corpus, not just vector search candidates. The retriever scrolls all indexed chunks and scores them with BM25, then merges with vector results. Scroll limit must exceed collection size a limit smaller than the collection silently truncates the corpus and causes BM25 to miss chunks beyond the cutoff.

**Cross-encoder reranking with chunk-type awareness**
Vector similarity scores aren't reliable enough for final ranking. A cross-encoder reads the question and each chunk together producing much more accurate relevance scores. We retrieve 20 candidates and rerank to the top 5. Code chunks receive a score bonus over doc chunks to prevent documentation from crowding out source code for implementation questions.

**Query intent routing**
Different questions need different retrieval strategies. "How do I run this?" should search docs, not source code. "Where is auth handled?" should search code only. The router detects intent and routes accordingly reducing noise in retrieved chunks.

**Dual indexes**
Code files and documentation are indexed separately. This lets the router search only the relevant index for each question type, improving both speed and accuracy.

---

## Evaluation

CodeSense is evaluated using RAGAS with GPT-4o as the judge LLM. The RAG pipeline itself runs fully locally only the evaluation judge uses an external API.

Evaluated on 10 questions against the FastAPI repository:

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | **0.889** | Are answers grounded in retrieved context? |
| Answer Relevancy | **0.434** | Do answers directly address the question? |
| Context Precision | **0.608** | Are retrieved chunks relevant? |
| Context Recall | **0.933** | Did retrieval find everything needed? |

**Notes on these scores:**

- Faithfulness (0.861) and context recall (0.900) are strong the retrieval pipeline surfaces the right code and the generator stays grounded in it.
- Answer relevancy (0.412) is the weakest metric and is constrained by the generator model size. Llama 3.1 8b tends to over-explain rather than answer directly. Improving this would require a larger local model or a hosted generator.
- Ground truths were generated synthetically using GPT-4o, which introduces evaluation circularity the same model family generates and judges. Scores are directionally valid but optimistic. Human-written ground truths would give more reliable numbers.
- Context precision dropped after enabling code-chunk boosting a deliberate tradeoff that improved faithfulness at the cost of occasionally surfacing less precise code chunks for conceptual questions.
- Ground truths are cached after first generation to ensure reproducible scores across runs. Regenerating ground truths each run introduced variance that made it impossible to isolate the effect of code changes on scores.

To run evaluation:

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "fastapi",
    "questions": ["How does routing work?", "Where is authentication handled?"]
  }'
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Repo ingestion | GitPython |
| Code parsing | tree-sitter (Python, JS, TS, Java, Go) |
| Vector database | Qdrant |
| Embeddings | nomic-embed-text via Ollama |
| Hybrid search | Qdrant + rank-bm25 |
| Reranking | sentence-transformers cross-encoder |
| LLM (generator) | Llama 3.1 via Ollama — fully local |
| LLM (evaluator) | GPT-4o via Azure OpenAI |
| Evaluation | RAGAS |
| Backend | FastAPI |
| Frontend | Next.js + Tailwind CSS |

**The RAG pipeline runs fully locally. An OpenAI API key is only required to run RAGAS evaluation.**

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker Desktop
- Ollama
- 16GB RAM recommended (8GB minimum)
- Azure OpenAI API key (evaluation only)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/codesense
cd codesense
```

### 2. Install Ollama and pull models

```bash
brew install ollama
ollama serve &
ollama pull llama3.1
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

### 5. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1
EMBEDDING_MODEL=nomic-embed-text
QDRANT_HOST=localhost
QDRANT_PORT=6333
TOP_K_RETRIEVAL=20
TOP_K_RERANK=5

# Required only for RAGAS evaluation
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

### 6. Start the backend

```bash
uvicorn backend.api.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### 7. Start the frontend

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
│   │   ├── router.py       # Query intent router
│   │   ├── retriever.py    # Hybrid search (vector + BM25)
│   │   └── reranker.py     # Cross-encoder reranker
│   ├── generation/
│   │   ├── generator.py    # LLM answer generation
│   │   └── memory.py       # Multi-turn chat memory
│   ├── evaluation/
│   │   └── evaluator.py    # RAGAS evaluation pipeline
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
- [x] Phase 4 — Evaluation pipeline (RAGAS metrics, GPT-4o judge)
- [x] Phase 5 — Next.js frontend
- [ ] Deployment (Docker + cloud)
- [ ] Support for private repositories
- [ ] VS Code extension

---

## License

MIT
