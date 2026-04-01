# Codebase RAG

> Chat with any GitHub repository using AI.

A production-grade RAG system that lets you ask natural language questions
about any codebase and get accurate answers with source citations.

## Architecture

- **Ingestion**: GitPython + tree-sitter AST chunking
- **Vector DB**: Qdrant
- **Embeddings**: nomic-embed-text (local, via Ollama)
- **Retrieval**: Hybrid search (vector + BM25) + cross-encoder reranking
- **LLM**: Llama 3.1 8B (local, via Ollama)
- **Backend**: FastAPI
- **Frontend**: Next.js (coming soon)

## Setup

### Prerequisites
- Python 3.11+
- Docker
- Ollama

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/codebase-rag
cd codebase-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run
```bash
# Start Qdrant
docker-compose up -d

# Start Ollama
ollama serve

# Start backend
uvicorn backend.api.main:app --reload
```

## Progress
- [x] Phase 1 — Ingestion pipeline
- [x] Phase 2 — Retrieval engine
- [ ] Phase 3 — Generation & memory
- [ ] Phase 4 — Evaluation dashboard
- [ ] Phase 5 — Frontend & deployment