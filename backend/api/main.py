import os
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.ingestion.cloner import RepoCloner
from backend.ingestion.walker import FileWalker
from backend.ingestion.chunker import ASTChunker
from backend.ingestion.embedder import Embedder
from backend.retrieval.retriever import HybridRetriever
from backend.retrieval.reranker import Reranker
from backend.generation.generator import Generator
from backend.generation.memory import ChatMemory
from backend.evaluation.evaluator import RAGEvaluator

load_dotenv()

app = FastAPI(
    title="CodeSense API",
    description="Chat with any GitHub repository",
    version="1.0.0"
)

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize all components once at startup
cloner   = RepoCloner()
walker   = FileWalker()
chunker  = ASTChunker()
embedder = Embedder()
retriever = HybridRetriever()
reranker  = Reranker()
generator = Generator()
evaluator = RAGEvaluator()

# In-memory session store: repo_id → ChatMemory
sessions: dict[str, ChatMemory] = {}


# ─── Request / Response Models ────────────────────────────────────────────────

class IngestRequest(BaseModel):
    github_url: str

class IngestResponse(BaseModel):
    repo_id: str
    total_chunks: int
    code_chunks: int
    doc_chunks: int
    message: str

class QueryRequest(BaseModel):
    repo_id: str
    question: str
    stream: bool = False

class QueryResponse(BaseModel):
    answer: str
    citations: list
    question: str

class EvalRequest(BaseModel):
    repo_id: str
    questions: list[str]
    ground_truths: Optional[list[str]] = None
EvalRequest.model_rebuild()

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "CodeSense"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Clone a GitHub repo, chunk it with AST, embed it, store in Qdrant.
    This is the first endpoint to call before asking questions.
    """
    try:
        # Run blocking IO in thread pool so we don't block the event loop
        loop = asyncio.get_event_loop()

        # 1. Clone
        print(f"\nIngesting: {request.github_url}")
        repo_path = await loop.run_in_executor(
            None, cloner.clone, request.github_url
        )

        # 2. Walk files
        files = await loop.run_in_executor(None, walker.walk, repo_path)

        # 3. Chunk with AST
        chunks = await loop.run_in_executor(
            None, chunker.chunk_many, files
        )

        # 4. Build repo_id from URL
        repo_id = request.github_url.rstrip("/").split("/")[-1].lower()
        repo_id = repo_id.replace(".git", "")

        # 5. Set up Qdrant collections and embed
        embedder.setup_collections(repo_id)
        await loop.run_in_executor(
            None,
            lambda: embedder.embed_chunks(chunks, repo_id)
        )

        # 6. Clean up cloned repo from disk
        cloner.cleanup(repo_path)

        # 7. Create a fresh memory session for this repo
        sessions[repo_id] = ChatMemory()

        code_chunks = len([c for c in chunks if c.chunk_type != "doc"])
        doc_chunks  = len([c for c in chunks if c.chunk_type == "doc"])

        return IngestResponse(
            repo_id=repo_id,
            total_chunks=len(chunks),
            code_chunks=code_chunks,
            doc_chunks=doc_chunks,
            message=f"Successfully indexed {repo_id}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question about an already-ingested repo.
    Returns answer + citations.
    """
    repo_id = request.repo_id

    # Check repo has been indexed
    try:
        embedder.get_collection_info(repo_id)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"Repo '{repo_id}' not found. Please ingest it first."
        )

    # Get or create memory session
    if repo_id not in sessions:
        sessions[repo_id] = ChatMemory()
    memory = sessions[repo_id]

    loop = asyncio.get_event_loop()

    # Retrieve + rerank
    chunks = await loop.run_in_executor(
        None,
        lambda: retriever.retrieve(request.question, repo_id)
    )
    top_chunks = await loop.run_in_executor(
        None,
        lambda: reranker.rerank(request.question, chunks)
    )

    # Add user message to memory
    memory.add_user_message(request.question)

    # Generate answer
    result = await loop.run_in_executor(
        None,
        lambda: generator.generate(
            question=request.question,
            chunks=top_chunks,
            chat_history=memory.get_history()
        )
    )

    # Store assistant response in memory
    memory.add_assistant_message(result["answer"], result["citations"])

    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        question=result["question"]
    )


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Same as /query but streams the response token by token.
    """
    repo_id = request.repo_id

    if repo_id not in sessions:
        sessions[repo_id] = ChatMemory()
    memory = sessions[repo_id]

    loop = asyncio.get_event_loop()

    chunks = await loop.run_in_executor(
        None,
        lambda: retriever.retrieve(request.question, repo_id)
    )
    top_chunks = await loop.run_in_executor(
        None,
        lambda: reranker.rerank(request.question, chunks)
    )

    memory.add_user_message(request.question)

    def stream_tokens():
        full_response = ""
        for token in generator.generate_stream(
            question=request.question,
            chunks=top_chunks,
            chat_history=memory.get_history()
        ):
            full_response += token
            yield token
        memory.add_assistant_message(full_response)

    return StreamingResponse(
        stream_tokens(),
        media_type="text/plain"
    )


@app.delete("/session/{repo_id}")
def clear_session(repo_id: str):
    """Clear chat memory for a repo."""
    if repo_id in sessions:
        sessions[repo_id].clear()
    return {"message": f"Session cleared for {repo_id}"}


@app.get("/repos")
def list_repos():
    """List all indexed repos."""
    collections = embedder.client.get_collections().collections
    repos = set()
    for c in collections:
        # Strip the _code_chunks / _docs_chunks suffix
        name = c.name.replace("_code_chunks", "").replace("_docs_chunks", "")
        repos.add(name)
    return {"repos": list(repos)}
@app.post("/evaluate")
async def evaluate_rag(body: EvalRequest):
    if body.repo_id not in sessions:
        sessions[body.repo_id] = ChatMemory()

    loop = asyncio.get_event_loop()
    questions = body.questions
    answers = []
    contexts = []

    for question in questions:
        chunks = await loop.run_in_executor(
            None,
            lambda q=question: retriever.retrieve(q, body.repo_id)
        )
        top_chunks = await loop.run_in_executor(
            None,
            lambda c=chunks, q=question: reranker.rerank(q, c)
        )
        result = await loop.run_in_executor(
            None,
            lambda c=top_chunks, q=question: generator.generate(
                question=q,
                chunks=c
            )
        )
        answers.append(result["answer"])
        contexts.append([c["content"] for c in top_chunks])

    scores = await loop.run_in_executor(
        None,
        lambda: evaluator.evaluate_responses(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=body.ground_truths
        )
    )

    return {
        "repo_id": body.repo_id,
        "num_questions": len(questions),
        "scores": scores,
        "questions_evaluated": questions
    }