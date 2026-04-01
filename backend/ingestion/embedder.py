import os
from typing import List
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, OptimizersConfigDiff
)
from langchain_ollama import OllamaEmbeddings
from backend.ingestion.chunker import Chunk

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# nomic-embed-text produces 768-dimensional vectors
VECTOR_SIZE = 768

# Two separate collections — one for code, one for docs
CODE_COLLECTION = "code_chunks"
DOCS_COLLECTION = "docs_chunks"


class Embedder:
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )

    def setup_collections(self, repo_id: str):
        """
        Create Qdrant collections for a repo.
        Uses repo_id as a prefix so multiple repos can coexist.
        """
        for collection_name in [
            f"{repo_id}_{CODE_COLLECTION}",
            f"{repo_id}_{DOCS_COLLECTION}"
        ]:
            # Delete if already exists (re-indexing)
            existing = [c.name for c in self.client.get_collections().collections]
            if collection_name in existing:
                self.client.delete_collection(collection_name)
                print(f"Deleted existing collection: {collection_name}")

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=0  # index immediately
                )
            )
            print(f"Created collection: {collection_name}")

    def embed_chunks(self, chunks: List[Chunk], repo_id: str, batch_size: int = 32):
        """
        Embed all chunks and store in Qdrant.
        Separates code and doc chunks into different collections.
        """
        code_chunks = [c for c in chunks if c.chunk_type != "doc"]
        doc_chunks  = [c for c in chunks if c.chunk_type == "doc"]

        print(f"Embedding {len(code_chunks)} code chunks "
              f"and {len(doc_chunks)} doc chunks...")

        self._embed_and_store(
            code_chunks,
            collection_name=f"{repo_id}_{CODE_COLLECTION}",
            batch_size=batch_size
        )
        self._embed_and_store(
            doc_chunks,
            collection_name=f"{repo_id}_{DOCS_COLLECTION}",
            batch_size=batch_size
        )

        print("Embedding complete!")

    def _embed_and_store(
        self,
        chunks: List[Chunk],
        collection_name: str,
        batch_size: int
    ):
        """Embed a list of chunks in batches and upsert into Qdrant."""
        if not chunks:
            return

        total = len(chunks)
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]

            # Build texts to embed — include file path and name for context
            MAX_CHARS = 2000  # Ollama has a max token limit, so we truncate long chunks
            texts = [
                (
                    f"File: {c.file_path}\n"
                    f"Name: {c.name}\n"
                    f"Type: {c.chunk_type}\n\n"
                    f"{c.content}"
                )[:MAX_CHARS]
                for c in batch
            ]

            # Get embeddings from Ollama
            vectors = self.embeddings.embed_documents(texts)

            # Build Qdrant points
            points = [
                PointStruct(
                    id=i + j,  # simple integer ID
                    vector=vector,
                    payload={
                        "content": chunk.content,
                        "file_path": chunk.file_path,
                        "language": chunk.language,
                        "chunk_type": chunk.chunk_type,
                        "name": chunk.name,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        **chunk.metadata
                    }
                )
                for j, (chunk, vector) in enumerate(zip(batch, vectors))
            ]

            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            print(f"  Stored {min(i + batch_size, total)}/{total} chunks "
                  f"in {collection_name}")

    def get_collection_info(self, repo_id: str):
        """Print stats about stored collections."""
        for suffix in [CODE_COLLECTION, DOCS_COLLECTION]:
            name = f"{repo_id}_{suffix}"
            try:
                info = self.client.get_collection(name)
                print(f"{name}: {info.points_count} points")
            except Exception:
                print(f"{name}: not found")