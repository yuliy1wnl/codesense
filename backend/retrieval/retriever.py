import os
import re
from typing import List
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi
from backend.retrieval.router import QueryRouter, RouteDecision

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 20))

CODE_COLLECTION = "code_chunks"
DOCS_COLLECTION = "docs_chunks"


def tokenize(text: str) -> list:
    return re.findall(r'\b\w+\b', text.lower())


class HybridRetriever:
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        self.router = QueryRouter()

    def retrieve(
        self,
        question: str,
        repo_id: str,
        top_k: int = TOP_K_RETRIEVAL
    ) -> List[dict]:
        decision = self.router.route(question)
        results = []

        if decision.search_code:
            code_results = self._hybrid_search(
                question=question,
                collection=f"{repo_id}_{CODE_COLLECTION}",
                top_k=top_k
            )
            results.extend(code_results)

        if decision.search_docs:
            doc_results = self._hybrid_search(
                question=question,
                collection=f"{repo_id}_{DOCS_COLLECTION}",
                top_k=top_k // 2
            )
            results.extend(doc_results)

        print(f"Retrieved {len(results)} total chunks")
        return results

    def _hybrid_search(
        self,
        question: str,
        collection: str,
        top_k: int
    ) -> List[dict]:

        # --- Vector search ---
        query_vector = self.embeddings.embed_query(question)
        vector_response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        vector_results = vector_response.points

        # --- BM25 keyword search ---
        all_points = self.client.scroll(
            collection_name=collection,
            limit=5000,
            with_payload=True,
            with_vectors=False
        )[0]

        bm25_results = []
        if all_points:
            corpus = [p.payload.get("content", "") for p in all_points]
            tokenized_corpus = [tokenize(doc) for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(tokenize(question))

            # STEP 1: boost scores for chunks whose name matches a query token
            # Must happen BEFORE sorting so boosted chunks rank higher
            query_tokens = set(tokenize(question))
            for i, point in enumerate(all_points):
                chunk_name = point.payload.get("name", "").lower()
                if chunk_name in query_tokens:
                    scores[i] *= 3.0
                    print(f"  [BOOST] {chunk_name} — {point.payload.get('file_path')} boosted to {scores[i]:.3f}")

            # STEP 2: sort AFTER boosting
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

            print(f"Top 3 BM25 results after boost:")
            for i in top_indices[:3]:
                print(f"  [{scores[i]:.3f}] {all_points[i].payload.get('name')} — {all_points[i].payload.get('file_path')}")

            bm25_results = [
                {
                    "content": all_points[i].payload.get("content", ""),
                    "file_path": all_points[i].payload.get("file_path", ""),
                    "chunk_type": all_points[i].payload.get("chunk_type", ""),
                    "name": all_points[i].payload.get("name", ""),
                    "start_line": all_points[i].payload.get("start_line", 0),
                    "end_line": all_points[i].payload.get("end_line", 0),
                    "score": float(scores[i]),
                    "source": "bm25"
                }
                for i in top_indices
                if scores[i] > 0
            ]

        # --- Merge: vector first, then BM25 ---
        vector_dicts = [
            {
                "content": r.payload.get("content", ""),
                "file_path": r.payload.get("file_path", ""),
                "chunk_type": r.payload.get("chunk_type", ""),
                "name": r.payload.get("name", ""),
                "start_line": r.payload.get("start_line", 0),
                "end_line": r.payload.get("end_line", 0),
                "score": r.score,
                "source": "vector"
            }
            for r in vector_results
        ]

        seen = set()
        merged = []
        for chunk in vector_dicts + bm25_results:
            key = (chunk["file_path"], chunk["start_line"])
            if key not in seen:
                seen.add(key)
                merged.append(chunk)

        return merged