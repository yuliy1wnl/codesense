import os
from typing import List
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 5))

class Reranker:
    def __init__(self):
        print("Loading reranker model...")
        # Small, fast cross-encoder that runs well on Apple Silicon
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        print("Reranker ready.")

    def rerank(
        self,
        question: str,
        chunks: List[dict],
        top_k: int = TOP_K_RERANK
    ) -> List[dict]:
        """
        Score each chunk against the question and return top_k.
        The cross-encoder reads both the question and chunk together
        — much more accurate than vector similarity alone.
        """
        if not chunks:
            return []

        # Build (question, chunk_content) pairs
        pairs = [(question, chunk["content"]) for chunk in chunks]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(
            chunks,
            key=lambda x: x["rerank_score"],
            reverse=True
        )[:top_k]

        print(f"Reranked {len(chunks)} chunks → kept top {len(reranked)}")
        for r in reranked:
            print(f"  [{r['rerank_score']:.3f}] {r['name']} — {r['file_path']}")

        return reranked