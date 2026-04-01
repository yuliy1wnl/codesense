import math
import os
import time
from typing import List
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class RAGEvaluator:
    def __init__(self):
        # RAGAS uses its own LLM + embeddings internally
        self.llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0
        )
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )

    def evaluate_responses(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> dict:
        """
        Run RAGAS evaluation on a set of Q&A pairs.

        Metrics:
        - faithfulness: is the answer grounded in the retrieved context?
        - answer_relevancy: does the answer address the question?
        - context_precision: are the retrieved chunks relevant?
        - context_recall: did we retrieve everything needed? (needs ground truth)
        """
        print("\nRunning RAGAS evaluation...")

        # Build RAGAS dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        # Pick metrics based on whether we have ground truths
        metrics = [faithfulness, answer_relevancy, context_precision]
        if ground_truths:
            metrics.append(context_recall)

        # Run evaluation
        start = time.time()
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )
        elapsed = time.time() - start

        scores = results.to_pandas().to_dict(orient="list")

        def safe_score(key):
            val = scores.get(key, [0])[0]
            try:
                f = float(val)
                return round(f, 3) if not math.isnan(f) else 0.0
            except (TypeError, ValueError):
                return 0.0

        summary = {
            "faithfulness":      safe_score("faithfulness"),
            "answer_relevancy":  safe_score("answer_relevancy"),
            "context_precision": safe_score("context_precision"),
            "eval_time_seconds": round(elapsed, 1)
        }
        if ground_truths:
            summary["context_recall"] = safe_score("context_recall")

        print(f"Evaluation complete in {elapsed:.1f}s")
        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: dict):
        print("\n── RAGAS Scores ──────────────────────")
        for metric, score in summary.items():
            if metric == "eval_time_seconds":
                continue
            try:
                bar = "█" * int(float(score) * 20)
                print(f"  {metric:<22} {score:.3f}  {bar}")
            except (ValueError, TypeError):
                print(f"  {metric:<22} N/A")
        print("──────────────────────────────────────")