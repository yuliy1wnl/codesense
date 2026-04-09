import json
import math
import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

GROUND_TRUTH_CACHE_PATH = os.getenv(
    "GROUND_TRUTH_CACHE_PATH",
    "backend/evaluation/ground_truth_cache.json"
)


class RAGEvaluator:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            temperature=0
        )
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )

    def _load_cache(self) -> dict:
        if os.path.exists(GROUND_TRUTH_CACHE_PATH):
            with open(GROUND_TRUTH_CACHE_PATH, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache: dict):
        os.makedirs(os.path.dirname(GROUND_TRUTH_CACHE_PATH), exist_ok=True)
        with open(GROUND_TRUTH_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"Ground truths saved to {GROUND_TRUTH_CACHE_PATH}")

    def _get_cache_key(self, repo_id: str, question: str) -> str:
        return f"{repo_id}::{question.strip().lower()}"

    def generate_ground_truths(
        self,
        questions: List[str],
        contexts: List[List[str]],
        repo_id: str = "default"
    ) -> List[str]:
        """
        Generate ground truths using GPT-4o, with caching.
        First run: generates and saves to disk.
        Subsequent runs: loads from cache — scores are now reproducible.
        NOTE: Introduces circularity — same model family generates and judges.
        """
        cache = self._load_cache()
        ground_truths = []
        new_entries = 0

        print("\nLoading/generating ground truths...")
        for i, (question, context_chunks) in enumerate(zip(questions, contexts)):
            cache_key = self._get_cache_key(repo_id, question)

            if cache_key in cache:
                ground_truths.append(cache[cache_key])
                print(f"  Loaded from cache {i+1}/{len(questions)}: {question[:50]}...")
            else:
                context_text = "\n\n".join(context_chunks)
                prompt = f"""You are an expert software engineer.
Given the following code context, write a concise and accurate ground truth answer to the question.
The answer should be factual, based only on the provided context.

Context:
{context_text}

Question: {question}

Ground truth answer (2-4 sentences, technical and precise):"""

                response = self.llm.invoke(prompt)
                gt = response.content.strip()
                ground_truths.append(gt)
                cache[cache_key] = gt
                new_entries += 1
                print(f"  Generated ground truth {i+1}/{len(questions)}: {question[:50]}...")

        if new_entries > 0:
            self._save_cache(cache)
            print(f"Saved {new_entries} new ground truths to cache.")
        else:
            print("All ground truths loaded from cache — scores are reproducible.")

        return ground_truths

    def evaluate_responses(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        repo_id: str = "default"
    ) -> dict:
        print("\nRunning RAGAS evaluation...")

        if not ground_truths:
            print("WARNING: Synthetic ground truths introduce evaluation circularity.")
            ground_truths = self.generate_ground_truths(questions, contexts, repo_id)

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)

        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]

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
            vals = scores.get(key, [])
            if not vals:
                return 0.0
            valid = []
            for v in vals:
                try:
                    f = float(v)
                    if not math.isnan(f):
                        valid.append(f)
                except (TypeError, ValueError):
                    pass
            return round(sum(valid) / len(valid), 3) if valid else 0.0

        summary = {
            "faithfulness":        safe_score("faithfulness"),
            "answer_relevancy":    safe_score("answer_relevancy"),
            "context_precision":   safe_score("context_precision"),
            "context_recall":      safe_score("context_recall"),
            "eval_time_seconds":   round(elapsed, 1),
            "num_samples":         len(questions),
            "ground_truth_source": "synthetic_gpt4o_cached"
        }

        print(f"Evaluation complete in {elapsed:.1f}s")
        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: dict):
        skip = {"eval_time_seconds", "num_samples", "ground_truth_source"}
        print("\n── RAGAS Scores ──────────────────────")
        for metric, score in summary.items():
            if metric in skip:
                continue
            try:
                bar = "█" * int(float(score) * 20)
                print(f"  {metric:<22} {score:.3f}  {bar}")
            except (ValueError, TypeError):
                print(f"  {metric:<22} N/A")
        print(f"\n  Samples evaluated: {summary.get('num_samples', '?')}")
        print(f"  Ground truths:     {summary.get('ground_truth_source', '?')}")
        print(f"  Eval time:         {summary.get('eval_time_seconds', '?')}s")
        print("──────────────────────────────────────")