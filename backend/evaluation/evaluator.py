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

# --- Judge LLM is GPT-4o, NOT your local Ollama ---
# Your RAG pipeline still runs locally (llama3.1 + nomic-embed-text)
# Only the RAGAS evaluation judge uses OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JUDGE_LLM_MODEL = os.getenv("JUDGE_LLM_MODEL", "gpt-4o")
JUDGE_EMBEDDING_MODEL = os.getenv("JUDGE_EMBEDDING_MODEL", "text-embedding-3-small")


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

    def generate_ground_truths(
        self,
        questions: List[str],
        contexts: List[List[str]]
    ) -> List[str]:
        """
        Synthetically generate ground truths using GPT-4o.

        NOTE: This introduces circularity — the same model family
        generates and judges. Scores will be optimistic. This is
        acceptable for a portfolio project but not for production evals.
        Always disclose this limitation when presenting results.
        """
        print("\nGenerating synthetic ground truths with GPT-4o...")
        ground_truths = []

        for i, (question, context_chunks) in enumerate(zip(questions, contexts)):
            context_text = "\n\n".join(context_chunks)
            prompt = f"""You are an expert software engineer.
Given the following code context, write a concise and accurate ground truth answer to the question.
The answer should be factual, based only on the provided context.

Context:
{context_text}

Question: {question}

Ground truth answer (2-4 sentences, technical and precise):"""

            response = self.llm.invoke(prompt)
            ground_truths.append(response.content.strip())
            print(f"  Generated ground truth {i+1}/{len(questions)}")

        return ground_truths

    def evaluate_responses(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> dict:
        """
        Run RAGAS evaluation on a set of Q&A pairs.

        Metrics:
        - faithfulness:       is the answer grounded in the retrieved context?
        - answer_relevancy:   does the answer address the question?
        - context_precision:  are the retrieved chunks relevant? (needs ground truth)
        - context_recall:     did we retrieve everything needed? (needs ground truth)

        If ground_truths is None, they are generated synthetically via GPT-4o.
        """
        print("\nRunning RAGAS evaluation...")

        # Auto-generate ground truths if not provided
        if not ground_truths:
            print("No ground truths provided — generating synthetically.")
            print("WARNING: Synthetic ground truths introduce evaluation circularity.")
            ground_truths = self.generate_ground_truths(questions, contexts)

        # Build RAGAS dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)

        # All 4 metrics — we always have ground truths now
        # BUG FIX: was metrics.append([...]) which nested a list inside a list
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]

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
            """
            Average scores across all samples.
            BUG FIX: original code only read index [0], silently
            dropping all rows except the first when evaluating multiple questions.
            """
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
            "ground_truth_source": "synthetic_gpt4o"
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