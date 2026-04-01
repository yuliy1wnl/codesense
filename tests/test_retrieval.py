from backend.retrieval.retriever import HybridRetriever
from backend.retrieval.reranker import Reranker

def test_retrieval():
    retriever = HybridRetriever()
    reranker = Reranker()

    repo_id = "fastapi_test"

    # Test different question types
    questions = [
        "Where is authentication handled?",
        "How do I run this project?",
        "What does the routing function do?",
        "Find bugs in the upload file handling",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print('='*60)

        chunks = retriever.retrieve(question, repo_id)
        top_chunks = reranker.rerank(question, chunks)

        print(f"\nTop answer sources:")
        for c in top_chunks:
            print(f"  {c['file_path']} — {c['name']} "
                  f"(lines {c['start_line']}-{c['end_line']})")

if __name__ == "__main__":
    test_retrieval()