from backend.retrieval.retriever import HybridRetriever
from backend.retrieval.reranker import Reranker
from backend.generation.generator import Generator
from backend.generation.memory import ChatMemory

def test_generation():
    retriever = HybridRetriever()
    reranker = Reranker()
    generator = Generator()
    memory = ChatMemory()

    repo_id = "fastapi_test"

    # Simulate a multi-turn conversation
    questions = [
        "What does the test_upload_file function do?",
        "Are there any edge cases handled in that test?",  # follow-up
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print('='*60)

        # Retrieve + rerank
        chunks = retriever.retrieve(question, repo_id)
        top_chunks = reranker.rerank(question, chunks)

        # Generate with memory
        memory.add_user_message(question)
        result = generator.generate(
            question=question,
            chunks=top_chunks,
            chat_history=memory.get_history()
        )

        memory.add_assistant_message(
            result["answer"],
            result["citations"]
        )

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nCitations:")
        for c in result["citations"]:
            print(f"  {c['file']} — {c['name']} (lines {c['lines']})")

    print(f"\nConversation turns in memory: {len(memory)}")

if __name__ == "__main__":
    test_generation()