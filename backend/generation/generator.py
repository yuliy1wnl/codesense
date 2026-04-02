import os
from typing import List, Generator
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


SYSTEM_PROMPT = """You are an expert software engineer helping a developer 
understand a codebase. You are given relevant code snippets and documentation 
retrieved from the repository.

Your job:
- Answer the question accurately using ONLY the provided context
- Always cite which file and function your answer comes from
- If the answer isn't in the context, say so honestly
- Keep answers concise and technical
- Format code examples with proper markdown code blocks
- Focus on SOURCE CODE files over translated documentation files
- Prefer files in the root or core directories over docs/ subdirectories

Context format:
Each snippet shows: [file path | function name | lines X-Y]
followed by the actual code or documentation.
"""

def format_chunks_as_context(chunks: List[dict]) -> str:
    """Format reranked chunks into a readable context block for the LLM."""
    if not chunks:
        return "No relevant code found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[{i}] {chunk['file_path']} | "
            f"{chunk['name']} | "
            f"lines {chunk['start_line']}-{chunk['end_line']}"
        )
        context_parts.append(f"{header}\n```\n{chunk['content']}\n```")

    return "\n\n".join(context_parts)


def format_citations(chunks: List[dict]) -> List[dict]:
    """Build a clean citations list from the top chunks."""
    return [
        {
            "file": chunk["file_path"],
            "name": chunk["name"],
            "lines": f"{chunk['start_line']}-{chunk['end_line']}",
            "type": chunk["chunk_type"]
        }
        for chunk in chunks
    ]


class Generator:
    def __init__(self):
        self.llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1
        )
        self.llm_stream = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            streaming=True
        )

    def generate(
        self,
        question: str,
        chunks: List[dict],
        chat_history: List[dict] = None
    ) -> dict:
        """
        Generate an answer from retrieved chunks.
        Returns answer text + citations.
        """
        context = format_chunks_as_context(chunks)
        citations = format_citations(chunks)
        messages = self._build_messages(question, context, chat_history)

        print(f"\nGenerating answer for: '{question}'")
        response = self.llm.invoke(messages)

        return {
            "answer": response.content,
            "citations": citations,
            "question": question
        }

    def generate_stream(
        self,
        question: str,
        chunks: List[dict],
        chat_history: List[dict] = None
    ):
        """
        Stream the answer token by token.
        Yields text chunks as they arrive.
        """
        context = format_chunks_as_context(chunks)
        messages = self._build_messages(question, context, chat_history)

        for token in self.llm_stream.stream(messages):
            yield token.content

    def _build_messages(
        self,
        question: str,
        context: str,
        chat_history: List[dict] = None
    ) -> list:
        """Build the full message list including history."""
        messages = [("system", SYSTEM_PROMPT)]

        # Add chat history for multi-turn conversations
        if chat_history:
            for turn in chat_history[-6:]:  # keep last 3 exchanges
                if turn["role"] == "user":
                    messages.append(("human", turn["content"]))
                else:
                    messages.append(("ai", turn["content"]))

        # Add current question with context
        messages.append((
            "human",
            f"Here is the relevant code context:\n\n{context}\n\n"
            f"Question: {question}"
        ))

        return messages