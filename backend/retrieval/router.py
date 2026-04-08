import os
from typing import Literal
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# All possible query intents
QueryIntent = Literal[
    "explain_code",      # "what does this function do?"
    "find_code",         # "where is authentication handled?"
    "explain_project",   # "what does this project do?"
    "find_bug",          # "find bugs in this file"
    "how_to_run",        # "how do I run this project?"
    "architecture",      # "summarize the architecture"
    "general",            # anything else
    "off_topic",          # not related to the codebase at all
    "explain_feature"    # "how does the search feature work?"
]

class RouteDecision(BaseModel):
    intent: QueryIntent = Field(
        description="The detected intent of the user query"
    )
    search_code: bool = Field(
        description="Whether to search the code index"
    )
    search_docs: bool = Field(
        description="Whether to search the docs index"
    )
    reasoning: str = Field(
        description="Brief reasoning for this routing decision"
    )

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query router for a codebase Q&A system.
Given a user question about a codebase, determine:
1. The intent of the question
2. Whether to search code files
3. Whether to search documentation files

Intent types:
- explain_code: asking to explain a specific function, class, or code snippet
- find_code: asking where something is implemented in the code
- explain_project: asking what the project does overall
- find_bug: asking to find issues or bugs in code
- how_to_run: asking how to set up or run the project
- architecture: asking about the overall structure or design
- general: any other question
- off_topic: question is not related to the codebase at all
- explain_feature: asking how a specific feature or concept works in this codebase

Rules:
- how_to_run and explain_project → search docs only
- explain_project → search both
- explain_feature → search both
- find_code, explain_code, find_bug → search code only
- architecture → search both
- general → search both
- off_topic → search neither

Respond in JSON with fields: intent, search_code, search_docs, reasoning"""),
    ("human", "Question: {question}")
])

class QueryRouter:
    def __init__(self):
        # Use JSON mode for structured output
        self.llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            format="json",
            temperature=0
        )
        self.chain = ROUTER_PROMPT | self.llm

    def route(self, question: str) -> RouteDecision:
        """Detect intent and decide which indexes to search."""
        print(f"\nRouting query: '{question}'")

        response = self.chain.invoke({"question": question})

        # Parse JSON response
        import json
        try:
            data = json.loads(response.content)
            decision = RouteDecision(**data)
        except Exception as e:
            # Fallback: search everything if parsing fails
            print(f"Router parsing failed: {e}, falling back to search all")
            decision = RouteDecision(
                intent="general",
                search_code=True,
                search_docs=True,
                reasoning="Fallback — search everything"
            )

        print(f"  Intent: {decision.intent}")
        print(f"  Search code: {decision.search_code}")
        print(f"  Search docs: {decision.search_docs}")
        print(f"  Reasoning: {decision.reasoning}")

        return decision