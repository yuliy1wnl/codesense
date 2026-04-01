from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChatTurn:
    role: str        # "user" or "assistant"
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    citations: List[dict] = field(default_factory=list)


class ChatMemory:
    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._history: List[ChatTurn] = []

    def add_user_message(self, content: str):
        self._history.append(ChatTurn(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str, citations: List[dict] = None):
        self._history.append(ChatTurn(
            role="assistant",
            content=content,
            citations=citations or []
        ))
        self._trim()

    def get_history(self) -> List[dict]:
        """Return history as a list of dicts for the generator."""
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self._history
        ]

    def get_full_history(self) -> List[ChatTurn]:
        """Return full history including citations."""
        return self._history

    def clear(self):
        self._history = []

    def _trim(self):
        """Keep only the last max_turns messages."""
        if len(self._history) > self.max_turns:
            self._history = self._history[-self.max_turns:]

    def __len__(self):
        return len(self._history)