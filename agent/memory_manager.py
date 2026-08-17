"""
In-memory memory manager.
Stores reflections and past results across iterations within a session.
Uses keyword matching for retrieval (no vector DB needed for this use case).
"""


class MemoryManager:
    """Per-session in-memory store with keyword-based retrieval."""

    def __init__(self, max_memories: int = 20):
        self._store: list[dict] = []
        self.max_memories = max_memories

    # ── Required interface (Milestone 2) ──────────────────────────────────

    def save(self, session_id: str, content: str, metadata: dict) -> None:
        """Persist a memory entry (session_id stored in metadata)."""
        self.add({"session_id": session_id, "content": content, **metadata})

    def recall(self, session_id: str, query: str, top_k: int = 3) -> list:
        """Return up to top_k memories for this session matching the query."""
        session_memories = [m for m in self._store if m.get("session_id") == session_id]
        return self._keyword_match(session_memories, query)[:top_k]

    def clear(self, session_id: str) -> None:
        """Remove all memories for a session."""
        self._store = [m for m in self._store if m.get("session_id") != session_id]

    # ── Internal helpers ──────────────────────────────────────────────────

    def add(self, memory: dict) -> None:
        """Append a memory, evicting the oldest if over capacity."""
        self._store.append(memory)
        if len(self._store) > self.max_memories:
            self._store.pop(0)

    def retrieve(self, query=None) -> list:
        """Return all memories, optionally filtered by keyword match."""
        if not query:
            return list(self._store)
        return self._keyword_match(self._store, query)

    def _keyword_match(self, memories: list, query: str) -> list:
        words = query.lower().split()
        matched = []
        for m in memories:
            text = str(m).lower()
            if any(w in text for w in words):
                matched.append(m)
        return matched

    def get_all(self) -> list:
        return list(self._store)
