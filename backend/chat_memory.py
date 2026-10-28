"""
RoboForge Chat Memory Store
Manages persistent, multi-turn conversational context and session state on disk.
Enables true multi-turn conversational memory so the AI assistant never loses context.
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("roboforge.chat_memory")

CHAT_MEMORY_DIR = Path("experiments/chat_memory")


class ChatMemoryStore:
    def __init__(self, storage_dir: Path = CHAT_MEMORY_DIR):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        safe_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_")).strip()
        if not safe_id:
            safe_id = "default_session"
        return self.storage_dir / f"{safe_id}.json"

    def get_history(self, session_id: str = "default_session", limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves persistent message history for a given session."""
        path = self._get_session_path(session_id)
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                messages = json.load(f)
            if isinstance(messages, list):
                return messages[-limit:]
            return []
        except Exception as e:
            logger.error("Failed to read chat history for '%s': %s", session_id, e)
            return []

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        response_data: Optional[Dict[str, Any]] = None,
        actions_executed: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Appends a single message turn to the persistent session file."""
        history = self.get_history(session_id, limit=100)
        entry: Dict[str, Any] = {
            "id": f"msg_{int(time.time() * 1000)}",
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }
        if response_data:
            entry["response"] = response_data
        if actions_executed:
            entry["actions_executed"] = actions_executed

        history.append(entry)
        path = self._get_session_path(session_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error("Failed to save chat message for '%s': %s", session_id, e)

    def sync_history(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Syncs the current state of messages to persistent storage."""
        path = self._get_session_path(session_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2)
        except Exception as e:
            logger.error("Failed to sync chat history for '%s': %s", session_id, e)

    def clear_history(self, session_id: str = "default_session") -> None:
        """Clears stored history for a session."""
        path = self._get_session_path(session_id)
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.error("Failed to clear chat session '%s': %s", session_id, e)


chat_memory = ChatMemoryStore()
