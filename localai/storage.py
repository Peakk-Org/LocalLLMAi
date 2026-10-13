from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .settings import CHAT_DIR, ensure_dirs

CHAT_FILE = CHAT_DIR / "history.json"

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def load_chats() -> list[dict[str, Any]]:
    ensure_dirs()
    try:
        value = json.loads(CHAT_FILE.read_text(encoding="utf-8"))
        if isinstance(value, list):
            chats = [x for x in value if isinstance(x, dict) and isinstance(x.get("id"), str)]
            for chat in chats:
                chat.setdefault("name", "New conversation")
                chat.setdefault("messages", [])
                chat.setdefault("updated", now_iso())
            return sorted(chats, key=lambda x: x.get("updated", ""), reverse=True)
    except (OSError, json.JSONDecodeError):
        pass
    return []

def save_chats(chats: list[dict[str, Any]]) -> None:
    ensure_dirs()
    temp = CHAT_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(chats, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(CHAT_FILE)

def new_chat() -> dict[str, Any]:
    return {"id": uuid.uuid4().hex[:12], "name": "New conversation", "messages": [], "updated": now_iso()}
