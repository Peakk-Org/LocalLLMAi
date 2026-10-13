from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
TRAINING_DIR = DATA_DIR / "training"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
CHAT_DIR = ROOT / "conversations"
SETTINGS_PATH = ROOT / "settings.json"

DEFAULTS: dict[str, Any] = {
    "backend": "Ollama",
    "endpoint": "http://127.0.0.1:11434",
    "model": "",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "use_knowledge": True,
    "remember_chats": True,
    "font_scale": 1.0,
    "theme": "Midnight",
}

def ensure_dirs() -> None:
    for path in (DATA_DIR, KNOWLEDGE_DIR, TRAINING_DIR, MODEL_DIR, LOG_DIR, CHAT_DIR):
        path.mkdir(parents=True, exist_ok=True)

def load_settings() -> dict[str, Any]:
    ensure_dirs()
    try:
        raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {**DEFAULTS, **raw}
    except (OSError, json.JSONDecodeError):
        pass
    return dict(DEFAULTS)

def save_settings(values: dict[str, Any]) -> None:
    ensure_dirs()
    payload = {**DEFAULTS, **values}
    temp = SETTINGS_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(SETTINGS_PATH)
