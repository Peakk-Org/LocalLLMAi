from __future__ import annotations
import json
import urllib.error
import urllib.request
from typing import Any, Callable

class EngineError(RuntimeError): pass

def _request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 30, headers: dict[str, str] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise EngineError(f"HTTP {exc.code}: {detail or exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise EngineError(f"Could not reach local model server: {exc}") from exc

def ollama_models(endpoint: str) -> list[str]:
    data = _request_json(endpoint.rstrip("/") + "/api/tags", timeout=4)
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

def ollama_chat(endpoint: str, model: str, messages: list[dict[str, str]], options: dict[str, Any], emit: Callable[[str], None]) -> str:
    url = endpoint.rstrip("/") + "/api/chat"
    payload = {"model": model, "messages": messages, "stream": True, "options": options}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    pieces = []
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                if not raw.strip(): continue
                try: item = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError: continue
                if item.get("error"): raise EngineError(str(item["error"]))
                token = item.get("message", {}).get("content", "")
                if token:
                    pieces.append(token); emit(token)
                if item.get("done"): break
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise EngineError(f"Ollama HTTP {exc.code}: {detail or exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise EngineError(f"Could not reach Ollama at {endpoint}: {exc}") from exc
    return "".join(pieces)

def openai_compatible_chat(endpoint: str, model: str, messages: list[dict[str, str]], options: dict[str, Any]) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/chat/completions"): url = base
    else: url = base + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": options.get("temperature", 0.7), "top_p": options.get("top_p", 0.9), "max_tokens": options.get("num_predict", 512), "stream": False}
    data = _request_json(url, payload, timeout=600)
    try: return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc: raise EngineError("Unexpected OpenAI-compatible response format") from exc

def build_teacher_examples(endpoint: str, model: str, source: str, options: dict[str, Any] | None = None) -> str:
    messages = [
        {"role": "system", "content": "Create high-quality student training text from provided source. Use only facts present in the source. Output 4 diverse question-and-answer pairs, each as 'User: ...\\nAssistant: ...'. If information is missing, do not invent it."},
        {"role": "user", "content": "SOURCE MATERIAL:\n" + source[:8000]},
    ]
    if options is None: options = {"temperature": 0.3, "top_p": 0.9, "num_predict": 900}
    return ollama_chat(endpoint, model, messages, options, lambda _: "")
