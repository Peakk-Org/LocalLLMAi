from __future__ import annotations
import csv
import hashlib
import html
import json
import re
import zipfile
import ipaddress
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import socket
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
from .settings import KNOWLEDGE_DIR, TRAINING_DIR, ensure_dirs

INDEX_PATH = KNOWLEDGE_DIR / "index.json"
SUPPORTED = {".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".json", ".jsonl", ".csv", ".pdf", ".docx"}

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "noscript", "svg"}: self.skip += 1
        elif tag.lower() in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "noscript", "svg"} and self.skip: self.skip -= 1
        elif tag.lower() in {"p", "div", "li", "h1", "h2", "h3", "tr"}: self.parts.append("\n")
    def handle_data(self, data):
        if not self.skip and data.strip(): self.parts.append(data.strip() + " ")

def _clean(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r", "\n")
    text = re.sub(r"[\t\u00a0 ]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support needs the optional package: pip install pypdf") from exc
        pages = []
        for page in PdfReader(str(path)).pages:
            pages.append(page.extract_text() or "")
        return _clean("\n\n".join(pages))
    if suffix == ".docx":
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ET.fromstring(xml)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paras = []
        for p in root.findall(".//w:p", ns):
            paras.append("".join(t.text or "" for t in p.findall(".//w:t", ns)))
        return _clean("\n".join(paras))
    data = path.read_text(encoding="utf-8-sig", errors="replace")
    if suffix in {".html", ".htm"}:
        parser = _TextExtractor()
        parser.feed(data)
        return _clean(" ".join(parser.parts))
    if suffix == ".json":
        try:
            obj = json.loads(data)
            if isinstance(obj, dict):
                vals = [str(v) for k, v in obj.items() if isinstance(v, (str, int, float))]
                return _clean("\n".join(vals) if vals else json.dumps(obj, ensure_ascii=False, indent=2))
            if isinstance(obj, list): return _clean("\n".join(json.dumps(x, ensure_ascii=False) for x in obj))
        except json.JSONDecodeError:
            pass
    if suffix == ".jsonl":
        pieces = []
        for line in data.splitlines():
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    candidates = [item.get(k) for k in ("text", "content", "body", "prompt", "completion", "answer", "question")]
                    pieces.append(" ".join(str(x) for x in candidates if x is not None))
                else: pieces.append(str(item))
            except json.JSONDecodeError:
                pieces.append(line)
        return _clean("\n".join(pieces))
    if suffix == ".csv":
        rows = []
        for row in csv.reader(data.splitlines()): rows.append(" | ".join(row))
        return _clean("\n".join(rows))
    return _clean(data)

def _chunk_text(text: str, max_chars: int = 1400, overlap: int = 180) -> list[str]:
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start + max_chars // 2, end), text.rfind(". ", start + max_chars // 2, end))
            if cut > start: end = cut + 1
        part = text[start:end].strip()
        if part: chunks.append(part)
        if end >= len(text): break
        start = max(end - overlap, start + 1)
    return chunks

def load_index() -> list[dict[str, Any]]:
    ensure_dirs()
    try:
        obj = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        return obj if isinstance(obj, list) else []
    except (OSError, json.JSONDecodeError): return []

def save_index(chunks: list[dict[str, Any]]) -> None:
    ensure_dirs()
    tmp = INDEX_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    tmp.replace(INDEX_PATH)

def index_files(paths: list[str | Path]) -> dict[str, Any]:
    existing = load_index()
    success, errors = [], []
    for raw in paths:
        path = Path(raw)
        try:
            if path.suffix.lower() not in SUPPORTED:
                raise ValueError("Unsupported file type")
            if not path.is_file(): raise FileNotFoundError("File not found")
            if path.stat().st_size > 100 * 1024 * 1024: raise ValueError("File larger than 100 MB is not supported")
            text = extract_text(path)
            if len(text) < 20: raise ValueError("Not enough extractable text (scanned PDFs need OCR, which is not built in)")
            # Replace previous chunks from this exact source, preventing stale duplicates.
            existing = [c for c in existing if c.get("source_path") != str(path.resolve())]
            digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
            for i, chunk in enumerate(_chunk_text(text)):
                existing.append({"id": f"{digest[:12]}-{i}", "source": path.name, "source_path": str(path.resolve()), "chunk": i + 1, "text": chunk, "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            success.append({"name": path.name, "chunks": len(_chunk_text(text)), "chars": len(text)})
        except Exception as exc:
            errors.append({"name": path.name, "error": str(exc)})
    save_index(existing)
    return {"success": success, "errors": errors, "chunks_total": len(existing)}

def remove_source(source_path: str) -> int:
    data = load_index()
    new = [c for c in data if c.get("source_path") != source_path]
    save_index(new)
    return len(data) - len(new)

def source_summary() -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for c in load_index():
        key = c.get("source_path", c.get("source", "unknown"))
        if key not in grouped: grouped[key] = {"path": key, "name": c.get("source", Path(key).name), "chunks": 0}
        grouped[key]["chunks"] += 1
    return sorted(grouped.values(), key=lambda x: x["name"].lower())

def search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    terms = set(re.findall(r"[a-zA-Z0-9_]{2,}|[^\W\d_]{2,}", query.lower()))
    if not terms: return []
    chunks = load_index()
    scored = []
    for c in chunks:
        text = c.get("text", "")
        low = text.lower()
        score = sum(low.count(t) * (1.0 + min(len(t), 10) / 20) for t in terms)
        if score > 0:
            # Slightly prefer compact matches over repeated keyword spam.
            score /= (max(len(text), 200) ** 0.18)
            scored.append((score, c))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [dict(c, score=round(score, 4)) for score, c in scored[:limit]]


def _assert_public_url(url: str) -> tuple[urllib.parse.ParseResult, str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter a complete http:// or https:// URL.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded credentials are not accepted.")
    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or "." not in host:
        raise ValueError("Local hostnames are not accepted for web import.")
    try:
        addr = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        addr = None
    if addr is not None and not addr.is_global:
        raise ValueError("Private, loopback, and link-local IP addresses are not accepted for web import.")
    try:
        resolved = {ipaddress.ip_address(answer[4][0].split("%", 1)[0]) for answer in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)}
        if not resolved or any(not address.is_global for address in resolved):
            raise ValueError("This hostname resolves to a private or non-public address; web import was blocked.")
    except socket.gaierror as exc:
        raise RuntimeError(f"Could not resolve host: {exc}") from exc
    return parsed, host

class _PublicRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _assert_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)

def import_web_url(url: str) -> Path:
    """Fetch one user-supplied public web page, consulting robots.txt and capping the response size."""
    ensure_dirs()
    parsed, host = _assert_public_url(url.strip())
    opener = urllib.request.build_opener(_PublicRedirectHandler())
    base = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = urllib.parse.urljoin(base, "/robots.txt")
    robot = urllib.robotparser.RobotFileParser()
    robot.set_url(robots_url)
    req = urllib.request.Request(robots_url, headers={"User-Agent": "LocalLLMAiStudio/2.0 (+user-triggered document import)"})
    try:
        with opener.open(req, timeout=8) as response:
            lines = response.read(256 * 1024).decode("utf-8", errors="replace").splitlines()
            robot.parse(lines)
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 410}:
            robot.parse([])
        else:
            raise RuntimeError(f"Could not verify robots.txt (HTTP {exc.code}); import cancelled for safety.") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not verify robots.txt; import cancelled for safety: {exc}") from exc
    if not robot.can_fetch("LocalLLMAiStudio/2.0", url):
        raise PermissionError("This page is disallowed by the site's robots.txt rules.")
    page_req = urllib.request.Request(url, headers={"User-Agent": "LocalLLMAiStudio/2.0 (+user-triggered document import)", "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5"})
    try:
        with opener.open(page_req, timeout=12) as response:
            content_type = response.headers.get("Content-Type", "")
            if not ("text/html" in content_type.lower() or "text/plain" in content_type.lower() or not content_type):
                raise ValueError(f"This page is not HTML or plain text ({content_type}).")
            raw = response.read(5 * 1024 * 1024 + 1)
            if len(raw) > 5 * 1024 * 1024:
                raise ValueError("Page response exceeded the 5 MB import limit.")
            charset = response.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Page request failed with HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not fetch the page: {exc}") from exc
    decoded = raw.decode(charset, errors="replace")
    if "text/html" in content_type.lower() or "<html" in decoded[:1000].lower():
        parser = _TextExtractor()
        parser.feed(decoded)
        text = _clean(" ".join(parser.parts))
    else:
        text = _clean(decoded)
    if len(text) < 50:
        raise ValueError("The page did not contain enough extractable text.")
    host_slug = re.sub(r"[^a-zA-Z0-9.-]+", "_", host)[:100]
    folder = KNOWLEDGE_DIR / "web_imports"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = folder / f"{host_slug}_{stamp}.txt"
    destination.write_text(f"Source URL: {url}\nImported at: {datetime.now().isoformat(timespec='seconds')}\n\n{text}\n", encoding="utf-8")
    return destination

def export_training_corpus(destination: Path | None = None) -> Path:
    ensure_dirs()
    destination = destination or (TRAINING_DIR / "knowledge_export.txt")
    chunks = load_index()
    with destination.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(f"Source: {c.get('source','unknown')}\n{c.get('text','')}\n\n")
    return destination
