"""Optional tiny byte-level GPT for learning experiments. Not a foundation model."""
from __future__ import annotations
from pathlib import Path
from typing import Callable

CONFIG = {"vocab_size": 256, "block_size": 128, "n_embd": 96, "n_head": 4, "n_layer": 2, "dropout": 0.1}

def _torch_components():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("The scratch trainer requires PyTorch. Install it from https://pytorch.org/get-started/locally/") from exc
    class ByteGPT(nn.Module):
        def __init__(self, config=CONFIG):
            super().__init__()
            self.config = dict(config)
            self.token = nn.Embedding(config["vocab_size"], config["n_embd"])
            self.position = nn.Embedding(config["block_size"], config["n_embd"])
            layer = nn.TransformerEncoderLayer(d_model=config["n_embd"], nhead=config["n_head"], dim_feedforward=config["n_embd"] * 4, dropout=config["dropout"], activation="gelu", batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, num_layers=config["n_layer"], enable_nested_tensor=False)
            self.norm = nn.LayerNorm(config["n_embd"])
            self.head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
            self.head.weight = self.token.weight
        def forward(self, idx, targets=None):
            b, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.token(idx) + self.position(pos)[None, :, :]
            mask = torch.triu(torch.ones((t, t), device=idx.device, dtype=torch.bool), diagonal=1)
            x = self.blocks(x, mask=mask)
            logits = self.head(self.norm(x))
            loss = None if targets is None else torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
    return torch, ByteGPT

def train(corpus_files: list[Path], checkpoint_path: Path, steps: int = 100, stop_event=None, callback: Callable[[dict], None] | None = None) -> dict:
    torch, ByteGPT = _torch_components()
    import random
    text_parts = []
    total_files = 0
    for path in corpus_files:
        try:
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".markdown", ".rst", ".jsonl", ".csv"}:
                content = path.read_text(encoding="utf-8", errors="replace").strip()
                if content: text_parts.append(content); total_files += 1
        except OSError:
            continue
    corpus = "\n\n".join(text_parts).encode("utf-8")
    if len(corpus) < 512:
        raise RuntimeError(f"Training needs at least 512 bytes of local text; found {len(corpus)} bytes in {total_files} files. Add more .txt/.md files first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(list(corpus), dtype=torch.long)
    split = max(1, int(len(data) * 0.95))
    train_data = data[:split]
    val_data = data[split:] if len(data) - split > CONFIG["block_size"] + 1 else data[:split]
    model = ByteGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    batch_size = 4
    block = CONFIG["block_size"]
    recent_loss = None
    for step in range(1, max(1, int(steps)) + 1):
        if stop_event is not None and stop_event.is_set(): break
        model.train()
        ix = torch.randint(0, max(1, len(train_data) - block - 1), (batch_size,))
        x = torch.stack([train_data[int(i):int(i) + block] for i in ix]).to(device)
        y = torch.stack([train_data[int(i) + 1:int(i) + block + 1] for i in ix]).to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        recent_loss = float(loss.detach().cpu())
        if step == 1 or step % 5 == 0 or step == steps:
            if callback: callback({"step": step, "steps": steps, "loss": recent_loss, "device": str(device), "corpus_bytes": len(corpus), "files": total_files})
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": CONFIG, "model": model.cpu().state_dict(), "step": step, "corpus_bytes": len(corpus), "files": total_files}, checkpoint_path)
    return {"step": step, "steps_requested": steps, "loss": recent_loss, "device": str(device), "corpus_bytes": len(corpus), "files": total_files, "checkpoint": str(checkpoint_path), "note": "Experimental byte-level model; fluency depends heavily on corpus and is not comparable to a pretrained LLM."}

def is_available(checkpoint_path: Path) -> bool:
    return checkpoint_path.exists()

def generate(checkpoint_path: Path, prompt: str, max_new_tokens: int = 160, temperature: float = 0.8) -> str:
    torch, ByteGPT = _torch_components()
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload.get("config", CONFIG)
    model = ByteGPT(config)
    model.load_state_dict(payload["model"])
    model.eval()
    context = torch.tensor(list(prompt.encode("utf-8", errors="replace"))[-config["block_size"]:], dtype=torch.long)[None, :]
    prompt_length = context.shape[1]
    generated_count = max(1, min(int(max_new_tokens), 512))
    with torch.no_grad():
        for _ in range(generated_count):
            logits, _ = model(context[:, -config["block_size"]:])
            logits = logits[:, -1, :] / max(float(temperature), 0.05)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
    new_bytes = bytes(context[0, prompt_length:].tolist())
    # Decode with replacement because independent byte sampling can form invalid UTF-8.
    return new_bytes.decode("utf-8", errors="replace")
