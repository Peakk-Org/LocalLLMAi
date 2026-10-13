# LocalLLMAi Studio 2.0

A more approachable local-AI desktop workspace inspired by [Peakk-Org/LocalLLMAi](https://github.com/Peakk-Org/LocalLLMAi). It keeps the local-first spirit, adds a document knowledge library, model-server integration, conversation management, diagnostics, and an optional educational from-scratch training lab.

> This is a new companion implementation, not a drop-in replacement for the original repository's SentencePiece checkpoints. The built-in Scratch GPT uses a separate byte-level architecture. The strongest general chat experience comes from connecting a suitable pretrained local model through Ollama or another OpenAI-compatible local server.

## Features

- Dark desktop workspace with Overview, Chat, Knowledge, Training Lab, and Models & Settings screens.
- Model backends: Ollama, OpenAI-compatible endpoint, and experimental Scratch GPT.
- Optional Ollama model download control inside Settings (runs the installed `ollama pull` command and displays progress).
- Chat history saved locally; create, switch, delete, and export conversations to Markdown.
- Streaming replies from Ollama, adjustable temperature/top-p/token settings, and optional document grounding.
- Document indexing for TXT, Markdown, RST, HTML, JSON, JSONL, CSV, DOCX, and text-based PDF (PDF requires `pypdf`).
- Explicit one-page web import with robots.txt checking, public-host validation, a response-size limit, and no automatic crawling.
- Basic retrieval search with source filenames and matching passage display.
- Teacher-generated Q&A data builder for turning indexed passages into additional local training text via Ollama. This is synthetic-data generation, **not** logit-level knowledge distillation.
- Optional scratch training of a tiny byte-level Transformer with progress reporting and a checkpoint.
- Diagnostics for Python, OS, CPU cores, available disk, optional PyTorch/CUDA and configured model-server status.
- No telemetry or automatic whole-PC scan. It indexes only files the user explicitly selects and sends prompts/documents only to the endpoint the user configures.

## Requirements

- Windows 10/11, macOS, or Linux with Python 3.10+ and Tkinter.
- For stronger local assistant responses: install [Ollama](https://ollama.com/) and pull a model that fits your hardware. Start Ollama, then open this app and use the refresh button beside the model selector.
- PDF text extraction: install `pypdf` with `python -m pip install pypdf`.
- From-scratch training: install a compatible PyTorch build using the official [PyTorch selector](https://pytorch.org/get-started/locally/). CUDA is used automatically only when PyTorch reports it is available.

## Start on Windows

1. Extract the folder.
2. Double-click `run.bat`, or open Command Prompt in this folder and run:

   ```bat
   python app.py
   ```

The desktop app itself has no mandatory third-party Python packages. If Tkinter is missing from your Python installation, install a Python distribution that includes Tcl/Tk.

## Connect a model

1. Install/start Ollama.
2. Pull an appropriate model in a terminal, for example `ollama pull qwen2.5:3b` (choose a size your RAM can handle).
3. Open **Models & Settings**, keep the default `http://127.0.0.1:11434`, choose backend `Ollama`, save, then return to Chat and refresh the model list. You can also enter a model identifier into the Settings download box and use **Pull model**; this requires the Ollama CLI to be on PATH.
4. Pick the installed model and send a message.

A custom endpoint can point somewhere else, including a remote server. If it is not local, your prompts and selected context may leave your PC. The app does not bypass the security or authentication of that endpoint.

## Use document knowledge

1. Open **Knowledge** → **Add files**, or choose **Import URL** for a public page. The Scratch GPT backend does not use retrieved document context; RAG is intended for connected chat models.
2. Select files you want to index. The original files remain where they are; extracted text chunks are stored in `data/knowledge/index.json`. Web imports first check the site's `robots.txt` and are limited to one page at a time.
3. Test keyword retrieval using Search your knowledge.
4. In Chat, enable **Use knowledge**. The app retrieves a few matching passages and includes them in the prompt, with source filenames.

Scanned-image PDFs do not have extractable text unless you OCR them separately. A document being indexed does not change model weights; retrieval-augmented generation (RAG) and training are separate processes.

## Training Lab

Place local corpus files (`.txt`, `.md`, `.markdown`, `.rst`, `.jsonl`, `.csv`) in `data/training/`. An example file is included. From Training Lab, choose a short run and start training. This creates `models/scratch_latest.pt`.

This small byte-level model is for learning and debugging only. It is not instruction-tuned and usually will not compare with a proper pretrained model. It is trained separately from the original project's model and cannot load its checkpoints. You can export indexed knowledge as corpus text or build teacher-generated Q&A examples from Ollama. The Q&A builder creates text for later training; it does not perform logit-matching distillation.

## Storage and privacy

- Settings: `settings.json`
- Conversation history: `conversations/history.json` (the opt-out stops future writes; it does not erase an existing history file)
- Knowledge index: `data/knowledge/index.json`
- Training text: `data/training/`
- Scratch checkpoint: `models/scratch_latest.pt`

These live inside the project folder. Keep this folder private if you add sensitive material. No background scanning or automatic upload is performed.

## Troubleshooting

- **Ollama offline:** ensure the Ollama app/service is running and the endpoint in Settings is correct.
- **No model appears:** run `ollama list` and ensure you have pulled a model; click refresh beside MODEL.
- **PDF extraction unavailable:** `python -m pip install pypdf`.
- **Scratch training fails with PyTorch import error:** install PyTorch for your Python and hardware; use the official selector instead of guessing a CUDA build.
- **Scratch training asks for more text:** add more supported files to `data/training/`. A tiny corpus often memorizes its examples.
- **Model response is low-quality:** use a suitable pretrained local model for everyday chat. Small from-scratch training is a research/learning feature, not a shortcut to a billion-parameter model.

## License

MIT. See `LICENSE`.
