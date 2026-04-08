# StackSeek

StackSeek is a semantic Stack Overflow search app built with Flask, FAISS, and GitHub Models embeddings.

## Features

- Semantic search over Stack Overflow Q&A records
- FAISS index for fast vector similarity lookup
- Simple web UI (`/`) and JSON API (`/search`)
- Persistent query cache to reduce repeated embedding calls (`data/query_cache.json`)

## Tech Stack

- Python 3.10+
- Flask
- FAISS (`faiss-cpu`)
- Hugging Face `datasets`
- OpenAI Python SDK (used with GitHub Models endpoint)

## 1) Setup

Clone the repo, then install dependencies:

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure Environment

Create `.env` from `.env.example` and fill in your token:

```env
GITHUB_TOKEN=your_token_here
GITHUB_MODELS_ENDPOINT=https://models.inference.ai.azure.com
CODESEARCH_EMBEDDING_MODEL=text-embedding-3-small
```

Notes:

- `GITHUB_TOKEN` (or `GITHUB_MODELS_TOKEN`) is required for embeddings.
- The app auto-loads values from project `.env`.

## 3) Build Dataset + FAISS Index (if needed)

If `data/index.faiss` and `data/records.json` are missing, generate them:

```bash
python app/dataset.py --output-dir data
```

Useful options:

```bash
python app/dataset.py --max-samples 4000 --batch-size 32 --model text-embedding-3-small --output-dir data
```

This creates:

- `data/index.faiss`
- `data/records.json`
- `data/config.json`

## 4) Run the App

```bash
python app/app.py
```

Then open:

- `http://localhost:5000` (UI)

API example:

```bash
curl -X POST http://localhost:5000/search -H "Content-Type: application/json" -d "{\"query\":\"python flask ci\", \"top_k\":8}"
```

## Project Structure

```text
app/
  app.py          # Flask server and routes
  dataset.py      # Dataset ingestion + embedding + FAISS build
  search.py       # Query embedding + semantic retrieval
  env_loader.py   # Lightweight .env loader
data/
  config.json
  index.faiss
  records.json
templates/
static/
```

## Troubleshooting

- `Missing GITHUB_TOKEN`:
  - Add `GITHUB_TOKEN` (or `GITHUB_MODELS_TOKEN`) to `.env`.
- `Search index files are missing`:
  - Run `python app/dataset.py --output-dir data`.
- Dependency install issues:
  - Verify Python version is 3.10+ and virtual environment is activated.
