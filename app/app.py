from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from env_loader import load_local_env
from search import SemanticSearchEngine

BASE_DIR = Path(__file__).resolve().parent.parent
load_local_env()
DATA_DIR = Path(os.getenv("CODESEARCH_DATA_DIR", BASE_DIR / "data"))
INDEX_PATH = DATA_DIR / "index.faiss"
RECORDS_PATH = DATA_DIR / "records.json"
EMBEDDING_MODEL = os.getenv("CODESEARCH_EMBEDDING_MODEL", "text-embedding-3-small")
CACHE_PATH = DATA_DIR / "query_cache.json"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

search_engine: SemanticSearchEngine | None = None
init_error: str | None = None
query_cache: dict[str, list[dict]] = {}
cache_lock = threading.Lock()


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().lower().split())


def _cache_key(query: str, top_k: int) -> str:
    return f"{_normalize_query(query)}|top_k={top_k}"


def _load_query_cache() -> None:
    global query_cache
    if not CACHE_PATH.exists():
        query_cache = {}
        return

    try:
        loaded = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            query_cache = {str(k): v for k, v in loaded.items() if isinstance(v, list)}
            return
    except Exception:
        pass

    query_cache = {}


def _write_query_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(query_cache, ensure_ascii=False, indent=2)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CACHE_PATH.parent, delete=False) as tmp_file:
        tmp_file.write(serialized)
        tmp_path = Path(tmp_file.name)
    tmp_path.replace(CACHE_PATH)


def init_search_engine() -> None:
    global search_engine, init_error

    if not INDEX_PATH.exists() or not RECORDS_PATH.exists():
        init_error = (
            "Search index files are missing. Run: "
            "python app/dataset.py --output-dir data"
        )
        return

    try:
        search_engine = SemanticSearchEngine(
            index_path=INDEX_PATH,
            records_path=RECORDS_PATH,
            embedding_model=EMBEDDING_MODEL,
        )
        _load_query_cache()
    except Exception as exc:
        init_error = f"Failed to initialize search engine: {exc}"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/results")
def results_page():
    query = request.args.get("q", "").strip()
    return render_template("results.html", query=query)


@app.route("/search", methods=["POST"])
def search_route():
    payload = request.get_json(silent=True) or request.form
    query = (payload.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    if search_engine is None:
        return jsonify({"error": init_error or "Search engine is not ready."}), 503

    top_k_value = payload.get("top_k", 8)
    try:
        top_k = int(top_k_value)
    except (TypeError, ValueError):
        top_k = 8

    top_k = max(1, min(top_k, 10))
    key = _cache_key(query=query, top_k=top_k)

    with cache_lock:
        cached_items = query_cache.get(key)
    if cached_items is not None:
        return jsonify({"query": query, "count": len(cached_items), "results": cached_items, "cached": True})

    items = search_engine.search(query=query, top_k=top_k)
    with cache_lock:
        query_cache[key] = items
        _write_query_cache()

    return jsonify({"query": query, "count": len(items), "results": items, "cached": False})


if __name__ == "__main__":
    init_search_engine()
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    init_search_engine()
