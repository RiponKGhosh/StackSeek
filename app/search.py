from __future__ import annotations

import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from env_loader import load_local_env


class SemanticSearchEngine:
    def __init__(
        self,
        index_path: str | Path,
        records_path: str | Path,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        load_local_env()
        self.index_path = Path(index_path)
        self.records_path = Path(records_path)
        self.embedding_model = embedding_model

        self.index = faiss.read_index(str(self.index_path))
        self.records = json.loads(self.records_path.read_text(encoding="utf-8"))

        if len(self.records) != self.index.ntotal:
            raise ValueError(
                f"Mismatch between records ({len(self.records)}) and FAISS index ({self.index.ntotal})."
            )

        token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_MODELS_TOKEN")
        if not token:
            raise RuntimeError("Missing GITHUB_TOKEN (or GITHUB_MODELS_TOKEN) for query embedding.")

        endpoint = os.getenv("GITHUB_MODELS_ENDPOINT", "https://models.inference.ai.azure.com")
        self.client = OpenAI(api_key=token, base_url=endpoint)

    def _embed_query(self, query: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.embedding_model, input=[query])
        vector = np.asarray(response.data[0].embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vector)
        return vector

    @staticmethod
    def _snippet(text: str, max_chars: int = 500) -> str:
        clean = " ".join(text.split())
        if len(clean) <= max_chars:
            return clean
        return clean[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _strip_html(text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text or "")

    def _matched_terms(self, query: str, row: dict, max_terms: int = 5) -> list[str]:
        query_terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}", query.lower()))
        if not query_terms:
            return []

        corpus = " ".join(
            [
                row.get("title", ""),
                row.get("body", ""),
                self._strip_html(row.get("answer", "")),
            ]
        ).lower()

        matched = [term for term in sorted(query_terms) if re.search(rf"\b{re.escape(term)}\b", corpus)]
        return matched[:max_terms]

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        top_k = max(1, min(top_k, 10))
        query_vec = self._embed_query(query)

        scores, indices = self.index.search(query_vec, top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            row = self.records[idx]
            matched_terms = self._matched_terms(query, row)
            explanation = (
                f"Semantic vector similarity score: {float(score):.4f}. "
                + (
                    f"Matched terms: {', '.join(matched_terms)}."
                    if matched_terms
                    else "No direct keyword overlap; semantic meaning matched."
                )
            )
            results.append(
                {
                    "title": row["title"],
                    "snippet": self._snippet(row["answer"]),
                    "full_answer": row.get("answer", ""),
                    "score": round(float(score), 4),
                    "question_id": row.get("question_id", ""),
                    "source_url": row.get("source_url", ""),
                    "matched_terms": matched_terms,
                    "match_explanation": explanation,
                }
            )

        return results
