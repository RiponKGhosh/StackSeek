from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from env_loader import load_local_env

TITLE_KEYS = ["question_title", "title", "pregunta_titulo", "question"]
BODY_KEYS = ["question_body", "body", "pregunta_cuerpo", "content", "text"]
ANSWER_KEYS = ["answer", "accepted_answer", "respuesta", "best_answer", "solution"]
URL_KEYS = ["question_url", "url", "link", "permalink", "stackoverflow_url"]
ID_KEYS = ["question_id", "id", "qid"]


def _pick_first(record: dict, keys: list[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def _build_embedding_text(title: str, body: str, answer: str) -> str:
    return f"Question: {title}\n\nDetails: {body}\n\nAnswer: {answer}"


def _normalize_source_url(raw_url: str, question_id: str) -> str:
    cleaned = raw_url.strip()
    if cleaned:
        if cleaned.startswith("http://") or cleaned.startswith("https://"):
            return cleaned
        return f"https://{cleaned.lstrip('/')}"
    if question_id:
        return f"https://stackoverflow.com/questions/{question_id}"
    return ""


def _get_openai_client() -> OpenAI:
    load_local_env()
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_MODELS_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing GITHUB_TOKEN (or GITHUB_MODELS_TOKEN). "
            "Create a GitHub token and export it before preprocessing."
        )

    endpoint = os.getenv("GITHUB_MODELS_ENDPOINT", "https://models.inference.ai.azure.com")
    return OpenAI(api_key=token, base_url=endpoint)


def _embed_batch(client: OpenAI, model: str, texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(model=model, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.asarray(vectors, dtype="float32")


def load_records(dataset_name: str, split: str, max_samples: int | None) -> list[dict]:
    dataset = load_dataset(dataset_name, split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    records: list[dict] = []
    for raw in dataset:
        title = _clean_text(_pick_first(raw, TITLE_KEYS))
        body = _clean_text(_pick_first(raw, BODY_KEYS))
        answer = _clean_text(_pick_first(raw, ANSWER_KEYS))
        source_url = _clean_text(_pick_first(raw, URL_KEYS))
        question_id = _clean_text(_pick_first(raw, ID_KEYS))
        if not question_id:
            question_id = str(raw.get("id") or "").strip()

        if not title or not answer:
            continue

        embedding_text = _build_embedding_text(title=title, body=body, answer=answer)
        records.append(
            {
                "title": title,
                "body": body,
                "answer": answer,
                "question_id": question_id,
                "source_url": _normalize_source_url(source_url, question_id),
                "embedding_text": embedding_text,
            }
        )

    if not records:
        raise RuntimeError(
            "No records were extracted from the dataset. "
            "Please inspect dataset column names and update key mappings."
        )

    return records


def build_index(records: list[dict], model: str, batch_size: int) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    client = _get_openai_client()

    all_vectors: list[np.ndarray] = []
    for start in tqdm(range(0, len(records), batch_size), desc="Embedding records"):
        end = start + batch_size
        batch_texts = [row["embedding_text"] for row in records[start:end]]
        batch_vectors = _embed_batch(client=client, model=model, texts=batch_texts)
        all_vectors.append(batch_vectors)

    vectors = np.vstack(all_vectors).astype("float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, vectors


def save_artifacts(output_dir: Path, index: faiss.IndexFlatIP, records: list[dict], model: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "index.faiss"
    records_path = output_dir / "records.json"
    config_path = output_dir / "config.json"

    clean_records = [
        {
            "title": r["title"],
            "body": r["body"],
            "answer": r["answer"],
            "question_id": r.get("question_id", ""),
            "source_url": r.get("source_url", ""),
        }
        for r in records
    ]

    faiss.write_index(index, str(index_path))
    records_path.write_text(json.dumps(clean_records, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "embedding_model": model,
                "total_records": len(clean_records),
                "index_path": str(index_path),
                "records_path": str(records_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess dataset and build FAISS index for CodeSearch")
    parser.add_argument(
        "--dataset",
        default="MartinElMolon/stackoverflow_preguntas_con_embeddings",
        help="Hugging Face dataset id",
    )
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=4000, help="Optional cap for faster local builds")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--model", default="text-embedding-3-small", help="GitHub Models embedding model")
    parser.add_argument("--output-dir", default="data", help="Output directory for FAISS + metadata")

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} [{args.split}]")
    records = load_records(args.dataset, args.split, args.max_samples)
    print(f"Records ready for indexing: {len(records)}")

    print("Building embeddings + FAISS index...")
    index, _ = build_index(records=records, model=args.model, batch_size=args.batch_size)

    save_artifacts(output_dir=Path(args.output_dir), index=index, records=records, model=args.model)
    print(f"Done. Artifacts saved in: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
