"""Retrieval for documents longer than the model context (docs/how-it-works.md).

Instead of cutting a long document at a page boundary, the lines most relevant to the
requested fields are selected until the context budget is full, and sent in document order.

Privacy: the vectors are derived from document content, so they live only in memory. A
ChromaDB `EphemeralClient` collection is created per document and deleted in `finally`;
nothing is persisted and no vector database runs as a service.

`chromadb` and `sentence_transformers` are imported lazily: importing the pipeline (API cold
start, Lambda) must not load them.
"""

import uuid
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any, Protocol

from schemalens_core.config import Settings
from schemalens_core.context import estimate_tokens
from schemalens_core.errors import SchemaLensError
from schemalens_core.models import OcrLine, OcrResult

Embedder = Callable[[list[str]], list[list[float]]]


class RetrievalUnavailable(SchemaLensError):
    """The embedding model is not available; the caller falls back to truncation."""

    code = "retrieval_unavailable"


class Retriever(Protocol):
    def select(self, ocr: OcrResult, schema: dict[str, Any], budget: int) -> list[OcrLine]: ...


@cache
def _sentence_model(path: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(path)


def sentence_embedder(settings: Settings) -> Embedder:
    """Multilingual sentence embeddings from the model in `EMBED_MODEL_DIR` (never downloaded)."""
    directory = Path(settings.embed_model_dir)

    def embed(texts: list[str]) -> list[list[float]]:
        if not (directory / "config.json").is_file():
            raise RetrievalUnavailable(f"embedding model not found in {directory}")
        model = _sentence_model(str(directory))
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, vector)) for vector in vectors]

    return embed


def field_queries(schema: dict[str, Any]) -> dict[str, str]:
    """One query per top-level field: its name and description, plus those of nested fields."""

    def describe(name: str, node: dict[str, Any]) -> list[str]:
        parts = [name.replace("_", " ")]
        if node.get("description"):
            parts.append(node["description"])
        child = node.get("items", node)
        for key, value in child.get("properties", {}).items():
            parts += describe(key, value)
        return parts

    return {name: ": ".join(describe(name, node)) for name, node in schema["properties"].items()}


def chunk_lines(lines: list[OcrLine], size: int, overlap: int) -> list[list[OcrLine]]:
    """Windows of `size` consecutive lines within a page, overlapping by `overlap`."""
    step = max(1, size - overlap)
    pages: dict[int, list[OcrLine]] = {}
    for line in lines:
        pages.setdefault(line.page, []).append(line)
    chunks = []
    for page_lines in pages.values():
        for start in range(0, len(page_lines), step):
            chunks.append(page_lines[start : start + size])
            if start + size >= len(page_lines):
                break
    return chunks


class ChromaRetriever:
    """Selects the lines most relevant to the schema fields, within a token budget."""

    def __init__(self, settings: Settings, embed: Embedder | None = None) -> None:
        self._settings = settings
        self._embed = embed or sentence_embedder(settings)

    def _ranked_chunks(
        self, chunks: list[list[OcrLine]], queries: dict[str, str]
    ) -> dict[str, list[int]]:
        """For every field, all chunk indexes from most to least similar."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        texts = [" ".join(line.text for line in chunk) for chunk in chunks]
        chunk_vectors = self._embed(texts)
        query_vectors = self._embed(list(queries.values()))

        client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
        name = f"doc-{uuid.uuid4().hex}"
        collection = client.create_collection(
            name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        try:
            collection.add(
                ids=[str(index) for index in range(len(chunks))],
                embeddings=chunk_vectors,  # type: ignore[arg-type]
            )
            found = collection.query(
                query_embeddings=query_vectors,  # type: ignore[arg-type]
                n_results=len(chunks),
                include=[],
            )
        finally:
            # Vectors of a client's document never outlive the request.
            client.delete_collection(name)
        return {
            field: [int(i) for i in ids] for field, ids in zip(queries, found["ids"], strict=True)
        }

    def select(self, ocr: OcrResult, schema: dict[str, Any], budget: int) -> list[OcrLine]:
        settings = self._settings
        lines = ocr.lines
        chars = settings.token_estimate_chars
        selected: set[str] = set()
        used = 0

        def take(group: list[OcrLine]) -> None:
            nonlocal used
            fresh = [line for line in group if line.id not in selected]
            text = "\n".join(f"[{line.id}] {line.text}" for line in fresh)
            cost = estimate_tokens(text, chars)
            if fresh and used + cost <= budget:
                selected.update(line.id for line in fresh)
                used += cost

        # The head (title, parties, dates, numbers) is often missed by semantic search; the
        # end matters for summaries. Both go in first.
        take(lines[: settings.rag_head_lines])
        if any(node.get("x-schemalens-derived") for node in schema["properties"].values()):
            take(lines[-settings.rag_tail_lines :])

        chunks = chunk_lines(lines, settings.rag_chunk_lines, settings.rag_chunk_overlap)
        ranked = self._ranked_chunks(chunks, field_queries(schema))
        # Round robin over the fields, best chunk first, so every field gets its share
        # before any field gets a second chunk.
        for rank in range(len(chunks)):
            for order in ranked.values():
                if rank < len(order):
                    take(chunks[order[rank]])
        return [line for line in lines if line.id in selected]
