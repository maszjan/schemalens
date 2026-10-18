"""Retrieval for long documents. Real ChromaDB in memory; a fake embedder unless the real
multilingual model is on disk (test marked `models`)."""

import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from schemalens_core.config import Settings
from schemalens_core.context import estimate_tokens
from schemalens_core.llm import LlmResponse
from schemalens_core.models import ExtractionOptions, OcrLine, OcrPage, OcrResult
from schemalens_core.pipeline import Pipeline
from schemalens_core.rag import (
    ChromaRetriever,
    RetrievalUnavailable,
    chunk_lines,
    field_queries,
    sentence_embedder,
)

SETTINGS = Settings(_env_file=None, local_model_fast="bielik")
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "data_zawarcia": {"type": "string", "format": "date", "description": "Data zawarcia umowy"},
        "wynagrodzenie": {"type": "number", "description": "Kwota wynagrodzenia"},
        "okres_wypowiedzenia": {"type": "string", "description": "Okres wypowiedzenia umowy"},
    },
}
FACTS = {
    "data_zawarcia": "Umowa zawarta dnia 1 września 2026 r. w Poznaniu",
    "wynagrodzenie": "Wynagrodzenie wynosi 5 000,00 zł netto miesięcznie",
    "okres_wypowiedzenia": "Okres wypowiedzenia wynosi jeden miesiąc",
}
FILLER = [
    "Strony zobowiązują się do zachowania poufności informacji handlowych",
    "Zleceniobiorca prowadzi dokumentację techniczną serwisu internetowego",
    "Wszelkie zmiany wymagają formy pisemnej pod rygorem nieważności",
    "Spory rozstrzyga sąd właściwy dla siedziby zleceniodawcy",
    "Zleceniodawca udostępni dostęp do środowiska testowego",
    "Raporty miesięczne przekazywane są pocztą elektroniczną",
]


def _long_document(pages: int = 5, lines_per_page: int = 40) -> OcrResult:
    """A long contract: filler everywhere, each fact on a different, later page."""
    lines = []
    fact_places = {
        (2, 17): "data_zawarcia",
        (4, 25): "wynagrodzenie",
        (5, 33): "okres_wypowiedzenia",
    }
    for page in range(1, pages + 1):
        for index in range(1, lines_per_page + 1):
            fact = fact_places.get((page, index))
            text = (
                FACTS[fact] if fact else f"§ {page}.{index} {FILLER[(page + index) % len(FILLER)]}"
            )
            lines.append(
                OcrLine(id=f"P{page}-L{index:03d}", page=page, text=text, bbox=None, conf=0.9)
            )
    pages_meta = [OcrPage(number=p, source="ocr", ocr_ms=1) for p in range(1, pages + 1)]
    return OcrResult(lines=lines, pages=pages_meta)


FACT_IDS = {"P2-L017", "P4-L025", "P5-L033"}


def fake_embedder(texts: list[str]) -> list[list[float]]:
    """Bag of word stems hashed into 256 dimensions: similar words, similar vectors."""
    vectors = []
    for text in texts:
        vector = [0.0] * 256
        for word in re.findall(r"\w+", text.lower()):
            bucket = int(hashlib.md5(word[:5].encode()).hexdigest(), 16) % 256
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


# --- building blocks -------------------------------------------------------------------


def test_chunks_are_overlapping_windows_within_a_page() -> None:
    lines = _long_document(pages=2, lines_per_page=7).lines

    chunks = chunk_lines(lines, size=3, overlap=1)

    ids = [[line.id for line in chunk] for chunk in chunks]
    assert ids[0] == ["P1-L001", "P1-L002", "P1-L003"]
    assert ids[1] == ["P1-L003", "P1-L004", "P1-L005"]
    assert ids[2] == ["P1-L005", "P1-L006", "P1-L007"]
    assert ids[3][0] == "P2-L001"  # no window spans two pages
    assert all(len({line.page for line in chunk}) == 1 for chunk in chunks)


def test_field_queries_use_names_descriptions_and_nested_fields() -> None:
    schema = {
        "type": "object",
        "properties": {
            "data_zawarcia": {"type": "string", "description": "Data zawarcia umowy"},
            "pozycje": {
                "type": "array",
                "items": {"type": "object", "properties": {"cena_netto": {"type": "number"}}},
            },
        },
    }
    assert field_queries(schema) == {
        "data_zawarcia": "data zawarcia: Data zawarcia umowy",
        "pozycje": "pozycje: cena netto",
    }


# --- selection -------------------------------------------------------------------------


def test_selection_finds_facts_deep_in_a_long_document_within_budget() -> None:
    ocr = _long_document()
    retriever = ChromaRetriever(SETTINGS, embed=fake_embedder)

    selected = retriever.select(ocr, SCHEMA, budget=600)

    ids = [line.id for line in selected]
    assert set(ids) >= FACT_IDS
    assert ids[:15] == [line.id for line in ocr.lines[:15]]  # the head always goes in
    assert ids == [line.id for line in ocr.lines if line.id in set(ids)]  # document order
    text = "\n".join(f"[{line.id}] {line.text}" for line in selected)
    assert estimate_tokens(text, SETTINGS.token_estimate_chars) <= 600
    assert len(selected) < len(ocr.lines)


def test_derived_fields_bring_the_end_of_the_document() -> None:
    ocr = _long_document()
    schema = {
        "type": "object",
        "properties": {
            **SCHEMA["properties"],
            "streszczenie": {"type": "string", "x-schemalens-derived": True},
        },
    }

    with_summary = ChromaRetriever(SETTINGS, embed=fake_embedder).select(ocr, schema, budget=600)
    without = ChromaRetriever(SETTINGS, embed=fake_embedder).select(ocr, SCHEMA, budget=600)

    assert ocr.lines[-1].id in {line.id for line in with_summary}
    assert ocr.lines[-1].id not in {line.id for line in without}


def test_collections_never_outlive_the_request() -> None:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    def broken(texts: list[str]) -> list[list[float]]:
        if len(texts) == 3:  # the queries: fail after the document was embedded
            raise RuntimeError("boom")
        return fake_embedder(texts)

    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    ChromaRetriever(SETTINGS, embed=fake_embedder).select(_long_document(), SCHEMA, budget=600)
    with pytest.raises(RuntimeError):
        ChromaRetriever(SETTINGS, embed=broken).select(_long_document(), SCHEMA, budget=600)

    assert client.list_collections() == []


def test_missing_embedding_model_is_reported_not_downloaded(tmp_path: Path) -> None:
    embed = sentence_embedder(Settings(_env_file=None, embed_model_dir=tmp_path / "nothing"))
    with pytest.raises(RetrievalUnavailable, match="embedding model not found"):
        embed(["tekst"])


# --- in the pipeline -------------------------------------------------------------------


def _blank_pdf(pages: int) -> bytes:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument.new()
    for _ in range(pages):
        document.new_page(595, 842)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


class LongOcr:
    def read(self, images: list[Image.Image]) -> OcrResult:
        return _long_document(pages=len(images))


class FakeLlm:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def extract(self, *, system: str, user: str, json_schema: Any, model: str) -> LlmResponse:
        self.prompts.append(user)
        answer = {
            "fields": {"wynagrodzenie": {"value": 5000, "evidence": ["P4-L025"], "confidence": 0.9}}
        }
        return LlmResponse(
            content=json.dumps(answer), prompt_tokens=1, completion_tokens=1, llm_ms=1, model=model
        )


def test_long_document_goes_through_retrieval_without_truncation() -> None:
    settings = Settings(_env_file=None, local_model_fast="bielik", local_direct_limit=600)
    llm = FakeLlm()
    retriever = ChromaRetriever(settings, embed=fake_embedder)

    result = Pipeline(settings, LongOcr(), llm, retriever=retriever).extract(
        _blank_pdf(5), None, SCHEMA
    )

    prompt = llm.prompts[0]
    assert all(f"[{line_id}]" in prompt for line_id in FACT_IDS)
    assert not any(warning.startswith("document truncated") for warning in result.warnings)
    assert any(
        warning.startswith("document longer than the model context") for warning in result.warnings
    )
    assert result.usage.context == "retrieval"
    assert result.data["wynagrodzenie"] == 5000 and result.fields["wynagrodzenie"].evidence == [
        "P4-L025"
    ]


def test_short_documents_are_sent_whole() -> None:
    llm = FakeLlm()
    retriever = ChromaRetriever(SETTINGS, embed=fake_embedder)

    result = Pipeline(SETTINGS, LongOcr(), llm, retriever=retriever).extract(
        _blank_pdf(1), None, SCHEMA
    )

    assert result.usage.context == "full"
    assert not any("context" in w or "truncated" in w for w in result.warnings)


def test_without_the_embedding_model_the_document_is_cut_and_says_why(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None, local_model_fast="bielik", local_direct_limit=600, embed_model_dir=tmp_path
    )
    result = Pipeline(settings, LongOcr(), FakeLlm(), retriever=ChromaRetriever(settings)).extract(
        _blank_pdf(5), None, SCHEMA, ExtractionOptions()
    )

    assert result.usage.context == "truncated"
    assert any(
        w.startswith("document truncated") and w.endswith("retrieval unavailable")
        for w in result.warnings
    )


def test_retrieval_can_be_switched_off() -> None:
    settings = Settings(
        _env_file=None, local_model_fast="bielik", local_direct_limit=600, rag_enabled=False
    )
    retriever = ChromaRetriever(settings, embed=fake_embedder)

    result = Pipeline(settings, LongOcr(), FakeLlm(), retriever=retriever).extract(
        _blank_pdf(5), None, SCHEMA
    )

    assert result.usage.context == "truncated"


# --- the real multilingual model (skipped when it is not on disk) -----------------------


@pytest.mark.models
def test_real_embeddings_recall_the_evidence_lines() -> None:
    """Done when: recall of the evidence lines >= 0.9."""
    if not (SETTINGS.embed_model_dir / "config.json").is_file():
        pytest.skip(f"embedding model not found in {SETTINGS.embed_model_dir}")
    ocr = _long_document()

    selected = ChromaRetriever(SETTINGS).select(ocr, SCHEMA, budget=1200)

    recall = len(FACT_IDS & {line.id for line in selected}) / len(FACT_IDS)
    assert recall >= 0.9


def test_document_budget_leaves_room_for_the_prompt_and_the_answer() -> None:
    settings = Settings(_env_file=None, local_model_fast="bielik")
    pipeline = Pipeline(settings, LongOcr(), FakeLlm())

    budget = pipeline._document_budget(SCHEMA)

    assert 8192 - 2000 - 1500 < budget < 8192 - 2000  # prompt overhead and margin come off
    assert budget < settings.local_direct_limit


def test_hosted_profiles_are_bounded_by_direct_limit_only() -> None:
    settings = Settings(
        _env_file=None, groq_api_key="k", groq_model_fast="m", groq_direct_limit=24_000
    )
    pipeline = Pipeline(settings, LongOcr(), FakeLlm(), profile=settings.llm_profile("groq"))
    assert pipeline._document_budget(SCHEMA) == 24_000


def test_progress_reports_each_stage_once_with_metadata_only() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    Pipeline(SETTINGS, LongOcr(), FakeLlm()).extract(
        _blank_pdf(1), None, SCHEMA, progress=lambda stage, details: events.append((stage, details))
    )
    assert [stage for stage, _ in events] == ["reading", "read", "extracting"]
    reading, read, extracting = (details for _, details in events)
    assert reading == {"pages": 1}
    assert set(read) == {"lines", "ocr_ms", "ocr_confidence", "language"}
    assert read["lines"] > 0
    assert extracting["fields"] == len(SCHEMA["properties"])
    # Counts, timings, the language and the model name: never the text the OCR read.
    strings = {v for _, details in events for v in details.values() if isinstance(v, str)}
    assert strings <= {read["language"], extracting["model"]}
