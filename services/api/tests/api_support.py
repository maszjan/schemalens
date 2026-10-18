"""Helpers for API tests: a fake pipeline and tiny synthetic documents."""

import io
import threading
import time
from typing import Any

import pypdfium2 as pdfium
from PIL import Image

from schemalens_api.settings import ApiSettings
from schemalens_core.models import ExtractionOptions, ExtractionResult, FieldMeta, Usage
from schemalens_core.preprocess import load_document
from schemalens_core.presets import resolve_schema
from schemalens_core.schemas import compile_schema

# Every call needs a key; the test settings configure this one (conftest.make_settings).
API_KEY = "test-key"
AUTH: dict[str, str] = {"Authorization": f"Bearer {API_KEY}"}

SCHEMA = {
    "type": "object",
    "properties": {"sklep": {"type": "string"}, "suma": {"type": "number"}},
    "required": ["sklep"],
}


def png(size: tuple[int, int] = (40, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", size, 255).save(buffer, format="PNG")
    return buffer.getvalue()


def pdf(pages: int) -> bytes:
    document = pdfium.PdfDocument.new()
    for _ in range(pages):
        document.new_page(595, 842)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


class FakePipeline:
    """Validates like the real pipeline (schema, document), then returns a canned result."""

    def __init__(self, settings: ApiSettings) -> None:
        self._settings = settings
        self.calls: list[dict[str, Any]] = []
        self.gate: threading.Event | None = None  # when set, extract() blocks until released
        self.entered = threading.Event()
        self.delay_s = 0.0
        self.error: Exception | None = None
        self.data: dict[str, Any] | None = None  # canned values, by field name

    def extract(
        self,
        document: bytes,
        mime: str | None,
        schema: dict[str, Any] | None,
        options: ExtractionOptions | None = None,
        *,
        preset: str | None = None,
        progress: Any = None,
    ) -> ExtractionResult:
        options = options or ExtractionOptions()
        self.stages: list[str] = []
        report = progress or (lambda stage, details: self.stages.append(stage))
        report("reading", {"pages": 1})
        report("read", {"lines": 12, "ocr_ms": 120, "ocr_confidence": 0.9, "language": "pl"})
        report("extracting", {"model": "fake-model", "fields": 3})
        self.calls.append(
            {"schema": schema, "preset": preset, "options": options, "bytes": len(document)}
        )
        resolved = resolve_schema(schema, preset, self._settings)
        compile_schema(resolved, self._settings)
        load_document(document, self._settings)
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(timeout=5)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        names = list(resolved["properties"])
        data: dict[str, Any] = dict.fromkeys(names)
        data[names[0]] = "SKLEP EXAMPLE"
        if self.data is not None:
            data = {name: self.data.get(name) for name in names}
        return ExtractionResult(
            data=data,
            fields={
                name: FieldMeta(
                    confidence=0.9 if data[name] else 0.0, evidence=["L001"] if data[name] else []
                )
                for name in names
            },
            warnings=["sklep: possible OCR noise"],
            usage=Usage(
                pages=1,
                ocr_ms=120,
                llm_ms=340,
                prompt_tokens=900,
                completion_tokens=200,
                llm_calls=1,
                model=f"model-{options.model_tier}",
                prompt_version="test",
            ),  # fmt: skip
            ocr_text=["[L001] SKLEP EXAMPLE"] if options.include_ocr_text else None,
        )
