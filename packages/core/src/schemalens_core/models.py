"""Data models shared across the pipeline."""

from typing import Any, Literal

from pydantic import BaseModel

BBox = tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels of the preprocessed page
LineSource = Literal["ocr", "pdf_text"]


class OcrLine(BaseModel):
    id: str  # "L001", or "P2-L014" for multi-page documents; used as evidence
    page: int  # 1-based
    text: str
    bbox: BBox | None  # None for lines taken from a PDF text layer
    conf: float
    source: LineSource = "ocr"


class OcrPage(BaseModel):
    number: int  # 1-based
    source: LineSource
    ocr_ms: int


class OcrResult(BaseModel):
    lines: list[OcrLine]
    pages: list[OcrPage]
    warnings: list[str] = []

    @property
    def ocr_ms(self) -> int:
        return sum(p.ocr_ms for p in self.pages)

    @property
    def full_text(self) -> str:
        """Numbered lines in reading order, as shown to the LLM."""
        return "\n".join(f"[{line.id}] {line.text}" for line in self.lines)


class FieldMeta(BaseModel):
    confidence: float
    evidence: list[str]  # ids of the OCR lines the value was taken from


class ExtractionOptions(BaseModel):
    model_tier: Literal["fast", "accurate"] = "fast"
    include_ocr_text: bool = False


class Usage(BaseModel):
    pages: int
    ocr_ms: int
    llm_ms: int
    prompt_tokens: int
    completion_tokens: int
    llm_calls: int
    model: str | None
    prompt_version: str
    reasoning_effort: str | None = None  # effort that produced the answer (reasoning models)
    # How the text met the model context: whole, selected by retrieval, or cut at a page.
    context: Literal["full", "retrieval", "truncated"] = "full"
    fewshot: str | None = None  # preset whose worked example was in the prompt
    # Mean OCR confidence (weighted by text length): how cleanly the document was read.
    ocr_confidence: float | None = None
    language: str | None = None  # ISO 639-1 code of the document's language, when it is clear
    effort_escalations: int = 0  # times the effort was raised after a failed generation


class ExtractionResult(BaseModel):
    data: dict[str, Any]  # every schema key present; missing -> None
    fields: dict[str, FieldMeta]
    warnings: list[str]
    usage: Usage
    ocr_text: list[str] | None = None  # numbered OCR lines, only with include_ocr_text
