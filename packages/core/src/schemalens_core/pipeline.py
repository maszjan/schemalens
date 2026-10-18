"""End-to-end extraction: bytes -> OCR -> LLM -> normalization -> validation -> result."""

from collections.abc import Callable
from functools import cache
from typing import Any, Literal

from schemalens_core.config import LlmProfile, Settings, get_settings
from schemalens_core.context import estimate_tokens, fit_to_context
from schemalens_core.errors import LlmFailed
from schemalens_core.fewshot import example_block, load_templates
from schemalens_core.language import detect_language
from schemalens_core.llm import LlmClient, OpenAICompatibleClient, extract_fields
from schemalens_core.models import ExtractionOptions, ExtractionResult, OcrLine, OcrResult, Usage
from schemalens_core.ocr import (
    EasyOcrEngine,
    OcrEngine,
    ensure_readable,
    mean_confidence,
    read_document,
)
from schemalens_core.preprocess import detect_kind, load_document
from schemalens_core.presets import resolve_schema
from schemalens_core.prompts import SYSTEM_PROMPT, build_user_prompt
from schemalens_core.rag import ChromaRetriever, RetrievalUnavailable, Retriever
from schemalens_core.schemas import compile_schema

# Stages reported to `progress` while a document is processed, each with metadata only:
#   reading     {"pages"}                                      OCR starts
#   read        {"lines", "ocr_ms", "ocr_confidence", "language"}  OCR done
#   extracting  {"model", "fields"}                            the LLM call starts
Stage = Literal["reading", "read", "extracting"]
Progress = Callable[[Stage, dict[str, Any]], None]

_MIME_TYPES = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "tiff": "image/tiff",
    "pdf": "application/pdf",
}


class Pipeline:
    """Document-type agnostic: all it knows comes from the schema.

    The OCR text goes to the LLM in one request. When it exceeds the context budget of the
    LLM profile, retrieval selects the relevant lines, or (without retrieval) the text is
    cut at a page boundary with a warning.
    """

    def __init__(
        self,
        settings: Settings,
        ocr: OcrEngine,
        llm: LlmClient,
        profile: LlmProfile | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self._settings = settings
        self._profile = profile or settings.llm_profile()
        self._retriever = retriever if settings.rag_enabled else None
        self._ocr = ocr
        self._llm = llm

    def warm_up(self) -> None:
        """Load what is slow to load (the OCR models), e.g. on a scheduled ping."""
        warm = getattr(self._ocr, "warm_up", None)
        if warm is not None:
            warm()

    def _example(self, preset: str | None, schema: dict[str, Any]) -> tuple[str | None, str | None]:
        """The worked example for a preset request, when few-shot is on and a template exists."""
        if not (preset and self._settings.fewshot_enabled):
            return None, None
        template = load_templates(self._settings.templates_dir).get(preset)
        if template is None:
            return None, None
        return example_block(template, schema), preset

    def _document_budget(self, schema: dict[str, Any], example: str | None = None) -> int:
        """Tokens the document may take: `direct_limit`, and never more than what the model's
        context window leaves after the prompt and the reserved answer."""
        profile = self._profile
        budget = profile.direct_limit
        if profile.context_window:
            prompt = SYSTEM_PROMPT + build_user_prompt([], schema, example)
            overhead = estimate_tokens(prompt, self._settings.token_estimate_chars)
            reserved = (profile.max_output_tokens or 0) + _CONTEXT_MARGIN
            budget = min(budget, profile.context_window - overhead - reserved)
        return max(budget, 0)

    def _fit(
        self, ocr: OcrResult, schema: dict[str, Any], example: str | None = None
    ) -> tuple[list[OcrLine], Literal["full", "retrieval", "truncated"], str | None]:
        """Lines for the model: all of them, a retrieved selection, or a page-boundary cut."""
        limit = self._document_budget(schema, example)
        lines, truncation = fit_to_context(ocr, limit, self._settings.token_estimate_chars)
        if truncation is None:
            return lines, "full", None
        if self._retriever is not None:
            try:
                selected = self._retriever.select(ocr, schema, limit)
            except RetrievalUnavailable:
                return lines, "truncated", f"{truncation}; retrieval unavailable"
            return selected, "retrieval", _context_note(len(selected), len(ocr.lines))
        return lines, "truncated", truncation

    def extract(
        self,
        document: bytes,
        mime: str | None,
        schema: dict[str, Any] | None,
        options: ExtractionOptions | None = None,
        *,
        preset: str | None = None,
        progress: Progress | None = None,
    ) -> ExtractionResult:
        """Extract fields from a document, described by `schema`, a `preset`, or both.

        With both, the schema extends the preset (docs/api.md).

        `mime` is only a declaration: the type is always detected from the content.
        `progress` is told about each stage (see `Stage`) with counts and timings, so a
        caller can show what is happening; it never gets document text or values.

        Raises `InvalidSchema`, `UnsupportedDocument` (`DocumentUnreadable` when the OCR
        finds too little readable text), `OcrFailed` or `LlmFailed`;
        everything else (missing fields, weak evidence, ...) is reported in `warnings`.
        """
        options = options or ExtractionOptions()
        settings = self._settings
        # Fail fast, before the expensive OCR.
        compiled = compile_schema(resolve_schema(schema, preset, settings), settings)
        model = (
            self._profile.model_accurate
            if options.model_tier == "accurate"
            else self._profile.model_fast
        )
        if not model:
            raise LlmFailed(
                f"LLM model for tier '{options.model_tier}' is not configured "
                f"in profile '{self._profile.name}'"
            )

        warnings: list[str] = []
        detected = _MIME_TYPES[detect_kind(document)]
        if mime and mime.split(";")[0].strip().lower() != detected:
            warnings.append(f"declared content type differs from the detected one ({detected})")

        report: Progress = progress or (lambda stage, details: None)
        loaded = load_document(document, settings)
        report("reading", {"pages": len(loaded.pages)})
        ocr = read_document(loaded, self._ocr)
        warnings += ocr.warnings
        ocr_confidence = round(mean_confidence(ocr), 3)
        language = detect_language(" ".join(line.text for line in ocr.lines))
        report(
            "read",
            {
                "lines": len(ocr.lines),
                "ocr_ms": ocr.ocr_ms,
                "ocr_confidence": ocr_confidence,
                "language": language,
            },
        )
        ensure_readable(ocr, settings)  # no LLM call for documents OCR could not read

        example, fewshot = self._example(preset, compiled.source)
        lines, context, note = self._fit(ocr, compiled.source, example)
        if note:
            warnings.append(note)
        report("extracting", {"model": model, "fields": len(compiled.source.get("properties", {}))})
        extraction = extract_fields(
            self._llm,
            compiled,
            lines,
            settings,
            model=model,
            profile=self._profile,
            example=example,
        )

        data, fields = extraction.data, extraction.fields
        warnings += extraction.warnings
        llm = extraction.usage
        usage = Usage(
            pages=len(ocr.pages),
            ocr_ms=ocr.ocr_ms,
            llm_ms=llm.llm_ms,
            prompt_tokens=llm.prompt_tokens,
            completion_tokens=llm.completion_tokens,
            llm_calls=llm.calls,
            model=llm.model,
            prompt_version=llm.prompt_version,
            reasoning_effort=llm.reasoning_effort,
            effort_escalations=llm.escalations,
            context=context,
            fewshot=fewshot,
            ocr_confidence=ocr_confidence,
            language=language,
        )

        warnings += [
            f"{name}: required field was not found in the document"
            for name in compiled.missing_required(data)
        ]
        return ExtractionResult(
            data=data,
            fields=fields,
            warnings=warnings,
            usage=usage,
            ocr_text=[f"[{line.id}] {line.text}" for line in ocr.lines]
            if options.include_ocr_text
            else None,
        )


# Room for the error of the token estimate (characters / 3.5 is only an approximation).
_CONTEXT_MARGIN = 512


def _context_note(selected: int, total: int) -> str:
    return (
        f"document longer than the model context: {selected} of {total} lines selected by retrieval"
    )


def build_pipeline(settings: Settings | None = None, profile: LlmProfile | None = None) -> Pipeline:
    """Real OCR and LLM client for `profile` (default: the active profile)."""
    settings = settings or get_settings()
    profile = profile or settings.llm_profile()
    llm = OpenAICompatibleClient(settings, profile)
    retriever = ChromaRetriever(settings) if settings.rag_enabled else None
    return Pipeline(settings, EasyOcrEngine(settings), llm, profile, retriever)


@cache
def _default_pipeline() -> Pipeline:
    return build_pipeline()


def extract(
    document: bytes,
    mime: str | None,
    schema: dict[str, Any] | None,
    options: ExtractionOptions | None = None,
    *,
    preset: str | None = None,
) -> ExtractionResult:
    """Extract with the process-wide pipeline configured from the environment."""
    return _default_pipeline().extract(document, mime, schema, options, preset=preset)
