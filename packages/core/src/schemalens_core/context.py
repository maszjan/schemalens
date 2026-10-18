"""Fitting a document into the context of the LLM profile. Never silently.

Direct mode sends the whole OCR text in one request. A document longer than the profile's
limit is cut at a page boundary and the cut is reported as a warning; retrieval (RAG) for
long documents is a later stage.
"""

import math

from schemalens_core.models import OcrLine, OcrResult


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return math.ceil(len(text) / chars_per_token)


def _cost(lines: list[OcrLine], chars_per_token: float) -> int:
    return estimate_tokens("\n".join(f"[{line.id}] {line.text}" for line in lines), chars_per_token)


def fit_to_context(
    ocr: OcrResult, limit: int, chars_per_token: float
) -> tuple[list[OcrLine], str | None]:
    """Lines that fit in `limit` tokens, whole pages first, and a warning if anything was cut."""
    if _cost(ocr.lines, chars_per_token) <= limit:
        return ocr.lines, None

    by_page: dict[int, list[OcrLine]] = {}
    for line in ocr.lines:
        by_page.setdefault(line.page, []).append(line)
    kept: list[OcrLine] = []
    kept_pages = 0
    for lines in by_page.values():
        if _cost([*kept, *lines], chars_per_token) > limit:
            break
        kept += lines
        kept_pages += 1

    note = ""
    if kept_pages == 0:
        # Even the first page is too long: keep as many of its lines as fit.
        first = next(iter(by_page.values()))
        while len(kept) < len(first) and _cost(first[: len(kept) + 1], chars_per_token) <= limit:
            kept.append(first[len(kept)])
        kept_pages = 1
        note = ", first page cut short"
    warning = f"document truncated to fit context ({kept_pages} of {len(by_page)} pages{note})"
    return kept, warning
