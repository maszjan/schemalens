"""OCR: EasyOCR engine, grouping of fragments into lines, reading order, line ids."""

import os
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from schemalens_core.config import Settings
from schemalens_core.errors import DocumentUnreadable, OcrFailed
from schemalens_core.models import BBox, OcrLine, OcrPage, OcrResult
from schemalens_core.preprocess import Document


@dataclass
class Fragment:
    """A single text box as returned by the OCR engine."""

    text: str
    bbox: BBox
    conf: float


class OcrEngine(Protocol):
    def read(self, images: list[Image.Image]) -> OcrResult: ...


def group_into_lines(fragments: list[Fragment], min_overlap: float) -> list[Fragment]:
    """Merge fragments into lines, top to bottom, left to right within a line.

    A fragment joins a line when their vertical overlap exceeds `min_overlap`
    of the smaller of the two heights.
    """
    rows: list[list[Fragment]] = []
    spans: list[tuple[float, float]] = []  # running mean (top, bottom) of each row
    for frag in sorted(fragments, key=lambda f: (f.bbox[1] + f.bbox[3]) / 2):
        if not frag.text.strip():
            continue
        top, bottom = frag.bbox[1], frag.bbox[3]
        if rows:
            row_top, row_bottom = spans[-1]
            overlap = min(bottom, row_bottom) - max(top, row_top)
            if overlap > min_overlap * min(bottom - top, row_bottom - row_top):
                n = len(rows[-1])
                rows[-1].append(frag)
                spans[-1] = ((row_top * n + top) / (n + 1), (row_bottom * n + bottom) / (n + 1))
                continue
        rows.append([frag])
        spans.append((top, bottom))

    lines = []
    for row in rows:
        row.sort(key=lambda f: f.bbox[0])
        weights = [len(f.text) for f in row]
        lines.append(
            Fragment(
                text=" ".join(f.text.strip() for f in row),
                bbox=(
                    min(f.bbox[0] for f in row),
                    min(f.bbox[1] for f in row),
                    max(f.bbox[2] for f in row),
                    max(f.bbox[3] for f in row),
                ),
                conf=sum(f.conf * w for f, w in zip(row, weights, strict=True)) / sum(weights),
            )
        )
    return lines


def _line_id(page: int, index: int, multi_page: bool) -> str:
    return f"P{page}-L{index:03d}" if multi_page else f"L{index:03d}"


def _text_layer_lines(text: str, page: int, multi_page: bool) -> list[OcrLine]:
    rows = [row.strip() for row in text.splitlines() if row.strip()]
    return [
        OcrLine(
            id=_line_id(page, i, multi_page),
            page=page,
            text=row,
            bbox=None,
            conf=1.0,
            source="pdf_text",
        )
        for i, row in enumerate(rows, start=1)
    ]


def read_document(document: Document, engine: OcrEngine) -> OcrResult:
    """OCR the image pages and merge them with PDF text-layer pages, in page order."""
    multi_page = len(document.pages) > 1
    image_pages = [(n, p.image) for n, p in enumerate(document.pages, start=1) if p.image]
    ocr = engine.read([image for _, image in image_pages])
    # The engine numbers the images it was given 1..n; map them back to document pages.
    page_numbers = {i: n for i, (n, _) in enumerate(image_pages, start=1)}
    ocr_lines: dict[int, list[OcrLine]] = {}
    for line in ocr.lines:
        ocr_lines.setdefault(page_numbers[line.page], []).append(line)
    ocr_ms = {page_numbers[p.number]: p.ocr_ms for p in ocr.pages}

    result = OcrResult(lines=[], pages=[], warnings=[*document.warnings, *ocr.warnings])
    for number, page in enumerate(document.pages, start=1):
        if page.text is not None:
            result.lines += _text_layer_lines(page.text, number, multi_page)
            result.pages.append(OcrPage(number=number, source="pdf_text", ocr_ms=0))
            continue
        result.lines += [
            line.model_copy(update={"id": _line_id(number, i, multi_page), "page": number})
            for i, line in enumerate(ocr_lines.get(number, []), start=1)
        ]
        result.pages.append(OcrPage(number=number, source="ocr", ocr_ms=ocr_ms[number]))
    return result


def mean_confidence(result: OcrResult) -> float:
    """Mean OCR confidence weighted by text length (0 when nothing was read)."""
    characters = sum(len(line.text) for line in result.lines)
    if not characters:
        return 0.0
    return sum(line.conf * len(line.text) for line in result.lines) / characters


def ensure_readable(result: OcrResult, settings: Settings) -> None:
    """Quality gate: refuse documents the OCR could not really read.

    Sending a few low-confidence fragments to the LLM only invites invented values, so too
    little text or a low mean confidence (weighted by text length) stops the pipeline here.
    """
    mean = mean_confidence(result)
    if len(result.lines) >= settings.ocr_min_lines and mean >= settings.ocr_min_mean_conf:
        return
    raise DocumentUnreadable(
        f"too little readable text was recognized ({len(result.lines)} lines, mean OCR confidence "
        f"{mean:.2f}; needed at least {settings.ocr_min_lines} lines and "
        f"{settings.ocr_min_mean_conf:g}). The document may be handwritten, illegible or "
        "photographed at too low a quality; printed text is supported."
    )


@cache
def _get_reader(langs: tuple[str, ...], gpu: bool, model_dir: Path, allow_download: bool) -> Any:
    # Imported lazily: pulls in torch, which unit tests and light adapters must not pay for.
    import easyocr
    import torch

    torch.set_num_threads(os.cpu_count() or 1)
    if allow_download:
        model_dir.mkdir(parents=True, exist_ok=True)
    try:
        return easyocr.Reader(
            list(langs),
            gpu=gpu,
            model_storage_directory=str(model_dir),
            download_enabled=allow_download,
            verbose=False,
        )
    except FileNotFoundError as exc:
        raise OcrFailed(
            f"EasyOCR models not found in {model_dir}; "
            "run once with OCR_ALLOW_DOWNLOAD=1 or bake them into the image"
        ) from exc


class EasyOcrEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _reader(self) -> Any:
        s = self._settings
        return _get_reader(s.ocr_langs, s.ocr_gpu, s.ocr_model_dir, s.ocr_allow_download)

    def warm_up(self) -> None:
        """Load the models now (a warm-up ping), not on the first document."""
        self._reader()

    def read(self, images: list[Image.Image]) -> OcrResult:
        import numpy as np

        if not images:
            return OcrResult(lines=[], pages=[])
        reader = self._reader()
        multi_page = len(images) > 1
        result = OcrResult(lines=[], pages=[])
        for number, image in enumerate(images, start=1):
            started = time.perf_counter()
            try:
                raw = reader.readtext(np.asarray(image), detail=1, paragraph=False)
            except Exception as exc:
                # Type name only, no chaining: engine messages could echo document data.
                raise OcrFailed(
                    f"OCR engine failed on page {number} ({type(exc).__name__})"
                ) from None
            fragments = [
                Fragment(
                    text=text,
                    bbox=(
                        int(min(x for x, _ in quad)),
                        int(min(y for _, y in quad)),
                        int(max(x for x, _ in quad)),
                        int(max(y for _, y in quad)),
                    ),
                    conf=float(conf),
                )
                for quad, text, conf in raw
            ]
            lines = group_into_lines(fragments, self._settings.ocr_line_overlap)
            result.lines += [
                OcrLine(
                    id=_line_id(number, i, multi_page),
                    page=number,
                    text=line.text,
                    bbox=line.bbox,
                    conf=round(line.conf, 4),
                )
                for i, line in enumerate(lines, start=1)
            ]
            ocr_ms = int((time.perf_counter() - started) * 1000)
            result.pages.append(OcrPage(number=number, source="ocr", ocr_ms=ocr_ms))
        return result
