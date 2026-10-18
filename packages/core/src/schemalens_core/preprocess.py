"""Document bytes -> pages ready for OCR. Everything stays in memory."""

import io
from dataclasses import dataclass, field
from typing import Literal

import pypdfium2 as pdfium
from PIL import Image, ImageOps, ImageSequence

from schemalens_core.config import Settings
from schemalens_core.errors import DocumentUnreadable, UnsupportedDocument

DocumentKind = Literal["jpeg", "png", "webp", "tiff", "pdf"]


@dataclass
class Page:
    """One page: either an image to OCR or text taken from a PDF text layer."""

    image: Image.Image | None = None
    text: str | None = None


@dataclass
class Document:
    pages: list[Page]
    warnings: list[str] = field(default_factory=list)


def detect_kind(data: bytes) -> DocumentKind:
    """Detect the document type from magic bytes (never from the file name)."""
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if data.startswith(b"%PDF-"):
        return "pdf"
    raise UnsupportedDocument("unsupported file type (expected JPEG, PNG, WEBP, TIFF or PDF)")


def count_pages(data: bytes) -> int:
    """Number of pages without rendering or OCR; cheap enough to enforce upload limits."""
    kind = detect_kind(data)
    if kind == "pdf":
        try:
            pdf = pdfium.PdfDocument(data)
        except pdfium.PdfiumError as exc:
            raise DocumentUnreadable("PDF could not be opened (corrupted or encrypted)") from exc
        try:
            return len(pdf)
        finally:
            pdf.close()
    if kind == "tiff":
        try:
            with Image.open(io.BytesIO(data)) as img:
                return int(getattr(img, "n_frames", 1))
        except (OSError, ValueError, SyntaxError) as exc:
            raise DocumentUnreadable("image could not be decoded") from exc
    return 1


def load_document(data: bytes, settings: Settings) -> Document:
    """Pages ready for reading."""
    if detect_kind(data) == "pdf":
        return _load_pdf(data, settings)
    return _load_image(data, settings)


def _truncation_warning(total: int, settings: Settings) -> list[str]:
    if total <= settings.max_pages:
        return []
    return [f"document has {total} pages, only the first {settings.max_pages} were processed"]


def _prepare_image(img: Image.Image, settings: Settings) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        # Flatten onto white, otherwise transparent areas turn black.
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    img = img.convert("L")
    if max(img.size) > settings.max_image_side:
        img.thumbnail((settings.max_image_side, settings.max_image_side), Image.Resampling.LANCZOS)
    return img


def _load_image(data: bytes, settings: Settings) -> Document:
    try:
        with Image.open(io.BytesIO(data)) as img:
            # Checked on the header, before any pixel data is decoded (decompression bombs).
            if img.width * img.height > settings.max_image_pixels:
                raise DocumentUnreadable("image dimensions exceed the allowed limit")
            total = getattr(img, "n_frames", 1)
            frames = ImageSequence.Iterator(img)
            pages = [
                Page(image=_prepare_image(frame.copy(), settings))
                for _, frame in zip(range(settings.max_pages), frames, strict=False)
            ]
    except UnsupportedDocument:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise DocumentUnreadable("image could not be decoded") from exc
    return Document(pages=pages, warnings=_truncation_warning(total, settings))


def _usable_text(text: str, settings: Settings) -> bool:
    """A text layer is trusted only if it is long enough and not mostly garbage glyphs."""
    chars = [c for c in text if not c.isspace()]
    if len(chars) < settings.pdf_text_min_chars:
        return False
    return sum(c.isalnum() for c in chars) / len(chars) >= settings.pdf_text_min_alnum_ratio


def _load_pdf(data: bytes, settings: Settings) -> Document:
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise DocumentUnreadable("PDF could not be opened (corrupted or encrypted)") from exc
    try:
        total = len(pdf)
        if total == 0:
            raise DocumentUnreadable("PDF has no pages")
        pages: list[Page] = []
        for index in range(min(total, settings.max_pages)):
            page = pdf[index]
            text = page.get_textpage().get_text_range()
            if _usable_text(text, settings):
                pages.append(Page(text=text))
                continue
            # Never render above max_image_side, whatever the page size claims to be.
            scale = min(
                settings.pdf_render_dpi / 72, settings.max_image_side / max(page.get_size())
            )
            image = page.render(scale=scale).to_pil()
            pages.append(Page(image=_prepare_image(image, settings)))
    except pdfium.PdfiumError as exc:
        raise DocumentUnreadable("PDF could not be rendered") from exc
    finally:
        pdf.close()
    return Document(pages=pages, warnings=_truncation_warning(total, settings))
