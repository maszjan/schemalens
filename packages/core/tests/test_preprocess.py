import io

import pypdfium2 as pdfium
import pytest
from fpdf import FPDF
from PIL import Image

from schemalens_core.config import Settings
from schemalens_core.errors import DocumentUnreadable, UnsupportedDocument
from schemalens_core.preprocess import count_pages, detect_kind, load_document


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def _image_bytes(size: tuple[int, int], fmt: str, mode: str = "RGB", **save_kwargs) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, "white").save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def _blank_pdf(pages: int, size: tuple[float, float] = (595, 842)) -> bytes:
    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(*size)
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


@pytest.mark.parametrize(
    ("fmt", "kind"), [("JPEG", "jpeg"), ("PNG", "png"), ("WEBP", "webp"), ("TIFF", "tiff")]
)
def test_detect_kind_images(fmt: str, kind: str) -> None:
    assert detect_kind(_image_bytes((10, 10), fmt)) == kind


def test_detect_kind_pdf() -> None:
    assert detect_kind(_blank_pdf(1)) == "pdf"


@pytest.mark.parametrize("data", [b"", b"GIF89a....", b"PK\x03\x04", b"hello world"])
def test_detect_kind_rejects_other_types(data: bytes) -> None:
    with pytest.raises(UnsupportedDocument):
        detect_kind(data)


def test_image_is_grayscale_and_downscaled(settings: Settings) -> None:
    doc = load_document(_image_bytes((4000, 1000), "JPEG"), settings)

    assert len(doc.pages) == 1
    image = doc.pages[0].image
    assert image is not None
    assert image.mode == "L"
    assert image.size == (2000, 500)
    assert doc.warnings == []


def test_small_image_is_not_upscaled(settings: Settings) -> None:
    doc = load_document(_image_bytes((300, 200), "PNG"), settings)
    assert doc.pages[0].image.size == (300, 200)


def test_exif_orientation_is_applied(settings: Settings) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW
    doc = load_document(_image_bytes((300, 100), "JPEG", exif=exif), settings)
    assert doc.pages[0].image.size == (100, 300)


def test_transparent_png_is_flattened_onto_white(settings: Settings) -> None:
    buf = io.BytesIO()
    Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(buf, format="PNG")
    doc = load_document(buf.getvalue(), settings)
    assert doc.pages[0].image.getpixel((5, 5)) == 255


def test_oversized_image_is_rejected_before_decoding() -> None:
    settings = Settings(_env_file=None, max_image_pixels=1000)
    with pytest.raises(UnsupportedDocument):
        load_document(_image_bytes((100, 100), "PNG"), settings)


def test_truncated_image_is_rejected(settings: Settings) -> None:
    with pytest.raises(UnsupportedDocument):
        load_document(_image_bytes((100, 100), "PNG")[:40], settings)


def test_pdf_without_text_layer_is_rendered(settings: Settings) -> None:
    doc = load_document(_blank_pdf(2), settings)

    assert len(doc.pages) == 2
    for page in doc.pages:
        assert page.text is None
        assert page.image is not None
        assert page.image.mode == "L"
        assert max(page.image.size) <= settings.max_image_side


def test_huge_pdf_page_is_rendered_within_limit(settings: Settings) -> None:
    doc = load_document(_blank_pdf(1, size=(14400, 14400)), settings)
    assert max(doc.pages[0].image.size) <= settings.max_image_side


def test_pdf_over_page_limit_is_truncated_with_warning(settings: Settings) -> None:
    doc = load_document(_blank_pdf(7), settings)

    assert len(doc.pages) == settings.max_pages
    assert doc.warnings == ["document has 7 pages, only the first 5 were processed"]


def test_broken_pdf_is_rejected(settings: Settings) -> None:
    with pytest.raises(UnsupportedDocument):
        load_document(b"%PDF-1.7\nnot really a pdf", settings)


def _text_pdf(pages: list[list[str]]) -> bytes:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for rows in pages:
        pdf.add_page()
        for row in rows:
            pdf.cell(text=row, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def test_pdf_text_layer_is_used_instead_of_rendering(settings: Settings) -> None:
    doc = load_document(_text_pdf([["Invoice FV/12/2026", "Total due: 1234.56 PLN"]]), settings)

    assert len(doc.pages) == 1
    assert doc.pages[0].image is None
    assert doc.pages[0].text is not None
    assert doc.pages[0].text.split("\r\n") == ["Invoice FV/12/2026", "Total due: 1234.56 PLN"]


def test_pdf_with_too_little_text_is_rendered(settings: Settings) -> None:
    doc = load_document(_text_pdf([["p. 1"]]), settings)

    assert doc.pages[0].text is None
    assert doc.pages[0].image is not None


def test_unreadable_documents_are_told_apart_from_unsupported_types(settings: Settings) -> None:
    with pytest.raises(DocumentUnreadable):
        load_document(_image_bytes((100, 100), "PNG")[:40], settings)
    with pytest.raises(DocumentUnreadable):
        load_document(b"%PDF-1.7\nnot really a pdf", settings)
    with pytest.raises(UnsupportedDocument) as excinfo:
        load_document(b"GIF89a....", settings)
    assert not isinstance(excinfo.value, DocumentUnreadable)
    assert DocumentUnreadable.code == "document_unreadable"


def test_count_pages_without_rendering() -> None:
    assert count_pages(_image_bytes((10, 10), "PNG")) == 1
    assert count_pages(_blank_pdf(3)) == 3
    assert count_pages(_text_pdf([["a"], ["b"]])) == 2
    with pytest.raises(DocumentUnreadable):
        count_pages(b"%PDF-1.7\nnot really a pdf")
    with pytest.raises(UnsupportedDocument):
        count_pages(b"hello")
