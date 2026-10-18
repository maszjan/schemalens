import io

import pytest
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

from schemalens_core.config import Settings
from schemalens_core.models import OcrLine, OcrPage, OcrResult
from schemalens_core.ocr import EasyOcrEngine, Fragment, group_into_lines, read_document
from schemalens_core.preprocess import Document, Page, load_document


def _frag(text: str, x: int, y: int, w: int = 100, h: int = 20, conf: float = 0.9) -> Fragment:
    return Fragment(text=text, bbox=(x, y, x + w, y + h), conf=conf)


def test_fragments_on_same_baseline_form_one_line_sorted_by_x() -> None:
    lines = group_into_lines(
        [_frag("123,45", 400, 102), _frag("Razem:", 10, 100), _frag("PLN", 520, 98)], 0.5
    )

    assert [line.text for line in lines] == ["Razem: 123,45 PLN"]
    assert lines[0].bbox == (10, 98, 620, 122)


def test_lines_are_ordered_top_to_bottom() -> None:
    lines = group_into_lines(
        [_frag("third", 10, 90), _frag("first", 10, 10), _frag("second", 300, 50)], 0.5
    )
    assert [line.text for line in lines] == ["first", "second", "third"]


def test_slight_vertical_overlap_does_not_merge_lines() -> None:
    lines = group_into_lines([_frag("upper", 10, 100), _frag("lower", 10, 115)], 0.5)
    assert [line.text for line in lines] == ["upper", "lower"]


def test_line_confidence_is_weighted_by_text_length() -> None:
    lines = group_into_lines(
        [_frag("aaaaaaaaa", 10, 10, conf=1.0), _frag("b", 200, 10, conf=0.0)], 0.5
    )
    assert lines[0].conf == pytest.approx(0.9)


def test_empty_fragments_are_dropped() -> None:
    assert group_into_lines([_frag("  ", 10, 10)], 0.5) == []


class FakeOcr:
    """Returns one line per image, numbered the way a real engine would."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def read(self, images: list[Image.Image]) -> OcrResult:
        self.calls.append(len(images))
        multi = len(images) > 1
        return OcrResult(
            lines=[
                OcrLine(
                    id=f"P{n}-L001" if multi else "L001",
                    page=n,
                    text=f"scan {n}",
                    bbox=(0, 0, 10, 10),
                    conf=0.8,
                )
                for n in range(1, len(images) + 1)
            ],
            pages=[OcrPage(number=n, source="ocr", ocr_ms=7) for n in range(1, len(images) + 1)],
        )


def _blank() -> Image.Image:
    return Image.new("L", (10, 10), 255)


def test_single_page_ids_have_no_page_prefix() -> None:
    result = read_document(Document(pages=[Page(image=_blank())]), FakeOcr())

    assert [line.id for line in result.lines] == ["L001"]
    assert result.full_text == "[L001] scan 1"
    assert result.ocr_ms == 7


def test_text_layer_pages_skip_ocr_and_keep_page_order() -> None:
    engine = FakeOcr()
    document = Document(
        pages=[
            Page(image=_blank()),
            Page(text="Invoice 1\n\n  Total 10  \n"),
            Page(image=_blank()),
        ],
        warnings=["truncated"],
    )

    result = read_document(document, engine)

    assert engine.calls == [2]
    assert [(line.id, line.text, line.source) for line in result.lines] == [
        ("P1-L001", "scan 1", "ocr"),
        ("P2-L001", "Invoice 1", "pdf_text"),
        ("P2-L002", "Total 10", "pdf_text"),
        ("P3-L001", "scan 2", "ocr"),
    ]
    assert [(p.number, p.source, p.ocr_ms) for p in result.pages] == [
        (1, "ocr", 7),
        (2, "pdf_text", 0),
        (3, "ocr", 7),
    ]
    assert result.warnings == ["truncated"]


ROWS = ["FAKTURA VAT 12/2026", "Sprzedawca: Example Sp. z o.o.", "Razem do zaplaty: 1234,56 PLN"]


def _synthetic_invoice() -> bytes:
    image = Image.new("RGB", (1000, 400), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    for i, row in enumerate(ROWS):
        draw.text((40, 40 + i * 110), row, fill="black", font=font)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.ocr
def test_easyocr_reads_synthetic_invoice() -> None:
    settings = Settings(_env_file=None)
    if not any(settings.ocr_model_dir.glob("*.pth")):
        pytest.skip(f"EasyOCR models not found in {settings.ocr_model_dir}")

    result = read_document(load_document(_synthetic_invoice(), settings), EasyOcrEngine(settings))

    assert [line.id for line in result.lines] == ["L001", "L002", "L003"]
    assert "FAKTURA" in result.lines[0].text.upper()
    assert "EXAMPLE" in result.lines[1].text.upper()
    assert "1234" in result.lines[2].text
    assert all(line.bbox is not None and 0 < line.conf <= 1 for line in result.lines)
    assert result.pages[0].ocr_ms > 0


def test_text_layer_pdf_end_to_end_needs_no_ocr() -> None:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for rows in (
        ["Invoice FV/12/2026", "Seller: Example Ltd."],
        ["Total due: 1234.56 PLN", "Thank you"],
    ):
        pdf.add_page()
        for row in rows:
            pdf.cell(text=row, new_x="LMARGIN", new_y="NEXT")
    engine = FakeOcr()

    result = read_document(load_document(bytes(pdf.output()), Settings(_env_file=None)), engine)

    assert engine.calls == [0]
    assert [(line.id, line.text) for line in result.lines] == [
        ("P1-L001", "Invoice FV/12/2026"),
        ("P1-L002", "Seller: Example Ltd."),
        ("P2-L001", "Total due: 1234.56 PLN"),
        ("P2-L002", "Thank you"),
    ]
    assert all(line.source == "pdf_text" and line.conf == 1.0 for line in result.lines)
    assert result.ocr_ms == 0
