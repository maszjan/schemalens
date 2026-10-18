"""Synthetic documents: text rows rendered as a slightly degraded scan. In memory only."""

import io
import json
import random
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_PATH = Path(__file__).parent.parent / "packages/core/tests/assets/fonts/DejaVuSans.ttf"

PRESETS_DIR = Path(__file__).parent.parent / "presets"

# Documents that fit no preset; used with generic_document or a custom schema.
EXTRA_DOCUMENTS: dict[str, list[str]] = {
    "meeting_note": [
        "Notes from the project team meeting",
        "Date: 8 October 2026",
        "Present: Anna Example, John Test, Eve Demo",
        "1. Release of the new app version moved to 20.10.2026.",
        "2. John Test will price the server by Friday.",
        "3. Testing budget: 12,000 GBP, supplier Example Cloud Ltd.",
        "Next meeting: 15.10.2026",
    ],
}

_FONT_SIZE = 30
_LINE_HEIGHT = 56
_MARGIN = 50


def render_scan(
    rows: list[str], *, fmt: Literal["PNG", "JPEG"] = "PNG", seed: int = 7, angle: float = 2.0
) -> bytes:
    """Render rows with DejaVu Sans, then add seeded noise, a small rotation and blur."""
    font = ImageFont.truetype(str(FONT_PATH), _FONT_SIZE)
    width = int(max(font.getlength(row) for row in rows)) + 2 * _MARGIN
    image = Image.new("L", (width, 2 * _MARGIN + _LINE_HEIGHT * len(rows)), 255)
    draw = ImageDraw.Draw(image)
    for i, row in enumerate(rows):
        draw.text((_MARGIN, _MARGIN + _LINE_HEIGHT * i), row, fill=20, font=font)
    # Seeded, so every run produces exactly the same pixels.
    noise = Image.frombytes(
        "L", image.size, random.Random(seed).randbytes(image.width * image.height)
    )
    image = Image.blend(image, noise, 0.12)
    image = image.rotate(angle, expand=True, fillcolor=255, resample=Image.Resampling.BICUBIC)
    image = image.filter(ImageFilter.GaussianBlur(0.7))
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **({"quality": 80} if fmt == "JPEG" else {}))
    return buffer.getvalue()


def preset_lines(name: str) -> list[str]:
    """Text lines of the synthetic example document shipped with a preset."""
    preset = json.loads((PRESETS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    lines: list[str] = preset["example"]["lines"]
    return lines


def preset_document(name: str, *, fmt: Literal["PNG", "JPEG"] = "JPEG", seed: int = 7) -> bytes:
    """The preset's example document rendered as a degraded scan."""
    return render_scan(preset_lines(name), fmt=fmt, seed=seed)


def render_text_pdf(pages: list[list[str]]) -> bytes:
    """A PDF with a real text layer (no OCR needed), one list of lines per page."""
    from fpdf import FPDF  # dev dependency: only evals and tests build PDFs

    pdf = FPDF()
    pdf.add_font("DejaVu", "", str(FONT_PATH))
    pdf.set_font("DejaVu", size=9)
    pdf.set_auto_page_break(False)
    for rows in pages:
        pdf.add_page()
        for row in rows:
            pdf.cell(text=row, new_x="LMARGIN", new_y="NEXT", h=5.6)
    return bytes(pdf.output())
