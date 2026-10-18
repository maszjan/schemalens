"""Synthetic documents for trying the panel by hand: python scripts/make_demo_docs.py

Writes scans of every preset's example (and a PDF) to samples/demo/ (gitignored). Every value
in them is made up; nothing here is a real document.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.synth import PRESETS_DIR, preset_document, preset_lines, render_text_pdf  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "samples" / "demo"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(path.stem for path in PRESETS_DIR.glob("*.json"))
    for name in names:
        (OUT / f"{name}.jpg").write_bytes(preset_document(name, fmt="JPEG", seed=7))
    # A second receipt (another "photo" of the same kind) and a PDF with a text layer.
    (OUT / "receipt-2.png").write_bytes(preset_document("receipt", fmt="PNG", seed=21))
    (OUT / "invoice.pdf").write_bytes(render_text_pdf([preset_lines("invoice")]))
    print(f"{len(names) + 2} files in {OUT}")


if __name__ == "__main__":
    main()
