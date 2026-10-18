"""Download OCR + embedding models at image build time (runtime is offline)."""

import sys
from pathlib import Path

import easyocr
from sentence_transformers import SentenceTransformer

target = Path(sys.argv[1] if len(sys.argv) > 1 else "/models")
(target / "easyocr").mkdir(parents=True, exist_ok=True)
easyocr.Reader(
    ["pl", "en"], gpu=False, model_storage_directory=str(target / "easyocr"), verbose=False
)
SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").save(
    str(target / "embed")
)
print(f"models ready in {target}")
