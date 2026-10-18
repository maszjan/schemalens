"""Heuristics for text that looks damaged by OCR.

Plain text fields are never corrected (we cannot know what the text should be), but a value
such as `Sp. Z 0.0` or `example. com` should not carry a high confidence. Everything here
only detects; the consequence (warning and penalty) is applied in scoring.
"""

import re
from typing import Any

from schemalens_core.normalize import semantic_type
from schemalens_core.schemas import type_of

# Digits and letters OCR commonly confuses with each other: 0/O, 1/l/I, 5/S, 8/B, 2/Z.
_DIGIT_LOOKALIKES = "01582"
_LETTER_LOOKALIKES = "OoIlSBZ"

_LETTER = r"[^\W\d_]"  # any Unicode letter
_WORD_START = r"(?<![^\W_])"  # not preceded by a letter or digit
_WORD_END = r"(?![^\W_])"

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # A look-alike digit inside a word: "P0lska", "Kowa1ski". Short codes (B2B, H2O) are skipped.
    re.compile(
        rf"{_WORD_START}(?=[^\W_]{{4,}}{_WORD_END}){_LETTER}+[{_DIGIT_LOOKALIKES}]{_LETTER}+{_WORD_END}"
    ),
    # A look-alike letter inside a number: "2O26", "1l5".
    re.compile(rf"{_WORD_START}\d+[{_LETTER_LOOKALIKES}]+\d+{_WORD_END}"),
    # Zeros read as the letter O at the end of a number: "12 0OO", "10O".
    re.compile(rf"{_WORD_START}\d+O+{_WORD_END}"),
    # "o.o." read as zeros next to a one- or two-letter word: "Sp. Z 0.0", "z0.0.".
    re.compile(rf"(?<!{_LETTER}){_LETTER}{{1,2}}\.?\s?(?:0\.[0Oo]|[Oo]\.0)(?!\.?\d)"),
    # A one-letter word followed by a lone zero: "Z 0".
    re.compile(rf"{_WORD_START}{_LETTER}\s0(?![.,]?\d){_WORD_END}"),
    # Space inside an e-mail address: "anna@example. com", "anna @example.com".
    re.compile(r"@\s|\s@|@\S*\.\s+[a-z]{2,6}\b"),
    # Space inside a URL or domain: "www. example.com", "example. com".
    re.compile(r"(?i)\b(?:www|https?:)\S*\.\s+[a-z]{2,}\b|\b[\w-]+\.\s(?:com|pl|org|net|eu|io)\b"),
    # A letter among the check digits of what is otherwise an account number: "PLG1 1090 1014".
    re.compile(r"\b[A-Z]{2}(?!\d\d)[0-9A-Z]{2}(?:\s?\d{4}){3,}"),
)


# A stray punctuation mark in the middle of a word: "Kowal;ski", "Fir|ma".
_STRAY_PUNCTUATION = re.compile(rf"{_LETTER}[,;:!?|*^~\"]{_LETTER}")
_ADDRESS_CHARACTERS = frozenset("/@=&")  # URLs and e-mails legitimately mix such marks


def looks_like_ocr_noise(text: str) -> bool:
    """True when the text contains a token typical of an OCR misread."""
    if any(pattern.search(text) for pattern in _PATTERNS):
        return True
    return any(
        _STRAY_PUNCTUATION.search(token)
        for token in text.split()
        if not _ADDRESS_CHARACTERS & set(token)
    )


def _is_plain_text(node: dict[str, Any]) -> bool:
    return (
        type_of(node) == "string"
        and "enum" not in node
        and "format" not in node
        and semantic_type(node) is None
    )


def has_noisy_text(value: Any, node: dict[str, Any]) -> bool:
    """Does any plain text under `value` (also inside lists and objects) look like OCR noise?"""
    kind = type_of(node)
    if kind == "object" and isinstance(value, dict):
        return any(
            has_noisy_text(value.get(key), child) for key, child in node["properties"].items()
        )
    if kind == "array" and isinstance(value, list):
        return any(has_noisy_text(item, node["items"]) for item in value)
    return isinstance(value, str) and _is_plain_text(node) and looks_like_ocr_noise(value)
