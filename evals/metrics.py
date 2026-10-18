"""Field-level comparison of an extraction result with the expected one."""

import difflib
import re
from statistics import quantiles
from typing import Any

OUTCOMES = ("correct", "miss", "wrong", "hallucination", "ocr_text_noise")
_NOISE_SIMILARITY = 0.8


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _is_plain_text(node: dict[str, Any]) -> bool:
    return (
        node.get("type") == "string"
        and "enum" not in node
        and "format" not in node
        and "x-schemalens-type" not in node
    )


def _same(value: Any, expected: Any, node: dict[str, Any], ocr_text: str | None) -> bool:
    """Equality after normalization. With `ocr_text`, plain text may differ by OCR noise only:
    the value must be in the OCR text verbatim and still be close to the expected one."""
    if expected is None or value is None:
        return value is None and expected is None
    kind = node.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return False
        return all(
            _same(value.get(key), expected.get(key), child, ocr_text)
            for key, child in node["properties"].items()
            if key in expected
        )
    if kind == "array":
        return isinstance(value, list) and list_f1(value, expected, node, ocr_text) == 1.0
    if kind in ("number", "integer"):
        return isinstance(value, int | float) and abs(value - expected) <= 0.01
    if isinstance(value, str) and isinstance(expected, str):
        if _fold(value) == _fold(expected):
            return True
        if ocr_text is None or not _is_plain_text(node):
            return False
        similar = difflib.SequenceMatcher(None, _fold(value), _fold(expected)).ratio()
        return _fold(value) in _fold(ocr_text) and similar >= _NOISE_SIMILARITY
    return bool(value == expected)


def list_f1(
    value: list[Any], expected: list[Any], node: dict[str, Any], ocr_text: str | None = None
) -> float:
    """Order-insensitive F1 of list elements."""
    if not value and not expected:
        return 1.0
    remaining = list(value)
    matched = 0
    for wanted in expected:
        for candidate in remaining:
            if _same(candidate, wanted, node["items"], ocr_text):
                remaining.remove(candidate)
                matched += 1
                break
    if matched == 0:
        return 0.0
    precision, recall = matched / len(value), matched / len(expected)
    return 2 * precision * recall / (precision + recall)


def field_outcome(
    value: Any, expected: Any, node: dict[str, Any], ocr_text: str, *, free_text: bool = False
) -> str:
    """One of OUTCOMES for a top-level field."""
    if expected is None:
        return "correct" if value is None else "hallucination"
    if value is None:
        return "miss"
    if free_text or _same(value, expected, node, None):
        return "correct"
    return "ocr_text_noise" if _same(value, expected, node, ocr_text) else "wrong"


def percentile(values: list[float], q: int) -> float:
    """q-th percentile (50 or 95); a single value is its own percentile."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(quantiles(values, n=100, method="inclusive")[q - 1])
