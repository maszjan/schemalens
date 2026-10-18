"""Grounding: is an extracted date, number or identifier actually present in its evidence?

A model can cite the right line and still return a wrong value (`30.06.2027` read as
`3006-07-20`). For value types we can parse deterministically, the evidence text is run
through the same normalizers as the field and the value must match one of the candidates.
Free text and derived fields are not checked.
"""

import re
from typing import Any

from schemalens_core.normalize import fix_ocr_digits, normalize_text, semantic_type
from schemalens_core.schemas import type_of

_COMPACT_TYPES = ("nip", "regon", "pesel", "iban", "phone", "postal_code_pl")
_EDGE_PUNCTUATION = " \t:;,.()[]<>\"'%"
_MAX_DATE_TOKENS = 4  # "15 października 2026 r."
_MAX_NUMBER_PARTS = 5  # "1 234 567,89"
_NUMBER_RUN = re.compile(r"[+-]?[0-9OlS][0-9OlS .,  ]*")


def _date_candidates(text: str) -> set[str]:
    tokens = text.split()
    found: set[str] = set()
    for start in range(len(tokens)):
        for size in range(1, _MAX_DATE_TOKENS + 1):
            chunk = " ".join(tokens[start : start + size]).strip(_EDGE_PUNCTUATION)
            result = normalize_text(chunk, kind="string", fmt="date", semantic=None)
            if result.value != chunk or re.fullmatch(r"\d{4}-\d{2}-\d{2}", chunk):
                found.add(result.value)
    return found


def _number_candidates(text: str) -> set[float]:
    found: set[float] = set()
    for run in _NUMBER_RUN.findall(text):
        parts = run.split()
        for start in range(len(parts)):
            for size in range(1, _MAX_NUMBER_PARTS + 1):
                chunk = " ".join(parts[start : start + size]).strip(_EDGE_PUNCTUATION)
                result = normalize_text(chunk, kind="number", fmt=None, semantic=None)
                if isinstance(result.value, float):
                    found.add(result.value)
    return found


def _compact(text: str, digits_only: bool) -> str:
    pattern = r"\D" if digits_only else r"[^0-9A-Za-z]"
    return re.sub(pattern, "", text).upper()


def _is_grounded(value: Any, node: dict[str, Any], text: str) -> bool:
    kind = type_of(node)
    semantic = semantic_type(node)
    if kind in ("number", "integer") and isinstance(value, int | float):
        return any(abs(candidate - value) < 0.005 for candidate in _number_candidates(text))
    if kind == "string" and node.get("format") == "date":
        return value in _date_candidates(text)
    if kind == "string" and semantic in _COMPACT_TYPES:
        digits_only = semantic != "iban"
        needle = _compact(str(value), digits_only)
        haystacks = [text] if semantic == "iban" else [text, fix_ocr_digits(text)]
        return bool(needle) and any(needle in _compact(h, digits_only) for h in haystacks)
    return True  # free text, enums, booleans: not checkable here


def ungrounded_paths(value: Any, node: dict[str, Any], path: str, text: str) -> list[str]:
    """Paths of checkable leaves under `value` that cannot be found in the evidence text."""
    kind = type_of(node)
    if value is None:
        return []
    if kind == "object" and isinstance(value, dict):
        return [
            missing
            for key, child in node["properties"].items()
            for missing in ungrounded_paths(value.get(key), child, f"{path}.{key}", text)
        ]
    if kind == "array" and isinstance(value, list):
        return [
            missing
            for i, item in enumerate(value)
            for missing in ungrounded_paths(item, node["items"], f"{path}[{i}]", text)
        ]
    return [] if _is_grounded(value, node, text) else [path]
