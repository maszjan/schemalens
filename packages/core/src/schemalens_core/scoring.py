"""Evidence checks and field confidence. The score is ours; the model's is only one input."""

from typing import Any

from schemalens_core.config import Settings
from schemalens_core.grounding import ungrounded_paths
from schemalens_core.models import FieldMeta, OcrLine
from schemalens_core.ocr_noise import has_noisy_text
from schemalens_core.schemas import type_of


def _evidence_limit(value: Any, node: dict[str, Any], total_lines: int, settings: Settings) -> int:
    """How many evidence lines a field may cite before the evidence stops meaning anything."""
    per_item = settings.conf_evidence_max_lines
    kind = type_of(node)
    if kind == "array" and isinstance(value, list):
        return per_item * max(len(value), 1)
    if kind == "object" and isinstance(value, dict):
        return per_item * max(sum(v is not None for v in value.values()), 1)
    # A scalar comes from one place; a single line is always fine, however short the document.
    return max(1, min(per_item, int(settings.conf_evidence_max_share * total_lines)))


def score_fields(
    schema: dict[str, Any],
    data: dict[str, Any],
    reported: dict[str, FieldMeta],
    lines: list[OcrLine],
    settings: Settings,
    *,
    normalized: frozenset[str] = frozenset(),
    retried: frozenset[str] = frozenset(),
    invalid: frozenset[str] = frozenset(),
) -> tuple[dict[str, FieldMeta], list[str]]:
    """Check evidence and compute confidence per top-level field (docs/how-it-works.md).

    Evidence: ids of lines that do not exist are removed; evidence citing too many lines
    is treated as no evidence at all.

    confidence = base x penalties, where
      base = w * mean OCR confidence of evidence lines + (1 - w) * model-reported confidence
             (without evidence: model-reported confidence * no-evidence factor;
              derived field without evidence: model-reported confidence, capped, no penalty)
      penalties: value changed by a normalizer, value obtained from the validation retry,
                 value that failed its checksum, date/number/identifier not found in the
                 evidence text (grounding), plain text that looks like an OCR misread.
    A missing value always scores 0.
    """
    by_id = {line.id: line for line in lines}
    fields: dict[str, FieldMeta] = {}
    warnings: list[str] = []
    for name, meta in reported.items():
        node = schema["properties"][name]
        derived = node.get("x-schemalens-derived") is True
        value = data[name]
        evidence = [line_id for line_id in meta.evidence if line_id in by_id]
        if len(evidence) < len(meta.evidence):
            warnings.append(f"{name}: evidence pointed to lines that do not exist, removed")
        if value is None:
            fields[name] = FieldMeta(confidence=0.0, evidence=evidence)
            continue

        too_broad = len(evidence) > _evidence_limit(value, node, len(lines), settings)
        if too_broad:
            warnings.append(f"{name}: evidence too broad")
            evidence = []

        if evidence:
            ocr_mean = sum(by_id[line_id].conf for line_id in evidence) / len(evidence)
            weight = settings.conf_ocr_weight
            confidence = weight * ocr_mean + (1 - weight) * meta.confidence
        elif derived:
            # Inferred from the content, so missing evidence is expected; just not verifiable.
            confidence = min(meta.confidence, settings.conf_derived_cap)
        else:
            if not too_broad:
                warnings.append(
                    f"{name}: value has no evidence in the document, treat it as unverified"
                )
            confidence = meta.confidence * settings.conf_no_evidence_factor

        if evidence and not derived:
            text = "\n".join(by_id[line_id].text for line_id in evidence)
            missing = ungrounded_paths(value, node, name, text)
            warnings += [f"{path}: value not found in evidence" for path in missing]
            if missing:
                confidence *= settings.conf_grounding_factor
        if not derived and has_noisy_text(value, node):
            # Plain text is never corrected, so the doubt has to show in the score.
            warnings.append(f"{name}: possible OCR noise")
            confidence *= settings.conf_suspicious_factor
        if name in normalized:
            confidence *= settings.conf_normalized_factor
        if name in retried:
            confidence *= settings.conf_retry_factor
        if name in invalid:
            confidence *= settings.conf_checksum_factor
        fields[name] = FieldMeta(confidence=round(confidence, 3), evidence=evidence)
    return fields, warnings
