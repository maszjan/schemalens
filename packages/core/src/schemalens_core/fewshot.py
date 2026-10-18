"""Few-shot examples per preset (docs/how-it-works.md).

A template is a synthetic document of a preset's type with its expected result, kept in
`templates/<preset>/template.json` and read-only at runtime. When a request uses a preset and
few-shot is enabled, the template is shown to the model as a worked example: its lines and
the answer in the exact wrapped format, with evidence ids. Templates are deliberately other
documents than the presets' own examples, which are eval cases.
"""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from schemalens_core.grounding import ungrounded_paths
from schemalens_core.schemas import type_of


@dataclass(frozen=True)
class Template:
    preset: str
    lines: list[str]
    data: dict[str, Any]


@cache
def _load(directory: Path) -> dict[str, Template]:
    templates = {}
    for path in sorted(directory.glob("*/template.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        templates[raw["preset"]] = Template(raw["preset"], raw["lines"], raw["data"])
    return templates


def load_templates(directory: Path) -> dict[str, Template]:
    return _load(directory.resolve())


def _supports(value: Any, node: dict[str, Any], name: str, text: str) -> bool:
    """Does this line hold (part of) the value? Used to give the example real evidence."""
    if value is None:
        return False
    if isinstance(value, list):
        return any(_supports(item, node.get("items", {}), name, text) for item in value)
    if isinstance(value, dict):
        children = node.get("properties", {})
        return any(_supports(v, children.get(k, {}), k, text) for k, v in value.items())
    if type_of(node) == "string" and "format" not in node and "x-schemalens-type" not in node:
        return str(value).casefold() in text.casefold()
    if isinstance(value, bool):
        return False
    return ungrounded_paths(value, node, name, text) == []


def example_block(template: Template, schema: dict[str, Any]) -> str:
    """The template as a worked example for `schema`: numbered lines and the wrapped answer."""
    ids = [f"L{index:03d}" for index in range(1, len(template.lines) + 1)]
    fields: dict[str, Any] = {}
    for name, node in schema["properties"].items():
        value = template.data.get(name)
        derived = node.get("x-schemalens-derived") is True
        evidence = [
            line_id
            for line_id, text in zip(ids, template.lines, strict=True)
            if not derived and _supports(value, node, name, text)
        ]
        fields[name] = {
            "value": value,
            "evidence": evidence,
            "confidence": 0.0 if value is None else 0.95,
        }
    lines = "\n".join(f"[{i}] {text}" for i, text in zip(ids, template.lines, strict=True))
    answer = json.dumps({"fields": fields}, ensure_ascii=False)
    return (
        f"<example_document>\n{lines}\n</example_document>\n"
        f"<example_answer>\n{answer}\n</example_answer>"
    )
