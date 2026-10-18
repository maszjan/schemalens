"""Schema presets: ready-made client schemas kept in `presets/*.json` (see presets/).

A preset is an ordinary client schema. It goes through the same validator and limits;
the pipeline has no preset-specific code path.
"""

import json
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from schemalens_core.config import Settings
from schemalens_core.errors import InvalidSchema


class PresetExample(BaseModel):
    lines: list[str]  # a synthetic document, one text line per entry
    data: dict[str, Any]  # expected result for those lines


class Preset(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    title: str
    description: str
    json_schema: dict[str, Any] = Field(alias="schema")
    example: PresetExample


@cache
def _load(directory: Path) -> dict[str, Preset]:
    presets: dict[str, Preset] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            preset = Preset.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, ValidationError) as exc:
            raise ValueError(f"preset file {path.name} is malformed") from exc
        if preset.name != path.stem:
            raise ValueError(f"preset file {path.name} declares the name '{preset.name}'")
        presets[preset.name] = preset
    return presets


def load_presets(settings: Settings) -> dict[str, Preset]:
    """All presets from `settings.presets_dir`, by name. Read once per process."""
    return _load(settings.presets_dir.resolve())


def resolve_schema(
    schema: dict[str, Any] | None, preset: str | None, settings: Settings
) -> dict[str, Any]:
    """Schema to extract with: the client's, a preset's, or a preset extended by the client's.

    With both, client properties are added to (or replace) the preset's and `required`
    lists are merged. The result still has to pass `compile_schema`.
    """
    if preset is None:
        if schema is None:
            raise InvalidSchema(["either 'schema' or 'preset' is required"])
        return schema
    presets = load_presets(settings)
    if preset not in presets:
        available = ", ".join(presets) or "none"
        raise InvalidSchema([f"preset '{preset}' does not exist (available: {available})"])
    base = presets[preset].json_schema
    if schema is None:
        return base
    extra = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(extra, dict):
        raise InvalidSchema(["schema extending a preset must be an object with 'properties'"])
    required = [*base.get("required", [])]
    extra_required = schema.get("required", [])
    if isinstance(extra_required, list):
        required += [name for name in extra_required if name not in required]
    return {**base, "properties": {**base["properties"], **extra}, "required": required}
