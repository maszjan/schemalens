"""Client JSON Schema: validation, the schema sent to the LLM, and the Pydantic response model.

The client schema is never sent to the LLM as-is. Every top-level field is wrapped in
`{value, evidence, confidence}` and every value is nullable, so "not in the document"
is always expressible as `null`. `required` from the client schema is only checked
after extraction (as warnings), never forced on the model.
"""

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
)

from schemalens_core.config import Settings
from schemalens_core.errors import InvalidSchema, LlmOutputInvalid
from schemalens_core.models import FieldMeta
from schemalens_core.normalize import SEMANTIC_TYPES, normalize_text, semantic_type

Schema = dict[str, Any]

_TYPES = ("string", "number", "integer", "boolean", "array", "object")
_FORMATS = ("date", "date-time", "email")
_COMMON_KEYWORDS = {"type", "enum", "title", "description", "examples"}
_TYPE_KEYWORDS: dict[str, set[str]] = {
    "string": {"format", "pattern", "minLength", "maxLength"},
    "number": {"minimum", "maximum"},
    "integer": {"minimum", "maximum"},
    "boolean": set(),
    "array": {"items"},
    "object": {"properties", "required", "additionalProperties"},
}
_ROOT_KEYWORDS = {"$schema", "$id"}
# Validated by Pydantic after the LLM call; strict mode providers reject them in the schema.
_CONSTRAINT_KEYWORDS = ("format", "pattern", "minLength", "maxLength", "minimum", "maximum")
_FORMAT_HINTS = {
    "date": "ISO 8601 date: YYYY-MM-DD",
    "date-time": "ISO 8601 date-time, e.g. 2026-05-15T13:45:00",
    "email": "e-mail address",
}
# Never sent as keywords to grammar-based decoders (llama.cpp behind Ollama):
# - pattern: common regex syntax such as `\d` is rejected with an HTTP 500;
# - format: the grammar forces the YYYY-MM-DD shape and the model pours the digits of
#   `03.10.2026` into it as `0310-10-20`. The date is normalized by our code instead.
_TEXT_ONLY_KEYWORDS = {"pattern", "format"}
_CONSTRAINT_LABELS = {
    "pattern": "must match regex",
    "minLength": "min length",
    "maxLength": "max length",
    "minimum": "minimum",
    "maximum": "maximum",
}
_EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
_MAX_NAME_LENGTH = 64


def type_of(node: Schema) -> str | None:
    """Resolve the node type: plain, `[T, "null"]`, or inferred from `enum` values."""
    declared = node.get("type")
    if isinstance(declared, list):
        non_null = [t for t in declared if t != "null"]
        declared = non_null[0] if len(non_null) == 1 else None
    if declared is None and isinstance(node.get("enum"), list) and node["enum"]:
        if all(isinstance(v, str) for v in node["enum"]):
            return "string"
        if all(isinstance(v, int) and not isinstance(v, bool) for v in node["enum"]):
            return "integer"
    return declared if isinstance(declared, str) and declared in _TYPES else None


# --- validation of the client schema -------------------------------------------------


class _Validator:
    def __init__(self, max_depth: int) -> None:
        self.max_depth = max_depth
        self.errors: list[str] = []
        self.field_count = 0

    def check(self, node: Any, path: str, depth: int) -> None:
        if not isinstance(node, dict):
            self.errors.append(f"{path}: must be a JSON object describing the field")
            return
        kind = type_of(node)
        if kind is None:
            self.errors.append(
                f"{path}: unsupported or missing type {node.get('type')!r} "
                f"(allowed: {', '.join(_TYPES)}, or enum)"
            )
            return

        allowed = (
            _COMMON_KEYWORDS | _TYPE_KEYWORDS[kind] | (_ROOT_KEYWORDS if path == "$" else set())
        )
        for keyword in node:
            if keyword not in allowed and not keyword.startswith("x-"):
                self.errors.append(f"{path}: unsupported keyword '{keyword}' for type {kind}")

        if "enum" in node:
            self._check_enum(node["enum"], kind, path)
        if "x-schemalens-type" in node and (
            kind != "string" or node["x-schemalens-type"] not in SEMANTIC_TYPES
        ):
            self.errors.append(
                f"{path}: x-schemalens-type must be one of {', '.join(SEMANTIC_TYPES)} "
                "and is only supported for string fields"
            )
        if not isinstance(node.get("x-schemalens-derived", False), bool):
            self.errors.append(f"{path}: x-schemalens-derived must be true or false")
        if kind == "string":
            self._check_string(node, path)
        elif kind == "object":
            self._check_object(node, path, depth)
        elif kind == "array":
            self._check_array(node, path, depth)

    def _check_enum(self, values: Any, kind: str, path: str) -> None:
        expected: type = str if kind == "string" else int
        if kind not in ("string", "integer"):
            self.errors.append(f"{path}: enum is only supported for string and integer fields")
        elif (
            not isinstance(values, list)
            or not values
            or not all(isinstance(v, expected) and not isinstance(v, bool) for v in values)
        ):
            self.errors.append(f"{path}: enum must be a non-empty list of {kind} values")
        elif len(set(values)) != len(values):
            self.errors.append(f"{path}: enum values must be unique")

    def _check_string(self, node: Schema, path: str) -> None:
        if "format" in node and node["format"] not in _FORMATS:
            self.errors.append(
                f"{path}: unsupported format {node['format']!r} (allowed: {', '.join(_FORMATS)})"
            )
        if "pattern" in node:
            try:
                re.compile(node["pattern"])
            except (re.error, TypeError):
                self.errors.append(f"{path}: pattern is not a valid regular expression")

    def _check_object(self, node: Schema, path: str, depth: int) -> None:
        if depth > self.max_depth:
            self.errors.append(f"{path}: nested deeper than {self.max_depth} levels")
            return
        properties = node.get("properties")
        if not isinstance(properties, dict) or not properties:
            self.errors.append(f"{path}: object must define at least one property")
            return
        required = node.get("required", [])
        if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
            self.errors.append(f"{path}: required must be a list of property names")
        else:
            for name in required:
                if name not in properties:
                    self.errors.append(f"{path}: required property '{name}' is not defined")
        for name, child in properties.items():
            self.field_count += 1
            child_path = name if path == "$" else f"{path}.{name}"
            if not name or len(name) > _MAX_NAME_LENGTH:
                self.errors.append(
                    f"{child_path}: property name must have 1-{_MAX_NAME_LENGTH} characters"
                )
            self.check(child, child_path, depth + 1)

    def _check_array(self, node: Schema, path: str, depth: int) -> None:
        items = node.get("items")
        if not isinstance(items, dict):
            self.errors.append(f"{path}: array must define items")
        elif type_of(items) == "array":
            self.errors.append(f"{path}: arrays of arrays are not supported")
        else:
            # Arrays do not add a nesting level; objects inside them do.
            self.check(items, f"{path}[]", depth)


def validate_client_schema(schema: Any, *, max_fields: int, max_depth: int) -> None:
    """Raise `InvalidSchema` with every problem found (not just the first one)."""
    if not isinstance(schema, dict) or type_of(schema) != "object":
        raise InvalidSchema(["$: schema must be a JSON object with type 'object'"])
    validator = _Validator(max_depth)
    validator.check(schema, "$", 1)
    if validator.field_count > max_fields:
        validator.errors.append(
            f"$: schema defines {validator.field_count} fields, the limit is {max_fields}"
        )
    if validator.errors:
        raise InvalidSchema(validator.errors)


# --- schema sent to the LLM ----------------------------------------------------------


def _llm_value_schema(node: Schema, strict: bool) -> Schema:
    """Non-null schema of a value; assumes a validated client schema."""
    kind = type_of(node)
    assert kind is not None
    out: Schema = {"type": kind}
    hints = [_FORMAT_HINTS.get(node.get("format", ""))]
    # Constraints that are not sent as keywords are passed on as text.
    hints += [
        f"{label} {node[k]}"
        for k, label in _CONSTRAINT_LABELS.items()
        if k in node and (strict or k in _TEXT_ONLY_KEYWORDS)
    ]
    hint = "; ".join(filter(None, hints))
    description = " ".join(filter(None, [node.get("description"), f"({hint})" if hint else None]))
    if description:
        out["description"] = description
    if "enum" in node:
        out["enum"] = list(node["enum"])
    if not strict:
        out.update(
            {k: node[k] for k in _CONSTRAINT_KEYWORDS if k in node and k not in _TEXT_ONLY_KEYWORDS}
        )
    if kind == "object":
        out["properties"] = {
            name: _nullable(_llm_value_schema(child, strict), strict)
            for name, child in node["properties"].items()
        }
        out["required"] = list(node["properties"])
        if strict:
            out["additionalProperties"] = False
    elif kind == "array":
        out["items"] = _llm_value_schema(node["items"], strict)
    return out


def _nullable(schema: Schema, strict: bool) -> Schema:
    if strict:
        return {"anyOf": [schema, {"type": "null"}]}
    out = {**schema, "type": [schema["type"], "null"]}
    if "enum" in out:
        out["enum"] = [*out["enum"], None]
    return out


def build_llm_schema(schema: Schema, *, strict: bool) -> Schema:
    """Build the `response_format` JSON Schema from a validated client schema.

    strict: for providers with constrained decoding. Every key is required,
        `additionalProperties: false` everywhere, nullability via `anyOf` with null,
        no `format`/`pattern`/bounds keywords: Pydantic enforces them later and the
        LLM sees them only as text in `description`.
    non-strict: for models that only take the schema as guidance (e.g. Ollama + Bielik).
        Keeps the length and range keywords as hints (`pattern` and `format` stay text)
        and uses the compact `["T", "null"]` form.
    """
    closed: Schema = {"additionalProperties": False} if strict else {}
    confidence: Schema = (
        {"type": "number"} if strict else {"type": "number", "minimum": 0, "maximum": 1}
    )
    fields = {
        name: {
            "type": "object",
            "properties": {
                "value": _nullable(_llm_value_schema(node, strict), strict),
                "evidence": {"type": "array", "items": {"type": "string"}},
                "confidence": confidence,
            },
            "required": ["value", "evidence", "confidence"],
            **closed,
        }
        for name, node in schema["properties"].items()
    }
    return {
        "type": "object",
        "properties": {
            "fields": {
                "type": "object",
                "properties": fields,
                "required": list(fields),
                **closed,
            }
        },
        "required": ["fields"],
        **closed,
    }


# --- Pydantic model validating the LLM response ---------------------------------------

_MODEL_CONFIG = ConfigDict(extra="ignore", regex_engine="python-re")
_Confidence = Annotated[float, AfterValidator(lambda v: min(max(v, 0.0), 1.0))]
_Evidence = Annotated[list[str], BeforeValidator(lambda v: [] if v is None else v)]


def _python_type(node: Schema) -> Any:
    kind = type_of(node)
    if "enum" in node:
        return Literal[tuple(node["enum"])]
    if kind == "string":
        fmt = node.get("format")
        if fmt == "date":
            return date
        if fmt == "date-time":
            return datetime
        constraints = {
            "pattern": node.get("pattern", _EMAIL_PATTERN if fmt == "email" else None),
            "min_length": node.get("minLength"),
            "max_length": node.get("maxLength"),
        }
        return _constrained(str, constraints)
    if kind in ("number", "integer"):
        base = float if kind == "number" else int
        return _constrained(base, {"ge": node.get("minimum"), "le": node.get("maximum")})
    if kind == "boolean":
        return bool
    if kind == "array":
        return list[_python_type(node["items"])]  # type: ignore[misc]
    return _object_model(
        "Object", {name: _python_type(c) for name, c in node["properties"].items()}
    )


def _constrained(base: type, constraints: dict[str, Any]) -> Any:
    active = {k: v for k, v in constraints.items() if v is not None}
    return Annotated[base, Field(**active)] if active else base


def _object_model(name: str, types: dict[str, Any]) -> type[BaseModel]:
    """Model with every field optional (missing -> None).

    Client property names become aliases of generated attribute names, so names like
    `schema`, `model_x` or `total-gross` cannot clash with Pydantic internals.
    """
    fields: dict[str, Any] = {
        f"f{i}": (annotation | None, Field(default=None, alias=alias))
        for i, (alias, annotation) in enumerate(types.items())
    }
    return create_model(name, __config__=_MODEL_CONFIG, **fields)


def build_response_model(schema: Schema) -> type[BaseModel]:
    """Pydantic model for the wrapped LLM response; assumes a validated client schema."""
    wrappers: dict[str, Any] = {}
    for name, node in schema["properties"].items():
        wrappers[name] = create_model(
            "FieldResult",
            __config__=_MODEL_CONFIG,
            value=(_python_type(node) | None, None),
            evidence=(_Evidence, Field(default_factory=list)),
            confidence=(_Confidence, 0.0),
        )
    fields_model = _object_model("Fields", wrappers)
    return create_model("LlmResponse", __config__=_MODEL_CONFIG, fields=(fields_model, ...))


# --- public entry point ---------------------------------------------------------------


@dataclass(frozen=True)
class ParsedExtraction:
    data: dict[str, Any]  # every schema key present; missing -> None
    fields: dict[str, FieldMeta]  # confidence here is still the one reported by the model
    warnings: list[str]  # normalizations applied and fields dropped
    normalized: frozenset[str]  # top-level fields changed by a normalizer
    invalid: frozenset[str]  # top-level fields whose value failed a checksum


class _Normalizer:
    """Deterministic clean-up of string values before validation; every change is reported."""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.changed: set[str] = set()  # top-level fields whose value was modified
        self.invalid: set[str] = set()  # top-level fields holding a value that failed its check

    def run(self, value: Any, node: Schema, path: str, root: str) -> Any:
        kind = type_of(node)
        if kind == "object" and isinstance(value, dict):
            properties = node["properties"]
            return {
                key: self.run(item, properties[key], f"{path}.{key}", root)
                if key in properties
                else item
                for key, item in value.items()
            }
        if kind == "array" and isinstance(value, list):
            return [
                self.run(item, node["items"], f"{path}[{i}]", root) for i, item in enumerate(value)
            ]
        if not isinstance(value, str):
            return value

        result = normalize_text(
            value, kind=kind, fmt=node.get("format"), semantic=semantic_type(node)
        )
        if "enum" in node and result.value not in node["enum"]:
            return value  # classification fields are only mapped onto an allowed value
        if result.problem:
            self.warnings.append(f"{path}: {result.problem}")
            self.invalid.add(root)
        if result.value == value:
            return value
        if result.noise_removed:
            self.warnings.append(f"{path}: removed non-digit noise")
        else:
            note = " (OCR characters corrected)" if result.ocr_fixed else ""
            self.warnings.append(f"{path}: normalized {value!r} to {result.value!r}{note}")
        self.changed.add(root)
        return result.value


def _field_errors(exc: ValidationError) -> tuple[list[str], set[str]]:
    """Readable `path: message` errors (no input values) and the top-level fields at fault."""
    errors: list[str] = []
    failed: set[str] = set()
    for error in exc.errors(include_input=False, include_url=False):
        # loc is ("fields", <field>, "value", ...); anything else is a structural problem
        loc = error["loc"][1:]
        failed.add(str(loc[0]) if loc else "")
        path = ""
        for i, part in enumerate(loc):
            if i == 1 and part == "value":
                continue
            path += f"[{part}]" if isinstance(part, int) else f".{part}" if path else str(part)
        errors.append(f"{path or 'fields'}: {error['msg']}")
    return errors, failed


@dataclass(frozen=True)
class CompiledSchema:
    source: Schema
    response_model: type[BaseModel]

    def llm_schema(self, *, strict: bool) -> Schema:
        return build_llm_schema(self.source, strict=strict)

    def parse(
        self, payload: str | bytes | dict[str, Any], *, drop_invalid: bool = False
    ) -> ParsedExtraction:
        """Normalize, then validate a raw LLM response (the order in docs/how-it-works.md).

        Raises `LlmOutputInvalid` when a field still fails after normalization. With
        `drop_invalid`, failing fields become `None` with a warning instead; only a
        response of the wrong overall shape still raises.
        """
        raw = payload
        if not isinstance(raw, dict):
            try:
                raw = json.loads(raw)
            except ValueError:
                raise LlmOutputInvalid(["response is not valid JSON"]) from None
        if not isinstance(raw, dict) or not isinstance(raw.get("fields"), dict):
            raise LlmOutputInvalid(["response must be a JSON object with a 'fields' object"])

        properties: dict[str, Schema] = self.source["properties"]
        normalizer = _Normalizer()
        warnings = normalizer.warnings
        wrappers = dict(raw["fields"])
        for name, node in properties.items():
            wrapper = wrappers.get(name)
            if isinstance(wrapper, dict) and "value" in wrapper:
                value = normalizer.run(wrapper["value"], node, name, name)
                wrappers[name] = {**wrapper, "value": value}

        try:
            response = self.response_model.model_validate({"fields": wrappers})
        except ValidationError as exc:
            errors, failed = _field_errors(exc)
            if not drop_invalid or not failed <= properties.keys():
                raise LlmOutputInvalid(errors, frozenset(failed & properties.keys())) from None
            for name in properties:
                if name in failed:
                    wrappers[name] = None
            warnings += [f"{error}; value set to null" for error in errors]
            response = self.response_model.model_validate({"fields": wrappers})

        dumped = response.model_dump(mode="json", by_alias=True)["fields"]
        data: dict[str, Any] = {}
        fields: dict[str, FieldMeta] = {}
        for name in properties:
            item = dumped[name] or {"value": None, "evidence": [], "confidence": 0.0}
            data[name] = item["value"]
            fields[name] = FieldMeta(confidence=item["confidence"], evidence=item["evidence"])
        return ParsedExtraction(
            data=data,
            fields=fields,
            warnings=warnings,
            normalized=frozenset(name for name in normalizer.changed if data[name] is not None),
            invalid=frozenset(name for name in normalizer.invalid if data[name] is not None),
        )

    @property
    def derived(self) -> frozenset[str]:
        """Top-level fields the model may infer from the content (`x-schemalens-derived`)."""
        return frozenset(
            name
            for name, node in self.source["properties"].items()
            if node.get("x-schemalens-derived") is True
        )

    def missing_required(self, data: dict[str, Any]) -> list[str]:
        """Top-level `required` fields that came back empty (reported as warnings)."""
        return [name for name in self.source.get("required", []) if data.get(name) is None]


def compile_schema(schema: Any, settings: Settings) -> CompiledSchema:
    """Validate a client schema and prepare everything derived from it."""
    validate_client_schema(
        schema, max_fields=settings.schema_max_fields, max_depth=settings.schema_max_depth
    )
    return CompiledSchema(source=schema, response_model=build_response_model(schema))
