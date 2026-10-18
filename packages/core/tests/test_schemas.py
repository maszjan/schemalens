import json
from typing import Any

import pytest

from schemalens_core.config import Settings
from schemalens_core.errors import InvalidSchema, LlmOutputInvalid
from schemalens_core.schemas import build_llm_schema, compile_schema

SETTINGS = Settings(_env_file=None)

INVOICE_PL: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sprzedawca_nazwa": {"type": "string", "description": "Nazwa sprzedawcy"},
        "sprzedawca_nip": {"type": "string", "pattern": r"^\d{10}$"},
        "data_wystawienia": {"type": "string", "format": "date"},
        "kwota_brutto": {"type": "number", "minimum": 0},
        "waluta": {"type": "string", "enum": ["PLN", "EUR", "USD"]},
        "czy_zaplacono": {"type": "boolean"},
        "email_kontaktowy": {"type": "string", "format": "email"},
        "pozycje": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nazwa": {"type": "string"},
                    "ilosc": {"type": "integer"},
                    "cena_netto": {"type": "number"},
                },
                "required": ["nazwa"],
            },
        },
    },
    "required": ["sprzedawca_nazwa", "kwota_brutto"],
}


def _field(value: Any, evidence: list[str] | None = None, confidence: float = 0.9) -> dict:
    return {"value": value, "evidence": evidence or [], "confidence": confidence}


def _errors(schema: Any, **limits: int) -> list[str]:
    settings = Settings(_env_file=None, **limits)
    with pytest.raises(InvalidSchema) as excinfo:
        compile_schema(schema, settings)
    return excinfo.value.errors


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for child in node.values():
            yield from _walk(child)
    elif isinstance(node, list):
        for child in node:
            yield from _walk(child)


# --- validation of the client schema -------------------------------------------------


def test_polish_invoice_schema_is_accepted() -> None:
    compiled = compile_schema(INVOICE_PL, SETTINGS)
    assert compiled.source is INVOICE_PL


@pytest.mark.parametrize("schema", [None, [], "x", {"type": "array", "items": {"type": "string"}}])
def test_root_must_be_an_object_schema(schema: Any) -> None:
    assert _errors(schema) == ["$: schema must be a JSON object with type 'object'"]


def test_too_deep_schema_is_rejected() -> None:
    def nest(levels: int) -> dict:
        inner: dict = {"type": "string"}
        for _ in range(levels):
            inner = {"type": "object", "properties": {"a": inner}}
        return inner

    compile_schema(nest(3), SETTINGS)
    assert _errors(nest(4)) == ["a.a.a: nested deeper than 3 levels"]


def test_objects_inside_arrays_count_as_a_level() -> None:
    item = {
        "type": "object",
        "properties": {"x": {"type": "object", "properties": {"y": {"type": "string"}}}},
    }
    schema = {"type": "object", "properties": {"rows": {"type": "array", "items": item}}}
    compile_schema(schema, SETTINGS)

    item["properties"]["x"]["properties"]["y"] = {
        "type": "object",
        "properties": {"z": {"type": "string"}},
    }
    assert _errors(schema) == ["rows[].x.y: nested deeper than 3 levels"]


def test_too_many_fields_are_rejected_counting_nested_ones() -> None:
    def flat(n: int) -> dict:
        return {"type": "object", "properties": {f"f{i}": {"type": "string"} for i in range(n)}}

    compile_schema(flat(50), SETTINGS)
    assert _errors(flat(51)) == ["$: schema defines 51 fields, the limit is 50"]

    nested = flat(48)
    nested["properties"]["rows"] = {"type": "array", "items": flat(2)}
    assert _errors(nested) == ["$: schema defines 51 fields, the limit is 50"]


def test_all_problems_are_reported_at_once_with_paths() -> None:
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "money"},
            "b": {"type": "string", "format": "uuid"},
            "c": {"type": "string", "$ref": "#/defs/x"},
            "d": {"type": "array"},
            "e": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
            "f": {"type": "number", "enum": [1.5]},
            "g": {"type": "string", "enum": []},
            "h": {"type": "string", "pattern": "("},
            "i": {"type": "object", "properties": {}},
            "rows": {"type": "array", "items": {"type": "object", "properties": {"x": "string"}}},
        },
        "required": ["a", "zzz"],
    }

    assert _errors(schema) == [
        "$: required property 'zzz' is not defined",
        "a: unsupported or missing type 'money' "
        "(allowed: string, number, integer, boolean, array, object, or enum)",
        "b: unsupported format 'uuid' (allowed: date, date-time, email)",
        "c: unsupported keyword '$ref' for type string",
        "d: array must define items",
        "e: arrays of arrays are not supported",
        "f: enum is only supported for string and integer fields",
        "g: enum must be a non-empty list of string values",
        "h: pattern is not a valid regular expression",
        "i: object must define at least one property",
        "rows[].x: must be a JSON object describing the field",
    ]


def test_nullable_type_lists_enum_without_type_and_extensions_are_accepted() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "note": {"type": ["string", "null"]},
            "currency": {"enum": ["PLN", "EUR"]},
            "nip": {"type": "string", "x-schemalens-type": "nip"},
        },
    }
    compiled = compile_schema(schema, SETTINGS)

    value = compiled.llm_schema(strict=True)["properties"]["fields"]["properties"]["note"]
    assert value["properties"]["value"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}


# --- schema sent to the LLM ----------------------------------------------------------


def test_strict_llm_schema_wraps_every_field() -> None:
    schema = build_llm_schema(INVOICE_PL, strict=True)

    assert schema["required"] == ["fields"]
    fields = schema["properties"]["fields"]
    assert fields["required"] == list(INVOICE_PL["properties"])
    assert fields["properties"]["waluta"] == {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [{"type": "string", "enum": ["PLN", "EUR", "USD"]}, {"type": "null"}]
            },
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
        "required": ["value", "evidence", "confidence"],
        "additionalProperties": False,
    }


def test_strict_llm_schema_is_closed_fully_required_and_has_no_constraints() -> None:
    schema = build_llm_schema(INVOICE_PL, strict=True)

    forbidden = {"format", "pattern", "minimum", "maximum", "minLength", "maxLength"}
    for node in _walk(schema):
        assert not forbidden & node.keys()
        assert not isinstance(node.get("type"), list)
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])


def test_strict_llm_schema_moves_format_into_description() -> None:
    fields = build_llm_schema(INVOICE_PL, strict=True)["properties"]["fields"]["properties"]

    assert fields["data_wystawienia"]["properties"]["value"]["anyOf"][0] == {
        "type": "string",
        "description": "(ISO 8601 date: YYYY-MM-DD)",
    }
    assert fields["sprzedawca_nazwa"]["properties"]["value"]["anyOf"][0] == {
        "type": "string",
        "description": "Nazwa sprzedawcy",
    }


def test_strict_llm_schema_passes_constraints_as_description_text() -> None:
    fields = build_llm_schema(INVOICE_PL, strict=True)["properties"]["fields"]["properties"]

    assert fields["sprzedawca_nip"]["properties"]["value"]["anyOf"][0] == {
        "type": "string",
        "description": r"(must match regex ^\d{10}$)",
    }
    assert fields["kwota_brutto"]["properties"]["value"]["anyOf"][0] == {
        "type": "number",
        "description": "(minimum 0)",
    }


def test_strict_llm_schema_nested_line_items() -> None:
    fields = build_llm_schema(INVOICE_PL, strict=True)["properties"]["fields"]["properties"]

    assert fields["pozycje"]["properties"]["value"] == {
        "anyOf": [
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "nazwa": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "ilosc": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                        "cena_netto": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                    },
                    # all keys, not the client's `required`: absence is expressed as null
                    "required": ["nazwa", "ilosc", "cena_netto"],
                    "additionalProperties": False,
                },
            },
            {"type": "null"},
        ]
    }


def test_non_strict_llm_schema_keeps_constraints_and_uses_type_lists() -> None:
    schema = build_llm_schema(INVOICE_PL, strict=False)
    fields = schema["properties"]["fields"]["properties"]

    assert all("anyOf" not in node and "additionalProperties" not in node for node in _walk(schema))
    # pattern is never a keyword: grammar-based decoders choke on regex escapes like \d
    assert fields["sprzedawca_nip"]["properties"]["value"] == {
        "type": ["string", "null"],
        "description": r"(must match regex ^\d{10}$)",
    }
    assert fields["kwota_brutto"]["properties"]["value"] == {
        "type": ["number", "null"],
        "minimum": 0,
    }
    assert fields["data_wystawienia"]["properties"]["value"] == {
        "type": ["string", "null"],
        "description": "(ISO 8601 date: YYYY-MM-DD)",  # format is text only, see schemas.py
    }
    assert fields["waluta"]["properties"]["value"] == {
        "type": ["string", "null"],
        "enum": ["PLN", "EUR", "USD", None],
    }
    assert fields["kwota_brutto"]["properties"]["confidence"] == {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
    }
    assert schema["properties"]["fields"]["required"] == list(INVOICE_PL["properties"])


# --- LLM response -> Pydantic -> dict ------------------------------------------------


def test_round_trip_full_response() -> None:
    compiled = compile_schema(INVOICE_PL, SETTINGS)
    response = {
        "fields": {
            "sprzedawca_nazwa": _field("Example Sp. z o.o.", ["L002"], 0.95),
            "sprzedawca_nip": _field("1234563218", ["L003"]),
            "data_wystawienia": _field("2026-05-15", ["L004"], 0.86),
            "kwota_brutto": _field(312.45, ["L020"]),
            "waluta": _field("PLN", ["L020"]),
            "czy_zaplacono": _field(True, ["L022"]),
            "email_kontaktowy": _field("biuro@example.com", ["L005"]),
            "pozycje": _field(
                [
                    {"nazwa": "Paliwo", "ilosc": 2, "cena_netto": 100.0},
                    {"nazwa": "Myjnia", "ilosc": None, "cena_netto": None},
                ],
                ["L010", "L011"],
            ),
        }
    }

    parsed = compiled.parse(json.dumps(response))

    assert parsed.data == {
        "sprzedawca_nazwa": "Example Sp. z o.o.",
        "sprzedawca_nip": "1234563218",
        "data_wystawienia": "2026-05-15",
        "kwota_brutto": 312.45,
        "waluta": "PLN",
        "czy_zaplacono": True,
        "email_kontaktowy": "biuro@example.com",
        "pozycje": [
            {"nazwa": "Paliwo", "ilosc": 2, "cena_netto": 100.0},
            {"nazwa": "Myjnia", "ilosc": None, "cena_netto": None},
        ],
    }
    assert parsed.fields["data_wystawienia"].model_dump() == {
        "confidence": 0.86,
        "evidence": ["L004"],
    }
    assert compiled.missing_required(parsed.data) == []


def test_round_trip_missing_data_becomes_none_with_all_keys_present() -> None:
    compiled = compile_schema(INVOICE_PL, SETTINGS)
    response = {
        "fields": {
            "sprzedawca_nazwa": _field("Example Sp. z o.o.", ["L002"]),
            "kwota_brutto": _field(None, [], 0),
            "waluta": None,
            "pozycje": _field([{"nazwa": "Paliwo"}], ["L010"]),
            "unknown_field": _field("ignored"),
        }
    }

    parsed = compiled.parse(response)

    assert list(parsed.data) == list(INVOICE_PL["properties"])
    assert parsed.data == {
        "sprzedawca_nazwa": "Example Sp. z o.o.",
        "sprzedawca_nip": None,
        "data_wystawienia": None,
        "kwota_brutto": None,
        "waluta": None,
        "czy_zaplacono": None,
        "email_kontaktowy": None,
        "pozycje": [{"nazwa": "Paliwo", "ilosc": None, "cena_netto": None}],
    }
    assert parsed.fields["sprzedawca_nip"].model_dump() == {"confidence": 0.0, "evidence": []}
    assert compiled.missing_required(parsed.data) == ["kwota_brutto"]


def test_confidence_is_clamped_and_null_evidence_tolerated() -> None:
    compiled = compile_schema(INVOICE_PL, SETTINGS)
    parsed = compiled.parse(
        {"fields": {"waluta": {"value": "EUR", "evidence": None, "confidence": 1.7}}}
    )
    assert parsed.fields["waluta"].model_dump() == {"confidence": 1.0, "evidence": []}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("waluta", "GBP"),
        ("sprzedawca_nip", "123-456-78-90"),
        ("data_wystawienia", "w zeszly wtorek"),
        ("kwota_brutto", "trzysta"),
        ("kwota_brutto", -5),
        ("email_kontaktowy", "not an email"),
        ("pozycje", [{"nazwa": "Paliwo", "ilosc": "dwa"}]),
    ],
)
def test_values_violating_the_client_schema_fail_validation(field: str, value: Any) -> None:
    compiled = compile_schema(INVOICE_PL, SETTINGS)
    with pytest.raises(LlmOutputInvalid):
        compiled.parse({"fields": {field: _field(value)}})


def test_response_without_fields_wrapper_fails_validation() -> None:
    compiled = compile_schema(INVOICE_PL, SETTINGS)
    with pytest.raises(LlmOutputInvalid):
        compiled.parse({"sprzedawca_nazwa": "Example"})


def test_property_names_clashing_with_pydantic_or_python_are_supported() -> None:
    schema = {
        "type": "object",
        "properties": {
            "schema": {"type": "string"},
            "model_config": {"type": "string"},
            "total-gross": {"type": "number"},
            "class": {"type": "object", "properties": {"json": {"type": "string"}}},
        },
    }
    compiled = compile_schema(schema, SETTINGS)

    parsed = compiled.parse(
        {
            "fields": {
                "schema": _field("a"),
                "total-gross": _field(1),
                "class": _field({"json": "x"}),
            }
        }
    )

    assert parsed.data == {
        "schema": "a",
        "model_config": None,
        "total-gross": 1.0,
        "class": {"json": "x"},
    }


# --- normalization before validation, per-field errors --------------------------------


def test_values_are_normalized_before_validation_with_warnings() -> None:
    compiled = compile_schema(INVOICE_PL, SETTINGS)

    parsed = compiled.parse(
        {
            "fields": {
                "data_wystawienia": _field("15 maja 2026 r."),
                "kwota_brutto": _field("1 234,56"),
                "pozycje": _field([{"nazwa": "Paliwo", "ilosc": "2", "cena_netto": "99,90"}]),
            }
        }
    )

    assert parsed.data["data_wystawienia"] == "2026-05-15"
    assert parsed.data["kwota_brutto"] == 1234.56
    assert parsed.data["pozycje"] == [{"nazwa": "Paliwo", "ilosc": 2, "cena_netto": 99.9}]
    assert parsed.warnings == [
        "data_wystawienia: normalized '15 maja 2026 r.' to '2026-05-15'",
        "kwota_brutto: normalized '1 234,56' to 1234.56",
        "pozycje[0].ilosc: normalized '2' to 2",
        "pozycje[0].cena_netto: normalized '99,90' to 99.9",
    ]


def test_clean_response_produces_no_warnings() -> None:
    compiled = compile_schema(INVOICE_PL, SETTINGS)
    parsed = compiled.parse(
        {"fields": {"data_wystawienia": _field("2026-05-15"), "kwota_brutto": _field(10.5)}}
    )
    assert parsed.warnings == []


def test_validation_errors_name_fields_without_echoing_values() -> None:
    compiled = compile_schema(INVOICE_PL, SETTINGS)

    with pytest.raises(LlmOutputInvalid) as excinfo:
        compiled.parse(
            {
                "fields": {
                    "waluta": _field("GBP"),
                    "sprzedawca_nip": _field("123-456-78-90"),
                    "pozycje": _field([{"nazwa": "Paliwo", "ilosc": "dwa"}]),
                    "kwota_brutto": _field(10),
                }
            }
        )

    assert excinfo.value.fields == {"waluta", "sprzedawca_nip", "pozycje"}
    assert excinfo.value.errors == [
        r"sprzedawca_nip: String should match pattern '^\d{10}$'",
        "waluta: Input should be 'PLN', 'EUR' or 'USD'",
        "pozycje[0].ilosc: Input should be a valid integer, unable to parse string as an integer",
    ]
    assert "GBP" not in str(excinfo.value.errors) and "dwa" not in str(excinfo.value.errors)


def test_drop_invalid_nulls_only_the_failing_fields() -> None:
    compiled = compile_schema(INVOICE_PL, SETTINGS)

    parsed = compiled.parse(
        {"fields": {"waluta": _field("GBP", ["L001"]), "kwota_brutto": _field("12,50", ["L002"])}},
        drop_invalid=True,
    )

    assert parsed.data["waluta"] is None
    assert parsed.fields["waluta"].model_dump() == {"confidence": 0.0, "evidence": []}
    assert parsed.data["kwota_brutto"] == 12.5
    assert parsed.warnings == [
        "kwota_brutto: normalized '12,50' to 12.5",
        "waluta: Input should be 'PLN', 'EUR' or 'USD'; value set to null",
    ]


@pytest.mark.parametrize("payload", ["not json", "[1, 2]", '{"fields": []}', '{"data": {}}'])
def test_wrong_overall_shape_raises_even_with_drop_invalid(payload: str) -> None:
    compiled = compile_schema(INVOICE_PL, SETTINGS)
    with pytest.raises(LlmOutputInvalid) as excinfo:
        compiled.parse(payload, drop_invalid=True)
    assert excinfo.value.fields == frozenset()


# --- x-schemalens-* extensions ---------------------------------------------------------


def test_schema_extensions_are_validated() -> None:
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string", "x-schemalens-type": "passport"},
            "b": {"type": "number", "x-schemalens-type": "nip"},
            "c": {"type": "string", "x-schemalens-derived": "yes"},
            "ok": {"type": "string", "x-schemalens-type": "pesel", "x-schemalens-derived": False},
            "other": {"type": "string", "x-custom": 1},
        },
    }
    expected_type_error = (
        "x-schemalens-type must be one of nip, regon, pesel, iban, phone, currency, country, "
        "postal_code_pl, url and is only supported for string fields"
    )
    assert _errors(schema) == [
        f"a: {expected_type_error}",
        f"b: {expected_type_error}",
        "c: x-schemalens-derived must be true or false",
    ]


def test_extensions_are_not_sent_to_the_llm_as_keywords() -> None:
    schema = {
        "type": "object",
        "properties": {
            "nip": {"type": "string", "x-schemalens-type": "nip"},
            "typ": {"type": "string", "x-schemalens-derived": True},
        },
    }
    for strict in (True, False):
        assert not any(
            key.startswith("x-")
            for node in _walk(build_llm_schema(schema, strict=strict))
            for key in node
        )
    assert compile_schema(schema, SETTINGS).derived == {"typ"}
