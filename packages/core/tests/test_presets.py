import json
from pathlib import Path

import pytest

from evals.synth import EXTRA_DOCUMENTS, preset_document, preset_lines, render_scan
from schemalens_core.config import Settings
from schemalens_core.errors import InvalidSchema
from schemalens_core.preprocess import detect_kind, load_document
from schemalens_core.presets import load_presets, resolve_schema
from schemalens_core.schemas import compile_schema

SETTINGS = Settings(_env_file=None)
PRESETS = load_presets(SETTINGS)
STARTING_SET = {
    "invoice",
    "receipt",
    "business_card",
    "cv",
    "product_label",
    "menu",
    "contract_terms",
    "generic_document",
    "lecture_notes",
}


def test_starting_presets_are_present() -> None:
    assert set(PRESETS) == STARTING_SET


@pytest.mark.parametrize("name", sorted(STARTING_SET))
def test_preset_passes_the_client_schema_validator(name: str) -> None:
    preset = PRESETS[name]
    compile_schema(preset.json_schema, SETTINGS)  # same validator and limits as client schemas
    assert preset.title and preset.description
    assert all(node.get("description") for node in preset.json_schema["properties"].values())


@pytest.mark.parametrize("name", sorted(STARTING_SET))
def test_preset_example_is_a_valid_normalized_result(name: str) -> None:
    preset = PRESETS[name]
    compiled = compile_schema(preset.json_schema, SETTINGS)
    assert set(preset.example.data) == set(preset.json_schema["properties"])

    response = {
        "fields": {
            field: {"value": value, "evidence": [], "confidence": 1.0}
            for field, value in preset.example.data.items()
        }
    }
    parsed = compiled.parse(response)

    assert parsed.data == preset.example.data  # already in normalized form
    assert parsed.warnings == []


@pytest.mark.parametrize("name", sorted(STARTING_SET))
def test_preset_example_renders_as_a_jpeg_scan(name: str) -> None:
    document = preset_document(name)

    assert detect_kind(document) == "jpeg"
    assert preset_document(name) == document  # deterministic
    page = load_document(document, SETTINGS).pages[0]
    assert page.image is not None and page.image.width > 300
    assert preset_lines(name) == PRESETS[name].example.lines


def test_extra_documents_render() -> None:
    assert detect_kind(render_scan(EXTRA_DOCUMENTS["meeting_note"], fmt="JPEG")) == "jpeg"


def test_no_presets_for_identity_or_medical_documents() -> None:
    banned = ("dowod", "paszport", "passport", "id_card", "recepta", "prescription", "medical")
    assert not [name for name in PRESETS if any(word in name for word in banned)]


def test_invoice_preset_declares_domain_types_explicitly() -> None:
    properties = PRESETS["invoice"].json_schema["properties"]
    assert "x-schemalens-type" not in properties["seller_tax_id"]  # VAT numbers vary by country
    assert properties["iban"]["x-schemalens-type"] == "iban"
    assert properties["currency"]["x-schemalens-type"] == "currency"


def test_invoice_preset_keeps_invoice_behaviour() -> None:
    compiled = compile_schema(PRESETS["invoice"].json_schema, SETTINGS)
    parsed = compiled.parse(
        {
            "fields": {
                "seller_tax_id": {"value": "GB123456789", "evidence": [], "confidence": 1},
                "iban": {
                    "value": "gb82 west 1234 5698 7654 32",
                    "evidence": [],
                    "confidence": 1,
                },
                "currency": {"value": "£", "evidence": [], "confidence": 1},
                "gross_total": {"value": "1,200.00", "evidence": [], "confidence": 1},
                "issue_date": {"value": "3 October 2026", "evidence": [], "confidence": 1},
            }
        }
    )
    assert parsed.data["seller_tax_id"] == "GB123456789"
    assert parsed.data["iban"] == "GB82WEST12345698765432"
    assert parsed.data["currency"] == "GBP"
    assert parsed.data["gross_total"] == 1200.0
    assert parsed.data["issue_date"] == "2026-10-03"


def test_resolve_schema_variants() -> None:
    own = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert resolve_schema(own, None, SETTINGS) is own
    assert resolve_schema(None, "menu", SETTINGS) is PRESETS["menu"].json_schema

    extra = {
        "properties": {"a": {"type": "string"}, "currency": {"type": "string"}},
        "required": ["a"],
    }
    merged = resolve_schema(extra, "menu", SETTINGS)
    assert list(merged["properties"])[-1] == "a"
    assert merged["properties"]["currency"] == {"type": "string"}  # client definition wins
    assert merged["required"] == ["items", "a"]
    assert "a" not in PRESETS["menu"].json_schema["properties"]  # preset itself is untouched


@pytest.mark.parametrize("extension", [{}, {"properties": []}, []])
def test_malformed_preset_extension_is_rejected(extension) -> None:
    with pytest.raises(InvalidSchema):
        resolve_schema(extension, "menu", SETTINGS)


def test_malformed_preset_file_is_reported_by_name(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text(json.dumps({"name": "broken"}))
    with pytest.raises(ValueError, match="broken.json is malformed"):
        load_presets(Settings(_env_file=None, presets_dir=tmp_path))


def test_preset_file_name_must_match_declared_name(tmp_path: Path) -> None:
    content = PRESETS["menu"].model_dump(by_alias=True)
    (tmp_path / "other.json").write_text(json.dumps(content))
    with pytest.raises(ValueError, match="declares the name 'menu'"):
        load_presets(Settings(_env_file=None, presets_dir=tmp_path))
