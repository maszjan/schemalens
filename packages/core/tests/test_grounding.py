from typing import Any

import pytest

from schemalens_core.config import Settings
from schemalens_core.grounding import ungrounded_paths
from schemalens_core.models import FieldMeta, OcrLine
from schemalens_core.presets import load_presets
from schemalens_core.scoring import score_fields

SETTINGS = Settings(_env_file=None)

DATE = {"type": "string", "format": "date"}
NUMBER = {"type": "number"}
INTEGER = {"type": "integer"}


def _typed(semantic: str) -> dict[str, Any]:
    return {"type": "string", "x-schemalens-type": semantic}


@pytest.mark.parametrize(
    ("value", "node", "text"),
    [
        ("2027-06-30", DATE, "Najlepiej spożyć przed: 30.06.2027"),
        ("2026-10-15", DATE, "Warszawa, 15 października 2026 r."),
        ("2026-10-20", DATE, "1. Wdrożenie przesunięte na 20.10.2026."),
        ("2026-05-15", DATE, "data: l5 maja 2O26"),  # OCR letters, same fix as the field gets
        (10.48, NUMBER, "SUMA PLN 10,48"),
        (6.49, NUMBER, "Chleb żytni 1 szt 6,49"),
        (1, INTEGER, "Chleb żytni 1 szt 6,49"),
        (1234.56, NUMBER, "Razem: 1 234,56 zł"),
        (12000.0, NUMBER, "3. Budżet na testy: 12 0OO zł, dostawca"),
        (545.0, NUMBER, "Wartość energetyczna w 100 g: 545 kcal"),
        (23, INTEGER, "VAT 23%: 230,00 zł"),
        ("1234563218", _typed("nip"), "NIP sprzedawcy: 123-456-32-18"),
        ("1234563218", _typed("nip"), "NIP l23-456-32-l8"),
        (
            "PL61109010140000071219812874",
            _typed("iban"),
            "Konto: PL61 1090 1014 0000 0712 1981 2874",
        ),
        ("+48600100200", _typed("phone"), "tel. +48 600 100 200"),
        ("+48600100200", _typed("phone"), "tel. 0048 600 100 200"),
        ("00-950", _typed("postal_code_pl"), "ul. Długa 5, 00-950 Warszawa"),
        ("44051401359", _typed("pesel"), "PESEL: 44051401359"),
    ],
)
def test_values_present_in_evidence_are_grounded(value: Any, node: dict, text: str) -> None:
    assert ungrounded_paths(value, node, "f", text) == []


@pytest.mark.parametrize(
    ("value", "node", "text"),
    [
        ("3006-07-20", DATE, "Najlepiej spożyć przed: 30.06.2027"),
        ("2026-10-16", DATE, "Warszawa, 15 października 2026 r."),
        (649.0, NUMBER, "Chleb żytni 1 szt 6,49"),
        (10.84, NUMBER, "SUMA PLN 10,48"),
        (7, INTEGER, "brak liczb w tej linii"),
        ("1234563218", _typed("nip"), "NIP sprzedawcy: 987-654-32-10"),
        (
            "PL61109010140000071219812874",
            _typed("iban"),
            "Konto: PL27 1140 2004 0000 3002 0135 5387",
        ),
        ("+48600100200", _typed("phone"), "tel. 601 202 303"),
        ("00-950", _typed("postal_code_pl"), "ul. Długa 5, 00-001 Łódź"),
    ],
)
def test_values_absent_from_evidence_are_not_grounded(value: Any, node: dict, text: str) -> None:
    assert ungrounded_paths(value, node, "f", text) == ["f"]


@pytest.mark.parametrize(
    ("value", "node"),
    [
        ("dowolny tekst, którego nie ma", {"type": "string"}),
        ("PLN", {"type": "string", "enum": ["PLN", "EUR"]}),
        ("PL", _typed("country")),
        ("PLN", _typed("currency")),
        (True, {"type": "boolean"}),
        (None, DATE),
    ],
)
def test_free_text_and_classifications_are_not_checked(value: Any, node: dict) -> None:
    assert ungrounded_paths(value, node, "f", "zupełnie inna treść 123") == []


def test_nested_values_are_reported_by_path() -> None:
    node = {
        "type": "array",
        "items": {"type": "object", "properties": {"nazwa": {"type": "string"}, "cena": NUMBER}},
    }
    value = [{"nazwa": "Chleb", "cena": 6.49}, {"nazwa": "Mleko", "cena": 9.39}, {"nazwa": "Ser"}]
    text = "Chleb żytni 1 szt 6,49\nMleko 2% 1 szt 3,99"

    assert ungrounded_paths(value, node, "pozycje", text) == ["pozycje[1].cena"]


# --- effect on confidence --------------------------------------------------------------

LABEL_LINES = [
    OcrLine(id="L001", page=1, text="CZEKOLADA MLECZNA Z ORZECHAMI", bbox=None, conf=0.96),
    OcrLine(id="L002", page=1, text="Masa netto: 100 g", bbox=None, conf=0.92),
    OcrLine(id="L003", page=1, text="Wartość energetyczna w 100 g: 545 kcal", bbox=None, conf=0.87),
    OcrLine(id="L004", page=1, text="Najlepiej spożyć przed: 30.06.2027", bbox=None, conf=0.98),
    OcrLine(id="L005", page=1, text="Producent: Słodka Fabryka Example", bbox=None, conf=0.9),
    OcrLine(id="L006", page=1, text="Kraj pochodzenia: Polska", bbox=None, conf=0.8),
]
LABEL_SCHEMA = load_presets(SETTINGS)["product_label"].json_schema


def _score_label(data: dict[str, Any], evidence: dict[str, list[str]]):
    full = {name: data.get(name) for name in LABEL_SCHEMA["properties"]}
    reported = {name: FieldMeta(confidence=1.0, evidence=evidence.get(name, [])) for name in full}
    return score_fields(LABEL_SCHEMA, full, reported, LABEL_LINES, SETTINGS)


def test_regression_misread_date_on_product_label() -> None:
    """Checkpoint B: the label says 30.06.2027, the model answered 3006-07-20 at 0.986."""
    fields, warnings = _score_label({"best_before": "3006-07-20"}, {"best_before": ["L004"]})

    assert warnings == ["best_before: value not found in evidence"]
    assert fields["best_before"].confidence < 0.4
    assert fields["best_before"].evidence == ["L004"]


def test_correct_date_keeps_its_confidence() -> None:
    fields, warnings = _score_label({"best_before": "2027-06-30"}, {"best_before": ["L004"]})

    assert warnings == []
    assert fields["best_before"].confidence == 0.986


def test_grounding_penalty_value_and_scope() -> None:
    fields, warnings = _score_label(
        {
            "energy_kcal": 454.0,  # digits swapped
            "product_name": "Czekolada gorzka",  # wrong, but free text is not checked
            "category": "żywność",  # derived
        },
        {"energy_kcal": ["L003"], "product_name": ["L001"], "category": ["L001"]},
    )

    assert warnings == ["energy_kcal: value not found in evidence"]
    assert fields["energy_kcal"].confidence == 0.273  # (0.7 * 0.87 + 0.3) * 0.3
    assert fields["product_name"].confidence == 0.972
    assert fields["category"].confidence == 0.972
