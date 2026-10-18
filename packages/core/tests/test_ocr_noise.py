from typing import Any

import pytest

from schemalens_core.config import Settings
from schemalens_core.models import FieldMeta, OcrLine
from schemalens_core.ocr_noise import has_noisy_text, looks_like_ocr_noise
from schemalens_core.scoring import score_fields

SETTINGS = Settings(_env_file=None)
TEXT = {"type": "string"}


@pytest.mark.parametrize(
    "text",
    [
        # values from evals/report.md
        "Testowa Firma Sp. Z 0.0",
        "Słodka Fabryka Example Sp. Z 0.0",
        "Example Sp.z0.0.",
        "Przykładowa Spółka Sp. z 0.0.",
        "PLG1 1090 1014 0000 0712 1981 2874",
        "anna.przykladowa@example. com",
        "3. Budżet na testy: 12 0OO zł",
        # the same kinds of misreads elsewhere
        "Sp. z o.0.",
        "Z 0",
        "P0lska",
        "Kowa1ski",
        "Sk1ep spożywczy",
        "rok 2O26",
        "1l5",
        "www. example.com",
        "anna @example.com",
        "Kowal;ski",
        "Fir|ma",
    ],
)
def test_ocr_misreads_are_detected(text: str) -> None:
    assert looks_like_ocr_noise(text)


@pytest.mark.parametrize(
    "text",
    [
        "Testowa Firma Sp. z o.o.",
        "Demo Systems S.A.",
        "FV/12/2026",
        "ZAM/455/2026",
        "ul. Długa 5, 00-950 Warszawa",
        "Mleko 2%",
        "100 g",
        "B2B",
        "H2O",
        "A4",
        "5a",
        "kod: A1B2",
        "Model X5",
        "Hala 3B",
        "5l",
        "MP3",
        "IPv4",
        "COVID-19",
        "Windows 11",
        "v1.0.0",
        "v0.0.1",
        "pkt 1.0 umowy",
        "wynik 2:0",
        "12:30",
        "1 szt.",
        "tak, proszę",
        "Etap 2: wdrożenie",
        "O'Neil",
        "2021-obecnie",
        "www.example.com",
        "https://example.com/a?b=1",
        "anna.przykladowa@example.com",
        "jan_testowy@example.org",
        "PL61 1090 1014 0000 0712 1981 2874",
        "Wspólnota Mieszkaniowa Przykładowa 7",
        "Notatka ze spotkania zespołu projektowego",
        "Python, FastAPI, PostgreSQL, Docker",
        "",
    ],
)
def test_ordinary_text_is_not_flagged(text: str) -> None:
    assert not looks_like_ocr_noise(text)


def test_known_blind_spots_stay_unflagged() -> None:
    """Misreads that still look like words cannot be detected; evals measure them instead."""
    assert not looks_like_ocr_noise("jantestowy@example.com")  # the dot was lost
    assert not looks_like_ocr_noise("Wdrozenie nowej wersji")  # the diacritic was lost
    assert not looks_like_ocr_noise("Www.example.com")


@pytest.mark.parametrize(
    ("value", "node", "expected"),
    [
        ("Firma Sp. Z 0.0", TEXT, True),
        ("Firma Sp. z o.o.", TEXT, False),
        # only plain text is judged: typed, formatted and enumerated values have their own checks
        (
            "PLG1 1090 1014 0000 0712 1981 2874",
            {"type": "string", "x-schemalens-type": "iban"},
            False,
        ),
        ("anna@example. com", {"type": "string", "format": "email"}, False),
        ("P0lska", {"type": "string", "enum": ["P0lska"]}, False),
        (12000.0, {"type": "number"}, False),
        (None, TEXT, False),
        (["Firma A", "Firma Sp. Z 0.0"], {"type": "array", "items": TEXT}, True),
        (["Firma A", "Firma B"], {"type": "array", "items": TEXT}, False),
        (
            [{"firma": "Example Sp.z0.0.", "okres": "2021-obecnie"}],
            {
                "type": "array",
                "items": {"type": "object", "properties": {"firma": TEXT, "okres": TEXT}},
            },
            True,
        ),
    ],
)
def test_has_noisy_text_walks_the_value_by_schema(value: Any, node: dict, expected: bool) -> None:
    assert has_noisy_text(value, node) is expected


LINES = [
    OcrLine(id="L001", page=1, text="Testowa Firma Sp. Z 0.0", bbox=None, conf=0.94),
    OcrLine(id="L002", page=1, text="Anna Przykładowa", bbox=None, conf=1.0),
    *[OcrLine(id=f"L{i:03d}", page=1, text="x", bbox=None, conf=0.9) for i in range(3, 9)],
]


def _score(data: dict[str, Any], nodes: dict[str, dict], evidence: dict[str, list[str]]):
    schema = {"type": "object", "properties": nodes}
    reported = {name: FieldMeta(confidence=1.0, evidence=evidence.get(name, [])) for name in data}
    return score_fields(schema, data, reported, LINES, SETTINGS)


def test_noisy_company_name_from_the_report_loses_its_high_confidence() -> None:
    """business_card `firma` scored 0.96 in the report while being an OCR misread."""
    fields, warnings = _score(
        {"firma": "Testowa Firma Sp. Z 0.0", "imie_nazwisko": "Anna Przykładowa"},
        {"firma": TEXT, "imie_nazwisko": TEXT},
        {"firma": ["L001"], "imie_nazwisko": ["L002"]},
    )

    assert warnings == ["firma: possible OCR noise"]
    assert fields["firma"].confidence == 0.575  # (0.7 * 0.94 + 0.3) * 0.6
    assert fields["firma"].confidence < 0.8 <= fields["imie_nazwisko"].confidence


def test_noise_inside_a_list_penalizes_the_field_once() -> None:
    fields, warnings = _score(
        {"strony": ["Example Sp. Z 0.0", "Druga Sp. Z 0.0"]},
        {"strony": {"type": "array", "items": TEXT}},
        {"strony": ["L001"]},
    )

    assert warnings == ["strony: possible OCR noise"]
    assert fields["strony"].confidence == 0.575


def test_derived_and_typed_fields_are_not_penalized_for_noise() -> None:
    fields, warnings = _score(
        {"streszczenie": "Umowa z firmą Sp. Z 0.0", "konto": "PLG1 1090 1014 0000 0712 1981 2874"},
        {
            "streszczenie": {"type": "string", "x-schemalens-derived": True},
            "konto": {"type": "string", "x-schemalens-type": "iban"},
        },
        {"streszczenie": ["L001"]},
    )

    assert not any("possible OCR noise" in warning for warning in warnings)
    assert fields["streszczenie"].confidence == 0.958
