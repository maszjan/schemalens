import pytest

from schemalens_core.language import detect_language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Faktura VAT nr 12/2026, data wystawienia 15 maja, razem do zapłaty 312,45 zł", "pl"),
        ("Invoice number 12, issue date 15 May, the total amount is due within 14 days", "en"),
        ("Rechnung Nr. 12 vom 15. Mai, der Gesamtbetrag ist in 14 Tagen fällig", "de"),
        ("Facture n° 12 du 15 mai, le montant total est à payer dans les 14 jours", "fr"),
    ],
)
def test_detects_common_languages(text: str, expected: str) -> None:
    assert detect_language(text) == expected


def test_too_little_text_is_not_guessed() -> None:
    assert detect_language("SUMA 12,50") is None
    assert detect_language("12345 67890 11 22 33 44") is None
