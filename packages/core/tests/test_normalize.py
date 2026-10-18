import pytest

from schemalens_core.normalize import (
    Normalized,
    fix_ocr_digits,
    is_valid_iban,
    is_valid_nip,
    is_valid_pesel,
    is_valid_regon,
    normalize_country,
    normalize_currency,
    normalize_date,
    normalize_number,
    normalize_phone,
    normalize_postal_code_pl,
    normalize_text,
    normalize_url,
    semantic_type,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-05-15", "2026-05-15"),
        ("2026-5-1", "2026-05-01"),
        ("2026.05.15", "2026-05-15"),
        ("15.05.2026", "2026-05-15"),
        ("15.05.2026 r.", "2026-05-15"),
        ("1/2/2026", "2026-02-01"),
        ("15-05-2026", "2026-05-15"),
        ("15 maja 2026", "2026-05-15"),
        ("15 maja 2026 roku", "2026-05-15"),
        ("3 października 2026 r.", "2026-10-03"),
        ("3 paź 2026", "2026-10-03"),
        ("3 paz. 2026", "2026-10-03"),
        ("7 pażdziernika 2026", "2026-10-07"),  # wrong diacritic, a common OCR misread
        ("7 pazdziernika 2026", "2026-10-07"),
        ("1 stycznia 2026", "2026-01-01"),
        ("28 lutego 2026", "2026-02-28"),
        ("7 września 2026", "2026-09-07"),
        ("  12 GRUDNIA 2026 ", "2026-12-12"),
        ("15 May 2026", "2026-05-15"),
        ("May 15, 2026", "2026-05-15"),
        ("Oct 3 2026", "2026-10-03"),
    ],
)
def test_normalize_date(text: str, expected: str) -> None:
    assert normalize_date(text) == expected


@pytest.mark.parametrize(
    "text", ["", "wczoraj", "31.02.2026", "15 foo 2026", "15.05.26", "2026", "maj 2026"]
)
def test_normalize_date_gives_up_instead_of_guessing(text: str) -> None:
    assert normalize_date(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("312,45", 312.45),
        ("312.45", 312.45),
        ("1 234,56", 1234.56),
        ("1\u00a0234,56", 1234.56),
        ("1\u202f234,56", 1234.56),
        ("1.234,56", 1234.56),
        ("1,234.56", 1234.56),
        ("1.234.567", 1234567.0),
        ("1,234,567", 1234567.0),
        ("-12,5", -12.5),
        ("100", 100.0),
    ],
)
def test_normalize_number(text: str, expected: float) -> None:
    assert normalize_number(text) == expected


@pytest.mark.parametrize("text", ["", "trzysta", "12,50 zł", "1e5", "12-34", ","])
def test_normalize_number_gives_up_instead_of_guessing(text: str) -> None:
    assert normalize_number(text) is None


# --- OCR character fixes ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2O26", "2026"),
        ("l5 maja 2O26", "15 maja 2026"),
        ("1O,48", "10,48"),
        ("S0", "50"),
        ("15 lipca 2026", "15 lipca 2026"),  # words are never touched
        ("O", "O"),  # a lone letter is not evidence of a digit
        ("SOS", "SOS"),
    ],
)
def test_fix_ocr_digits_only_touches_tokens_with_digits(text: str, expected: str) -> None:
    assert fix_ocr_digits(text) == expected


# --- identifiers -----------------------------------------------------------------------


@pytest.mark.parametrize("nip", ["1234563218", "5260250995"])
def test_valid_nip(nip: str) -> None:
    assert is_valid_nip(nip)


@pytest.mark.parametrize("nip", ["1234567890", "123456321", "12345632189", "abcdefghij", ""])
def test_invalid_nip(nip: str) -> None:
    assert not is_valid_nip(nip)


@pytest.mark.parametrize("regon", ["123456785", "12345678512347"])
def test_valid_regon(regon: str) -> None:
    assert is_valid_regon(regon)


@pytest.mark.parametrize("regon", ["123456789", "12345678512345", "1234", ""])
def test_invalid_regon(regon: str) -> None:
    assert not is_valid_regon(regon)


@pytest.mark.parametrize(
    "iban",
    ["PL61109010140000071219812874", "61109010140000071219812874", "DE89370400440532013000"],
)
def test_valid_iban(iban: str) -> None:
    assert is_valid_iban(iban)


@pytest.mark.parametrize(
    "iban",
    ["PL61109010140000071219812875", "PL6110901014000007121981287", "DE00370400440532013000", ""],
)
def test_invalid_iban(iban: str) -> None:
    assert not is_valid_iban(iban)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("zł", "PLN"),
        ("ZŁ", "PLN"),
        ("zl.", "PLN"),
        ("złotych", "PLN"),
        ("pln", "PLN"),
        ("€", "EUR"),
        ("Euro", "EUR"),
        ("$", "USD"),
        ("usd", "USD"),
        ("£", "GBP"),
        ("Kč", "CZK"),
        ("jpy", "JPY"),
        ("INR", "INR"),
        ("₹", "INR"),
        ("R$", "BRL"),
        ("Swiss franc", "CHF"),
        ("yen", "JPY"),
        ("zar", "ZAR"),
    ],
)
def test_normalize_currency(text: str, expected: str) -> None:
    assert normalize_currency(text) == expected


@pytest.mark.parametrize("text", ["", "bitcoin", "XYZ", "12", "kr", "¥"])  # unknown or ambiguous
def test_unknown_currency_is_left_alone(text: str) -> None:
    assert normalize_currency(text) is None


@pytest.mark.parametrize(
    "declared",
    ["nip", "regon", "pesel", "iban", "phone", "currency", "country", "postal_code_pl"],
)
def test_semantic_type_is_read_from_the_extension_keyword(declared: str) -> None:
    assert semantic_type({"type": "string", "x-schemalens-type": declared}) == declared


@pytest.mark.parametrize("node", [{}, {"x-schemalens-type": "unknown"}, {"description": "NIP"}])
def test_semantic_type_is_never_guessed(node: dict) -> None:
    assert semantic_type(node) is None


# --- normalize_text: one value, by field type -----------------------------------------


def _text(text: str, kind: str = "string", fmt: str | None = None, semantic=None) -> Normalized:
    return normalize_text(text, kind=kind, fmt=fmt, semantic=semantic)


def test_number_with_ocr_letters_is_fixed_only_when_it_then_parses() -> None:
    assert _text("1O,48", "number") == Normalized(10.48, ocr_fixed=True)
    assert _text("10,48", "number") == Normalized(10.48)
    assert _text("okolo 1O", "number") == Normalized("okolo 1O")
    assert _text("l2", "integer") == Normalized(12, ocr_fixed=True)
    assert _text("12,5", "integer") == Normalized("12,5")


def test_date_with_ocr_letters_is_fixed_only_when_it_then_parses() -> None:
    assert _text("l5 maja 2O26", fmt="date") == Normalized("2026-05-15", ocr_fixed=True)
    assert _text("15 lipca 2026", fmt="date") == Normalized("2026-07-15")
    assert _text("3l.O2.2026", fmt="date") == Normalized("3l.O2.2026")  # 31 Feb is not a date


def test_plain_strings_are_never_touched() -> None:
    assert _text("SOS 2O26 l5") == Normalized("SOS 2O26 l5")


def test_nip_is_cleaned_and_checked() -> None:
    assert _text("PL 123-456-32-18", semantic="nip") == Normalized("1234563218")
    assert _text("1234563218", semantic="nip") == Normalized("1234563218")


def test_nip_ocr_fix_requires_a_valid_checksum() -> None:
    assert _text("l234S632l8", semantic="nip") == Normalized("1234563218", ocr_fixed=True)
    assert _text("123-456-78-90", semantic="nip") == Normalized(
        "123-456-78-90", problem="NIP checksum is invalid"
    )
    assert _text("l234S67890", semantic="nip") == Normalized(
        "l234S67890", problem="NIP checksum is invalid"
    )


def test_regon_and_iban() -> None:
    assert _text("123-456-785", semantic="regon") == Normalized("123456785")
    assert _text("123456789", semantic="regon") == Normalized(
        "123456789", problem="REGON checksum is invalid"
    )
    assert _text("pl61 1090 1014 0000 0712 1981 2874", semantic="iban") == Normalized(
        "PL61109010140000071219812874"
    )
    assert _text("PLG1 1090 1014 0000 0712 1981 2874", semantic="iban") == Normalized(
        "PLG1 1090 1014 0000 0712 1981 2874", problem="IBAN checksum is invalid"
    )


def test_currency() -> None:
    assert _text("zł", semantic="currency") == Normalized("PLN")
    assert _text("bitcoin", semantic="currency") == Normalized("bitcoin")


# --- pesel, phone, country, postal code ------------------------------------------------


def test_pesel_checksum() -> None:
    assert is_valid_pesel("44051401359")
    assert not is_valid_pesel("44051401358")
    assert not is_valid_pesel("4405140135")
    assert _text("440514 01359", semantic="pesel") == Normalized("44051401359")
    assert _text("44051401358", semantic="pesel") == Normalized(
        "44051401358", problem="PESEL checksum is invalid"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("+48 600 100 200", "+48600100200"),
        ("600-100-200", "600100200"),
        ("(22) 123 45 67", "221234567"),
        ("0048 600 100 200", "+48600100200"),
        ("601.202.303", "601202303"),
    ],
)
def test_normalize_phone_never_adds_a_country_code(text: str, expected: str) -> None:
    assert normalize_phone(text) == expected


@pytest.mark.parametrize("text", ["", "12345", "tel. 600 100 200", "600 100 200 wew. 12"])
def test_unparseable_phone_is_left_alone(text: str) -> None:
    assert normalize_phone(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Polska", "PL"),
        ("poland", "PL"),
        ("pl", "PL"),
        ("Niemcy", "DE"),
        ("Włochy", "IT"),
        ("USA", "US"),
    ],
)
def test_normalize_country(text: str, expected: str) -> None:
    assert normalize_country(text) == expected


def test_unknown_country_is_left_alone() -> None:
    assert normalize_country("Atlantyda") is None
    assert normalize_country("XX") is None


@pytest.mark.parametrize(
    ("text", "expected"), [("00-950", "00-950"), ("00950", "00-950"), ("00 950", "00-950")]
)
def test_normalize_postal_code_pl(text: str, expected: str) -> None:
    assert normalize_postal_code_pl(text) == expected


def test_postal_code_ocr_fix_and_rejects() -> None:
    assert _text("O0-9S0", semantic="postal_code_pl") == Normalized("00-950", ocr_fixed=True)
    assert _text("0-950", semantic="postal_code_pl") == Normalized("0-950")
    assert _text("Polska", semantic="country") == Normalized("PL")
    assert _text("+48 600 100 200", semantic="phone") == Normalized("+48600100200")


# --- stray non-digit characters in identifiers -----------------------------------------


def test_regression_stray_letter_inside_a_nip_is_removed_when_the_checksum_agrees() -> None:
    """From the review of real documents: OCR read a NIP as '9876543 b210'."""
    assert _text("9876543 b210", semantic="nip") == Normalized("9876543210", noise_removed=True)


@pytest.mark.parametrize(
    ("text", "semantic", "expected"),
    [
        ("NIP: 123-456-32-18.", "nip", "1234563218"),
        ("1234563218 r", "nip", "1234563218"),
        ("123456785/", "regon", "123456785"),
        ("44051401359 (PESEL)", "pesel", "44051401359"),
    ],
)
def test_noise_is_removed_from_nip_regon_and_pesel(text: str, semantic: str, expected: str) -> None:
    assert _text(text, semantic=semantic) == Normalized(expected, noise_removed=True)


@pytest.mark.parametrize(
    ("text", "semantic", "label"),
    [
        ("9876543 b211", "nip", "NIP"),  # digits left over fail the checksum
        ("98765 b210", "nip", "NIP"),  # wrong length
        ("abc", "nip", "NIP"),
        ("12345678x9", "regon", "REGON"),
        ("4405140135x8", "pesel", "PESEL"),
    ],
)
def test_noise_removal_needs_the_right_length_and_checksum(
    text: str, semantic: str, label: str
) -> None:
    assert _text(text, semantic=semantic) == Normalized(
        text, problem=f"{label} checksum is invalid"
    )


def test_clean_identifiers_and_ocr_fixes_are_not_reported_as_noise() -> None:
    assert _text("123-456-32-18", semantic="nip") == Normalized("1234563218")
    assert _text("l234S632l8", semantic="nip") == Normalized("1234563218", ocr_fixed=True)
    assert (
        _text("PL61 1090 1014 0000 0712 1981 2874 x", semantic="iban").problem is not None
    )  # IBAN keeps letters


@pytest.mark.parametrize(
    ("text", "expected", "ocr_fixed"),
    [
        ("www.example.com", "www.example.com", False),
        ("https://Example.com/kontakt", "example.com/kontakt", False),
        ("WWW.Example.PL", "www.example.pl", False),
        ("example . com", "example.com", False),
        # misreads seen on printed business cards
        ("Wwvwebsitename com", "www.websitename.com", True),
        ("vvw.example.com", "www.example.com", True),
        ("wvw example pl", "www.example.pl", True),
        ("www.example, com", "www.example.com", True),
    ],
)
def test_normalize_url(text: str, expected: str, ocr_fixed: bool) -> None:
    assert normalize_url(text) == (expected, ocr_fixed)


@pytest.mark.parametrize("text", ["", "not an address", "example", "anna@example.com"])
def test_unparseable_url_is_left_alone_with_a_problem(text: str) -> None:
    assert normalize_url(text) is None
    if text:
        result = normalize_text(text, kind="string", fmt=None, semantic="url")
        assert result.value == text and result.problem == "web address is not well-formed"
