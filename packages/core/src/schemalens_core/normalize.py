"""Deterministic normalizers (code, not LLM): dates, numbers and declared domain types.

Nothing here guesses. A value is changed only when the result is unambiguous; OCR
character fixes are accepted only when the corrected text parses or passes its checksum.
"""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

SemanticType = Literal[
    "nip", "regon", "pesel", "iban", "phone", "currency", "country", "postal_code_pl", "url"
]  # fmt: skip

_MONTHS = {
    # Polish, by the first three letters of any inflected form (maja, października, paź.)
    "sty": 1, "lut": 2, "mar": 3, "kwi": 4, "maj": 5, "cze": 6,
    "lip": 7, "sie": 8, "wrz": 9, "paz": 10, "lis": 11, "gru": 12,
    # English
    "jan": 1, "feb": 2, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip

_YEAR_SUFFIX = re.compile(r"(?<=\d)\s*(?:roku|r\.?)$", re.IGNORECASE)
_YMD = re.compile(r"^(\d{4})[-./](\d{1,2})[-./](\d{1,2})$")
_DMY = re.compile(r"^(\d{1,2})[-./](\d{1,2})[-./](\d{4})$")
_D_MONTH_Y = re.compile(r"^(\d{1,2})\.?\s+([^\W\d_]+)\.?,?\s+(\d{4})$")
_MONTH_D_Y = re.compile(r"^([^\W\d_]+)\.?\s+(\d{1,2}),?\s+(\d{4})$")
_NUMBER = re.compile(r"^[+-]?\d[\d.,]*$")

# Letters OCR commonly reads in place of digits.
_OCR_DIGITS = str.maketrans("OlS", "015")

# Symbols and names that point to one currency. Ambiguous ones ("kr" = SEK/NOK/DKK,
# "¥" = JPY/CNY, "R" = ZAR/...) are left out on purpose: such a value stays as it was.
_CURRENCIES = {
    "zł": "PLN", "zl": "PLN", "zło": "PLN", "zlo": "PLN", "złoty": "PLN", "zloty": "PLN",
    "złote": "PLN", "zlote": "PLN", "złotych": "PLN", "zlotych": "PLN",
    "€": "EUR", "euro": "EUR", "euros": "EUR",
    "$": "USD", "us$": "USD", "dollar": "USD", "dollars": "USD", "dolar": "USD", "dolarów": "USD",
    "dolarow": "USD", "£": "GBP", "pound": "GBP", "pounds": "GBP", "sterling": "GBP", "funt": "GBP",
    "funtów": "GBP", "kč": "CZK", "kc": "CZK", "fr": "CHF", "frank": "CHF", "franków": "CHF",
    "swiss franc": "CHF", "c$": "CAD", "ca$": "CAD", "a$": "AUD", "au$": "AUD", "nz$": "NZD",
    "hk$": "HKD", "s$": "SGD", "r$": "BRL", "mx$": "MXN", "₹": "INR", "rupee": "INR",
    "rupees": "INR", "₩": "KRW", "won": "KRW", "₽": "RUB", "rouble": "RUB", "ruble": "RUB",
    "₺": "TRY", "lira": "TRY", "₴": "UAH", "hryvnia": "UAH", "₪": "ILS", "shekel": "ILS",
    "₫": "VND", "₱": "PHP", "฿": "THB", "baht": "THB", "₦": "NGN", "naira": "NGN",
    "ft": "HUF", "forint": "HUF", "lei": "RON", "leu": "RON", "yen": "JPY", "yuan": "CNY",
    "renminbi": "CNY", "rmb": "CNY", "zar": "ZAR", "rand": "ZAR",
}  # fmt: skip
# Active ISO 4217 codes: any of them is accepted as written.
_ISO_4217 = {
    "AED", "AFN", "ALL", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG", "AZN", "BAM", "BBD", "BDT",
    "BGN", "BHD", "BIF", "BMD", "BND", "BOB", "BRL", "BSD", "BTN", "BWP", "BYN", "BZD", "CAD",
    "CDF", "CHF", "CLP", "CNY", "COP", "CRC", "CUP", "CVE", "CZK", "DJF", "DKK", "DOP", "DZD",
    "EGP", "ERN", "ETB", "EUR", "FJD", "FKP", "GBP", "GEL", "GHS", "GIP", "GMD", "GNF", "GTQ",
    "GYD", "HKD", "HNL", "HTG", "HUF", "IDR", "ILS", "INR", "IQD", "IRR", "ISK", "JMD", "JOD",
    "JPY", "KES", "KGS", "KHR", "KMF", "KPW", "KRW", "KWD", "KYD", "KZT", "LAK", "LBP", "LKR",
    "LRD", "LSL", "LYD", "MAD", "MDL", "MGA", "MKD", "MMK", "MNT", "MOP", "MRU", "MUR", "MVR",
    "MWK", "MXN", "MYR", "MZN", "NAD", "NGN", "NIO", "NOK", "NPR", "NZD", "OMR", "PAB", "PEN",
    "PGK", "PHP", "PKR", "PLN", "PYG", "QAR", "RON", "RSD", "RUB", "RWF", "SAR", "SBD", "SCR",
    "SDG", "SEK", "SGD", "SHP", "SLE", "SOS", "SRD", "SSP", "STN", "SVC", "SYP", "SZL", "THB",
    "TJS", "TMT", "TND", "TOP", "TRY", "TTD", "TWD", "TZS", "UAH", "UGX", "USD", "UYU", "UZS",
    "VES", "VND", "VUV", "WST", "XAF", "XCD", "XOF", "XPF", "YER", "ZAR", "ZMW", "ZWL",
}  # fmt: skip

_COUNTRIES = {
    "polska": "PL", "poland": "PL", "niemcy": "DE", "germany": "DE", "deutschland": "DE",
    "francja": "FR", "france": "FR", "włochy": "IT", "wlochy": "IT", "italy": "IT",
    "hiszpania": "ES", "spain": "ES", "czechy": "CZ", "czech republic": "CZ", "czechia": "CZ",
    "słowacja": "SK", "slowacja": "SK", "slovakia": "SK", "ukraina": "UA", "ukraine": "UA",
    "litwa": "LT", "lithuania": "LT", "holandia": "NL", "netherlands": "NL", "belgia": "BE",
    "belgium": "BE", "austria": "AT", "szwajcaria": "CH", "switzerland": "CH",
    "szwecja": "SE", "sweden": "SE", "norwegia": "NO", "norway": "NO", "dania": "DK",
    "denmark": "DK", "wielka brytania": "GB", "united kingdom": "GB", "uk": "GB",
    "stany zjednoczone": "US", "united states": "US", "usa": "US", "chiny": "CN", "china": "CN",
    "turcja": "TR", "turkey": "TR", "węgry": "HU", "wegry": "HU", "hungary": "HU",
    "rumunia": "RO", "romania": "RO", "irlandia": "IE", "ireland": "IE",
    "portugalia": "PT", "portugal": "PT", "grecja": "GR", "greece": "GR",
}  # fmt: skip
_COUNTRY_CODES = set(_COUNTRIES.values())

# --- dates and numbers ----------------------------------------------------------------


def _month(name: str) -> int | None:
    # Diacritics are dropped first: OCR and vision models write "pazdziernika" or "pażdziernika".
    plain = unicodedata.normalize("NFKD", name.lower().replace("ł", "l"))
    return _MONTHS.get("".join(c for c in plain if not unicodedata.combining(c))[:3])


def normalize_date(text: str) -> str | None:
    """Parse common Polish/English date notations into ISO 8601 (`YYYY-MM-DD`).

    Numeric dates are read day-first (`15.05.2026`), as written in Poland.
    """
    cleaned = _YEAR_SUFFIX.sub("", text.strip())
    year: str | int
    month: str | int | None
    day: str | int
    if match := _YMD.match(cleaned):
        year, month, day = match.groups()
    elif match := _DMY.match(cleaned):
        day, month, year = match.groups()
    elif match := _D_MONTH_Y.match(cleaned):
        day, name, year = match.groups()
        month = _month(name)
    elif match := _MONTH_D_Y.match(cleaned):
        name, day, year = match.groups()
        month = _month(name)
    else:
        return None
    if month is None:
        return None
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def normalize_number(text: str) -> float | None:
    """Parse `1 234,56`, `1.234,56`, `1,234.56` or `312,45` into a float.

    A single comma is a decimal separator (Polish notation).
    """
    cleaned = "".join(text.split())  # also drops non-breaking and narrow spaces
    if not _NUMBER.match(cleaned):
        return None
    if "," in cleaned and "." in cleaned:
        decimal, thousands = (",", ".") if cleaned.rfind(",") > cleaned.rfind(".") else (".", ",")
        cleaned = cleaned.replace(thousands, "").replace(decimal, ".")
    elif cleaned.count(",") == 1:
        cleaned = cleaned.replace(",", ".")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", "")
    elif cleaned.count(".") > 1:
        cleaned = cleaned.replace(".", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def fix_ocr_digits(text: str) -> str:
    """Replace O/l/S with 0/1/5, but only inside tokens that already contain a digit.

    Month names and other words are left alone, and so is a lone letter: `2O26` is
    clearly a number, a bare `O` is not.
    """

    def fix(match: re.Match[str]) -> str:
        token = match.group()
        return token.translate(_OCR_DIGITS) if any(c.isdigit() for c in token) else token

    return re.sub(r"[^\W_]+", fix, text)


# --- identifiers and currencies -------------------------------------------------------


def _weighted_sum(digits: str, weights: tuple[int, ...]) -> int:
    return sum(int(d) * w for d, w in zip(digits, weights, strict=False))


def clean_nip(text: str) -> str:
    cleaned = re.sub(r"[\s\-]", "", text)
    return cleaned[2:] if cleaned[:2].upper() == "PL" else cleaned


def is_valid_nip(nip: str) -> bool:
    if not re.fullmatch(r"\d{10}", nip):
        return False
    return _weighted_sum(nip, (6, 5, 7, 2, 3, 4, 5, 6, 7)) % 11 == int(nip[9])


def clean_regon(text: str) -> str:
    return re.sub(r"[\s\-]", "", text)


def is_valid_regon(regon: str) -> bool:
    if re.fullmatch(r"\d{9}", regon):
        weights: tuple[int, ...] = (8, 9, 2, 3, 4, 5, 6, 7)
    elif re.fullmatch(r"\d{14}", regon):
        weights = (2, 4, 8, 5, 0, 9, 7, 3, 6, 1, 2, 4, 8)
    else:
        return False
    return _weighted_sum(regon, weights) % 11 % 10 == int(regon[-1])


def is_valid_pesel(pesel: str) -> bool:
    if not re.fullmatch(r"\d{11}", pesel):
        return False
    return (10 - _weighted_sum(pesel, (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)) % 10) % 10 == int(pesel[10])


def clean_iban(text: str) -> str:
    return re.sub(r"[\s\-]", "", text).upper()


def is_valid_iban(iban: str) -> bool:
    """ISO 13616 mod-97 check. A bare 26-digit Polish account number (NRB) is checked as PL."""
    if re.fullmatch(r"\d{26}", iban):
        iban = "PL" + iban
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", iban):
        return False
    if iban.startswith("PL") and len(iban) != 28:
        return False
    rearranged = iban[4:] + iban[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


def normalize_currency(text: str) -> str | None:
    """Map any ISO 4217 code, or an unambiguous symbol or name, to the ISO 4217 code."""
    cleaned = text.strip().rstrip(".").lower()
    if cleaned.upper() in _ISO_4217:
        return cleaned.upper()
    return _CURRENCIES.get(cleaned)


SEMANTIC_TYPES: tuple[SemanticType, ...] = (
    "nip", "regon", "pesel", "iban", "phone", "currency", "country", "postal_code_pl", "url"
)  # fmt: skip


def normalize_phone(text: str) -> str | None:
    """Compact phone number: digits with an optional leading `+`. No country code is added."""
    cleaned = re.sub(r"[\s\-./()]", "", text)
    if cleaned.startswith("00"):
        cleaned = "+" + cleaned[2:]
    return cleaned if re.fullmatch(r"\+?\d{7,15}", cleaned) else None


def normalize_country(text: str) -> str | None:
    """Map a country name (Polish or English) or code to ISO 3166-1 alpha-2."""
    cleaned = text.strip().rstrip(".")
    if cleaned.upper() in _COUNTRY_CODES:
        return cleaned.upper()
    return _COUNTRIES.get(cleaned.lower())


def normalize_postal_code_pl(text: str) -> str | None:
    digits = re.sub(r"[\s\-]", "", text)
    return f"{digits[:2]}-{digits[2:]}" if re.fullmatch(r"\d{5}", digits) else None


_URL = re.compile(r"(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,24}(?:/\S*)?")
# OCR reads the w of "www" as v (or vv): "Wwvexample.com", "vvw.example.com".
_WWW_MISREAD = re.compile(r"^(?!www)[wv]{3}(?:[.,]\s*|\s+|(?=[a-z0-9]))", re.IGNORECASE)
_DOT_MISREAD = re.compile(r"\s*[,\s]\s*(?=[a-z]{2,6}(?:/|$))")  # "example com", "example, pl"


def normalize_url(text: str) -> tuple[str, bool] | None:
    """A web address as `www.example.com/path` (scheme dropped, host in lower case).

    OCR fixes are made only for misreads seen on printed addresses: a mangled `www`
    and a dot read as a space or comma before the domain ending. The result must look like
    an address, or nothing is changed. Returns the address and whether OCR was corrected.
    """
    cleaned = text.strip().strip(".,;")
    cleaned = re.sub(r"^https?://", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\.\s*", ".", cleaned)  # "example . com"
    host, slash, path = cleaned.partition("/")
    fixed = _DOT_MISREAD.sub(".", _WWW_MISREAD.sub("www.", host))
    address = fixed.lower() + slash + path
    if not _URL.fullmatch(address):
        return None
    return address, fixed.lower() != host.lower()


def semantic_type(node: dict[str, Any]) -> SemanticType | None:
    """Domain type of a field. Declared only through `x-schemalens-type`, never guessed."""
    declared = node.get("x-schemalens-type")
    return declared if declared in SEMANTIC_TYPES else None


# --- one entry point for the schema walker ---------------------------------------------


_MAPPERS: dict[str, Callable[[str], str | None]] = {
    "currency": normalize_currency,
    "country": normalize_country,
    "phone": normalize_phone,
}


@dataclass(frozen=True)
class Normalized:
    value: Any
    ocr_fixed: bool = False
    problem: str | None = None  # reported as a warning; the value is left as it was
    noise_removed: bool = False  # stray non-digit characters were dropped from an identifier


def _with_ocr_fix(text: str, parse: Callable[[str], Any]) -> Normalized | None:
    parsed = parse(text)
    if parsed is not None:
        return Normalized(parsed)
    fixed = fix_ocr_digits(text)
    if fixed != text and (parsed := parse(fixed)) is not None:
        return Normalized(parsed, ocr_fixed=True)
    return None


def _identifier(
    text: str, clean: Callable[[str], str], is_valid: Callable[[str], bool], label: str
) -> Normalized:
    cleaned = clean(text)
    if is_valid(cleaned):
        return Normalized(cleaned)
    fixed = cleaned.translate(_OCR_DIGITS)
    if fixed != cleaned and is_valid(fixed):
        return Normalized(fixed, ocr_fixed=True)
    # OCR sometimes drops a stray character into the number ("9876543 b210"). The checksum
    # decides whether what is left is the real identifier; if not, nothing is changed.
    digits = re.sub(r"\D", "", cleaned)
    if digits != cleaned and is_valid(digits):
        return Normalized(digits, noise_removed=True)
    return Normalized(text, problem=f"{label} checksum is invalid")


def _integer(text: str) -> int | None:
    number = normalize_number(text)
    return int(number) if number is not None and number.is_integer() else None


def normalize_text(
    text: str, *, kind: str | None, fmt: str | None, semantic: SemanticType | None
) -> Normalized:
    """Normalize one string value according to the type of the schema field it belongs to."""
    result: Normalized | None = None
    if kind == "number":
        result = _with_ocr_fix(text, normalize_number)
    elif kind == "integer":
        result = _with_ocr_fix(text, _integer)
    elif kind == "string" and fmt == "date":
        result = _with_ocr_fix(text, normalize_date)
    elif kind == "string" and semantic == "nip":
        result = _identifier(text, clean_nip, is_valid_nip, "NIP")
    elif kind == "string" and semantic == "regon":
        result = _identifier(text, clean_regon, is_valid_regon, "REGON")
    elif kind == "string" and semantic == "iban":
        # No OCR fix here: letters are legitimate in an IBAN.
        cleaned = clean_iban(text)
        valid = is_valid_iban(cleaned)
        result = Normalized(
            cleaned if valid else text, problem=None if valid else "IBAN checksum is invalid"
        )
    elif kind == "string" and semantic == "pesel":
        result = _identifier(text, clean_regon, is_valid_pesel, "PESEL")
    elif kind == "string" and semantic == "postal_code_pl":
        result = _with_ocr_fix(text, normalize_postal_code_pl)
    elif kind == "string" and semantic == "url":
        url = normalize_url(text)
        result = (
            Normalized(url[0], ocr_fixed=url[1])
            if url
            else Normalized(text, problem="web address is not well-formed")
        )
    elif kind == "string" and semantic in _MAPPERS:
        mapped = _MAPPERS[semantic](text)
        result = Normalized(mapped) if mapped else None
    return result or Normalized(text)
