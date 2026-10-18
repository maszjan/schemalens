"""Language of a document's text, from its most common short words.

Cheap and local (no model, no dependency): enough to tell the reader of a result which
language the document was in. The OCR languages and the prompt do not depend on it.
"""

import re


# A few dozen of the most frequent words per language; diacritics also count (below).
def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


_STOPWORDS: dict[str, frozenset[str]] = {
    "pl": _words(
        "i w na z do nie się że jest to o od za po przez dla oraz lub jak ale czy który która "
        "które tak jego jej ich być był była sp zł ul nr data kwota faktura razem"
    ),
    "en": _words(
        "the and of to in is for on with as by at from this that be are was or an it not "
        "your our we you total date invoice amount number street"
    ),
    "de": _words(
        "der die das und ist nicht mit von zu den des im ein eine für auf dem sie es auch "
        "rechnung datum betrag gesamt straße"
    ),
    "fr": _words(
        "le la les et de des du un une est pour dans en sur au aux avec par pas que qui "
        "facture date montant total rue"
    ),
    "es": _words(
        "el la los las y de del en un una es por para con no que se al su factura fecha "
        "importe total calle"
    ),
    "it": _words(
        "il lo la gli le e di del della in un una è per con non che si al fattura data "
        "importo totale via"
    ),
}
_LETTERS = {
    "pl": set("ąćęłńóśźż"),
    "de": set("äöüß"),
    "fr": set("àâçéèêëîïôûùœ"),
    "es": set("ñáíóúü¿¡"),
}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(text: str, *, min_words: int = 5) -> str | None:
    """ISO 639-1 code of the most likely language, or None when the text is too short."""
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < min_words:
        return None
    scores = {lang: sum(w in stop for w in words) for lang, stop in _STOPWORDS.items()}
    chars = set(text.lower())
    for lang, letters in _LETTERS.items():
        scores[lang] += 3 * len(chars & letters)
    best = max(scores, key=lambda lang: scores[lang])
    return best if scores[best] >= 2 else None
