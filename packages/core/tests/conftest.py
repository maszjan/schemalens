"""Synthetic documents shared by tests. Everything is generated in memory."""

import pytest

from evals.synth import render_scan

RECEIPT_ROWS = [
    "SKLEP SPOŻYWCZY EXAMPLE",
    "ul. Testowa 1, 00-001 Łódź",
    "NIP 123-456-32-18",
    "PARAGON FISKALNY",
    "15 października 2026 r.",
    "Chleb żytni 1 szt 6,49",
    "Mleko 2% 1 szt 3,99",
    "SUMA PLN 10,48",
]


@pytest.fixture(scope="session")
def receipt_png() -> bytes:
    return render_scan(RECEIPT_ROWS)
