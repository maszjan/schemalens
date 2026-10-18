"""The eval tooling itself: comparison rules, golden set integrity, report hygiene."""

from typing import Any

import pytest

from evals import run as eval_run
from evals.cases import load_cases
from evals.metrics import field_outcome, list_f1, percentile
from evals.run import (
    CachedLlm,
    FieldRow,
    QuotaExhausted,
    RunResult,
    build_report,
    cache_key,
    estimate_run,
    resolve_run,
    run_profile,
    select_cases,
)
from schemalens_core.config import Settings
from schemalens_core.errors import LlmFailed, LlmRateLimited, ProfileUnavailable
from schemalens_core.llm import LlmResponse
from schemalens_core.models import OcrLine, OcrPage, OcrResult
from schemalens_core.presets import resolve_schema
from schemalens_core.schemas import compile_schema

SETTINGS = Settings(_env_file=None)
TEXT = {"type": "string"}
NUMBER = {"type": "number"}
OCR = "Sprzedawca: Example Sp.z0.0.\nSUMA PLN 10,48"


@pytest.mark.parametrize(
    ("value", "expected", "node", "outcome"),
    [
        ("Example Sp. z o.o.", "Example Sp. z o.o.", TEXT, "correct"),
        ("  example  SP. Z O.O. ", "Example Sp. z o.o.", TEXT, "correct"),
        (10.484, 10.48, NUMBER, "correct"),
        (10.84, 10.48, NUMBER, "wrong"),
        (None, None, TEXT, "correct"),
        (None, "x", TEXT, "miss"),
        ("x", None, TEXT, "hallucination"),
        ("Example Sp.z0.0.", "Example Sp. z o.o.", TEXT, "ocr_text_noise"),
        ("Inna Firma S.A.", "Example Sp. z o.o.", TEXT, "wrong"),
        ("Example Sp.z0.0.", "Example Sp. z o.o.", {"type": "string", "enum": ["a"]}, "wrong"),
        ("2026-10-16", "2026-10-15", {"type": "string", "format": "date"}, "wrong"),
        (True, True, {"type": "boolean"}, "correct"),
        (False, True, {"type": "boolean"}, "wrong"),
    ],
)
def test_field_outcome(value: Any, expected: Any, node: dict, outcome: str) -> None:
    assert field_outcome(value, expected, node, OCR) == outcome


def test_free_text_fields_only_need_a_value() -> None:
    assert (
        field_outcome("dowolne streszczenie", "(dowolne)", TEXT, OCR, free_text=True) == "correct"
    )
    assert field_outcome(None, "(dowolne)", TEXT, OCR, free_text=True) == "miss"


def test_lists_are_compared_without_order() -> None:
    node = {"type": "array", "items": TEXT}
    assert field_outcome(["b", "a"], ["a", "b"], node, OCR) == "correct"
    assert field_outcome(["a"], ["a", "b"], node, OCR) == "wrong"
    assert list_f1(["a"], ["a", "b"], node) == pytest.approx(2 / 3)
    assert list_f1([], [], node) == 1.0


def test_lists_of_objects_and_noise_inside_them() -> None:
    node = {
        "type": "array",
        "items": {"type": "object", "properties": {"nazwa": TEXT, "cena": NUMBER}},
    }
    expected = [{"nazwa": "Example Sp. z o.o.", "cena": 10.48}]
    assert (
        field_outcome([{"nazwa": "Example Sp. z o.o.", "cena": 10.48}], expected, node, OCR)
        == "correct"
    )
    assert (
        field_outcome([{"nazwa": "Example Sp.z0.0.", "cena": 10.48}], expected, node, OCR)
        == "ocr_text_noise"
    )
    assert (
        field_outcome([{"nazwa": "Example Sp.z0.0.", "cena": 1.0}], expected, node, OCR) == "wrong"
    )


def test_percentile() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([7], 95) == 7.0
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0


def test_golden_set_covers_every_preset_and_the_special_cases() -> None:
    cases = load_cases()
    categories = {case.category for case in cases}

    assert len(cases) >= 12
    assert categories >= {
        "invoice", "receipt", "business_card", "cv", "product_label", "menu",
        "contract_terms", "generic_document", "other",
    }  # fmt: skip
    assert sum(case.category == "other" for case in cases) >= 2
    assert any("injection" in case.tags for case in cases)
    assert any("absent_field" in case.tags for case in cases)
    assert len({case.id for case in cases}) == len(cases)


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case.id)
def test_every_case_has_a_valid_schema_and_complete_expectations(case) -> None:
    schema = resolve_schema(case.schema, case.preset, SETTINGS)
    compile_schema(schema, SETTINGS)
    assert set(case.expected) == set(schema["properties"])
    assert case.free_text <= set(schema["properties"])
    expected_start = b"%PDF" if case.pdf_pages else b"\xff\xd8\xff"  # text PDF or JPEG
    assert case.document().startswith(expected_start)


def test_absent_field_cases_expect_null() -> None:
    for case in load_cases():
        if "absent_field" in case.tags:
            assert None in case.expected.values(), case.id


def test_resolve_run_supports_effort_variants() -> None:
    settings = Settings(_env_file=None, groq_api_key="gsk_test", groq_model_fast="m")
    assert resolve_run("groq", settings)[1].reasoning_effort == "medium"
    name, profile = resolve_run("groq:low", settings)
    assert name == "groq:low" and profile.reasoning_effort == "low"
    assert resolve_run("local", settings)[1].name == "local"
    with pytest.raises(ProfileUnavailable):
        resolve_run("groq", Settings(_env_file=None))


def test_report_contains_metrics_but_no_document_content_or_keys() -> None:
    cases = load_cases()[:2]
    run = RunResult(name="groq", model="some-model", reasoning_effort="low")
    for case in cases:
        for i, name in enumerate(case.expected):
            outcome = "wrong" if i == 0 else "correct"
            run.rows.append(
                FieldRow(case.id, case.category, name, outcome, 0.9, 0.95, i == 0, False)
            )
    run.ocr_ms, run.llm_ms, run.tokens_per_page, run.completion_tokens = (
        [900],
        [2000],
        [2500.0],
        [800],
    )

    report = build_report([run], cases, ["przebieg `local` — niedostępny"])

    assert "| groq | `some-model` | low |" in report
    assert "Pominięto: przebieg `local`" in report
    assert "## Kalibracja confidence" in report and "## Grounding i limit evidence" in report
    for case in cases:
        assert case.id in report
        for line in case.lines:
            assert line not in report
    assert "gsk_" not in report


# --- hosted profiles on a free tier: cap, cache, estimate, clean stop -------------------

GROQ_SETTINGS = Settings(_env_file=None, groq_api_key="gsk_test", groq_model_fast="oss-20b")
GROQ = GROQ_SETTINGS.llm_profile("groq")
LOCAL = Settings(_env_file=None, local_model_fast="bielik").llm_profile("local")
REQUEST = {"system": "s", "user": "u", "json_schema": {"type": "object"}, "model": "oss-20b"}


class FakeOcr:
    def read(self, images: list) -> OcrResult:
        lines = [
            OcrLine(id=f"L{i:03d}", page=1, text=f"linia dokumentu numer {i}", bbox=None, conf=0.9)
            for i in range(1, 6)
        ]
        return OcrResult(lines=lines, pages=[OcrPage(number=1, source="ocr", ocr_ms=5)])


class CountingClient:
    def __init__(self, fail_after: int | None = None, error: Exception | None = None) -> None:
        self.calls = 0
        self._fail_after, self._error = fail_after, error

    def extract(self, **kwargs: Any) -> LlmResponse:
        if self._fail_after is not None and self.calls >= self._fail_after:
            raise self._error or LlmRateLimited("quota", retry_after=868.0)
        self.calls += 1
        return LlmResponse(
            content='{"fields": {}}', prompt_tokens=900, completion_tokens=300, llm_ms=1200,
            model=kwargs["model"], reasoning_effort="medium",
        )  # fmt: skip


def test_evals_default_to_the_local_profile() -> None:
    assert eval_run.DEFAULT_PROFILE == "local"
    assert eval_run.DEFAULT_MAX_DOCS == 4


def test_hosted_runs_are_capped_and_take_the_priority_cases_first() -> None:
    cases = load_cases()

    capped = select_cases(cases, GROQ, 4)

    assert [case.id for case in capped] == list(eval_run.PRIORITY)
    assert len(select_cases(cases, GROQ, 2)) == 2
    assert select_cases(cases, LOCAL, 4) == cases  # the local profile always runs everything
    assert {"injection"} <= {tag for case in capped for tag in case.tags}


def test_cache_key_depends_on_prompt_model_effort_and_prompt_version(monkeypatch) -> None:
    base = cache_key(GROQ, **REQUEST)

    assert cache_key(GROQ, **REQUEST) == base
    assert cache_key(GROQ, **{**REQUEST, "user": "other"}) != base
    assert cache_key(GROQ, **{**REQUEST, "model": "oss-120b"}) != base
    assert cache_key(GROQ.model_copy(update={"reasoning_effort": "low"}), **REQUEST) != base
    monkeypatch.setattr(eval_run, "PROMPT_VERSION", "next")
    assert cache_key(GROQ, **REQUEST) != base


def test_cached_answers_are_not_requested_again(tmp_path) -> None:
    client = CountingClient()
    first = CachedLlm(client, GROQ, tmp_path)
    again = CachedLlm(client, GROQ, tmp_path)

    original = first.extract(**REQUEST)
    replayed = again.extract(**REQUEST)
    other = again.extract(**{**REQUEST, "user": "another document"})

    assert client.calls == 2 and (first.hits, again.hits) == (0, 1)
    assert replayed == original and other.model == "oss-20b"
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert "gsk_test" not in "".join(p.read_text() for p in tmp_path.glob("*.json"))


def test_estimate_counts_only_documents_that_are_not_cached(tmp_path) -> None:
    cases = select_cases(load_cases(), GROQ, 2)

    tokens, cached = estimate_run(cases, GROQ_SETTINGS, GROQ, FakeOcr(), tmp_path)
    assert cached == 0 and 2 * 1500 < tokens < 2 * 6000  # prompt + ~1500 answer tokens each

    client = CachedLlm(CountingClient(), GROQ, tmp_path)
    run_profile("groq", GROQ, cases[:1], GROQ_SETTINGS, GROQ_SETTINGS, FakeOcr(), client)
    remaining, cached = estimate_run(cases, GROQ_SETTINGS, GROQ, FakeOcr(), tmp_path)

    assert cached == 1 and 0 < remaining < tokens


def test_run_profile_counts_documents_and_cache_hits(tmp_path) -> None:
    cases = select_cases(load_cases(), GROQ, 2)
    backend = CountingClient()

    first = run_profile(
        "groq",
        GROQ,
        cases,
        GROQ_SETTINGS,
        GROQ_SETTINGS,
        FakeOcr(),
        CachedLlm(backend, GROQ, tmp_path),
    )
    second = run_profile(
        "groq",
        GROQ,
        cases,
        GROQ_SETTINGS,
        GROQ_SETTINGS,
        FakeOcr(),
        CachedLlm(backend, GROQ, tmp_path),
    )

    assert (first.documents, first.cache_hits, backend.calls) == (2, 0, 2)
    assert (second.documents, second.cache_hits) == (2, 2)
    assert [row.outcome for row in second.rows] == [row.outcome for row in first.rows]


def test_exhausted_quota_stops_the_run_instead_of_failing_every_document(tmp_path) -> None:
    cases = select_cases(load_cases(), GROQ, 4)
    backend = CountingClient(fail_after=1)

    with pytest.raises(QuotaExhausted) as excinfo:
        run_profile(
            "groq",
            GROQ,
            cases,
            GROQ_SETTINGS,
            GROQ_SETTINGS,
            FakeOcr(),
            CachedLlm(backend, GROQ, tmp_path),
        )

    assert excinfo.value.retry_after == 868.0
    assert backend.calls == 1  # nothing was sent after the refusal
    assert len(list(tmp_path.glob("*.json"))) == 1  # the answer that did arrive is kept


def test_other_llm_failures_still_count_as_a_failed_document(tmp_path) -> None:
    cases = select_cases(load_cases(), GROQ, 2)
    backend = CountingClient(fail_after=1, error=LlmFailed("bad output"))

    run = run_profile("groq", GROQ, cases, GROQ_SETTINGS, GROQ_SETTINGS, FakeOcr(), backend)

    assert run.documents == 2 and run.errors == [f"{cases[1].id}: llm_failed"]


def test_report_shows_document_counts_cache_hits_and_uncovered_categories() -> None:
    cases = load_cases()
    capped = RunResult(name="groq", model="m", reasoning_effort="medium", documents=1, cache_hits=1)
    capped.rows.append(
        FieldRow(cases[0].id, cases[0].category, "f", "correct", 0.9, 0.9, False, False)
    )

    report = build_report(
        [capped], cases, ["przebieg `groq:low` przerwany: wyczerpany limit dostawcy"]
    )

    assert "| groq | `m` | medium | 1 | 1 | 0 | 1 |" in report
    assert f"| {cases[0].category} | 100.0% (1) |" in report
    assert "| other | — |" in report
    assert "Pominięto: przebieg `groq:low` przerwany" in report


def test_evals_build_the_same_optional_parts_as_the_service(tmp_path) -> None:
    """Regression: the runner built pipelines without retrieval."""
    assert eval_run.build_retriever(Settings(_env_file=None)) is not None
    assert eval_run.build_retriever(Settings(_env_file=None, rag_enabled=False)) is None
