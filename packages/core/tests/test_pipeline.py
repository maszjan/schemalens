import io
import json
import re
from typing import Any

import pypdfium2 as pdfium
import pytest
from PIL import Image

from schemalens_core import cli
from schemalens_core.config import LlmProfile, Settings
from schemalens_core.context import estimate_tokens, fit_to_context
from schemalens_core.errors import DocumentUnreadable, InvalidSchema, LlmFailed, UnsupportedDocument
from schemalens_core.llm import LlmResponse, OpenAICompatibleClient
from schemalens_core.models import ExtractionOptions, FieldMeta, OcrLine, OcrPage, OcrResult
from schemalens_core.ocr import EasyOcrEngine
from schemalens_core.pipeline import Pipeline
from schemalens_core.prompts import PROMPT_VERSION
from schemalens_core.scoring import score_fields

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sklep": {"type": "string"},
        "nip": {"type": "string", "x-schemalens-type": "nip"},
        "data": {"type": "string", "format": "date"},
        "suma": {"type": "number"},
        "waluta": {"type": "string", "enum": ["PLN", "EUR"], "x-schemalens-type": "currency"},
    },
    "required": ["sklep", "suma"],
}


def _settings(**overrides: Any) -> Settings:
    values = {
        "active_llm_profile": "local",
        "local_model_fast": "fast-model",
        "local_model_accurate": "accurate-model",
    }
    return Settings(_env_file=None, **{**values, **overrides})


def _field(value: Any, evidence: list[str] | None = None, confidence: float = 0.9) -> dict:
    return {"value": value, "evidence": evidence or [], "confidence": confidence}


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (40, 40), 255).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeOcr:
    def __init__(self, *lines: tuple[str, float]) -> None:
        self._lines = lines

    def read(self, images: list[Image.Image]) -> OcrResult:
        return OcrResult(
            lines=[
                OcrLine(id=f"L{i:03d}", page=1, text=text, bbox=(0, 0, 1, 1), conf=conf)
                for i, (text, conf) in enumerate(self._lines, start=1)
            ],
            pages=[OcrPage(number=1, source="ocr", ocr_ms=42)],
        )


class FakeLlm:
    """Answers with a canned dict, or with a function of the user prompt. No network."""

    def __init__(self, *answers: Any) -> None:
        self._answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    def extract(self, *, system: str, user: str, json_schema: dict, model: str) -> LlmResponse:
        self.calls.append({"user": user, "model": model})
        answer = self._answers.pop(0)
        content = answer(user) if callable(answer) else answer
        return LlmResponse(
            content=json.dumps(content),
            prompt_tokens=100,
            completion_tokens=20,
            llm_ms=5,
            model=model,
        )


RECEIPT_OCR = FakeOcr(("SKLEP EXAMPLE", 0.9), ("NIP 123-456-32-18", 0.8), ("SUMA PLN 1O,48", 0.6))


# --- scoring ---------------------------------------------------------------------------


SCORE_LINES = [
    OcrLine(id="L001", page=1, text="Razem 10,48 dnia 30.06.2027", bbox=None, conf=0.9),
    OcrLine(id="L002", page=1, text="b", bbox=None, conf=0.5),
    *[OcrLine(id=f"L{i:03d}", page=1, text="x", bbox=None, conf=0.8) for i in range(3, 11)],
]


def _score(
    data: dict, reported: dict, nodes: dict | None = None, **flags: Any
) -> tuple[dict[str, FieldMeta], list[str]]:
    """`reported`: field -> (model confidence, evidence). Fields are strings unless in `nodes`."""
    schema = {
        "type": "object",
        "properties": {name: {"type": "string"} for name in data} | (nodes or {}),
    }
    metas = {name: FieldMeta(confidence=c, evidence=e) for name, (c, e) in reported.items()}
    return score_fields(schema, data, metas, SCORE_LINES, _settings(), **flags)


def test_confidence_blends_ocr_and_model_confidence() -> None:
    fields, warnings = _score(
        {"a": "x", "b": "y"}, {"a": (1.0, ["L001"]), "b": (0.8, ["L001", "L002"])}
    )

    assert fields["a"].confidence == 0.93  # 0.7 * 0.9 + 0.3 * 1.0
    assert fields["b"].confidence == 0.73  # 0.7 * mean(0.9, 0.5) + 0.3 * 0.8
    assert warnings == []


def test_model_cannot_talk_a_weak_ocr_line_into_high_confidence() -> None:
    fields, _ = _score({"a": "x"}, {"a": (1.0, ["L002"])})
    assert fields["a"].confidence == 0.65


def test_unknown_evidence_ids_are_removed_with_warning() -> None:
    fields, warnings = _score({"a": "x"}, {"a": (1.0, ["L001", "L999"])})

    assert fields["a"].evidence == ["L001"]
    assert fields["a"].confidence == 0.93
    assert warnings == ["a: evidence pointed to lines that do not exist, removed"]


def test_value_without_evidence_is_penalized_and_flagged() -> None:
    fields, warnings = _score({"a": "x", "b": "y"}, {"a": (1.0, []), "b": (1.0, ["L777"])})

    assert fields["a"].confidence == 0.5
    assert fields["b"].model_dump() == {"confidence": 0.5, "evidence": []}
    assert warnings == [
        "a: value has no evidence in the document, treat it as unverified",
        "b: evidence pointed to lines that do not exist, removed",
        "b: value has no evidence in the document, treat it as unverified",
    ]


def test_normalization_and_retry_penalties_multiply() -> None:
    reported = {"a": (1.0, ["L001"])}
    assert _score({"a": "x"}, reported, normalized=frozenset({"a"}))[0]["a"].confidence == 0.837
    assert _score({"a": "x"}, reported, retried=frozenset({"a"}))[0]["a"].confidence == 0.79
    both = _score({"a": "x"}, reported, normalized=frozenset({"a"}), retried=frozenset({"a"}))
    assert both[0]["a"].confidence == 0.711


def test_failed_checksum_halves_confidence() -> None:
    reported = {"a": (1.0, ["L001"])}
    assert _score({"a": "x"}, reported, invalid=frozenset({"a"}))[0]["a"].confidence == 0.465


def test_scalar_evidence_over_three_lines_is_too_broad() -> None:
    fields, warnings = _score(
        {"a": "x", "b": "y"},
        {"a": (1.0, ["L003", "L004", "L005"]), "b": (1.0, ["L003", "L004", "L005", "L006"])},
    )

    assert fields["a"].model_dump() == {"confidence": 0.86, "evidence": ["L003", "L004", "L005"]}
    assert fields["b"].model_dump() == {"confidence": 0.5, "evidence": []}  # no-evidence penalty
    assert warnings == ["b: evidence too broad"]


def test_scalar_evidence_over_half_of_a_short_document_is_too_broad() -> None:
    lines = [OcrLine(id=f"L00{i}", page=1, text="x", bbox=None, conf=0.8) for i in (1, 2, 3)]
    schema = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
    reported = {
        "a": FieldMeta(confidence=1.0, evidence=["L001"]),
        "b": FieldMeta(confidence=1.0, evidence=["L001", "L002"]),
    }

    fields, warnings = score_fields(schema, {"a": "x", "b": "y"}, reported, lines, _settings())

    assert fields["a"].evidence == ["L001"]
    assert fields["b"].evidence == []
    assert warnings == ["b: evidence too broad"]


def test_single_line_evidence_is_fine_even_in_a_one_line_document() -> None:
    lines = [OcrLine(id="L001", page=1, text="x", bbox=None, conf=0.8)]
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    reported = {"a": FieldMeta(confidence=1.0, evidence=["L001"])}

    fields, warnings = score_fields(schema, {"a": "x"}, reported, lines, _settings())

    assert fields["a"].evidence == ["L001"] and warnings == []


def test_list_evidence_limit_scales_with_the_number_of_elements() -> None:
    nodes = {"a": {"type": "array", "items": {"type": "string"}}}
    six = [f"L{i:03d}" for i in range(3, 9)]
    seven = [f"L{i:03d}" for i in range(3, 10)]

    fine, no_warnings = _score({"a": ["x", "y"]}, {"a": (1.0, six)}, nodes)
    broad, warnings = _score({"a": ["x", "y"]}, {"a": (1.0, seven)}, nodes)

    assert fine["a"].evidence == six and no_warnings == []
    assert broad["a"].evidence == [] and warnings == ["a: evidence too broad"]


def test_derived_field_citing_the_whole_document_falls_back_to_the_ceiling() -> None:
    nodes = {"a": {"type": "string", "x-schemalens-derived": True}}
    everything = [line.id for line in SCORE_LINES]

    fields, warnings = _score({"a": "notatka"}, {"a": (1.0, everything)}, nodes)

    assert fields["a"].model_dump() == {"confidence": 0.7, "evidence": []}
    assert warnings == ["a: evidence too broad"]


def test_missing_value_scores_zero_without_warning() -> None:
    fields, warnings = _score({"a": None}, {"a": (0.9, [])})
    assert fields["a"].model_dump() == {"confidence": 0.0, "evidence": []}
    assert warnings == []


# --- pipeline with fakes ---------------------------------------------------------------


def test_pipeline_end_to_end_with_fakes() -> None:
    llm = FakeLlm(
        {
            "fields": {
                "sklep": _field("SKLEP EXAMPLE", ["L001"], 1.0),
                "nip": _field("123-456-32-18", ["L002"], 1.0),
                "suma": _field("1O,48", ["L003"], 1.0),
                "waluta": _field("zł", ["L003", "L042"], 1.0),
                "data": _field(None),
            }
        }
    )
    pipeline = Pipeline(_settings(), RECEIPT_OCR, llm)

    result = pipeline.extract(_png(), "image/png", SCHEMA)

    assert result.data == {
        "sklep": "SKLEP EXAMPLE",
        "nip": "1234563218",
        "data": None,
        "suma": 10.48,
        "waluta": "PLN",
    }
    assert result.warnings == [
        "nip: normalized '123-456-32-18' to '1234563218'",
        "suma: normalized '1O,48' to 10.48 (OCR characters corrected)",
        "waluta: normalized 'zł' to 'PLN'",
        "waluta: evidence pointed to lines that do not exist, removed",
    ]
    assert {name: meta.model_dump() for name, meta in result.fields.items()} == {
        "sklep": {"confidence": 0.93, "evidence": ["L001"]},
        "nip": {"confidence": 0.774, "evidence": ["L002"]},  # (0.7*0.8 + 0.3) * 0.9
        "data": {"confidence": 0.0, "evidence": []},
        "suma": {"confidence": 0.648, "evidence": ["L003"]},  # (0.7*0.6 + 0.3) * 0.9
        "waluta": {"confidence": 0.648, "evidence": ["L003"]},
    }
    assert result.usage.model_dump() == {
        "pages": 1,
        "ocr_ms": 42,
        "llm_ms": 5,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "llm_calls": 1,
        "model": "fast-model",
        "prompt_version": PROMPT_VERSION,
        "reasoning_effort": None,
        "effort_escalations": 0,
        "context": "full",
        "fewshot": None,
        "ocr_confidence": 0.766,  # weighted by text length
        "language": None,  # too little text to tell
    }
    assert result.ocr_text is None
    assert "[L003] SUMA PLN 1O,48" in llm.calls[0]["user"]


def test_domain_types_apply_only_when_declared_in_the_schema() -> None:
    plain = {
        "type": "object",
        "properties": {"nip": {"type": "string"}, "waluta": {"type": "string"}},
    }
    answer = {
        "fields": {"nip": _field("123-456-32-18", ["L002"]), "waluta": _field("zł", ["L003"])}
    }

    result = Pipeline(_settings(), RECEIPT_OCR, FakeLlm(answer)).extract(_png(), None, plain)

    assert result.data == {"nip": "123-456-32-18", "waluta": "zł"}
    assert result.warnings == []


def test_invalid_checksum_keeps_the_value_and_lowers_confidence() -> None:
    answer = {"fields": {"nip": _field("123-456-78-90", ["L001"], 1.0)}}
    ocr = FakeOcr(("NIP 123-456-78-90", 0.8), ("Sprzedawca", 0.8))

    result = Pipeline(_settings(), ocr, FakeLlm(answer)).extract(_png(), None, SCHEMA)

    assert result.data["nip"] == "123-456-78-90"
    assert "nip: NIP checksum is invalid" in result.warnings
    assert result.fields["nip"].confidence == 0.43  # (0.7 * 0.8 + 0.3) * 0.5


def test_missing_required_fields_are_warnings_not_errors() -> None:
    llm = FakeLlm({"fields": {"sklep": _field("SKLEP EXAMPLE", ["L001"])}})
    result = Pipeline(_settings(), RECEIPT_OCR, llm).extract(_png(), None, SCHEMA)

    assert result.data["suma"] is None
    assert result.warnings == ["suma: required field was not found in the document"]


def test_options_select_model_tier_and_ocr_text() -> None:
    llm = FakeLlm({"fields": {}})
    options = ExtractionOptions(model_tier="accurate", include_ocr_text=True)

    result = Pipeline(_settings(), RECEIPT_OCR, llm).extract(_png(), None, SCHEMA, options)

    assert llm.calls[0]["model"] == "accurate-model"
    assert result.usage.model == "accurate-model"
    assert result.ocr_text == [
        "[L001] SKLEP EXAMPLE",
        "[L002] NIP 123-456-32-18",
        "[L003] SUMA PLN 1O,48",
    ]


def test_unconfigured_tier_is_a_clear_error() -> None:
    pipeline = Pipeline(_settings(local_model_accurate=""), RECEIPT_OCR, FakeLlm())
    with pytest.raises(LlmFailed, match="tier 'accurate'"):
        pipeline.extract(_png(), None, SCHEMA, ExtractionOptions(model_tier="accurate"))


# --- OCR quality gate ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lines", "reason"),
    [
        ((), "0 lines, mean OCR confidence 0.00"),  # blank page
        ((("x9", 0.72),), "1 lines, mean OCR confidence 0.72"),  # one stray fragment
        (
            (("qxj", 0.4), ("zvk", 0.3), ("wqz", 0.36)),
            "3 lines, mean OCR confidence 0.35",
        ),  # scribbles
        ((("a long blurred line of text", 0.28), ("zz", 0.9)), "2 lines, mean OCR confidence 0.32"),
    ],
)
def test_unreadable_documents_stop_before_the_llm(lines: tuple, reason: str) -> None:
    llm = FakeLlm()

    with pytest.raises(DocumentUnreadable) as excinfo:
        Pipeline(_settings(), FakeOcr(*lines), llm).extract(_png(), None, SCHEMA)

    assert llm.calls == []
    message = str(excinfo.value)
    assert reason in message
    assert "handwritten, illegible or photographed at too low a quality" in message
    assert excinfo.value.code == "document_unreadable"
    assert not any(text in message for text, _ in lines)  # numbers only, no OCR text


def test_readable_documents_pass_the_gate_at_the_thresholds() -> None:
    llm = FakeLlm({"fields": {}})
    ocr = FakeOcr(("pierwsza linia", 0.5), ("druga linia!!", 0.5))

    Pipeline(_settings(), ocr, llm).extract(_png(), None, SCHEMA)

    assert len(llm.calls) == 1


def test_gate_thresholds_come_from_settings() -> None:
    one_line = FakeOcr(("jedna linia", 0.45))
    strict = Pipeline(_settings(), one_line, FakeLlm())
    relaxed = Pipeline(
        _settings(ocr_min_lines=1, ocr_min_mean_conf=0.4), one_line, FakeLlm({"fields": {}})
    )

    with pytest.raises(DocumentUnreadable):
        strict.extract(_png(), None, SCHEMA)
    relaxed.extract(_png(), None, SCHEMA)
    assert Settings(_env_file=None).ocr_min_lines == 2
    assert Settings(_env_file=None).ocr_min_mean_conf == 0.5


def test_declared_mime_mismatch_is_a_warning() -> None:
    llm = FakeLlm({"fields": {"sklep": _field("S", ["L001"]), "suma": _field(10.48, ["L003"])}})
    result = Pipeline(_settings(), RECEIPT_OCR, llm).extract(_png(), "application/pdf", SCHEMA)
    assert result.warnings == ["declared content type differs from the detected one (image/png)"]


def test_invalid_schema_fails_before_ocr_and_llm() -> None:
    class ExplodingOcr:
        def read(self, images: list[Image.Image]) -> OcrResult:
            raise AssertionError("OCR must not run for an invalid schema")

    with pytest.raises(InvalidSchema):
        Pipeline(_settings(), ExplodingOcr(), FakeLlm()).extract(_png(), None, {"type": "string"})


def test_unsupported_file_is_rejected() -> None:
    with pytest.raises(UnsupportedDocument):
        Pipeline(_settings(), RECEIPT_OCR, FakeLlm()).extract(b"GIF89a", None, SCHEMA)


# --- CLI -------------------------------------------------------------------------------


built_with: list[LlmProfile] = []


def _run_cli(
    monkeypatch, tmp_path, capsys, llm: FakeLlm, *extra: str, schema: Any = SCHEMA, **settings
):
    """Run the CLI with fakes; `schema=None` leaves --schema out."""
    built_with.clear()
    pipeline = Pipeline(_settings(), RECEIPT_OCR, llm)
    monkeypatch.setattr(cli, "get_settings", lambda: _settings(**settings))
    monkeypatch.setattr(
        cli, "build_pipeline", lambda _, profile: built_with.append(profile) or pipeline
    )
    (tmp_path / "doc.png").write_bytes(_png())
    argv = [str(tmp_path / "doc.png"), *extra]
    if schema is not None:
        (tmp_path / "schema.json").write_text(json.dumps(schema))
        argv += ["--schema", str(tmp_path / "schema.json")]
    code = cli.main(argv)
    return code, capsys.readouterr()


def test_cli_prints_result_json_with_null_fields(monkeypatch, tmp_path, capsys) -> None:
    llm = FakeLlm(
        {"fields": {"sklep": _field("SKLEP EXAMPLE", ["L001"]), "suma": _field(10.48, ["L003"])}}
    )

    code, out = _run_cli(monkeypatch, tmp_path, capsys, llm)

    assert code == 0
    output = json.loads(out.out)
    assert output["data"] == {
        "sklep": "SKLEP EXAMPLE",
        "nip": None,
        "data": None,
        "suma": 10.48,
        "waluta": None,
    }
    assert set(output) == {"data", "fields", "warnings", "usage"}
    assert llm.calls[0]["model"] == "fast-model"


def test_cli_tier_and_ocr_text_flags(monkeypatch, tmp_path, capsys) -> None:
    llm = FakeLlm({"fields": {}})

    code, out = _run_cli(monkeypatch, tmp_path, capsys, llm, "--tier", "accurate", "--ocr-text")

    assert code == 0
    assert json.loads(out.out)["ocr_text"][0] == "[L001] SKLEP EXAMPLE"
    assert llm.calls[0]["model"] == "accurate-model"


def test_cli_reports_invalid_schema_on_stderr(monkeypatch, tmp_path, capsys) -> None:
    code, out = _run_cli(
        monkeypatch,
        tmp_path,
        capsys,
        FakeLlm(),
        schema={"type": "object", "properties": {"a": {"type": "money"}}},
    )

    assert code == 2
    assert out.out == ""
    assert "invalid schema" in out.err
    assert "a: unsupported or missing type 'money'" in out.err


def test_cli_reports_missing_file(tmp_path, capsys) -> None:
    code = cli.main([str(tmp_path / "nope.png"), "--schema", str(tmp_path / "nope.json")])
    assert code == 2
    assert "cannot read input" in capsys.readouterr().err


# --- e2e on a synthetic scan: real OCR, fake LLM --------------------------------------


class SpyOcr:
    def __init__(self, engine: EasyOcrEngine) -> None:
        self._engine = engine
        self.result: OcrResult | None = None

    def read(self, images: list[Image.Image]) -> OcrResult:
        self.result = self._engine.read(images)
        return self.result


def _answer_from_prompt(user: str) -> dict:
    """Plays the LLM: quotes values the way they appear on the receipt, with real line ids."""

    def ids(pattern: str) -> list[str]:
        return re.findall(rf"\[(L\d+)\] [^\n]*{pattern}", user, flags=re.IGNORECASE)[:1]

    return {
        "fields": {
            "sklep": _field(re.search(r"\] (SKLEP[^\n]*)", user).group(1), ids("SKLEP"), 1.0),
            "nip": _field("123-456-32-18", ids("NIP"), 1.0),
            "data": _field("15 października 2026 r.", ids("2026"), 1.0),
            "suma": _field("10,48", ids("SUMA"), 1.0),
            "waluta": _field("PLN", ids("SUMA"), 1.0),
        }
    }


@pytest.fixture
def real_ocr() -> SpyOcr:
    settings = Settings(_env_file=None)
    if not any(settings.ocr_model_dir.glob("*.pth")):
        pytest.skip(f"EasyOCR models not found in {settings.ocr_model_dir}")
    return SpyOcr(EasyOcrEngine(settings))


@pytest.mark.ocr
def test_e2e_synthetic_receipt_scan_with_fake_llm(receipt_png: bytes, real_ocr: SpyOcr) -> None:
    pipeline = Pipeline(_settings(), real_ocr, FakeLlm(_answer_from_prompt))

    result = pipeline.extract(
        receipt_png, "image/png", SCHEMA, ExtractionOptions(include_ocr_text=True)
    )

    assert result.data == {
        "sklep": "SKLEP SPOŻYWCZY EXAMPLE",  # Polish diacritics survive real OCR
        "nip": "1234563218",
        "data": "2026-10-15",
        "suma": 10.48,
        "waluta": "PLN",
    }
    assert result.warnings == [
        "nip: normalized '123-456-32-18' to '1234563218'",
        "data: normalized '15 października 2026 r.' to '2026-10-15'",
        "suma: normalized '10,48' to 10.48",
    ]
    assert real_ocr.result is not None
    ocr_conf = {line.id: line.conf for line in real_ocr.result.lines}
    assert result.ocr_text is not None and len(result.ocr_text) == len(ocr_conf) >= 8
    for name, meta in result.fields.items():
        assert len(meta.evidence) == 1, name  # the value was found in the OCR text
        expected = 0.7 * ocr_conf[meta.evidence[0]] + 0.3
        if name in ("nip", "data", "suma"):
            expected *= 0.9
        assert meta.confidence == pytest.approx(expected, abs=0.001)
        assert 0 < meta.confidence < 1
    assert result.usage.pages == 1
    assert result.usage.ocr_ms > 0
    assert result.usage.llm_calls == 1


# --- live: same scan, real OCR + Ollama/Bielik from .env (pytest -m live -s) ----------


@pytest.mark.live
def test_live_synthetic_receipt_scan(receipt_png: bytes) -> None:
    settings = Settings()
    profile = settings.llm_profile("local")
    llm = OpenAICompatibleClient(settings, profile)
    pipeline = Pipeline(settings, EasyOcrEngine(settings), llm, profile)

    result = pipeline.extract(receipt_png, "image/png", SCHEMA)

    # Synthetic receipt, so printing the result is fine here.
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    assert result.data["suma"] == 10.48
    assert result.data["waluta"] == "PLN"
    assert result.data["nip"] == "1234563218"
    assert result.data["data"] == "2026-10-15"
    assert result.data["sklep"] == "SKLEP SPOŻYWCZY EXAMPLE"
    assert 0 < result.fields["suma"].confidence < 1


# --- presets and profiles through the pipeline and CLI --------------------------------


def test_pipeline_extracts_with_a_preset() -> None:
    llm = FakeLlm({"fields": {"full_name": _field("Anna Przykładowa", ["L001"])}})

    result = Pipeline(_settings(), RECEIPT_OCR, llm).extract(
        _png(), None, None, preset="business_card"
    )

    assert result.data["full_name"] == "Anna Przykładowa"
    assert "phone" in result.data and "postal_code" in result.data
    assert "- phone (string): Phone number" in llm.calls[0]["user"]


def test_pipeline_preset_extended_with_client_fields() -> None:
    extra = {
        "type": "object",
        "properties": {"branza": {"type": "string", "x-schemalens-derived": True}},
        "required": ["branza"],
    }
    llm = FakeLlm({"fields": {"full_name": _field("Anna Przykładowa", ["L001"])}})

    result = Pipeline(_settings(), RECEIPT_OCR, llm).extract(
        _png(), None, extra, preset="business_card"
    )

    assert list(result.data)[-1] == "branza"
    assert "- branza (string, derived)" in llm.calls[0]["user"]
    assert result.warnings == ["branza: required field was not found in the document"]


def test_pipeline_rejects_unknown_preset_and_missing_schema() -> None:
    pipeline = Pipeline(_settings(), RECEIPT_OCR, FakeLlm())
    with pytest.raises(InvalidSchema) as unknown:
        pipeline.extract(_png(), None, None, preset="passport")
    assert unknown.value.errors[0].startswith(
        "preset 'passport' does not exist (available: business_card"
    )
    with pytest.raises(InvalidSchema) as missing:
        pipeline.extract(_png(), None, None)
    assert missing.value.errors == ["either 'schema' or 'preset' is required"]


def test_derived_field_without_evidence_is_capped_not_penalized() -> None:
    schema = {
        "type": "object",
        "properties": {
            "typ": {"type": "string", "x-schemalens-derived": True},
            "tytul": {"type": "string"},
        },
    }
    llm = FakeLlm(
        {"fields": {"typ": _field("paragon", [], 1.0), "tytul": _field("SKLEP", [], 1.0)}}
    )

    result = Pipeline(_settings(), RECEIPT_OCR, llm).extract(_png(), None, schema)

    assert result.fields["typ"].confidence == 0.7  # ceiling, no warning
    assert result.fields["tytul"].confidence == 0.5  # extracted field: penalty and warning
    assert result.warnings == [
        "tytul: value has no evidence in the document, treat it as unverified"
    ]


def test_derived_field_with_evidence_uses_the_regular_formula() -> None:
    schema = {
        "type": "object",
        "properties": {"typ": {"type": "string", "x-schemalens-derived": True}},
    }
    llm = FakeLlm({"fields": {"typ": _field("paragon", ["L001"], 1.0)}})
    result = Pipeline(_settings(), RECEIPT_OCR, llm).extract(_png(), None, schema)
    assert result.fields["typ"].confidence == 0.93


def test_cli_preset_without_schema(monkeypatch, tmp_path, capsys) -> None:
    llm = FakeLlm({"fields": {"full_name": _field("Anna Przykładowa", ["L001"])}})

    code, out = _run_cli(
        monkeypatch, tmp_path, capsys, llm, "--preset", "business_card", schema=None
    )

    assert code == 0
    assert json.loads(out.out)["data"]["full_name"] == "Anna Przykładowa"


def test_cli_requires_schema_or_preset(monkeypatch, tmp_path, capsys) -> None:
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, tmp_path, capsys, FakeLlm(), schema=None)
    assert "one of --schema or --preset is required" in capsys.readouterr().err


def test_cli_groq_profile_builds_the_pipeline_with_groq_settings(
    monkeypatch, tmp_path, capsys
) -> None:
    llm = FakeLlm({"fields": {}})

    code, out = _run_cli(
        monkeypatch, tmp_path, capsys, llm, "--profile", "groq",
        groq_api_key="gsk_secret_value", groq_model_fast="groq-fast",
    )  # fmt: skip

    assert code == 0
    assert built_with[0].name == "groq"
    assert built_with[0].model_fast == "groq-fast"
    assert "gsk_secret_value" not in out.out + out.err


def test_cli_groq_profile_without_key_explains_how_to_enable_it(
    monkeypatch, tmp_path, capsys
) -> None:
    code, out = _run_cli(monkeypatch, tmp_path, capsys, FakeLlm(), "--profile", "groq")

    assert code == 1
    assert out.out == ""
    assert "LLM profile 'groq' is not configured: set GROQ_API_KEY" in out.err


def test_cli_uses_the_active_profile_by_default_and_local_on_request(
    monkeypatch, tmp_path, capsys
) -> None:
    _run_cli(monkeypatch, tmp_path, capsys, FakeLlm({"fields": {}}))
    assert built_with[0].name == "local"  # ACTIVE_LLM_PROFILE in the test settings

    _run_cli(
        monkeypatch, tmp_path, capsys, FakeLlm({"fields": {}}), "--profile", "local",
        active_llm_profile="groq", groq_api_key="gsk_test",
    )  # fmt: skip
    assert built_with[0].name == "local"


def test_pipeline_takes_models_and_strictness_from_its_profile() -> None:
    settings = _settings(groq_api_key="gsk_test", groq_model_fast="groq-fast")
    llm = FakeLlm({"fields": {}})

    Pipeline(settings, RECEIPT_OCR, llm, settings.llm_profile("groq")).extract(_png(), None, SCHEMA)

    assert llm.calls[0]["model"] == "groq-fast"


# --- private documents stay on the local profile ---------------------------------------


def _run_cli_on(monkeypatch, capsys, path, *extra: str, **settings):
    llm = FakeLlm({"fields": {}})
    built_with.clear()
    pipeline = Pipeline(_settings(), RECEIPT_OCR, llm)
    monkeypatch.setattr(cli, "get_settings", lambda: _settings(**settings))
    monkeypatch.setattr(
        cli, "build_pipeline", lambda _, profile: built_with.append(profile) or pipeline
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png())
    code = cli.main([str(path), "--preset", "receipt", *extra])
    return code, capsys.readouterr(), llm


def test_cli_refuses_private_documents_on_the_groq_profile(monkeypatch, tmp_path, capsys) -> None:
    private = tmp_path / "samples" / "private" / "umowa.png"

    code, out, llm = _run_cli_on(
        monkeypatch, capsys, private, "--profile", "groq", groq_api_key="gsk_test"
    )

    assert code == 2
    assert out.out == ""
    assert "private sample" in out.err and "Use --profile local" in out.err
    assert llm.calls == [] and built_with == []


def test_cli_refuses_private_documents_when_groq_is_merely_the_active_profile(
    monkeypatch, tmp_path, capsys
) -> None:
    private = tmp_path / "samples" / "private" / "nested" / "umowa.png"

    code, out, llm = _run_cli_on(
        monkeypatch, capsys, private, active_llm_profile="groq", groq_api_key="gsk_test"
    )

    assert code == 2 and llm.calls == []


def test_cli_processes_private_documents_with_the_local_profile(
    monkeypatch, tmp_path, capsys
) -> None:
    private = tmp_path / "samples" / "private" / "umowa.png"

    code, out, llm = _run_cli_on(
        monkeypatch, capsys, private, "--profile", "local",
        active_llm_profile="groq", groq_api_key="gsk_test",
    )  # fmt: skip

    assert code == 0 and len(llm.calls) == 1
    assert built_with[0].name == "local"


def test_cli_private_check_cannot_be_dodged_with_a_relative_or_symlinked_path(
    monkeypatch, tmp_path, capsys
) -> None:
    private = tmp_path / "samples" / "private" / "umowa.png"
    _run_cli_on(monkeypatch, capsys, private, "--profile", "local", groq_api_key="gsk_test")
    link = tmp_path / "link.png"
    link.symlink_to(private)
    monkeypatch.chdir(tmp_path / "samples")

    by_relative_path = cli.main(["private/umowa.png", "--preset", "receipt", "--profile", "groq"])
    by_symlink = cli.main([str(link), "--preset", "receipt", "--profile", "groq"])

    assert (by_relative_path, by_symlink) == (2, 2)
    capsys.readouterr()


def test_cli_other_documents_are_not_affected(monkeypatch, tmp_path, capsys) -> None:
    public = tmp_path / "samples" / "public" / "doc.png"
    code, _, llm = _run_cli_on(
        monkeypatch, capsys, public, "--profile", "groq", groq_api_key="gsk_test"
    )
    assert code == 0 and len(llm.calls) == 1


def test_cli_treats_private_extra_like_private(monkeypatch, tmp_path, capsys) -> None:
    extra = tmp_path / "samples" / "private_extra" / "doc.png"
    code, out, llm = _run_cli_on(
        monkeypatch, capsys, extra, "--profile", "groq", groq_api_key="gsk_test"
    )
    assert code == 2 and llm.calls == []


# --- context limit per profile ---------------------------------------------------------


def _multipage(
    pages: int, lines_per_page: int = 4, text: str = "linia tekstu dokumentu"
) -> OcrResult:
    lines = [
        OcrLine(id=f"P{p}-L{i:03d}", page=p, text=f"{text} {p}.{i}", bbox=None, conf=0.9)
        for p in range(1, pages + 1)
        for i in range(1, lines_per_page + 1)
    ]
    return OcrResult(
        lines=lines, pages=[OcrPage(number=p, source="ocr", ocr_ms=1) for p in range(1, pages + 1)]
    )


def test_document_within_the_limit_is_untouched() -> None:
    ocr = _multipage(3)
    lines, warning = fit_to_context(ocr, limit=10_000, chars_per_token=3.5)
    assert lines == ocr.lines and warning is None


def test_long_document_is_cut_at_a_page_boundary_with_a_warning() -> None:
    ocr = _multipage(5)
    one_page = estimate_tokens(
        "\n".join(f"[{line.id}] {line.text}" for line in ocr.lines if line.page == 1), 3.5
    )

    lines, warning = fit_to_context(ocr, limit=one_page * 2 + 5, chars_per_token=3.5)

    assert {line.page for line in lines} == {1, 2}
    assert len(lines) == 8  # whole pages, never half of page 3
    assert warning == "document truncated to fit context (2 of 5 pages)"


def test_single_page_longer_than_the_limit_is_cut_by_lines_and_says_so() -> None:
    ocr = _multipage(2, lines_per_page=30)

    lines, warning = fit_to_context(ocr, limit=60, chars_per_token=3.5)

    assert 0 < len(lines) < 30 and {line.page for line in lines} == {1}
    assert lines == ocr.lines[: len(lines)]
    assert warning == "document truncated to fit context (1 of 2 pages, first page cut short)"


def test_profiles_have_their_own_context_limits() -> None:
    settings = _settings(groq_api_key="gsk_test")
    assert settings.llm_profile("groq").direct_limit == 24_000
    assert settings.llm_profile("local").direct_limit == 6_000
    assert _settings(local_direct_limit=123).llm_profile("local").direct_limit == 123


def _blank_pdf(pages: int) -> bytes:
    document = pdfium.PdfDocument.new()
    for _ in range(pages):
        document.new_page(595, 842)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_pipeline_truncates_for_the_profile_and_reports_it() -> None:
    class LongOcr:
        def read(self, images: list[Image.Image]) -> OcrResult:
            return _multipage(len(images), lines_per_page=20)

    settings = _settings(local_direct_limit=450)
    llm = FakeLlm({"fields": {"sklep": _field("linia", ["P1-L001"]), "suma": _field(None)}})

    result = Pipeline(settings, LongOcr(), llm).extract(
        _blank_pdf(4), None, SCHEMA, ExtractionOptions(include_ocr_text=True)
    )

    prompt = llm.calls[0]["user"]
    assert "[P1-L001]" in prompt and "[P2-L020]" in prompt
    assert "[P3-L001]" not in prompt  # the cut pages never reach the LLM
    assert "document truncated to fit context (2 of 4 pages)" in result.warnings
    assert result.usage.pages == 4
    assert len(result.ocr_text or []) == 80  # the OCR text itself is complete


def test_short_documents_produce_no_truncation_warning() -> None:
    llm = FakeLlm(
        {"fields": {"sklep": _field("SKLEP EXAMPLE", ["L001"]), "suma": _field(10.48, ["L003"])}}
    )
    result = Pipeline(_settings(), RECEIPT_OCR, llm).extract(_png(), None, SCHEMA)
    assert not any("truncated" in warning for warning in result.warnings)


def test_noise_in_a_nip_gives_its_own_warning_and_keeps_grounding() -> None:
    ocr = FakeOcr(("NIP nabywcy: 9876543 b210", 0.8), ("Sprzedawca Example", 0.9))
    llm = FakeLlm({"fields": {"nip": _field("9876543 b210", ["L001"], 1.0)}})

    result = Pipeline(_settings(), ocr, llm).extract(_png(), None, SCHEMA)

    assert result.data["nip"] == "9876543210"
    assert result.warnings == [
        "nip: removed non-digit noise",
        "sklep: required field was not found in the document",
        "suma: required field was not found in the document",
    ]
    assert (
        result.fields["nip"].confidence == 0.774
    )  # (0.7 * 0.8 + 0.3) * 0.9, as for any normalization


def test_noise_that_fails_the_checksum_leaves_the_value_and_halves_confidence() -> None:
    ocr = FakeOcr(("NIP nabywcy: 9876543 b211", 0.8), ("Sprzedawca Example", 0.9))
    llm = FakeLlm({"fields": {"nip": _field("9876543 b211", ["L001"], 1.0)}})

    result = Pipeline(_settings(), ocr, llm).extract(_png(), None, SCHEMA)

    assert result.data["nip"] == "9876543 b211"
    assert "nip: NIP checksum is invalid" in result.warnings
    assert result.fields["nip"].confidence == 0.43  # (0.7 * 0.8 + 0.3) * CONF_CHECKSUM_FACTOR
