import json
from typing import Any

import pytest
from PIL import Image

from schemalens_core.config import Settings
from schemalens_core.fewshot import example_block, load_templates
from schemalens_core.llm import LlmResponse
from schemalens_core.models import OcrLine, OcrPage, OcrResult
from schemalens_core.pipeline import Pipeline
from schemalens_core.presets import load_presets
from schemalens_core.schemas import compile_schema

SETTINGS = Settings(_env_file=None, local_model_fast="bielik")
PRESETS = load_presets(SETTINGS)
TEMPLATES = load_templates(SETTINGS.templates_dir)


def test_every_preset_has_a_template() -> None:
    assert set(TEMPLATES) == set(PRESETS)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_template_answer_is_a_valid_normalized_result(name: str) -> None:
    template = TEMPLATES[name]
    schema = PRESETS[name].json_schema
    assert set(template.data) == set(schema["properties"])

    response = {
        "fields": {
            k: {"value": v, "evidence": [], "confidence": 1} for k, v in template.data.items()
        }
    }
    parsed = compile_schema(schema, SETTINGS).parse(response)

    assert parsed.data == template.data and parsed.warnings == []


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_template_is_not_the_document_the_evals_score(name: str) -> None:
    """The preset's own example is an eval case: showing it in the prompt would leak the answer.
    Only section headings (a line in capitals, like EXPERIENCE) may repeat."""
    shared = set(TEMPLATES[name].lines) & set(PRESETS[name].example.lines)
    assert all(line.isupper() for line in shared), shared


def test_example_block_has_numbered_lines_and_evidence() -> None:
    block = example_block(TEMPLATES["receipt"], PRESETS["receipt"].json_schema)

    lines = block.split("<example_answer>")[0]
    answer = json.loads(block.split("<example_answer>\n")[1].split("\n</example_answer>")[0])
    fields = answer["fields"]

    assert "[L001] EXAMPLE PHARMACY" in lines
    assert fields["seller"] == {
        "value": "EXAMPLE PHARMACY",
        "evidence": ["L001"],
        "confidence": 0.95,
    }
    assert fields["total"]["evidence"] == ["L008"]
    assert fields["tax_id"]["evidence"] == ["L003"]
    assert fields["date"]["evidence"] == ["L005"]  # date found in another notation
    assert fields["payment_method"]["evidence"] == []  # derived: no evidence claimed


def test_example_block_follows_the_requested_schema() -> None:
    schema = {**PRESETS["receipt"].json_schema}
    schema["properties"] = {**schema["properties"], "kasjer": {"type": "string"}}

    fields = json.loads(
        example_block(TEMPLATES["receipt"], schema)
        .split("<example_answer>\n")[1]
        .split("\n</example_answer>")[0]
    )["fields"]

    assert fields["kasjer"] == {"value": None, "evidence": [], "confidence": 0.0}


class Ocr:
    def read(self, images: list[Image.Image]) -> OcrResult:
        lines = [
            OcrLine(id=f"L{i:03d}", page=1, text=f"linia {i} paragonu", bbox=None, conf=0.9)
            for i in range(1, 6)
        ]
        return OcrResult(lines=lines, pages=[OcrPage(number=1, source="ocr", ocr_ms=1)])


class Llm:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def extract(self, *, system: str, user: str, json_schema: Any, model: str) -> LlmResponse:
        self.prompts.append(user)
        return LlmResponse(
            content='{"fields": {}}', prompt_tokens=1, completion_tokens=1, llm_ms=1, model=model
        )


def _png() -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("L", (20, 20), 255).save(buffer, format="PNG")
    return buffer.getvalue()


def test_pipeline_adds_the_example_only_when_enabled_and_for_presets() -> None:
    on = Settings(_env_file=None, local_model_fast="bielik", fewshot_enabled=True)

    off_llm, on_llm, custom_llm = Llm(), Llm(), Llm()
    off = Pipeline(SETTINGS, Ocr(), off_llm).extract(_png(), None, None, preset="receipt")
    enabled = Pipeline(on, Ocr(), on_llm).extract(_png(), None, None, preset="receipt")
    custom = Pipeline(on, Ocr(), custom_llm).extract(_png(), None, PRESETS["receipt"].json_schema)

    assert "<example_document>" not in off_llm.prompts[0] and off.usage.fewshot is None
    assert "<example_document>" in on_llm.prompts[0] and enabled.usage.fewshot == "receipt"
    assert on_llm.prompts[0].index("</example_answer>") < on_llm.prompts[0].index("<document>")
    assert "never copy its values" in on_llm.prompts[0]
    assert "<example_document>" not in custom_llm.prompts[0] and custom.usage.fewshot is None


def test_example_tokens_come_out_of_the_document_budget() -> None:
    on = Settings(_env_file=None, local_model_fast="bielik", fewshot_enabled=True)
    pipeline = Pipeline(on, Ocr(), Llm())
    schema = PRESETS["invoice"].json_schema
    example, _ = pipeline._example("invoice", schema)

    assert pipeline._document_budget(schema, example) < pipeline._document_budget(schema) - 200
