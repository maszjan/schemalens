"""Request and response models. Their examples feed the Scalar docs and the SDKs."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from schemalens_core.models import ExtractionOptions, ExtractionResult

_INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "seller_name": {"type": "string", "description": "Name of the seller"},
        "issue_date": {"type": "string", "format": "date"},
        "total_gross": {"type": "number"},
        "currency": {"type": "string", "x-schemalens-type": "currency"},
        "cost_category": {
            "type": "string",
            "enum": ["fuel", "equipment", "services", "other"],
            "x-schemalens-derived": True,
        },
    },
    "required": ["seller_name", "total_gross"],
}
_PRESET_EXTENSION = {
    "type": "object",
    "properties": {
        "industry": {
            "type": "string",
            "description": "Industry of the company",
            "x-schemalens-derived": True,
        }
    },
}
_RESULT_EXAMPLE = {
    "data": {
        "seller_name": "Example Sp. z o.o.",
        "issue_date": "2026-05-15",
        "total_gross": 312.45,
    },
    "fields": {
        "seller_name": {"confidence": 0.93, "evidence": ["L002"]},
        "issue_date": {"confidence": 0.86, "evidence": ["L004"]},
        "total_gross": {"confidence": 0.9, "evidence": ["L011"]},
    },
    "warnings": ["issue_date: normalized '15 maja 2026 r.' to '2026-05-15'"],
    "usage": {
        "pages": 1,
        "ocr_ms": 2310,
        "llm_ms": 640,
        "prompt_tokens": 925,
        "completion_tokens": 270,
        "llm_calls": 1,
        "model": "openai/gpt-oss-20b",
        "prompt_version": "2026-10-13.3",
        "reasoning_effort": "medium",
        "effort_escalations": 0,
    },
}


class ExtractJsonRequest(BaseModel):
    """JSON variant of an extraction request. Send `schema`, `preset`, or both."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {"document_base64": "<base64>", "preset": "business_card"},
                {"document_base64": "<base64>", "schema": _INVOICE_SCHEMA},
                {
                    "document_base64": "<base64>",
                    "preset": "business_card",
                    "schema": _PRESET_EXTENSION,
                    "options": {"model_tier": "accurate", "include_ocr_text": True},
                },
            ]
        },
    )

    document_base64: str = Field(description="The document (JPEG, PNG, WEBP, TIFF or PDF), base64")
    json_schema: dict[str, Any] | None = Field(
        default=None,
        alias="schema",
        description="JSON Schema of the fields to extract. With `preset`, it extends the preset.",
    )
    preset: str | None = Field(
        default=None, description="Name of a preset from GET /api/v1/presets"
    )
    options: ExtractionOptions = Field(default_factory=ExtractionOptions)


class ExtractMultipartRequest(BaseModel):
    """Multipart variant of an extraction request (documentation only)."""

    model_config = ConfigDict(populate_by_name=True)

    file: bytes = Field(description="The document (JPEG, PNG, WEBP, TIFF or PDF)")
    json_schema: str | None = Field(
        default=None, alias="schema", description="JSON Schema as a JSON string"
    )
    preset: str | None = Field(
        default=None, description="Name of a preset from GET /api/v1/presets"
    )
    options: str | None = Field(
        default=None,
        description='Options as a JSON string, e.g. {"model_tier": "accurate"}',
    )


class ExtractResponse(ExtractionResult):
    model_config = ConfigDict(json_schema_extra={"examples": [_RESULT_EXAMPLE]})


class PresetField(BaseModel):
    name: str
    type: str
    description: str | None = None
    derived: bool = Field(default=False, description="Inferred from the content, not copied")
    semantic_type: str | None = Field(default=None, description="Value of x-schemalens-type")
    required: bool = False


class PresetInfo(BaseModel):
    name: str
    version: str
    title: str
    description: str
    fields: list[PresetField]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "business_card",
                    "version": "2.0.0",
                    "title": "Business card",
                    "description": (
                        "A business card: the person, their job title, the company "
                        "and contact details."
                    ),
                    "fields": [
                        {
                            "name": "phone",
                            "type": "string",
                            "description": "Phone number",
                            "derived": False,
                            "semantic_type": "phone",
                            "required": False,
                        }
                    ],
                }
            ]
        }
    )


class PresetList(BaseModel):
    presets: list[PresetInfo]


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class RenderRequest(BaseModel):
    """A result the client already has, turned into a file. Nothing is extracted or stored."""

    model_config = ConfigDict(populate_by_name=True)

    result: ExtractionResult = Field(description="The response of POST /api/v1/extract")
    json_schema: dict[str, Any] | None = Field(
        default=None, alias="schema", description="The schema sent with the extraction, if any"
    )
    preset: str | None = Field(
        default=None, description="The preset sent with the extraction, if any"
    )
    labels: dict[str, str] | None = Field(
        default=None,
        description=(
            "XLSX only: readable names for the column headings, by field path "
            '(e.g. {"gross_total": "Gross total"}); without it the headings are the paths'
        ),
    )


class DemoInfo(BaseModel):
    """What the landing site needs to know about this deployment."""

    max_upload_bytes: int
    max_pages: int
    daily_documents: int
