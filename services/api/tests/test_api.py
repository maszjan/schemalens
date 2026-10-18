import asyncio
import base64
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from api_support import AUTH, SCHEMA, FakePipeline, pdf, png

from schemalens_api.deps import get_pipeline
from schemalens_api.main import create_app
from schemalens_core.errors import LlmFailed, LlmRateLimited

PROBLEM = "application/problem+json"


def multipart(document: bytes | None = None, **fields: Any) -> dict[str, Any]:
    data = {
        name: value if isinstance(value, str) else json.dumps(value)
        for name, value in fields.items()
    }
    return {
        "files": {"file": ("doc.png", document if document is not None else png(), "image/png")},
        "data": data,
    }


def assert_problem(response: httpx.Response, status: int, slug: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == PROBLEM
    body = response.json()
    assert body["type"] == f"https://schemalens.space/docs#error-{slug}"
    assert body["status"] == status and body["title"] and body["detail"]
    assert body["instance"] == response.headers["x-request-id"]
    assert "Traceback" not in response.text
    return body


# --- service, docs, middleware ---------------------------------------------------------


async def test_healthz_needs_no_key(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


async def test_request_id_and_server_timing_headers(client: httpx.AsyncClient) -> None:
    generated = await client.get("/healthz")
    assert generated.headers["x-request-id"].startswith("req_")
    assert generated.headers["server-timing"].startswith("app;dur=")

    echoed = await client.get("/healthz", headers={"X-Request-ID": "trace-123.abc"})
    assert echoed.headers["x-request-id"] == "trace-123.abc"

    replaced = await client.get("/healthz", headers={"X-Request-ID": "bad id\twith spaces"})
    assert replaced.headers["x-request-id"].startswith("req_")


async def test_docs_are_served_by_scalar(client: httpx.AsyncClient) -> None:
    response = await client.get("/docs")
    assert response.status_code == 200 and "scalar" in response.text.lower()
    assert (await client.get("/redoc")).status_code == 404


async def test_operation_ids_are_stable(app) -> None:
    operations = {
        (method, path): operation["operationId"]
        for path, methods in app.openapi()["paths"].items()
        for method, operation in methods.items()
    }
    assert operations == {
        ("get", "/healthz"): "healthz",
        ("get", "/api/v1/presets"): "list_presets",
        ("post", "/api/v1/extract"): "extract_document",
        ("get", "/api/v1/demo"): "demo_info",
        ("post", "/api/v1/render"): "render_result",
    }


async def test_openapi_documents_both_request_variants_with_examples(app) -> None:
    schema = app.openapi()
    for path in ("/api/v1/extract",):
        content = schema["paths"][path]["post"]["requestBody"]["content"]
        assert set(content) == {"multipart/form-data", "application/json"}
    examples = schema["components"]["schemas"]["ExtractJsonRequest"]["examples"]
    assert any("preset" in e and "schema" not in e for e in examples)  # preset only
    assert any("schema" in e and "preset" not in e for e in examples)  # own schema only
    assert any("schema" in e and "preset" in e for e in examples)  # preset + extension


def test_committed_openapi_json_is_up_to_date() -> None:
    root = Path(__file__).parents[3]
    check = subprocess.run(
        [sys.executable, str(root / "scripts/export_openapi.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert check.returncode == 0, check.stderr


def test_importing_the_api_does_not_load_the_ml_stack() -> None:
    code = (
        "import sys, schemalens_api.main\n"
        "heavy = ('torch', 'easyocr', 'sentence_transformers', 'chromadb')\n"
        "print([m for m in heavy if m in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# --- presets ---------------------------------------------------------------------------


async def test_list_presets(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/presets", headers=AUTH)

    presets = {preset["name"]: preset for preset in response.json()["presets"]}
    assert set(presets) >= {"invoice", "receipt", "business_card", "generic_document"}
    card = presets["business_card"]
    assert card["version"] == "2.0.0" and card["title"] and card["description"]
    phone = next(field for field in card["fields"] if field["name"] == "phone")
    assert phone == {
        "name": "phone",
        "type": "string",
        "description": "Phone number",
        "derived": False,
        "semantic_type": "phone",
        "required": False,
    }
    derived = [f["name"] for f in presets["generic_document"]["fields"] if f["derived"]]
    assert "summary" in derived


# --- POST /api/v1/extract: happy paths -----------------------------------------------------


async def test_extract_multipart_with_own_schema(
    client: httpx.AsyncClient, pipeline: FakePipeline
) -> None:
    response = await client.post("/api/v1/extract", headers=AUTH, **multipart(schema=SCHEMA))

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    assert body["data"] == {"sklep": "SKLEP EXAMPLE", "suma": None}  # null fields are kept
    assert body["fields"]["suma"] == {"confidence": 0.0, "evidence": []}
    assert body["warnings"] == ["sklep: possible OCR noise"]
    assert body["usage"]["model"] == "model-fast" and body["usage"]["pages"] == 1
    assert "ocr_text" not in body
    assert pipeline.calls[0]["schema"] == SCHEMA and pipeline.calls[0]["preset"] is None
    assert "ocr;dur=120" in response.headers["server-timing"]
    assert "llm;dur=340" in response.headers["server-timing"]


async def test_extract_with_preset(client: httpx.AsyncClient, pipeline: FakePipeline) -> None:
    response = await client.post(
        "/api/v1/extract", headers=AUTH, **multipart(preset="business_card")
    )

    assert response.status_code == 200
    assert "full_name" in response.json()["data"] and "phone" in response.json()["data"]
    assert pipeline.calls[0]["preset"] == "business_card" and pipeline.calls[0]["schema"] is None


async def test_extract_with_preset_extended_by_schema(
    client: httpx.AsyncClient, pipeline: FakePipeline
) -> None:
    extension = {
        "type": "object",
        "properties": {"branza": {"type": "string", "x-schemalens-derived": True}},
    }

    response = await client.post(
        "/api/v1/extract", headers=AUTH, **multipart(preset="business_card", schema=extension)
    )

    data = response.json()["data"]
    assert response.status_code == 200
    assert list(data)[-1] == "branza" and "full_name" in data
    assert pipeline.calls[0] == {
        **pipeline.calls[0],
        "preset": "business_card",
        "schema": extension,
    }


async def test_extract_json_with_base64_document_and_options(
    client: httpx.AsyncClient, pipeline: FakePipeline
) -> None:
    payload = {
        "document_base64": base64.b64encode(png()).decode(),
        "schema": SCHEMA,
        "options": {"model_tier": "accurate", "include_ocr_text": True},
    }

    response = await client.post("/api/v1/extract", headers=AUTH, json=payload)

    body = response.json()
    assert response.status_code == 200, response.text
    assert body["usage"]["model"] == "model-accurate"  # the client picks the tier, not the profile
    assert body["ocr_text"] == ["[L001] SKLEP EXAMPLE"]
    assert pipeline.calls[0]["options"].model_tier == "accurate"


async def test_multipart_options_as_json_string(
    client: httpx.AsyncClient, pipeline: FakePipeline
) -> None:
    response = await client.post(
        "/api/v1/extract",
        headers=AUTH,
        **multipart(schema=SCHEMA, options={"model_tier": "accurate"}),
    )
    assert response.status_code == 200 and pipeline.calls[0]["options"].model_tier == "accurate"


# --- POST /api/v1/extract: errors ----------------------------------------------------------


async def test_413_when_the_document_is_too_big(make_settings) -> None:
    settings = make_settings(max_upload_bytes=2_000)
    app = create_app(settings)
    fake = FakePipeline(settings)
    app.dependency_overrides[get_pipeline] = lambda: fake
    big = png((600, 600)) + b"\0" * 3_000

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as http:
        over_part = await http.post(
            "/api/v1/extract", headers=AUTH, **multipart(big, schema=SCHEMA)
        )
        huge = await http.post(
            "/api/v1/extract", headers=AUTH, **multipart(b"\0" * 400_000, schema=SCHEMA)
        )
        as_json = await http.post(
            "/api/v1/extract",
            headers=AUTH,
            json={"document_base64": base64.b64encode(big).decode(), "schema": SCHEMA},
        )

    for response in (over_part, huge, as_json):
        assert_problem(response, 413, "document-too-large")
    assert fake.calls == []


async def test_413_when_the_document_has_too_many_pages(
    client: httpx.AsyncClient, pipeline: FakePipeline
) -> None:
    ok = await client.post("/api/v1/extract", headers=AUTH, **multipart(pdf(3), schema=SCHEMA))
    too_long = await client.post(
        "/api/v1/extract", headers=AUTH, **multipart(pdf(4), schema=SCHEMA)
    )

    assert ok.status_code == 200
    body = assert_problem(too_long, 413, "document-too-large")
    assert "4 pages" in body["detail"]
    assert len(pipeline.calls) == 1


async def test_415_for_unsupported_document_and_request_types(client: httpx.AsyncClient) -> None:
    gif = await client.post(
        "/api/v1/extract", headers=AUTH, **multipart(b"GIF89a" + b"\0" * 50, schema=SCHEMA)
    )
    plain = await client.post(
        "/api/v1/extract", headers={**AUTH, "Content-Type": "text/plain"}, content=b"hello"
    )

    assert_problem(gif, 415, "unsupported-document")
    assert_problem(plain, 415, "unsupported-media-type")


async def test_422_lists_every_schema_error(client: httpx.AsyncClient) -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "money"}, "b": {"type": "string", "format": "uuid"}},
    }

    response = await client.post("/api/v1/extract", headers=AUTH, **multipart(schema=schema))

    body = assert_problem(response, 422, "invalid-schema")
    assert len(body["errors"]) == 2
    assert body["errors"][0].startswith("a: unsupported or missing type 'money'")


async def test_422_for_unknown_preset_and_missing_schema(client: httpx.AsyncClient) -> None:
    unknown = await client.post("/api/v1/extract", headers=AUTH, **multipart(preset="passport"))
    neither = await client.post("/api/v1/extract", headers=AUTH, **multipart())

    assert (
        "preset 'passport' does not exist"
        in assert_problem(unknown, 422, "invalid-schema")["errors"][0]
    )
    assert assert_problem(neither, 422, "invalid-schema")["errors"] == [
        "either 'schema' or 'preset' is required"
    ]


async def test_422_for_an_unreadable_document(client: httpx.AsyncClient) -> None:
    truncated = png((200, 200))[:60]
    broken_pdf = b"%PDF-1.7\nnot really a pdf"

    image = await client.post(
        "/api/v1/extract", headers=AUTH, **multipart(truncated, schema=SCHEMA)
    )
    document = await client.post(
        "/api/v1/extract", headers=AUTH, **multipart(broken_pdf, schema=SCHEMA)
    )

    assert_problem(image, 422, "document-unreadable")
    assert_problem(document, 422, "document-unreadable")


async def test_422_for_malformed_requests_without_echoing_input(client: httpx.AsyncClient) -> None:
    bad_schema_json = await client.post(
        "/api/v1/extract", headers=AUTH, **multipart(schema="{not json")
    )
    bad_options = await client.post(
        "/api/v1/extract",
        headers=AUTH,
        **multipart(schema=SCHEMA, options={"model_tier": "SECRET-TIER"}),
    )
    no_file = await client.post(
        "/api/v1/extract", headers=AUTH, data={"schema": json.dumps(SCHEMA)}, files={"x": b"1"}
    )
    bad_base64 = await client.post(
        "/api/v1/extract",
        headers=AUTH,
        json={"document_base64": "@@not-base64@@", "schema": SCHEMA},
    )
    bad_body = await client.post("/api/v1/extract", headers=AUTH, json={"schema": SCHEMA})

    assert "not valid JSON" in assert_problem(bad_schema_json, 422, "invalid-request")["detail"]
    options = assert_problem(bad_options, 422, "invalid-request")
    assert (
        options["errors"][0].startswith("options.model_tier:")
        and "SECRET-TIER" not in bad_options.text
    )
    assert "'file' part" in assert_problem(no_file, 422, "invalid-request")["detail"]
    assert "base64" in assert_problem(bad_base64, 422, "invalid-request")["detail"]
    assert assert_problem(bad_body, 422, "invalid-request")["errors"][0].startswith(
        "document_base64:"
    )


async def test_429_when_every_slot_is_busy(make_settings) -> None:
    settings = make_settings(max_concurrent_jobs=1, busy_retry_after_s=7)
    app = create_app(settings)
    fake = FakePipeline(settings)
    fake.gate = threading.Event()
    app.dependency_overrides[get_pipeline] = lambda: fake

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as http:
        first = asyncio.create_task(
            http.post("/api/v1/extract", headers=AUTH, **multipart(schema=SCHEMA))
        )
        await asyncio.to_thread(fake.entered.wait, 5)
        second = await http.post("/api/v1/extract", headers=AUTH, **multipart(schema=SCHEMA))
        fake.gate.set()
        first_response = await first

    assert_problem(second, 429, "too-many-requests")
    assert second.headers["retry-after"] == "7"
    assert first_response.status_code == 200 and len(fake.calls) == 1


async def test_504_when_processing_takes_too_long(make_settings) -> None:
    settings = make_settings(sync_timeout_s=0.05)
    app = create_app(settings)
    fake = FakePipeline(settings)
    fake.delay_s = 0.4
    app.dependency_overrides[get_pipeline] = lambda: fake

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as http:
        response = await http.post("/api/v1/extract", headers=AUTH, **multipart(schema=SCHEMA))
        assert_problem(response, 504, "extraction-timeout")
        await asyncio.sleep(0.5)  # the abandoned work finishes and frees its slot
        fake.delay_s = 0
        again = await http.post("/api/v1/extract", headers=AUTH, **multipart(schema=SCHEMA))
        assert again.status_code in (200, 504)


async def test_503_with_retry_after_when_the_llm_provider_rate_limits(
    client: httpx.AsyncClient, pipeline: FakePipeline
) -> None:
    pipeline.error = LlmRateLimited("provider said: SECRET DOCUMENT TEXT", retry_after=465.0)

    response = await client.post("/api/v1/extract", headers=AUTH, **multipart(schema=SCHEMA))

    assert_problem(response, 503, "llm-rate-limited")
    assert response.headers["retry-after"] == "465"
    assert "SECRET" not in response.text


async def test_502_and_500_never_leak_internals(
    client: httpx.AsyncClient, pipeline: FakePipeline
) -> None:
    pipeline.error = LlmFailed("output unusable: SECRET DOCUMENT TEXT")
    upstream = await client.post("/api/v1/extract", headers=AUTH, **multipart(schema=SCHEMA))
    pipeline.error = RuntimeError("SECRET DOCUMENT TEXT in a traceback")
    crash = await client.post("/api/v1/extract", headers=AUTH, **multipart(schema=SCHEMA))

    assert_problem(upstream, 502, "llm-failed")
    assert_problem(crash, 500, "internal-error")
    assert "SECRET" not in upstream.text + crash.text


async def test_unknown_routes_are_problems_too(client: httpx.AsyncClient) -> None:
    assert_problem(await client.get("/v1/nothing"), 404, "not-found")
    assert_problem(await client.get("/api/v1/extract", headers=AUTH), 405, "method-not-allowed")


# --- logs ------------------------------------------------------------------------------


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.fixture
def log_lines():
    handler = _Capture()
    logger = logging.getLogger("schemalens.api")
    logger.addHandler(handler)
    yield handler.lines
    logger.removeHandler(handler)


async def test_logs_are_json_with_metadata_only(
    client: httpx.AsyncClient, pipeline: FakePipeline, log_lines: list[str]
) -> None:
    document = png()
    await client.post("/api/v1/extract", headers=AUTH, **multipart(document, schema=SCHEMA))
    await client.post("/api/v1/extract", headers=AUTH, **multipart(schema={"type": "object"}))
    pipeline.error = RuntimeError("SECRET DOCUMENT TEXT")
    await client.post("/api/v1/extract", headers=AUTH, **multipart(schema=SCHEMA))

    entries = [json.loads(line) for line in log_lines]
    ok = entries[0]
    assert ok["event"] == "request" and ok["method"] == "POST" and ok["path"] == "/api/v1/extract"
    assert ok["status"] == 200 and ok["ms"] >= 0 and ok["request_id"].startswith("req_")
    assert ok["bytes"] == len(document) and ok["pages"] == 1
    assert (ok["ocr_ms"], ok["llm_ms"], ok["model"]) == (120, 340, "model-fast")
    assert set(ok) <= {
        "event", "request_id", "method", "path", "status", "ms",
        "key_prefix", "bytes", "pages", "ocr_ms", "llm_ms", "model", "error_code",
    }  # fmt: skip
    assert any(e.get("status") == 422 and e.get("error_code") == "invalid-schema" for e in entries)
    assert any(
        e.get("event") == "unhandled_exception" and e["error_type"] == "RuntimeError"
        for e in entries
    )
    assert entries[-1]["status"] == 500 and entries[-1]["error_code"] == "internal-error"
    everything = "\n".join(log_lines)
    for forbidden in ("SKLEP EXAMPLE", "SECRET", "possible OCR noise", "sklep"):
        assert forbidden not in everything
