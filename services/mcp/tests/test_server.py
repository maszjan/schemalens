"""The MCP tools against a fake SchemaLens API: requests, results and readable errors."""

import json
from typing import Any

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from schemalens_mcp import server

PRESETS = {
    "presets": [
        {
            "name": "receipt",
            "version": "2.0.0",
            "title": "Receipt",
            "description": "A till receipt.",
            "fields": [
                {
                    "name": "total",
                    "type": "number",
                    "description": "Total",
                    "derived": False,
                    "semantic_type": None,
                    "required": True,
                },
            ],
        }
    ]
}
RESULT = {
    "data": {"total": 10.48},
    "fields": {"total": {"confidence": 0.9, "evidence": ["L003"]}},
    "warnings": [],
    "usage": {"pages": 1},
}


@pytest.fixture
def api(monkeypatch):
    """Route the tools' HTTP calls to a handler; collect the requests."""
    requests: list[httpx.Request] = []
    answers: dict[str, httpx.Response] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return answers.get(request.url.path, httpx.Response(404, json={"title": "Not found"}))

    real = server._client

    def fake(settings: server.McpSettings) -> httpx.Client:
        client = real(settings)
        client._transport = httpx.MockTransport(handle)
        return client

    monkeypatch.setattr(server, "_client", fake)
    return requests, answers


@pytest.fixture
def receipt(tmp_path):
    path = tmp_path / "receipt.jpg"
    path.write_bytes(b"\xff\xd8\xff fake jpeg")
    return path


async def test_the_server_offers_both_tools() -> None:
    names = {tool.name for tool in await server.server.list_tools()}
    assert names == {"list_presets", "extract_document"}


def test_list_presets_returns_names_and_fields(api) -> None:
    requests, answers = api
    answers["/api/v1/presets"] = httpx.Response(200, json=PRESETS)
    presets = server.list_presets()
    assert presets[0]["name"] == "receipt"
    assert presets[0]["fields"] == [
        {"name": "total", "type": "number", "description": "Total", "derived": False}
    ]


def test_extract_sends_the_file_preset_and_fields(api, receipt, monkeypatch) -> None:
    requests, answers = api
    monkeypatch.setenv("SCHEMALENS_API_KEY", "k-one")
    answers["/api/v1/extract"] = httpx.Response(200, json=RESULT)
    fields: dict[str, Any] = {"cashier": {"type": "string"}}

    result = server.extract_document(str(receipt), preset="receipt", fields=fields)

    assert result == RESULT
    request = requests[-1]
    assert request.headers["authorization"] == "Bearer k-one"
    body = request.content.decode("latin-1")
    assert 'name="preset"' in body and "receipt" in body
    assert json.dumps({"type": "object", "properties": fields}) in body
    assert 'filename="receipt.jpg"' in body


def test_api_problems_become_readable_tool_errors(api, receipt) -> None:
    requests, answers = api
    answers["/api/v1/extract"] = httpx.Response(
        422, json={"title": "Document unreadable", "detail": "Too little readable text."}
    )
    with pytest.raises(ToolError, match=r"Document unreadable \(422\): Too little readable text."):
        server.extract_document(str(receipt), preset="receipt")


def test_bad_input_is_refused_before_any_call(api, receipt, tmp_path) -> None:
    requests, _ = api
    with pytest.raises(ToolError, match="No such file"):
        server.extract_document(str(tmp_path / "missing.pdf"), preset="receipt")
    text = tmp_path / "notes.txt"
    text.write_text("hello")
    with pytest.raises(ToolError, match="Unsupported file type"):
        server.extract_document(str(text), preset="receipt")
    with pytest.raises(ToolError, match="preset, your own fields"):
        server.extract_document(str(receipt))
    assert requests == []


def test_an_unreachable_api_is_reported(monkeypatch, receipt) -> None:
    monkeypatch.setenv("SCHEMALENS_API_URL", "http://127.0.0.1:9")
    with pytest.raises(ToolError, match="Cannot reach the SchemaLens API"):
        server.extract_document(str(receipt), preset="receipt")
