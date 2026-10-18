"""MCP server for SchemaLens: AI agents (Claude, Cursor, ...) extract fields from documents.

For local work: it runs on your machine next to the SchemaLens you set up yourself (Docker on
localhost by default, or your own server) and calls its HTTP API, so it installs in seconds
(no OCR, no models). Nothing is
logged except what the MCP runtime logs about tool calls; document content never is.
"""

import json
from pathlib import Path
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".pdf"}


class McpSettings(BaseSettings):
    """Where the API is. `SCHEMALENS_API_URL`, `SCHEMALENS_API_KEY`, `SCHEMALENS_TIMEOUT_S`."""

    model_config = SettingsConfigDict(env_prefix="SCHEMALENS_", extra="ignore")

    api_url: str = "http://localhost:8000"
    api_key: SecretStr = SecretStr("")
    # A local model needs about a minute per document; a hosted one a few seconds.
    timeout_s: float = 300.0
    max_file_bytes: int = 20 * 1024 * 1024


INSTRUCTIONS = """\
SchemaLens extracts structured fields from printed documents (invoices, receipts, CVs, business
cards, contracts, product labels, menus, any other document) using OCR and a language model.
Call list_presets to see ready-made schemas, then extract_document with a local file path and a
preset, your own fields, or both; for a folder, call it once per file.
Every value comes with a confidence (0-1) and the OCR lines it
was read from; a value missing from the document is null, never guessed. Check fields with a low
confidence or a warning before relying on them."""

server = MCPServer("SchemaLens", instructions=INSTRUCTIONS)


def _client(settings: McpSettings) -> httpx.Client:
    headers = {"Accept": "application/json"}
    if key := settings.api_key.get_secret_value():
        headers["Authorization"] = f"Bearer {key}"
    return httpx.Client(
        base_url=settings.api_url.rstrip("/"), headers=headers, timeout=settings.timeout_s
    )


def _raise_for_problem(response: httpx.Response) -> None:
    """The API's RFC 9457 problem, as a tool error the agent can read and act on."""
    if response.is_success:
        return
    try:
        problem = response.json()
        message = (
            f"{problem.get('title', 'Error')} ({response.status_code}): {problem.get('detail', '')}"
        )
    except ValueError:
        message = f"The SchemaLens API answered {response.status_code}."
    raise ToolError(message.strip())


@server.tool()
def list_presets() -> list[dict[str, Any]]:
    """Ready-made schemas (invoice, receipt, CV, business card, contract terms, ...) with their
    fields: name, type, description, whether a field is inferred ("derived")."""
    settings = McpSettings()
    try:
        with _client(settings) as client:
            response = client.get("/api/v1/presets")
    except httpx.HTTPError as error:
        raise ToolError(f"Cannot reach the SchemaLens API at {settings.api_url}: {error}") from None
    _raise_for_problem(response)
    return [
        {
            "name": preset["name"],
            "title": preset["title"],
            "description": preset["description"],
            "fields": [
                {key: field[key] for key in ("name", "type", "description", "derived")}
                for field in preset["fields"]
            ],
        }
        for preset in response.json()["presets"]
    ]


@server.tool()
def extract_document(
    file_path: str,
    preset: str | None = None,
    fields: dict[str, Any] | None = None,
    include_ocr_text: bool = False,
) -> dict[str, Any]:
    """Extract fields from a printed document (JPEG, PNG, WEBP, TIFF or PDF on this machine).

    Give a `preset` name (see list_presets), your own `fields` as JSON Schema properties
    (e.g. {"order_number": {"type": "string", "description": "Order number"}}), or both: your
    fields then extend the preset. Mark inferred fields with "x-schemalens-derived": true.

    Returns `data` (one value per field, null when the document does not contain it), `fields`
    (confidence and evidence line ids per field), `warnings` and `usage`.
    """
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise ToolError(f"No such file: {path}")
    if path.suffix.lower() not in SUFFIXES:
        raise ToolError(f"Unsupported file type {path.suffix!r}; use JPEG, PNG, WEBP, TIFF or PDF.")
    settings = McpSettings()
    if path.stat().st_size > settings.max_file_bytes:
        raise ToolError(f"The file is larger than {settings.max_file_bytes // (1024 * 1024)} MB.")
    if not preset and not fields:
        raise ToolError("Give a preset, your own fields, or both.")

    form: dict[str, str] = {"options": json.dumps({"include_ocr_text": include_ocr_text})}
    if preset:
        form["preset"] = preset
    if fields:
        form["schema"] = json.dumps({"type": "object", "properties": fields})
    try:
        with _client(settings) as client, path.open("rb") as file:
            response = client.post("/api/v1/extract", data=form, files={"file": (path.name, file)})
    except httpx.TimeoutException:
        raise ToolError(f"The extraction took longer than {settings.timeout_s:g} s.") from None
    except httpx.HTTPError as error:
        raise ToolError(f"Cannot reach the SchemaLens API at {settings.api_url}: {error}") from None
    _raise_for_problem(response)
    return response.json()


def main() -> None:
    """`schemalens-mcp`: serve over stdio, as MCP clients (Claude, Cursor, ...) expect."""
    server.run()


if __name__ == "__main__":
    main()
