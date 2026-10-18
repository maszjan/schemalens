"""Reading an extraction request: multipart or JSON, in memory, within size limits."""

import base64
import binascii
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from pydantic import ValidationError
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from schemalens_api.models import ExtractJsonRequest
from schemalens_api.problems import ApiError
from schemalens_core.models import ExtractionOptions

_MB = 1024 * 1024
# Room for multipart boundaries and form fields, or base64 (4/3) and the JSON around it.
_ENVELOPE_FACTOR = 1.4
_ENVELOPE_EXTRA = 256 * 1024
_MAX_FIELD_BYTES = 512 * 1024  # schema and options as form fields


class _InMemoryMultiPartParser(MultiPartParser):
    # Starlette spools uploads over 1 MB to a temporary file. Documents must never touch
    # the disk (zero retention), and the body is already capped, so keep them in memory.
    spool_max_size = 1024 * _MB


@dataclass
class ExtractionRequest:
    document: bytes
    schema: dict[str, Any] | None
    preset: str | None
    options: ExtractionOptions


def too_large(detail: str) -> ApiError:
    return ApiError(413, "document-too-large", "Document too large", detail)


def _invalid(detail: str, errors: list[str] | None = None) -> ApiError:
    return ApiError(422, "invalid-request", "Invalid request", detail, errors=errors)


async def _capped_stream(
    request: Request, limit: int, max_bytes: int
) -> AsyncGenerator[bytes, None]:
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > limit:
            raise too_large(_size_detail(max_bytes))
        yield chunk


def _size_detail(max_bytes: int) -> str:
    return f"The document exceeds the limit of {max_bytes / _MB:g} MB."


def _json_field(text: str | None, name: str) -> Any:
    if text is None or not text.strip():
        return None
    try:
        return json.loads(text)
    except ValueError:
        raise _invalid(f"Form field '{name}' is not valid JSON.") from None


def _options(raw: Any) -> ExtractionOptions:
    try:
        return ExtractionOptions.model_validate(raw or {})
    except ValidationError as exc:
        raise _invalid(
            "The options are not valid, see errors.", _messages(exc, "options")
        ) from None


def _messages(exc: ValidationError, prefix: str = "") -> list[str]:
    # Locations and messages only, never the rejected input.
    return [
        ".".join(str(part) for part in ([prefix] if prefix else []) + list(error["loc"]))
        + f": {error['msg']}"
        for error in exc.errors(include_input=False, include_url=False)
    ]


async def read_extraction_request(request: Request, max_bytes: int) -> ExtractionRequest:
    """Parse the request without writing anything to disk; 413 as soon as it is too big."""
    limit = int(max_bytes * _ENVELOPE_FACTOR) + _ENVELOPE_EXTRA
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise too_large(_size_detail(max_bytes))
    stream = _capped_stream(request, limit, max_bytes)
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()

    if content_type == "multipart/form-data":
        parsed = await _read_multipart(request, stream)
    elif content_type == "application/json":
        parsed = _read_json(b"".join([chunk async for chunk in stream]))
    else:
        raise ApiError(
            415,
            "unsupported-media-type",
            "Unsupported media type",
            "Send multipart/form-data with a 'file' part or application/json with document_base64.",
        )
    if not parsed.document:
        raise _invalid("The document is empty.")
    if len(parsed.document) > max_bytes:
        raise too_large(_size_detail(max_bytes))
    return parsed


async def _read_multipart(
    request: Request, stream: AsyncGenerator[bytes, None]
) -> ExtractionRequest:
    parser = _InMemoryMultiPartParser(
        request.headers, stream, max_files=1, max_fields=4, max_part_size=_MAX_FIELD_BYTES
    )
    try:
        form = await parser.parse()
    except MultiPartException:
        raise _invalid("The multipart body could not be parsed.") from None
    try:
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise _invalid("Multipart requests need a 'file' part with the document.")
        document = await upload.read()
        fields = {name: form.get(name) for name in ("schema", "preset", "options")}
    finally:
        await form.close()
    text = {name: value if isinstance(value, str) else None for name, value in fields.items()}
    schema = _json_field(text["schema"], "schema")
    if schema is not None and not isinstance(schema, dict):
        raise _invalid("Form field 'schema' must be a JSON object.")
    return ExtractionRequest(
        document=document,
        schema=schema,
        preset=(text["preset"] or "").strip() or None,
        options=_options(_json_field(text["options"], "options")),
    )


def _read_json(body: bytes) -> ExtractionRequest:
    try:
        payload = ExtractJsonRequest.model_validate_json(body)
    except ValidationError as exc:
        raise _invalid("The request body is not valid, see errors.", _messages(exc)) from None
    try:
        document = base64.b64decode(payload.document_base64, validate=True)
    except (binascii.Error, ValueError):
        raise _invalid("document_base64 is not valid base64.") from None
    return ExtractionRequest(document, payload.json_schema, payload.preset, payload.options)
