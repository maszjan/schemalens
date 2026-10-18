"""Errors as RFC 9457 problem documents. Never a traceback, never document content."""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from schemalens_core.errors import (
    DocumentUnreadable,
    InvalidSchema,
    LlmRateLimited,
    ProfileUnavailable,
    SchemaLensError,
    UnsupportedDocument,
)

PROBLEM_JSON = "application/problem+json"
ERROR_DOCS = "https://schemalens.space/docs#error-"


class Problem(BaseModel):
    """RFC 9457 problem details."""

    type: str
    title: str
    status: int
    detail: str
    instance: str | None = None  # the request id
    errors: list[str] | None = None  # e.g. every problem found in a schema

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": ERROR_DOCS + "invalid-schema",
                    "title": "Invalid schema",
                    "status": 422,
                    "detail": "The schema was rejected, see errors.",
                    "instance": "req_0f8d79d2c1a4",
                    "errors": ["total: unsupported or missing type 'money'"],
                }
            ]
        }
    }


class ApiError(Exception):
    """An error with a fixed public description."""

    def __init__(
        self,
        status: int,
        slug: str,
        title: str,
        detail: str,
        *,
        errors: list[str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(slug)
        self.status = status
        self.slug = slug
        self.title = title
        self.detail = detail
        self.errors = errors
        self.headers = headers or {}


def from_core_error(exc: SchemaLensError) -> ApiError:
    """Public description of a core error. Core messages are metadata only by contract."""
    if isinstance(exc, InvalidSchema):
        detail = "The schema was rejected, see errors."
        return ApiError(422, "invalid-schema", "Invalid schema", detail, errors=exc.errors)
    if isinstance(exc, DocumentUnreadable):
        return ApiError(422, "document-unreadable", "Document unreadable", str(exc))
    if isinstance(exc, UnsupportedDocument):
        return ApiError(415, "unsupported-document", "Unsupported document type", str(exc))
    if isinstance(exc, LlmRateLimited):
        headers = {"Retry-After": str(int(exc.retry_after))} if exc.retry_after else {}
        detail = "The language model provider is rate limiting requests. Try again later."
        return ApiError(503, "llm-rate-limited", "Upstream rate limit", detail, headers=headers)
    if isinstance(exc, ProfileUnavailable):
        detail = "The service is not configured to reach a language model."
        return ApiError(503, "llm-unavailable", "Language model unavailable", detail)
    slug = exc.code.replace("_", "-")
    if exc.code == "llm_failed":
        detail = "The language model did not return a usable answer."
        return ApiError(502, slug, "Language model failed", detail)
    if exc.code == "ocr_failed":
        return ApiError(500, slug, "Text recognition failed", "The document could not be read.")
    return ApiError(500, "internal-error", "Internal error", "The request could not be processed.")


def problem_body(error: ApiError, request_id: str | None) -> dict[str, Any]:
    body = Problem(
        type=ERROR_DOCS + error.slug,
        title=error.title,
        status=error.status,
        detail=error.detail,
        instance=request_id,
        errors=error.errors,
    )
    return body.model_dump(exclude_none=True)


INTERNAL_ERROR = from_core_error(SchemaLensError())


def _respond(request: Request, error: ApiError) -> JSONResponse:
    request.state.error_code = error.slug
    body = problem_body(error, getattr(request.state, "request_id", None))
    return JSONResponse(body, error.status, headers=error.headers, media_type=PROBLEM_JSON)


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _respond(request, exc)

    @app.exception_handler(SchemaLensError)
    async def core_error(request: Request, exc: SchemaLensError) -> JSONResponse:
        return _respond(request, from_core_error(exc))

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Locations and messages only: the rejected input itself is never echoed.
        errors = [
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        ]
        detail = "The request is not valid, see errors."
        return _respond(
            request, ApiError(422, "invalid-request", "Invalid request", detail, errors=errors)
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        titles = {404: "Not found", 405: "Method not allowed"}
        title = titles.get(exc.status_code, "Request failed")
        slug = title.lower().replace(" ", "-")
        return _respond(request, ApiError(exc.status_code, slug, title, title + "."))
