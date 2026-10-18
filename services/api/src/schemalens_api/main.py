"""FastAPI application. A thin HTTP adapter: all extraction logic lives in schemalens_core."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from pydantic.json_schema import models_json_schema
from scalar_fastapi import get_scalar_api_reference

from schemalens_api.deps import require_key
from schemalens_api.execution import Executor
from schemalens_api.limits import build_limiter
from schemalens_api.middleware import RequestContextMiddleware
from schemalens_api.models import ExtractJsonRequest, ExtractMultipartRequest
from schemalens_api.problems import install_handlers
from schemalens_api.routes import extract, health, presets
from schemalens_api.settings import ApiSettings, get_api_settings

DESCRIPTION = """\
Send a document (photo, scan, PDF, screenshot) with a preset or a JSON Schema and get \
validated JSON back, with a confidence and evidence for every field.

Every call except `/healthz` and the docs needs an API key: `Authorization: Bearer <key>` or \
`X-API-Key`. Documents are processed in memory and never stored. Errors are RFC 9457 problem \
documents.
"""
NO_KEYS = (
    "No API key is configured, so nobody could call this API. Generate one with "
    "`schemalens-api keys new --env-file .env` (or set API_KEYS / DEMO_KEYS) and start again."
)
_REF = "#/components/schemas/{model}"


def _configure_logging() -> None:
    logger = logging.getLogger("schemalens.api")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))  # the message is already JSON
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False


def _document_request_bodies(schema: dict[str, Any]) -> None:
    """The extract endpoint reads its body itself (multipart or JSON); describe it for the docs."""
    _, definitions = models_json_schema(
        [(model, "validation") for model in (ExtractJsonRequest, ExtractMultipartRequest)],
        ref_template=_REF,
    )
    schema["components"]["schemas"].update(definitions["$defs"])
    schema["paths"]["/api/v1/extract"]["post"]["requestBody"] = {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {"$ref": _REF.format(model="ExtractMultipartRequest")}
            },
            "application/json": {"schema": {"$ref": _REF.format(model="ExtractJsonRequest")}},
        },
    }


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or get_api_settings()
    _configure_logging()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Fail at start, not with a 401 on every call. Building the app (OpenAPI export) is fine.
        if not settings.keys_configured():
            raise RuntimeError(NO_KEYS)
        yield

    app = FastAPI(
        lifespan=lifespan,
        title="SchemaLens API",
        version="1.0.0",
        description=DESCRIPTION,
        docs_url=None,
        redoc_url=None,
        dependencies=[Depends(require_key)],
    )
    app.state.settings = settings
    app.state.executor = Executor(settings.max_concurrent_jobs, settings.busy_retry_after_s)
    app.state.pipeline = None  # built on first use, see deps.get_pipeline
    app.state.limiter = build_limiter(settings)

    app.add_middleware(RequestContextMiddleware)
    # The landing site (another origin, e.g. Vercel) calls the API from the browser.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origin_list(),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Accept", "Authorization", "X-API-Key"],
        expose_headers=["Content-Disposition", "Server-Timing", "X-Request-ID", "Retry-After"],
        max_age=600,
    )
    install_handlers(app)
    for module in (health, presets, extract):
        app.include_router(module.router)

    @app.get("/docs", include_in_schema=False)
    async def docs() -> HTMLResponse:
        return get_scalar_api_reference(openapi_url=app.openapi_url, title=app.title)

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            schema = get_openapi(
                title=app.title,
                version=app.version,
                description=app.description,
                routes=app.routes,
            )
            _document_request_bodies(schema)
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]
    return app


app = create_app()
