"""Request dependencies: settings, the (lazily built) pipeline, limits, the client address."""

import hashlib
import hmac
import json
import threading
from collections.abc import Callable
from typing import Annotated, Any, Protocol

from fastapi import Depends, Request

from schemalens_api.execution import Executor
from schemalens_api.limits import Limiter, client_key
from schemalens_api.problems import ApiError
from schemalens_api.settings import ApiSettings
from schemalens_core.models import ExtractionOptions, ExtractionResult

_pipeline_lock = threading.Lock()


class Extractor(Protocol):
    def extract(
        self,
        document: bytes,
        mime: str | None,
        schema: dict[str, Any] | None,
        options: ExtractionOptions | None = None,
        *,
        preset: str | None = None,
        progress: Callable[[Any, dict[str, Any]], None] | None = None,
    ) -> ExtractionResult: ...


def get_settings(request: Request) -> ApiSettings:
    settings: ApiSettings = request.app.state.settings
    return settings


def get_executor(request: Request) -> Executor:
    executor: Executor = request.app.state.executor
    return executor


def get_limiter(request: Request) -> Limiter:
    limiter: Limiter = request.app.state.limiter
    return limiter


def get_pipeline(request: Request) -> Extractor:
    """The extraction pipeline, built on first use.

    The import lives here so that importing the API (cold start, docs, health checks)
    never loads the OCR and LLM stack.
    """
    state = request.app.state
    with _pipeline_lock:
        if getattr(state, "pipeline", None) is None:
            from schemalens_core.pipeline import build_pipeline

            state.pipeline = build_pipeline(state.settings)
    pipeline: Extractor = state.pipeline
    return pipeline


def client_address(request: Request) -> str:
    """The caller's address. On AWS it comes from the request context of the Function URL
    invocation (set by AWS and passed on by the Lambda Web Adapter, not by the client); the
    socket peer there is the adapter itself."""
    settings: ApiSettings = request.app.state.settings
    if settings.lambda_request_context:
        try:
            context = json.loads(request.headers.get("x-amzn-request-context", ""))
            return str(context["http"]["sourceIp"])
        except (ValueError, KeyError, TypeError):
            return "unknown"
    return request.client.host if request.client else "unknown"


def too_many(detail: str) -> ApiError:
    return ApiError(429, "limit-reached", "Limit reached", detail, headers={"Retry-After": "3600"})


# Paths open without a key: health checks and the docs.
_OPEN_PATHS = ("/healthz", "/docs", "/openapi.json", "/internal/warm-up")


def _given_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def _match(given: str, keys: list[str]) -> str | None:
    return next((key for key in keys if given and hmac.compare_digest(given, key)), None)


def _gateway_key_id(request: Request, settings: ApiSettings) -> str | None:
    """The id of the key API Gateway checked, from the request context AWS sets (REST API).

    A call through the demo's Function URL has no such id, whatever headers the client sends.
    """
    if not (settings.gateway_keys and settings.lambda_request_context):
        return None
    try:
        context = json.loads(request.headers.get("x-amzn-request-context", ""))
        key_id = context["identity"]["apiKeyId"]
    except (ValueError, KeyError, TypeError):
        return None
    return str(key_id) if key_id else None


def _prefix(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:8]


def require_key(request: Request) -> None:
    """Every call except health and docs carries a key: a service key (`API_KEYS`), a demo key
    (`DEMO_KEYS`) or, on AWS, one API Gateway has already checked.

    Keys are compared in constant time and never logged: the log gets a short hash prefix.
    """
    settings: ApiSettings = request.app.state.settings
    if request.url.path in _OPEN_PATHS:
        return
    given = _given_key(request)
    if service := _match(given, settings.key_list()):
        request.state.key_kind, request.state.api_key = "service", service
        request.state.key_prefix = _prefix(service)
        return
    if demo := _match(given, settings.demo_key_list()):
        request.state.key_kind = "demo"
        request.state.key_prefix = _prefix(demo)
        return
    if key_id := _gateway_key_id(request, settings):
        request.state.key_kind = "gateway"
        request.state.key_prefix = _prefix(key_id)
        return
    raise ApiError(
        401,
        "unauthorized",
        "Unauthorized",
        "Send a valid API key: Authorization: Bearer <key> or X-API-Key.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def take_daily(request: Request, kind: str, limit: int, global_limit: int | None = None) -> None:
    """Count one use against the caller's daily limit; 429 past it.

    A service key counts per key (`KEY_DAILY_DOCUMENTS`, 0 = no limit); a demo key per client
    address and in total; a key API Gateway checked is limited by its usage plan, not here.
    """
    settings: ApiSettings = request.app.state.settings
    limiter = get_limiter(request)
    salt = settings.ip_salt.get_secret_value()
    key_kind = getattr(request.state, "key_kind", None)
    if key_kind == "service":
        daily = settings.key_daily_documents
        if daily and not limiter.take(
            client_key(kind, f"key:{request.state.api_key}", salt), daily
        ):
            raise too_many(f"This API key allows {daily} documents per day.")
        return
    if key_kind != "demo":
        return
    if global_limit is not None and not limiter.take(f"{kind}:all", global_limit):
        raise too_many("The demo has reached today's limit. Please try again tomorrow.")
    if not limiter.take(client_key(kind, client_address(request), salt), limit):
        raise too_many(
            f"This demo allows {limit} per day from one address. Please try again tomorrow."
        )


SettingsDep = Annotated[ApiSettings, Depends(get_settings)]
PipelineDep = Annotated[Extractor, Depends(get_pipeline)]
ExecutorDep = Annotated[Executor, Depends(get_executor)]
