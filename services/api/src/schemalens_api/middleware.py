"""Request id, Server-Timing and one structured log line per request (metadata only)."""

import json
import logging
import re
import time
import uuid
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from schemalens_api.problems import INTERNAL_ERROR, PROBLEM_JSON, problem_body

logger = logging.getLogger("schemalens.api")

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# The only request facts that may be logged No bodies, no document content.
_LOGGED_STATE = ("key_prefix", "bytes", "pages", "ocr_ms", "llm_ms", "model", "error_code")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _REQUEST_ID.match(incoming) else f"req_{uuid.uuid4().hex[:16]}"
        state: dict[str, Any] = scope.setdefault("state", {})
        state["request_id"] = request_id
        started = time.perf_counter()
        status = 500

        async def send_with_headers(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                timings = [f"app;dur={(time.perf_counter() - started) * 1000:.1f}"]
                for name in ("ocr", "llm"):
                    if state.get(f"{name}_ms") is not None:
                        timings.append(f"{name};dur={state[f'{name}_ms']}")
                headers["Server-Timing"] = ", ".join(timings)
            await send(message)

        started_response = False

        async def tracked_send(message: Message) -> None:
            nonlocal started_response
            started_response = started_response or message["type"] == "http.response.start"
            await send_with_headers(message)

        try:
            await self.app(scope, receive, tracked_send)
        except Exception as exc:
            # Handled here and not re-raised: the server would otherwise log a traceback,
            # and exception text may describe a document. Only the type is recorded.
            state["error_code"] = INTERNAL_ERROR.slug
            logger.error(
                json.dumps({"event": "unhandled_exception", "request_id": request_id,
                            "error_type": type(exc).__name__})
            )  # fmt: skip
            if not started_response:
                response = JSONResponse(
                    problem_body(INTERNAL_ERROR, request_id), 500, media_type=PROBLEM_JSON
                )
                await response(scope, receive, send_with_headers)
        finally:
            entry = {
                "event": "request",
                "request_id": request_id,
                "method": scope["method"],
                "path": _loggable_path(scope["path"]),
                "status": status,
                "ms": round((time.perf_counter() - started) * 1000, 1),
            }
            entry.update({key: state[key] for key in _LOGGED_STATE if state.get(key) is not None})
            logger.info(json.dumps(entry))


def _loggable_path(path: str) -> str:
    """Upload URLs carry their credential in the path; it never goes to the log."""
    if path.startswith(_UPLOAD_PREFIX):
        return _UPLOAD_PREFIX + "{token}"
    return path


_UPLOAD_PREFIX = "/api/v1/uploads/"
