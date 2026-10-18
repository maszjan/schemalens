"""The public demo: one document in, validated fields out. Nothing is stored."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from functools import partial
from typing import Any

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from schemalens_api.deps import ExecutorDep, PipelineDep, SettingsDep, take_daily
from schemalens_api.execution import Executor
from schemalens_api.formats import MEDIA_TYPES, negotiate, render
from schemalens_api.intake import read_extraction_request, too_large
from schemalens_api.models import DemoInfo, ExtractResponse, RenderRequest
from schemalens_api.problems import INTERNAL_ERROR, ApiError, Problem, from_core_error, problem_body
from schemalens_core.errors import SchemaLensError
from schemalens_core.models import ExtractionResult
from schemalens_core.preprocess import count_pages
from schemalens_core.presets import resolve_schema
from schemalens_core.schemas import compile_schema

logger = logging.getLogger("schemalens.api")
router = APIRouter(tags=["extraction"])

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    406: {"model": Problem, "description": "Unknown format or unsupported Accept"},
    413: {"model": Problem, "description": "Document too large for the demo"},
    415: {"model": Problem, "description": "Unsupported document or request type"},
    422: {"model": Problem, "description": "Invalid schema, request or unreadable document"},
    429: {"model": Problem, "description": "Daily demo limit reached, or all slots busy"},
    502: {"model": Problem, "description": "The language model failed"},
    503: {"model": Problem, "description": "The language model provider is rate limiting"},
}

FORMAT_QUERY = Query(
    default=None,
    alias="format",
    description="Result format; wins over the Accept header. Default: json.",
    json_schema_extra={"enum": list(MEDIA_TYPES)},
)
STREAM_QUERY = Query(
    default=False,
    description=(
        "JSON only: answer with `application/x-ndjson`, one event per line as the work goes "
        "on (`accepted`, `reading`, `read`, `extracting`, then `result` or `error`). Events "
        "carry counts and timings, never document text; `result` is the usual JSON body."
    ),
)
NDJSON = "application/x-ndjson"
INCLUDE_META_QUERY = Query(
    default=False,
    description="CSV only: add a `<field>__confidence` column for every scalar field.",
)
# Documented next to the JSON model; csv and xlsx carry the extracted data only.
_CSV_EXAMPLE = "seller_name,total_gross,items.name,items.price\r\nExample,312.45,Paliwo,300.0\r\n"
FILE_RESPONSES: dict[str, Any] = {
    "text/csv": {"schema": {"type": "string"}, "example": _CSV_EXAMPLE},
    MEDIA_TYPES["xlsx"]: {"schema": {"type": "string", "format": "binary"}},
}


def result_body(result: ExtractionResult) -> dict[str, Any]:
    """Result as JSON; `ocr_text` only when it was asked for. Null fields are kept."""
    body = result.model_dump(mode="json")
    if body["ocr_text"] is None:
        del body["ocr_text"]
    return body


def note_usage(request: Request, result: ExtractionResult) -> None:
    """Metadata for the access log and Server-Timing."""
    usage = result.usage
    request.state.pages = usage.pages
    request.state.ocr_ms = usage.ocr_ms
    request.state.llm_ms = usage.llm_ms
    request.state.model = usage.model


@router.get("/api/v1/demo", operation_id="demo_info", summary="Limits of this demo deployment")
def demo_info(settings: SettingsDep) -> DemoInfo:
    return DemoInfo(
        max_upload_bytes=settings.max_upload_bytes,
        max_pages=settings.max_pages,
        daily_documents=settings.demo_daily_documents,
    )


@router.post(
    "/api/v1/extract",
    operation_id="extract_document",
    summary="Extract fields from a document",
    response_model=ExtractResponse,
    responses={200: {"content": FILE_RESPONSES}, **ERROR_RESPONSES, 504: {"model": Problem}},
)
async def extract_document(
    request: Request,
    settings: SettingsDep,
    pipeline: PipelineDep,
    executor: ExecutorDep,
    result_format: str | None = FORMAT_QUERY,
    include_meta: bool = INCLUDE_META_QUERY,
    stream: bool = STREAM_QUERY,
) -> Response:
    """Send a document with a `preset`, a `schema`, or both (the schema extends the preset).

    The document is processed in memory and forgotten: nothing is written to disk, logged or
    kept. The demo takes small documents (see `GET /api/v1/demo`) and a few per day per address.

    The result comes as JSON (with confidence, evidence, timings and the detected language),
    CSV or XLSX (`?format=` or `Accept`). With `?stream=true` the JSON result is preceded
    by progress events (NDJSON), so a client can show what is happening while it waits.
    """
    fmt = negotiate(result_format, request.headers.get("accept"))  # 406 before any work
    incoming = await read_extraction_request(request, settings.max_upload_bytes)
    request.state.bytes = len(incoming.document)
    resolved = resolve_schema(incoming.schema, incoming.preset, settings)
    compile_schema(resolved, settings)  # a bad schema costs nothing from the daily limit
    pages = count_pages(incoming.document)
    if pages > settings.max_pages:
        raise too_large(
            f"The document has {pages} pages; the demo takes up to {settings.max_pages}."
        )
    take_daily(
        request, "extract", settings.demo_daily_documents, settings.demo_global_daily_documents
    )

    work = partial(
        pipeline.extract,
        incoming.document,
        None,
        incoming.schema,
        incoming.options,
        preset=incoming.preset,
    )
    if stream and fmt == "json":
        executor.ensure_free()  # still a plain 429 while no byte of the stream has gone out
        accepted = {"bytes": len(incoming.document), "pages": pages}
        events = stream_extraction(request, work, executor, settings.sync_timeout_s, accepted)
        return StreamingResponse(
            events,
            media_type=NDJSON,
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )
    result: ExtractionResult = await executor.run_now(work, settings.sync_timeout_s)
    note_usage(request, result)
    body = result_body(result)
    if fmt == "json":
        return JSONResponse(body)
    return render(fmt, body, resolved, include_meta=include_meta, filename="extraction")


def _line(event: str, **payload: Any) -> bytes:
    return (json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n").encode()


async def stream_extraction(
    request: Request,
    work: Callable[..., ExtractionResult],
    executor: Executor,
    timeout_s: float,
    accepted: dict[str, Any],
) -> AsyncIterator[bytes]:
    """Progress events from the pipeline thread, then the result (or the problem) as the last line.

    The status is already 200 once the first line is out, so a failure arrives as an `error`
    event carrying the same problem body a plain request would get.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue()

    def progress(stage: str, details: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, _line(stage, **details))

    yield _line("accepted", **accepted)
    task = asyncio.create_task(executor.run_now(partial(work, progress=progress), timeout_s))
    while not task.done():
        waiter = asyncio.ensure_future(queue.get())
        await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if waiter.done():
            yield waiter.result()
        else:
            waiter.cancel()
    while not queue.empty():
        yield queue.get_nowait()

    request_id = getattr(request.state, "request_id", None)
    try:
        result: ExtractionResult = task.result()
    except ApiError as error:
        request.state.error_code = error.slug
        yield _line("error", problem=problem_body(error, request_id))
        return
    except SchemaLensError as error:
        problem = from_core_error(error)
        request.state.error_code = problem.slug
        yield _line("error", problem=problem_body(problem, request_id))
        return
    except Exception as error:
        # The type only: messages may quote the document.
        logger.error(json.dumps({"event": "stream_failed", "error": type(error).__name__}))
        request.state.error_code = INTERNAL_ERROR.slug
        yield _line("error", problem=problem_body(INTERNAL_ERROR, request_id))
        return
    note_usage(request, result)
    yield _line("result", result=result_body(result))


@router.post(
    "/api/v1/render",
    operation_id="render_result",
    summary="Turn a result into CSV or XLSX",
    responses={200: {"content": FILE_RESPONSES}, 406: {"model": Problem}, 422: {"model": Problem}},
)
async def render_result(
    request: Request,
    payload: RenderRequest,
    settings: SettingsDep,
    result_format: str | None = FORMAT_QUERY,
    include_meta: bool = INCLUDE_META_QUERY,
) -> Response:
    """The export buttons of the demo: the result the browser already has, as a file. No
    extraction runs again, and nothing is stored or logged."""
    fmt = negotiate(result_format, request.headers.get("accept"))
    resolved = resolve_schema(payload.json_schema, payload.preset, settings)
    compile_schema(resolved, settings)
    body = result_body(payload.result)
    if fmt == "json":
        return JSONResponse(
            body, headers={"Content-Disposition": 'attachment; filename="extraction.json"'}
        )
    return render(
        fmt,
        body,
        resolved,
        include_meta=include_meta,
        filename="extraction",
        labels=payload.labels,
    )
