import asyncio

from fastapi import APIRouter, Request

from schemalens_api.models import Health

router = APIRouter(tags=["service"])


@router.get("/healthz", operation_id="healthz", summary="Liveness check")
async def healthz() -> Health:
    return Health()


def _warm(request: Request) -> None:
    from schemalens_core.pipeline import build_pipeline

    state = request.app.state
    if state.pipeline is None:
        state.pipeline = build_pipeline(state.settings)
    state.pipeline.warm_up()


@router.post("/internal/warm-up", include_in_schema=False)
async def warm_up(request: Request) -> dict[str, bool]:
    """The scheduled EventBridge ping on AWS (the Lambda Web Adapter posts non-HTTP events
    here): builds the pipeline and loads the OCR models, so a visitor rarely waits for a cold
    start. Harmless if called by anyone: it only loads what a first request would load."""
    await asyncio.to_thread(_warm, request)
    return {"warm": True}
