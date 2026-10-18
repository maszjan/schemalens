"""API tests run the real app over ASGI with a fake pipeline: no OCR, no LLM, no network."""

from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from api_support import API_KEY, FakePipeline

from schemalens_api.deps import get_pipeline
from schemalens_api.main import create_app
from schemalens_api.settings import ApiSettings


@pytest.fixture
def make_settings(tmp_path) -> Callable[..., ApiSettings]:
    def build(**overrides: Any) -> ApiSettings:
        values: dict[str, Any] = {
            "active_llm_profile": "local",
            "api_keys": API_KEY,
        }
        return ApiSettings(_env_file=None, **{**values, **overrides})

    return build


@pytest.fixture
def settings(make_settings: Callable[..., ApiSettings]) -> ApiSettings:
    return make_settings()


@pytest.fixture
def pipeline(settings: ApiSettings) -> FakePipeline:
    return FakePipeline(settings)


@pytest.fixture
def app(settings: ApiSettings, pipeline: FakePipeline):
    application = create_app(settings)
    application.dependency_overrides[get_pipeline] = lambda: pipeline
    return application


@pytest.fixture
async def client(app) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as http:
        yield http
