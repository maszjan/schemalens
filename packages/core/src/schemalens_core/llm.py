"""LLM access: OpenAI-compatible client (Groq / Ollama) and extraction with a validation retry."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import openai

from schemalens_core.config import LlmProfile, Settings
from schemalens_core.errors import LlmFailed, LlmOutputInvalid, LlmRateLimited
from schemalens_core.models import FieldMeta, OcrLine
from schemalens_core.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_retry_prompt,
    build_user_prompt,
)
from schemalens_core.schemas import CompiledSchema, ParsedExtraction
from schemalens_core.scoring import score_fields


@dataclass(frozen=True)
class LlmResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    llm_ms: int
    model: str
    reasoning_effort: str | None = None  # the effort that produced this answer, if any was sent
    escalations: int = 0  # times the effort was raised after a failed constrained generation


class LlmClient(Protocol):
    def extract(
        self, *, system: str, user: str, json_schema: dict[str, Any], model: str
    ) -> LlmResponse: ...


class OpenAICompatibleClient:
    """Chat Completions with `response_format: json_schema` against one LLM profile."""

    def __init__(
        self,
        settings: Settings,
        profile: LlmProfile | None = None,
        *,
        http_client: Any = None,  # the SDK's HTTP client type; injected in tests
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._profile = profile or settings.llm_profile()
        self._sleep = sleep
        self._client = openai.OpenAI(
            base_url=self._profile.base_url,
            api_key=self._profile.api_key.get_secret_value(),
            timeout=self._profile.timeout_s,
            max_retries=0,  # retries are handled in extract()
            http_client=http_client,
        )

    def _retry_delay(self, exc: openai.APIError, attempt: int) -> float | None:
        """Seconds to wait before another attempt, or None when the error is not retryable."""
        backoff: float = self._settings.llm_retry_backoff_s * 2**attempt
        if isinstance(exc, openai.RateLimitError):
            asked = _retry_after(exc)
            if asked is None:
                return backoff
            # A longer wait means a quota (e.g. tokens per day), not a burst: fail now
            # instead of sleeping through attempts that cannot succeed.
            return asked if asked <= self._settings.llm_retry_after_max_s else None
        if isinstance(exc, openai.APITimeoutError) and self._profile.name == "local":
            # A local model that ran out of time is overloaded or the request is too big;
            # asking again only multiplies the wait.
            return None
        if isinstance(exc, openai.InternalServerError | openai.APITimeoutError):
            return backoff
        if _generation_failed(exc):
            return 0.0
        return None

    def extract(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any],
        model: str,
    ) -> LlmResponse:
        response_format: Any = {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction",
                "strict": self._profile.strict_schema,
                "schema": json_schema,
            },
        }
        # Only reasoning models accept the parameter; others reject the request.
        supported = any(part in model for part in self._settings.reasoning_effort_models)
        effort: str | None = self._profile.reasoning_effort if supported else None
        escalations = 0
        request_seconds = 0.0  # time spent in requests; waiting between attempts is not LLM time
        for attempt in range(self._settings.llm_max_retries + 1):
            started = time.perf_counter()
            options: Any = {"reasoning_effort": effort} if effort else {}
            if self._profile.max_output_tokens:
                options["max_tokens"] = self._profile.max_output_tokens
            options["response_format"] = response_format
            messages: Any = [{"role": "user", "content": user}]
            if system:
                messages.insert(0, {"role": "system", "content": system})
            try:
                completion = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    **options,
                )
                request_seconds += time.perf_counter() - started
                break
            except openai.APIError as exc:
                request_seconds += time.perf_counter() - started
                delay = self._retry_delay(exc, attempt)
                if delay is None or attempt == self._settings.llm_max_retries:
                    message = _describe_error(exc, attempt + 1)
                    if isinstance(exc, openai.RateLimitError):
                        raise LlmRateLimited(message, _retry_after(exc)) from None
                    raise LlmFailed(message) from None
                if delay:
                    self._sleep(delay)
                if _generation_failed(exc) and (higher := _next_effort(effort)):
                    # More reasoning makes the model follow the schema; each level is tried once.
                    effort = higher
                    escalations += 1

        usage = completion.usage
        return LlmResponse(
            content=completion.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            llm_ms=int(request_seconds * 1000),
            model=completion.model,
            reasoning_effort=effort,
            escalations=escalations,
        )


# Groq's constrained decoding can still fail to produce a matching document and answers
# HTTP 400 with this code. A retry with a higher reasoning effort usually succeeds.
_GENERATION_FAILED = "json_validate_failed"
_EFFORTS = ("low", "medium", "high")


def _generation_failed(exc: openai.APIError) -> bool:
    return isinstance(exc, openai.BadRequestError) and exc.code == _GENERATION_FAILED


def _next_effort(effort: str | None) -> str | None:
    if effort not in _EFFORTS[:-1]:
        return None
    return _EFFORTS[_EFFORTS.index(effort) + 1]


def _retry_after(exc: openai.RateLimitError) -> float | None:
    try:
        return float(exc.response.headers.get("retry-after", ""))
    except ValueError:
        return None


def _describe_error(exc: openai.APIError, attempts: int) -> str:
    # Type and status only: provider error bodies can echo the prompt (document content).
    status = getattr(exc, "status_code", None)
    detail = f"{type(exc).__name__}, HTTP {status}" if status else type(exc).__name__
    if isinstance(exc, openai.RateLimitError):
        retry_after = exc.response.headers.get("retry-after")
        if retry_after:
            detail += f", provider asks to retry after {retry_after} s"
    return f"LLM request failed after {attempts} attempt(s) ({detail})"


@dataclass(frozen=True)
class LlmUsage:
    model: str
    prompt_tokens: int
    completion_tokens: int
    llm_ms: int
    calls: int
    prompt_version: str
    reasoning_effort: str | None
    escalations: int


@dataclass(frozen=True)
class LlmExtraction:
    data: dict[str, Any]
    fields: dict[str, FieldMeta]
    warnings: list[str]
    usage: LlmUsage


def request_with_retry(
    send: Callable[[list[str] | None], LlmResponse],
    schema: CompiledSchema,
) -> tuple[ParsedExtraction, list[LlmResponse], frozenset[str]]:
    """Ask once; if the answer fails validation, ask again with the errors.

    `send(errors)` performs a request (errors of the previous answer, or None the first
    time). Returns the parsed answer, every response received and the fields that needed the
    retry. Fields still
    failing after the retry come back as `None` with a warning.
    """
    responses = [send(None)]
    try:
        return schema.parse(responses[0].content), responses, frozenset()
    except LlmOutputInvalid as first:
        # Fields that failed the first time; a response of the wrong shape taints all of them.
        retried = first.fields or frozenset(schema.source["properties"])
        responses.append(send(first.errors))
        try:
            return schema.parse(responses[1].content, drop_invalid=True), responses, retried
        except LlmOutputInvalid as second:
            raise LlmFailed(
                f"LLM output unusable after retry: {'; '.join(second.errors)}"
            ) from None


def usage_of(responses: list[LlmResponse]) -> "LlmUsage":
    return LlmUsage(
        model=responses[-1].model,
        prompt_tokens=sum(r.prompt_tokens for r in responses),
        completion_tokens=sum(r.completion_tokens for r in responses),
        llm_ms=sum(r.llm_ms for r in responses),
        calls=len(responses),
        prompt_version=PROMPT_VERSION,
        reasoning_effort=responses[-1].reasoning_effort,
        escalations=sum(r.escalations for r in responses),
    )


def extract_fields(
    client: LlmClient,
    schema: CompiledSchema,
    lines: list[OcrLine],
    settings: Settings,
    *,
    model: str | None = None,
    profile: LlmProfile | None = None,
    example: str | None = None,
) -> LlmExtraction:
    """Ask the LLM for the schema fields; one more call if the output fails validation.

    Values are normalized before validation, so the retry happens only for what code
    could not fix. Fields still failing after the retry come back as `None` with a warning.
    Evidence is checked against `lines` and confidence is computed here, not by the model.
    """
    profile = profile or settings.llm_profile()
    model = model or profile.model_fast
    if not model:
        raise LlmFailed(f"LLM model is not configured ({profile.name.upper()}_MODEL_FAST)")
    json_schema = schema.llm_schema(strict=profile.strict_schema)
    user = build_user_prompt(lines, schema.source, example)

    def send(errors: list[str] | None) -> LlmResponse:
        prompt = build_retry_prompt(user, errors) if errors else user
        return client.extract(
            system=SYSTEM_PROMPT, user=prompt, json_schema=json_schema, model=model
        )

    parsed, responses, retried = request_with_retry(send, schema)

    fields, evidence_warnings = score_fields(
        schema.source,
        parsed.data,
        parsed.fields,
        lines,
        settings,
        normalized=parsed.normalized,
        retried=retried,
        invalid=parsed.invalid,
    )
    return LlmExtraction(
        data=parsed.data,
        fields=fields,
        warnings=[*parsed.warnings, *evidence_warnings],
        usage=usage_of(responses),
    )
