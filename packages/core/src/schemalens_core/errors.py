"""Exception hierarchy. Messages must never contain document content."""


class SchemaLensError(Exception):
    code = "internal_error"


class InvalidSchema(SchemaLensError):
    """Client schema rejected; `errors` is a human-readable list for the 422 response."""

    code = "invalid_schema"

    def __init__(self, errors: list[str]) -> None:
        super().__init__(f"invalid schema: {len(errors)} error(s)")
        self.errors = errors


class LlmOutputInvalid(SchemaLensError):
    """LLM response failed validation. `errors` holds field paths and messages, never values."""

    code = "llm_output_invalid"

    def __init__(self, errors: list[str], fields: frozenset[str] = frozenset()) -> None:
        super().__init__(f"LLM output failed validation: {len(errors)} error(s)")
        self.errors = errors
        self.fields = fields  # top-level fields at fault; empty when the whole shape is wrong


class LlmFailed(SchemaLensError):
    code = "llm_failed"


class LlmRateLimited(LlmFailed):
    """The LLM provider refused the request because of a rate limit or an exhausted quota."""

    code = "llm_rate_limited"

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after  # seconds, as asked by the provider


class ProfileUnavailable(SchemaLensError):
    code = "profile_unavailable"


class UnsupportedDocument(SchemaLensError):
    code = "unsupported_document"


class OcrFailed(SchemaLensError):
    code = "ocr_failed"


class DocumentUnreadable(UnsupportedDocument):
    """A supported file type whose content cannot be decoded (corrupted, encrypted, empty)."""

    code = "document_unreadable"
