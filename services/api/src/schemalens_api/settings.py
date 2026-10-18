"""API settings: the core settings plus what only the HTTP adapter needs."""

from functools import cache

from pydantic import SecretStr

from schemalens_core.config import Settings


class ApiSettings(Settings):
    # Every /v1 call needs a key: `Authorization: Bearer <key>` or `X-API-Key`. Comma-separated;
    # generate one with `schemalens-api keys new`. The API refuses to start without any.
    # api_keys: for your systems; the daily limit counts per key (`key_daily_documents`, 0 = none).
    api_keys: SecretStr = SecretStr("")
    key_daily_documents: int = 0
    # demo_keys: for a public web page, where anyone can read the key in the browser. It only
    # says which page calls, so its calls are limited per client address and in total (below).
    demo_keys: SecretStr = SecretStr("")
    # On AWS, API Gateway checks the key and applies the usage plan: the function accepts what the
    # gateway let through (the request context carries the key id). Needs lambda_request_context.
    gateway_keys: bool = False

    max_upload_bytes: int = 4 * 1024 * 1024  # Lambda Function URL bodies stop at 6 MB
    max_pages: int = 3
    sync_timeout_s: float = 60.0
    # Calls with a demo key: per client address and all visitors together, per UTC day.
    demo_daily_documents: int = 10
    demo_global_daily_documents: int = 500

    # Browser origins allowed to call the API (the landing site), comma-separated.
    allowed_origins: str = "http://localhost:3000"

    # Rate limits: in memory locally, a DynamoDB table (counters only) on AWS.
    limits_table: str = ""  # empty = in memory
    # Client addresses are hashed with this salt before they are counted; never stored plain.
    ip_salt: SecretStr = SecretStr("")
    aws_region: str = "eu-central-1"
    # On AWS the app runs behind the Lambda Web Adapter: the client address is in the
    # `x-amzn-request-context` header the adapter fills in from the invocation. Off elsewhere,
    # where any client could send that header.
    lambda_request_context: bool = False

    # OCR is CPU-bound: how many extractions may run at once, and what to tell the rest
    max_concurrent_jobs: int = 2
    busy_retry_after_s: int = 5

    def key_list(self) -> list[str]:
        return _split(self.api_keys.get_secret_value())

    def demo_key_list(self) -> list[str]:
        return _split(self.demo_keys.get_secret_value())

    def keys_configured(self) -> bool:
        """Whether any caller can get in."""
        gateway = self.gateway_keys and self.lambda_request_context
        return bool(self.key_list() or self.demo_key_list() or gateway)

    def origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@cache
def get_api_settings() -> ApiSettings:
    return ApiSettings()
