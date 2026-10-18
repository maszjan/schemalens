"""Per-address daily limits for the public demo.

Only counters are kept: a salted hash of the client address, a day and a number, expiring
after two days. Locally they live in memory; on AWS in one DynamoDB table (on-demand, TTL).
"""

import hashlib
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from schemalens_api.settings import ApiSettings

_TTL_S = 2 * 24 * 3600


class Limiter(Protocol):
    def take(self, key: str, limit: int) -> bool:
        """Count one use of `key` today; False (and nothing counted) when the limit is reached."""
        ...


def day(clock: Callable[[], float] = time.time) -> str:
    return datetime.fromtimestamp(clock(), UTC).strftime("%Y-%m-%d")


def client_key(kind: str, address: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{address}".encode()).hexdigest()[:32]
    return f"{kind}:{digest}"


class MemoryLimiter:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._counts: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    def take(self, key: str, limit: int) -> bool:
        today = day(self._clock)
        with self._lock:
            self._counts = {k: v for k, v in self._counts.items() if k[1] == today}
            count = self._counts.get((key, today), 0)
            if count >= limit:
                return False
            self._counts[(key, today)] = count + 1
            return True


class DynamoLimiter:
    """One item per key and day: an atomic conditional increment, so concurrent Lambdas agree."""

    def __init__(
        self, settings: ApiSettings, client: Any = None, clock: Callable[[], float] = time.time
    ) -> None:
        if client is None:
            import boto3  # type: ignore[import-untyped]

            client = boto3.client("dynamodb", region_name=settings.aws_region)
        self._client = client
        self._table = settings.limits_table
        self._clock = clock

    def take(self, key: str, limit: int) -> bool:
        from botocore.exceptions import ClientError  # type: ignore[import-untyped]

        try:
            self._client.update_item(
                TableName=self._table,
                Key={"pk": {"S": f"{key}#{day(self._clock)}"}},
                UpdateExpression="ADD uses :one SET expires_at = if_not_exists(expires_at, :ttl)",
                ConditionExpression="attribute_not_exists(uses) OR uses < :limit",
                ExpressionAttributeValues={
                    ":one": {"N": "1"},
                    ":limit": {"N": str(limit)},
                    ":ttl": {"N": str(int(self._clock()) + _TTL_S)},
                },
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                return False
            raise
        return True


def build_limiter(settings: ApiSettings) -> Limiter:
    return DynamoLimiter(settings) if settings.limits_table else MemoryLimiter()
