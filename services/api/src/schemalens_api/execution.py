"""Running the blocking pipeline: off the event loop, a bounded number at a time."""

import asyncio
from collections.abc import Callable
from typing import Any

from schemalens_api.problems import ApiError


class Executor:
    """OCR is CPU-bound, so only `slots` extractions run at once.

    Requests are turned away when every slot is taken.
    """

    def __init__(self, slots: int, retry_after_s: int) -> None:
        self._semaphore = asyncio.Semaphore(slots)
        self._retry_after_s = retry_after_s
        self._tasks: set[asyncio.Task[Any]] = set()

    def ensure_free(self) -> None:
        """429 when every slot is taken."""
        if self._semaphore.locked():
            raise ApiError(
                429,
                "too-many-requests",
                "Too many requests",
                "All extraction slots are busy. Retry shortly.",
                headers={"Retry-After": str(self._retry_after_s)},
            )

    async def run(self, work: Callable[[], Any], on_start: Callable[[], None] | None = None) -> Any:
        """Wait for a slot, then run `work` in a thread."""
        async with self._semaphore:
            if on_start:
                on_start()
            return await asyncio.to_thread(work)

    async def run_now(self, work: Callable[[], Any], timeout_s: float) -> Any:
        """Run `work` if a slot is free (else 429); wait at most `timeout_s` for it (else 504)."""
        self.ensure_free()
        # A thread cannot be cancelled: on timeout the work finishes in the background and
        # keeps its slot until then, so the CPU bound still holds.
        task = asyncio.create_task(self.run(work))
        self._tasks.add(task)
        task.add_done_callback(self._forget)
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout_s)
        except TimeoutError:
            raise ApiError(
                504,
                "extraction-timeout",
                "Extraction timed out",
                f"The document was not processed within {timeout_s:g} s.",
            ) from None

    def _forget(self, task: "asyncio.Task[Any]") -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()  # mark as retrieved; an abandoned result is dropped on purpose
