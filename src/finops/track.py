"""Drop-in instrumentation for the Anthropic SDK client.

Wrap your client once and every call lands in the ledger with its cost:

    from anthropic import Anthropic
    from finops import TrackedClient

    client = TrackedClient(Anthropic(), tags={"service": "support-bot"})
    msg = client.messages.create(model="claude-opus-5", max_tokens=1024, ...)

The wrapper is transparent: unknown attributes pass through to the underlying
client, and the return values are the SDK's own objects. Recording failures are
swallowed by design -- a broken ledger must never take down a request path.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

from .ledger import Ledger, UsageRecord
from .usage import TokenUsage, _get

__all__ = ["TrackedClient", "TrackedMessages"]


class _TrackedStream:
    """Context manager that records usage once a stream completes."""

    def __init__(self, inner, recorder, tags: dict[str, str]) -> None:
        self._inner = inner
        self._recorder = recorder
        self._tags = tags
        self._stream = None
        self._started = 0.0

    def __enter__(self):
        self._started = time.perf_counter()
        self._stream = self._inner.__enter__()
        return self._stream

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None and self._stream is not None:
                message = self._stream.get_final_message()
                self._recorder(message, self._tags, self._started)
        except Exception as err:  # pragma: no cover - defensive
            warnings.warn(f"finops: could not record streamed usage: {err}", stacklevel=2)
        return self._inner.__exit__(exc_type, exc, tb)


class TrackedMessages:
    """Proxy around ``client.messages`` that records every completed call."""

    def __init__(self, inner, ledger: Ledger, tags: dict[str, str], batch: bool) -> None:
        self._inner = inner
        self._ledger = ledger
        self._tags = tags
        self._batch = batch

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _record(self, message, tags: dict[str, str], started: float) -> UsageRecord | None:
        try:
            return self._ledger.record(
                getattr(message, "model", None) or "unknown",
                TokenUsage.from_api(getattr(message, "usage", None)),
                tags=tags,
                request_id=getattr(message, "id", None),
                duration_ms=(time.perf_counter() - started) * 1000,
                batch=self._batch,
                speed=_get(getattr(message, "usage", None), "speed", "standard") or "standard",
            )
        except Exception as err:
            # Never let accounting break the caller's request.
            warnings.warn(f"finops: could not record usage: {err}", stacklevel=2)
            return None

    def _call_tags(self, extra: dict[str, str] | None) -> dict[str, str]:
        return {**self._tags, **(extra or {})}

    def create(self, *args, finops_tags: dict[str, str] | None = None, **kwargs):
        tags = self._call_tags(finops_tags)
        started = time.perf_counter()
        message = self._inner.create(*args, **kwargs)
        self._record(message, tags, started)
        return message

    def stream(self, *args, finops_tags: dict[str, str] | None = None, **kwargs):
        return _TrackedStream(
            self._inner.stream(*args, **kwargs),
            lambda message, tags, started: self._record(message, tags, started),
            self._call_tags(finops_tags),
        )

    def parse(self, *args, finops_tags: dict[str, str] | None = None, **kwargs):
        tags = self._call_tags(finops_tags)
        started = time.perf_counter()
        message = self._inner.parse(*args, **kwargs)
        self._record(message, tags, started)
        return message


class TrackedClient:
    """A cost-aware wrapper around an Anthropic client.

    ``tags`` are attached to every call made through this wrapper; use
    :meth:`tagged` to derive a narrower-scoped client for a subsystem.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        ledger: Ledger | str | None = None,
        tags: dict[str, str] | None = None,
        batch: bool = False,
    ) -> None:
        if client is None:
            from anthropic import Anthropic  # optional dependency, imported lazily

            client = Anthropic()
        self._client = client
        self.ledger = ledger if isinstance(ledger, Ledger) else Ledger(ledger)
        self.tags = dict(tags or {})
        self._batch = batch
        self._messages: TrackedMessages | None = None

    @property
    def messages(self) -> TrackedMessages:
        if self._messages is None:
            self._messages = TrackedMessages(
                self._client.messages, self.ledger, self.tags, self._batch
            )
        return self._messages

    def tagged(self, **tags: str) -> "TrackedClient":
        """A new wrapper over the same client and ledger with extra tags."""
        return TrackedClient(
            self._client,
            ledger=self.ledger,
            tags={**self.tags, **tags},
            batch=self._batch,
        )

    def report(self, **kwargs):
        return self.ledger.report(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
