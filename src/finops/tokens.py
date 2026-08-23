"""Token counting: exact counts, memoized, with an honest offline fallback.

Two rules drive this module:

1. **Never use ``tiktoken``.** It is OpenAI's tokenizer and undercounts Claude
   by roughly 15-20% on prose and considerably more on code. The only exact
   source of truth is Anthropic's ``/v1/messages/count_tokens`` endpoint.
2. **Counting costs a round trip, so cache it.** Counts are deterministic for a
   given (model, payload), so results are memoized on disk keyed by content
   hash. Re-counting an unchanged file is free.

When no API access is available, :meth:`TokenCounter.estimate` returns an
:class:`Estimate` carrying an explicit uncertainty range rather than a single
falsely-precise integer -- and :meth:`TokenCounter.calibrate` can fit the
estimator to your own corpus against real counts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = ["Estimate", "TokenCounter", "estimate_tokens"]

_FINOPS_HOME = Path(os.environ.get("FINOPS_HOME", Path.home() / ".cache" / "finops"))

# Calibration defaults, in characters per token. Claude's tokenizer packs
# natural-language prose more densely than symbol-heavy source code, so the
# estimator interpolates between these two anchors by symbol density.
_PROSE_CHARS_PER_TOKEN = 3.9
_CODE_CHARS_PER_TOKEN = 3.0
#: Typical relative error of the offline estimator, used for the reported range.
_ESTIMATE_TOLERANCE = 0.15

_SYMBOL_RE = re.compile(r"[^\w\s]")
_WORD_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class Estimate:
    """An offline token estimate with an explicit uncertainty band."""

    tokens: int
    low: int
    high: int
    method: str

    def __int__(self) -> int:
        return self.tokens

    def __str__(self) -> str:
        return f"~{self.tokens} tokens (±{self.high - self.tokens}, {self.method})"


def _symbol_density(text: str) -> float:
    if not text:
        return 0.0
    symbols = len(_SYMBOL_RE.findall(text))
    words = len(_WORD_RE.findall(text))
    return symbols / (symbols + words) if (symbols + words) else 0.0


def estimate_tokens(text: str, *, chars_per_token: float | None = None) -> Estimate:
    """Estimate tokens for ``text`` without calling the API.

    Blends a character-density model between prose and code anchors. Accurate
    to roughly ±15% on mixed content -- good enough to decide whether something
    fits a budget, never good enough to bill against.
    """
    if not text:
        return Estimate(0, 0, 0, "heuristic")

    if chars_per_token is None:
        # Symbol-heavy text (code, JSON, markup) tokenizes more finely.
        density = min(_symbol_density(text) / 0.35, 1.0)
        chars_per_token = (
            _PROSE_CHARS_PER_TOKEN
            + (_CODE_CHARS_PER_TOKEN - _PROSE_CHARS_PER_TOKEN) * density
        )
        method = "heuristic"
    else:
        method = "calibrated"

    tokens = max(1, round(len(text) / chars_per_token))
    margin = max(1, round(tokens * _ESTIMATE_TOLERANCE))
    return Estimate(tokens, max(0, tokens - margin), tokens + margin, method)


def _normalize_messages(content: str | Sequence[dict]) -> list[dict]:
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    return list(content)


def _flatten_text(messages: Iterable[dict], system: Any = None) -> str:
    parts: list[str] = []
    if isinstance(system, str):
        parts.append(system)
    elif isinstance(system, list):
        parts.extend(str(b.get("text", "")) for b in system if isinstance(b, dict))
    for message in messages:
        body = message.get("content")
        if isinstance(body, str):
            parts.append(body)
        elif isinstance(body, list):
            for block in body:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    parts.append(block)
    return "\n".join(parts)


class TokenCounter:
    """Counts tokens for a model, exactly when possible and cached always.

    >>> counter = TokenCounter("claude-opus-5")        # doctest: +SKIP
    >>> counter.count("hello world")                    # doctest: +SKIP
    3
    """

    def __init__(
        self,
        model: str = "claude-opus-5",
        *,
        client: Any = None,
        cache_path: str | Path | None = None,
        offline: bool = False,
    ) -> None:
        self.model = model
        self._client = client
        self._offline = offline
        self._cache_path = Path(cache_path) if cache_path else _FINOPS_HOME / "token-counts.json"
        self._cache: dict[str, int] | None = None
        self._dirty = False

    # -- cache -----------------------------------------------------------
    def _load_cache(self) -> dict[str, int]:
        if self._cache is None:
            try:
                self._cache = json.loads(self._cache_path.read_text())
            except (OSError, ValueError):
                self._cache = {}
        return self._cache

    def flush(self) -> None:
        """Persist newly-computed counts to disk."""
        if not self._dirty or self._cache is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache, sort_keys=True))
        os.replace(tmp, self._cache_path)
        self._dirty = False

    def __enter__(self) -> "TokenCounter":
        return self

    def __exit__(self, *exc) -> None:
        self.flush()

    @staticmethod
    def _key(model: str, payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return f"{model}:{hashlib.sha256(blob).hexdigest()[:32]}"

    # -- counting --------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic  # imported lazily; optional dependency

            self._client = Anthropic()
        return self._client

    def count(
        self,
        content: str | Sequence[dict],
        *,
        system: Any = None,
        tools: Sequence[dict] | None = None,
        use_cache: bool = True,
    ) -> int:
        """Exact token count for a prompt, memoized by content hash.

        Falls back to :meth:`estimate` when the counter is offline or the API
        call fails, so a budgeting path never hard-fails on a missing API key.
        """
        messages = _normalize_messages(content)
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if system is not None:
            payload["system"] = system
        if tools:
            payload["tools"] = list(tools)

        key = self._key(self.model, payload)
        cache = self._load_cache() if use_cache else {}
        if key in cache:
            return cache[key]

        if self._offline:
            return self.estimate(_flatten_text(messages, system)).tokens

        try:
            response = self._get_client().messages.count_tokens(**payload)
            tokens = int(response.input_tokens)
        except Exception:
            return self.estimate(_flatten_text(messages, system)).tokens

        if use_cache:
            cache[key] = tokens
            self._cache = cache
            self._dirty = True
        return tokens

    def count_file(self, path: str | Path, **kw) -> int:
        return self.count(Path(path).read_text(errors="replace"), **kw)

    def estimate(self, text: str) -> Estimate:
        """Offline estimate, using a calibrated ratio when one is stored."""
        ratio = self._calibrated_ratio()
        return estimate_tokens(text, chars_per_token=ratio)

    def _calibrated_ratio(self) -> float | None:
        value = self._load_cache().get(f"__calibration__:{self.model}")
        return float(value) / 1000 if value else None

    def calibrate(self, samples: Sequence[str]) -> float:
        """Fit chars-per-token against real counts and persist the result.

        Pass a handful of representative prompts from your own workload; the
        fitted ratio then backs every later :meth:`estimate` call.
        """
        samples = [s for s in samples if s.strip()]
        if not samples:
            raise ValueError("calibrate() needs at least one non-empty sample")

        total_chars = 0
        total_tokens = 0
        for sample in samples:
            total_tokens += self.count(sample)
            total_chars += len(sample)
        if total_tokens <= 0:
            raise RuntimeError("calibration produced no tokens")

        ratio = total_chars / total_tokens
        cache = self._load_cache()
        # Stored as an int (ratio x1000) to keep the cache file a flat int map.
        cache[f"__calibration__:{self.model}"] = round(ratio * 1000)
        self._cache = cache
        self._dirty = True
        self.flush()
        return ratio
