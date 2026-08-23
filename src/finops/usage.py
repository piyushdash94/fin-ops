"""Token usage accounting and cost math.

The core type is :class:`TokenUsage`, which mirrors the five ways a Claude
request can bill tokens: fresh input, output, cache reads, and cache writes at
each of the two TTLs. :func:`price` turns that into a :class:`CostBreakdown`
that also reports what the same request *would* have cost without caching --
the number that makes a caching strategy worth defending in a budget review.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass, field

from . import models
from .models import (
    BATCH_MULTIPLIER,
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    MTOK,
    ModelSpec,
)

__all__ = [
    "TokenUsage",
    "CostBreakdown",
    "price",
    "cache_breakeven_requests",
    "cache_projection",
]


def _get(obj, name, default=0):
    """Read ``name`` off an SDK object or a plain dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None else value


@dataclass(frozen=True)
class TokenUsage:
    """Tokens billed by a single request, split by billing category."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0

    @classmethod
    def from_api(cls, usage) -> "TokenUsage":
        """Build from an SDK ``message.usage`` object or the equivalent dict.

        Handles both shapes of cache-write reporting: the flat
        ``cache_creation_input_tokens`` total and the per-TTL
        ``cache_creation.{ephemeral_5m,ephemeral_1h}_input_tokens`` breakdown.
        When only the flat total is present it is attributed to the 5m TTL,
        which is the API default.
        """
        creation = _get(usage, "cache_creation", None)
        write_5m = int(_get(creation, "ephemeral_5m_input_tokens", 0))
        write_1h = int(_get(creation, "ephemeral_1h_input_tokens", 0))
        flat = int(_get(usage, "cache_creation_input_tokens", 0))
        if write_5m + write_1h == 0:
            write_5m = flat

        return cls(
            input_tokens=int(_get(usage, "input_tokens", 0)),
            output_tokens=int(_get(usage, "output_tokens", 0)),
            cache_read_tokens=int(_get(usage, "cache_read_input_tokens", 0)),
            cache_write_5m_tokens=write_5m,
            cache_write_1h_tokens=write_1h,
        )

    @property
    def cache_write_tokens(self) -> int:
        return self.cache_write_5m_tokens + self.cache_write_1h_tokens

    @property
    def prompt_tokens(self) -> int:
        """Every input-side token, cached or not -- the context actually sent."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Share of input-side tokens served from cache (0.0 when no input)."""
        return self.cache_read_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_5m_tokens + other.cache_write_5m_tokens,
            self.cache_write_1h_tokens + other.cache_write_1h_tokens,
        )

    __radd__ = __add__

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CostBreakdown:
    """Per-category cost in USD, plus what caching saved."""

    model: str
    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0
    #: What this exact request would have cost with no prompt caching at all.
    uncached_equivalent: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.input_cost
            + self.output_cost
            + self.cache_read_cost
            + self.cache_write_cost
        )

    @property
    def cache_savings(self) -> float:
        """USD saved versus sending the same prompt uncached.

        Negative on a cache-write-heavy request that never gets reused -- that
        is the honest signal that the caching strategy is not paying off.
        """
        return self.uncached_equivalent - self.total

    def as_dict(self) -> dict:
        data = asdict(self)
        data["total"] = self.total
        data["cache_savings"] = self.cache_savings
        return data


def price(
    usage: TokenUsage,
    model: str | ModelSpec,
    *,
    day: dt.date | None = None,
    speed: str = "standard",
    batch: bool = False,
) -> CostBreakdown:
    """Cost a :class:`TokenUsage` against a model's published rates."""
    spec = model if isinstance(model, ModelSpec) else models.get(model)
    rates = spec.rates_on(day, speed=speed)
    discount = BATCH_MULTIPLIER if batch else 1.0
    per_input = rates.input * discount / MTOK
    per_output = rates.output * discount / MTOK

    cache_write_cost = (
        usage.cache_write_5m_tokens * per_input * CACHE_WRITE_MULTIPLIER["5m"]
        + usage.cache_write_1h_tokens * per_input * CACHE_WRITE_MULTIPLIER["1h"]
    )
    return CostBreakdown(
        model=spec.id,
        input_cost=usage.input_tokens * per_input,
        output_cost=usage.output_tokens * per_output,
        cache_read_cost=usage.cache_read_tokens * per_input * CACHE_READ_MULTIPLIER,
        cache_write_cost=cache_write_cost,
        uncached_equivalent=usage.prompt_tokens * per_input
        + usage.output_tokens * per_output,
    )


def cache_breakeven_requests(ttl: str = "5m") -> int:
    """Minimum requests over one cached prefix before caching is cheaper.

    Writing costs ``w x`` the base input rate and each later read costs
    ``0.1 x``, so ``n`` requests cost ``w + 0.1(n-1)`` versus ``n`` uncached.
    Solving ``w + 0.1(n-1) <= n`` gives 2 requests at the 5m TTL and 3 at 1h.
    """
    try:
        write = CACHE_WRITE_MULTIPLIER[ttl]
    except KeyError:
        raise ValueError(f"unknown ttl {ttl!r}; expected one of {sorted(CACHE_WRITE_MULTIPLIER)}") from None
    n = (write - CACHE_READ_MULTIPLIER) / (1 - CACHE_READ_MULTIPLIER)
    return max(1, math.ceil(n - 1e-9))


@dataclass(frozen=True)
class CacheProjection:
    model: str
    ttl: str
    prefix_tokens: int
    requests: int
    uncached_cost: float
    cached_cost: float
    breakeven_requests: int

    @property
    def savings(self) -> float:
        return self.uncached_cost - self.cached_cost

    @property
    def worth_it(self) -> bool:
        return self.savings > 0

    def as_dict(self) -> dict:
        data = asdict(self)
        data["savings"] = self.savings
        data["worth_it"] = self.worth_it
        return data


def cache_projection(
    prefix_tokens: int,
    requests: int,
    model: str | ModelSpec,
    *,
    ttl: str = "5m",
    day: dt.date | None = None,
    writes: int = 1,
) -> CacheProjection:
    """Project the cost of caching a shared prefix across ``requests`` calls.

    ``writes`` is how many times the prefix has to be re-written -- once for a
    steady stream inside the TTL, more if traffic is bursty enough that entries
    expire between bursts.
    """
    if requests < 1:
        raise ValueError("requests must be >= 1")
    if writes < 1 or writes > requests:
        raise ValueError("writes must be between 1 and requests")

    spec = model if isinstance(model, ModelSpec) else models.get(model)
    per_input = spec.rates_on(day).input / MTOK
    uncached = requests * prefix_tokens * per_input
    cached = prefix_tokens * per_input * (
        writes * CACHE_WRITE_MULTIPLIER[ttl]
        + (requests - writes) * CACHE_READ_MULTIPLIER
    )
    return CacheProjection(
        model=spec.id,
        ttl=ttl,
        prefix_tokens=prefix_tokens,
        requests=requests,
        uncached_cost=uncached,
        cached_cost=cached,
        breakeven_requests=cache_breakeven_requests(ttl),
    )
