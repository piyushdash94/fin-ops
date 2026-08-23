"""Model registry: context limits and published pricing for Claude models.

Pricing here is a *cached snapshot* (see ``SNAPSHOT_DATE``). Rates are the
Anthropic first-party API rates in USD per million tokens. Bedrock and Vertex
are partner-operated with separate pricing -- register those yourself with
:func:`register` rather than trusting these numbers.

Deliberate design choice: an unknown model raises :class:`UnknownModel` instead
of falling back to a guessed rate. A cost tool that quietly invents prices is
worse than one that refuses to answer.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

__all__ = [
    "MTOK",
    "CACHE_READ_MULTIPLIER",
    "CACHE_WRITE_MULTIPLIER",
    "BATCH_MULTIPLIER",
    "SNAPSHOT_DATE",
    "Rates",
    "ModelSpec",
    "UnknownModel",
    "get",
    "register",
    "all_models",
    "normalize_model_id",
    "refresh_limits_from_api",
]

MTOK = 1_000_000

#: Prompt-cache pricing, expressed as multipliers of the base *input* rate.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER: dict[str, float] = {"5m": 1.25, "1h": 2.0}
#: The Message Batches API bills at 50% of standard rates.
BATCH_MULTIPLIER = 0.5

#: When the bundled price table was last verified.
SNAPSHOT_DATE = dt.date(2026, 6, 24)

_LIMITS_OVERRIDE = Path(
    os.environ.get("FINOPS_HOME", Path.home() / ".cache" / "finops")
) / "model-limits.json"


class UnknownModel(KeyError):
    """Raised when a model id has no registered pricing."""

    def __init__(self, model_id: str, known: list[str]) -> None:
        self.model_id = model_id
        super().__init__(
            f"no pricing registered for model {model_id!r}. "
            f"Known models: {', '.join(sorted(known))}. "
            f"Use finops.models.register(...) to add custom or partner rates."
        )


@dataclass(frozen=True)
class Rates:
    """USD per million tokens."""

    input: float
    output: float

    def scaled(self, factor: float) -> "Rates":
        return Rates(self.input * factor, self.output * factor)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    context_window: int
    max_output_tokens: int
    rates: Rates
    #: Fast mode (research preview) rates, where the model supports it.
    fast_rates: Rates | None = None
    #: Promotional rates, in effect through ``promo_until`` inclusive.
    promo_rates: Rates | None = None
    promo_until: dt.date | None = None
    aliases: tuple[str, ...] = ()

    def rates_on(
        self, day: dt.date | None = None, *, speed: str = "standard"
    ) -> Rates:
        """Rates in effect on ``day`` (default: today) for the given speed."""
        if speed == "fast":
            if self.fast_rates is None:
                raise ValueError(f"{self.id} does not support fast mode")
            return self.fast_rates
        if speed != "standard":
            raise ValueError(f"unknown speed {speed!r}")
        if self.promo_rates is not None and self.promo_until is not None:
            if (day or dt.date.today()) <= self.promo_until:
                return self.promo_rates
        return self.rates


def _spec(
    model_id: str,
    context_window: int,
    max_output_tokens: int,
    input_rate: float,
    output_rate: float,
    **kw,
) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        rates=Rates(input_rate, output_rate),
        **kw,
    )


_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec, *, overwrite: bool = True) -> ModelSpec:
    """Add or replace a model in the registry (custom or partner pricing)."""
    if not overwrite and spec.id in _REGISTRY:
        raise ValueError(f"{spec.id} already registered")
    _REGISTRY[spec.id] = spec
    for alias in spec.aliases:
        _REGISTRY.setdefault(alias, spec)
    return spec


for _s in (
    _spec("claude-fable-5", 1_000_000, 128_000, 10.00, 50.00),
    _spec("claude-mythos-5", 1_000_000, 128_000, 10.00, 50.00),
    _spec(
        "claude-opus-5",
        1_000_000,
        128_000,
        5.00,
        25.00,
        fast_rates=Rates(10.00, 50.00),
    ),
    _spec(
        "claude-opus-4-8",
        1_000_000,
        128_000,
        5.00,
        25.00,
        fast_rates=Rates(10.00, 50.00),
    ),
    _spec("claude-opus-4-7", 1_000_000, 128_000, 5.00, 25.00),
    _spec("claude-opus-4-6", 1_000_000, 128_000, 5.00, 25.00),
    _spec(
        "claude-sonnet-5",
        1_000_000,
        128_000,
        3.00,
        15.00,
        promo_rates=Rates(2.00, 10.00),
        promo_until=dt.date(2026, 8, 31),
    ),
    _spec("claude-sonnet-4-6", 1_000_000, 128_000, 3.00, 15.00),
    _spec("claude-haiku-4-5", 200_000, 64_000, 1.00, 5.00),
):
    register(_s)


# Provider prefixes seen in the wild; stripped before registry lookup so the
# same ledger can hold first-party, Bedrock and Vertex records.
_PROVIDER_PREFIXES = ("anthropic.", "us.anthropic.", "eu.anthropic.", "apac.anthropic.")


def normalize_model_id(model_id: str) -> str:
    """Map a wire model id onto a registry key.

    Strips provider prefixes (``anthropic.claude-opus-5``) and dated-snapshot
    suffixes (``claude-opus-4-5@20251101``, ``...-20251101``) so historical log
    lines still price correctly.
    """
    name = model_id.strip()
    for prefix in sorted(_PROVIDER_PREFIXES, key=len, reverse=True):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    name = name.split("@", 1)[0]
    if name in _REGISTRY:
        return name
    head, sep, tail = name.rpartition("-")
    if sep and len(tail) == 8 and tail.isdigit():
        name = head
    return name


def get(model_id: str) -> ModelSpec:
    """Look up a model spec, tolerating provider prefixes and date suffixes."""
    if model_id in _REGISTRY:
        return _REGISTRY[model_id]
    name = normalize_model_id(model_id)
    if name not in _REGISTRY:
        raise UnknownModel(model_id, all_model_ids())
    return _REGISTRY[name]


def all_model_ids() -> list[str]:
    return sorted({s.id for s in _REGISTRY.values()})


def all_models() -> list[ModelSpec]:
    seen: dict[str, ModelSpec] = {s.id: s for s in _REGISTRY.values()}
    return [seen[k] for k in sorted(seen)]


def _apply_limit_overrides() -> None:
    try:
        data = json.loads(_LIMITS_OVERRIDE.read_text())
    except (OSError, ValueError):
        return
    for model_id, limits in data.items():
        if model_id not in _REGISTRY:
            continue
        spec = _REGISTRY[model_id]
        register(
            replace(
                spec,
                context_window=int(limits.get("context_window", spec.context_window)),
                max_output_tokens=int(
                    limits.get("max_output_tokens", spec.max_output_tokens)
                ),
            )
        )


def refresh_limits_from_api(client=None) -> dict[str, dict[str, int]]:
    """Refresh context/output limits from the Models API and cache them.

    Only *limits* are refreshed. The Models API does not publish pricing, so
    rates are never auto-updated -- that stays an explicit human decision.
    """
    if client is None:  # pragma: no cover - requires network
        from anthropic import Anthropic

        client = Anthropic()

    updates: dict[str, dict[str, int]] = {}
    for model in client.models.list():
        model_id = getattr(model, "id", None)
        if model_id not in _REGISTRY:
            continue
        limits = {}
        max_input = getattr(model, "max_input_tokens", None)
        max_output = getattr(model, "max_tokens", None)
        if max_input:
            limits["context_window"] = int(max_input)
        if max_output:
            limits["max_output_tokens"] = int(max_output)
        if limits:
            updates[_REGISTRY[model_id].id] = limits

    if updates:
        _LIMITS_OVERRIDE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _LIMITS_OVERRIDE.with_suffix(".tmp")
        tmp.write_text(json.dumps(updates, indent=2, sort_keys=True))
        os.replace(tmp, _LIMITS_OVERRIDE)
        _apply_limit_overrides()
    return updates


_apply_limit_overrides()
