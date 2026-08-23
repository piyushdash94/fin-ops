"""finops -- token accounting and cost control for Claude-powered applications.

Four things, each usable on its own:

* :mod:`finops.usage` -- what a request cost, and what caching saved.
* :mod:`finops.tokens` -- exact token counts (memoized) with an honest fallback.
* :mod:`finops.budget` -- fit context into a window; order it for cache hits.
* :mod:`finops.ledger` -- an append-only usage log and reports over it.

:class:`~finops.track.TrackedClient` ties them together: wrap your Anthropic
client once and every call is priced and logged.
"""

from .budget import BudgetExceeded, ContextBudget, Item, PackResult, order_for_cache
from .ledger import Ledger, Report, UsageRecord
from .models import ModelSpec, Rates, UnknownModel, all_models, get as get_model, register
from .tokens import Estimate, TokenCounter, estimate_tokens
from .track import TrackedClient
from .usage import (
    CostBreakdown,
    TokenUsage,
    cache_breakeven_requests,
    cache_projection,
    price,
)

__version__ = "0.1.0"

__all__ = [
    "BudgetExceeded",
    "ContextBudget",
    "CostBreakdown",
    "Estimate",
    "Item",
    "Ledger",
    "ModelSpec",
    "PackResult",
    "Rates",
    "Report",
    "TokenCounter",
    "TokenUsage",
    "TrackedClient",
    "UnknownModel",
    "UsageRecord",
    "all_models",
    "cache_breakeven_requests",
    "cache_projection",
    "estimate_tokens",
    "get_model",
    "order_for_cache",
    "price",
    "register",
    "__version__",
]
