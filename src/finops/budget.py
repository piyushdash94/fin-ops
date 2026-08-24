"""Context budgeting: decide what fits, and order it so the cache can hit.

Two problems this solves:

*What fits.* A context window is a hard wall, and the failure mode is a 400 in
production rather than a graceful degradation. :class:`ContextBudget` reserves
room for output and system overhead, then packs prioritized items into what is
left -- dropping the least valuable content instead of truncating mid-document.

*What order.* Prompt caching is a strict prefix match: one byte changing early
invalidates everything after it. :func:`order_for_cache` sorts stable content
ahead of volatile content so the cacheable prefix is as long as possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

from . import models
from .models import ModelSpec
from .tokens import estimate_tokens

__all__ = ["Item", "PackResult", "ContextBudget", "BudgetExceeded", "order_for_cache"]


class BudgetExceeded(ValueError):
    """Raised when required items alone overflow the budget."""


@dataclass
class Item:
    """A candidate piece of context.

    ``priority`` is a relative value, not a rank -- an item at 10.0 is worth ten
    times one at 1.0 when the packer decides what to drop.
    """

    name: str
    content: str | None = None
    tokens: int = 0
    priority: float = 1.0
    required: bool = False
    #: True for content that is byte-stable across requests (system prompt,
    #: tool definitions, reference docs). False for anything that changes per
    #: request (timestamps, the user's question, retrieved chunks).
    stable: bool = True

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            if self.content is None:
                raise ValueError(f"item {self.name!r} needs content or a token count")
            self.tokens = estimate_tokens(self.content).tokens
        if self.priority < 0:
            raise ValueError("priority must be non-negative")


@dataclass
class PackResult:
    included: list[Item] = field(default_factory=list)
    dropped: list[Item] = field(default_factory=list)
    tokens_used: int = 0
    budget: int = 0

    @property
    def tokens_free(self) -> int:
        return self.budget - self.tokens_used

    @property
    def utilization(self) -> float:
        return self.tokens_used / self.budget if self.budget else 0.0

    def __bool__(self) -> bool:
        return not self.dropped

    def summary(self) -> str:
        head = (
            f"{len(self.included)} items, {self.tokens_used:,}/{self.budget:,} tokens "
            f"({self.utilization:.0%} of budget)"
        )
        if self.dropped:
            names = ", ".join(i.name for i in self.dropped[:5])
            more = f" +{len(self.dropped) - 5} more" if len(self.dropped) > 5 else ""
            head += f"; dropped {len(self.dropped)}: {names}{more}"
        return head


class ContextBudget:
    """A token budget derived from a model's real context window.

    >>> budget = ContextBudget("claude-opus-5", reserve_output=8000)
    >>> budget.available > 900_000
    True
    """

    def __init__(
        self,
        model: str | ModelSpec,
        *,
        reserve_output: int = 8192,
        reserve_overhead: int = 512,
        headroom: float = 0.05,
        limit: int | None = None,
    ) -> None:
        spec = model if isinstance(model, ModelSpec) else models.get(model)
        self.spec = spec
        if reserve_output > spec.max_output_tokens:
            raise ValueError(
                f"reserve_output {reserve_output} exceeds {spec.id} max output "
                f"of {spec.max_output_tokens}"
            )
        if not 0.0 <= headroom < 1.0:
            raise ValueError("headroom must be in [0, 1)")

        window = min(limit, spec.context_window) if limit else spec.context_window
        usable = window - reserve_output - reserve_overhead
        if usable <= 0:
            raise ValueError(
                f"nothing left for context: a {window:,}-token window minus "
                f"{reserve_output:,} reserved for output and {reserve_overhead:,} "
                f"for overhead leaves {usable:,} tokens"
            )
        self.available = int(usable * (1.0 - headroom))
        self.reserve_output = reserve_output
        self.reserve_overhead = reserve_overhead
        self.headroom = headroom

    def fits(self, tokens: int) -> bool:
        return tokens <= self.available

    def pack(self, items: Sequence[Item]) -> PackResult:
        """Fit items into the budget, dropping the least valuable first.

        Required items are placed first. The rest are chosen greedily by value
        density (priority per token), the standard approximation for the
        0/1 knapsack this is -- near-optimal in practice and, unlike an exact
        solver, linearithmic in the number of items.

        Included items are returned in their original order, so callers keep
        whatever structure they built into the sequence.
        """
        order = {id(item): i for i, item in enumerate(items)}
        required = [i for i in items if i.required]
        optional = [i for i in items if not i.required]

        used = sum(i.tokens for i in required)
        if used > self.available:
            raise BudgetExceeded(
                f"required items need {used:,} tokens but only {self.available:,} "
                f"are available in {self.spec.id}"
            )

        chosen = list(required)
        dropped: list[Item] = []
        for item in sorted(
            optional,
            key=lambda i: (-(i.priority / max(i.tokens, 1)), -i.priority, i.tokens),
        ):
            if item.priority > 0 and used + item.tokens <= self.available:
                chosen.append(item)
                used += item.tokens
            else:
                dropped.append(item)

        chosen.sort(key=lambda i: order[id(i)])
        dropped.sort(key=lambda i: order[id(i)])
        return PackResult(chosen, dropped, used, self.available)


def order_for_cache(items: Iterable[Item]) -> list[Item]:
    """Reorder so byte-stable content forms the longest possible cache prefix.

    Relative order is preserved within each group, so a caller's own ordering of
    stable items (and of volatile ones) still holds.
    """
    items = list(items)
    return [i for i in items if i.stable] + [i for i in items if not i.stable]
