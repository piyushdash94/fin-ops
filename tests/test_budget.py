import pytest

from finops.budget import (
    BudgetExceeded,
    ContextBudget,
    Item,
    order_for_cache,
)


def test_item_estimates_tokens_from_content():
    item = Item("doc", content="hello world " * 100)
    assert item.tokens > 0


def test_item_requires_content_or_tokens():
    with pytest.raises(ValueError):
        Item("empty")


def test_budget_reserves_output_and_headroom():
    budget = ContextBudget(
        "claude-haiku-4-5", reserve_output=8000, reserve_overhead=0, headroom=0.0
    )
    assert budget.available == 200_000 - 8000

    with_headroom = ContextBudget(
        "claude-haiku-4-5", reserve_output=8000, reserve_overhead=0, headroom=0.1
    )
    assert with_headroom.available == pytest.approx(192_000 * 0.9, abs=1)


def test_budget_rejects_impossible_configuration():
    with pytest.raises(ValueError, match="nothing left for context"):
        ContextBudget("claude-haiku-4-5", reserve_output=8000, limit=1000)
    with pytest.raises(ValueError, match="exceeds"):
        ContextBudget("claude-haiku-4-5", reserve_output=999_999)


def test_pack_keeps_required_and_drops_lowest_value_density():
    budget = ContextBudget(
        "claude-haiku-4-5", reserve_output=1000, reserve_overhead=0,
        headroom=0.0, limit=3000,
    )  # 2000 tokens available
    items = [
        Item("system", tokens=500, required=True),
        Item("cheap-and-useful", tokens=100, priority=5.0),
        Item("bulky-and-dull", tokens=1800, priority=1.0),
        Item("question", tokens=200, required=True, stable=False),
    ]
    result = budget.pack(items)

    names = [i.name for i in result.included]
    assert "system" in names and "question" in names
    assert "cheap-and-useful" in names
    assert [i.name for i in result.dropped] == ["bulky-and-dull"]
    assert result.tokens_used == 800
    assert not result  # falsy: something was dropped


def test_pack_preserves_caller_ordering():
    budget = ContextBudget(
        "claude-haiku-4-5", reserve_output=100, reserve_overhead=0,
        headroom=0.0, limit=10_000,
    )
    items = [Item(f"i{n}", tokens=10, priority=n + 1.0) for n in range(5)]
    assert [i.name for i in budget.pack(items).included] == ["i0", "i1", "i2", "i3", "i4"]


def test_pack_raises_when_required_items_alone_overflow():
    budget = ContextBudget(
        "claude-haiku-4-5", reserve_output=100, reserve_overhead=0,
        headroom=0.0, limit=1000,
    )
    with pytest.raises(BudgetExceeded):
        budget.pack([Item("huge", tokens=5000, required=True)])


def test_zero_priority_items_are_never_included():
    budget = ContextBudget(
        "claude-haiku-4-5", reserve_output=100, reserve_overhead=0,
        headroom=0.0, limit=10_000,
    )
    result = budget.pack([Item("worthless", tokens=1, priority=0.0)])
    assert result.included == []


def test_order_for_cache_puts_stable_content_first_and_is_stable():
    items = [
        Item("question", tokens=1, stable=False),
        Item("system", tokens=1),
        Item("timestamp", tokens=1, stable=False),
        Item("tools", tokens=1),
    ]
    assert [i.name for i in order_for_cache(items)] == [
        "system", "tools", "question", "timestamp"
    ]
