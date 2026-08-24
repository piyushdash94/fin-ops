import pytest

from finops.models import MTOK
from finops.usage import (
    TokenUsage,
    cache_breakeven_requests,
    cache_projection,
    price,
)


def test_from_api_accepts_dict_and_object():
    class Obj:
        input_tokens = 10
        output_tokens = 20
        cache_read_input_tokens = 30
        cache_creation_input_tokens = 40
        cache_creation = None

    from_obj = TokenUsage.from_api(Obj())
    from_dict = TokenUsage.from_api(
        {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 40,
        }
    )
    assert from_obj == from_dict
    # A flat cache_creation total is attributed to the default 5m TTL.
    assert from_obj.cache_write_5m_tokens == 40
    assert from_obj.cache_write_1h_tokens == 0


def test_from_api_prefers_per_ttl_breakdown():
    usage = TokenUsage.from_api(
        {
            "input_tokens": 5,
            "cache_creation_input_tokens": 300,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 100,
                "ephemeral_1h_input_tokens": 200,
            },
        }
    )
    assert (usage.cache_write_5m_tokens, usage.cache_write_1h_tokens) == (100, 200)


def test_from_api_tolerates_missing_and_null_fields():
    assert TokenUsage.from_api(None) == TokenUsage()
    assert TokenUsage.from_api({"input_tokens": None, "output_tokens": 7}).output_tokens == 7


def test_price_matches_hand_computed_rates():
    usage = TokenUsage(input_tokens=MTOK, output_tokens=MTOK)
    cost = price(usage, "claude-opus-5")
    assert cost.input_cost == pytest.approx(5.00)
    assert cost.output_cost == pytest.approx(25.00)
    assert cost.total == pytest.approx(30.00)


def test_cache_read_and_write_multipliers():
    read = price(TokenUsage(cache_read_tokens=MTOK), "claude-opus-5")
    assert read.total == pytest.approx(0.5)  # 0.1x of $5

    write_5m = price(TokenUsage(cache_write_5m_tokens=MTOK), "claude-opus-5")
    assert write_5m.total == pytest.approx(6.25)  # 1.25x of $5

    write_1h = price(TokenUsage(cache_write_1h_tokens=MTOK), "claude-opus-5")
    assert write_1h.total == pytest.approx(10.00)  # 2x of $5


def test_batch_pricing_halves_the_bill():
    usage = TokenUsage(input_tokens=MTOK, output_tokens=MTOK)
    assert price(usage, "claude-opus-5", batch=True).total == pytest.approx(15.00)


def test_cache_savings_is_negative_when_writes_are_never_reused():
    # A prefix written once and never read costs more than sending it fresh.
    cost = price(TokenUsage(cache_write_5m_tokens=100_000), "claude-opus-5")
    assert cost.cache_savings < 0


def test_prompt_tokens_counts_every_input_side_token():
    usage = TokenUsage(
        input_tokens=10, output_tokens=99, cache_read_tokens=20, cache_write_5m_tokens=30
    )
    assert usage.prompt_tokens == 60
    assert usage.total_tokens == 159
    assert usage.cache_hit_rate == pytest.approx(20 / 60)


def test_usage_is_summable():
    a = TokenUsage(input_tokens=1, output_tokens=2)
    b = TokenUsage(input_tokens=3, cache_read_tokens=4)
    assert sum([a, b], TokenUsage()) == TokenUsage(
        input_tokens=4, output_tokens=2, cache_read_tokens=4
    )


def test_breakeven_matches_published_thresholds():
    assert cache_breakeven_requests("5m") == 2
    assert cache_breakeven_requests("1h") == 3
    with pytest.raises(ValueError):
        cache_breakeven_requests("2h")


def test_cache_projection_flips_at_the_breakeven_point():
    single = cache_projection(50_000, 1, "claude-opus-5")
    assert not single.worth_it
    assert cache_projection(50_000, 2, "claude-opus-5").worth_it


def test_cache_projection_rejects_impossible_write_counts():
    with pytest.raises(ValueError):
        cache_projection(1000, 2, "claude-opus-5", writes=3)
