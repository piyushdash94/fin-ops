import datetime as dt

import pytest

from finops import models


def test_lookup_strips_provider_prefix_and_date_suffix():
    assert models.get("anthropic.claude-opus-5").id == "claude-opus-5"
    assert models.get("us.anthropic.claude-opus-5").id == "claude-opus-5"
    assert models.get("claude-opus-5-20260101").id == "claude-opus-5"
    assert models.get("claude-opus-5@20260101").id == "claude-opus-5"


def test_unknown_model_raises_rather_than_guessing():
    with pytest.raises(models.UnknownModel) as err:
        models.get("gpt-4o")
    assert "no pricing registered" in str(err.value)


def test_promotional_rates_expire():
    sonnet = models.get("claude-sonnet-5")
    assert sonnet.rates_on(dt.date(2026, 8, 31)).input == 2.00
    assert sonnet.rates_on(dt.date(2026, 9, 1)).input == 3.00


def test_fast_mode_rates_only_where_supported():
    assert models.get("claude-opus-5").rates_on(speed="fast").input == 10.00
    with pytest.raises(ValueError):
        models.get("claude-sonnet-4-6").rates_on(speed="fast")


def test_register_custom_partner_pricing():
    spec = models.ModelSpec(
        id="vendor-model-x",
        context_window=100_000,
        max_output_tokens=4096,
        rates=models.Rates(1.5, 7.5),
    )
    models.register(spec)
    assert models.get("vendor-model-x").rates_on().output == 7.5
