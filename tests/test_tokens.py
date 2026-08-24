import json

import pytest

from finops.tokens import Estimate, TokenCounter, estimate_tokens


class FakeCounter:
    """Stands in for client.messages.count_tokens, counting invocations."""

    def __init__(self, tokens=42):
        self.tokens = tokens
        self.calls = 0

    def count_tokens(self, **kwargs):
        self.calls += 1
        return type("Resp", (), {"input_tokens": self.tokens})()


class FakeClient:
    def __init__(self, tokens=42):
        self.messages = FakeCounter(tokens)


def test_estimate_reports_a_range_not_false_precision():
    est = estimate_tokens("The quick brown fox jumps over the lazy dog. " * 20)
    assert isinstance(est, Estimate)
    assert est.low < est.tokens < est.high
    assert int(est) == est.tokens


def test_estimate_of_empty_text_is_zero():
    assert estimate_tokens("").tokens == 0


def test_code_estimates_denser_than_prose_of_equal_length():
    prose = "the quick brown fox jumps over a lazy dog and keeps on running " * 10
    code = ('x={"a":[1,2,3],"b":(4,5)};y=x["a"][0]+len(x)*2;#!@$%^&*()<>?/\\|' * 10)[: len(prose)]
    assert estimate_tokens(code).tokens > estimate_tokens(prose).tokens


def test_count_uses_api_and_memoizes(tmp_path):
    client = FakeClient(tokens=123)
    counter = TokenCounter("claude-opus-5", client=client, cache_path=tmp_path / "c.json")
    assert counter.count("hello") == 123
    assert counter.count("hello") == 123
    assert client.messages.calls == 1  # second lookup served from cache


def test_cache_persists_across_instances(tmp_path):
    cache = tmp_path / "c.json"
    first = TokenCounter("claude-opus-5", client=FakeClient(7), cache_path=cache)
    with first:
        first.count("hello")

    reused = FakeClient(999)
    second = TokenCounter("claude-opus-5", client=reused, cache_path=cache)
    assert second.count("hello") == 7
    assert reused.messages.calls == 0


def test_cache_key_separates_models(tmp_path):
    cache = tmp_path / "c.json"
    client = FakeClient(10)
    TokenCounter("claude-opus-5", client=client, cache_path=cache).count("hi")
    TokenCounter("claude-haiku-4-5", client=client, cache_path=cache).count("hi")
    assert client.messages.calls == 2


def test_system_and_tools_change_the_count_key(tmp_path):
    client = FakeClient(5)
    counter = TokenCounter("claude-opus-5", client=client, cache_path=tmp_path / "c.json")
    counter.count("hi")
    counter.count("hi", system="You are terse.")
    counter.count("hi", tools=[{"name": "t", "input_schema": {}}])
    assert client.messages.calls == 3


def test_offline_counter_never_calls_the_api(tmp_path):
    client = FakeClient()
    counter = TokenCounter(
        "claude-opus-5", client=client, cache_path=tmp_path / "c.json", offline=True
    )
    assert counter.count("some text here") > 0
    assert client.messages.calls == 0


def test_api_failure_degrades_to_an_estimate(tmp_path):
    class Broken:
        class messages:
            @staticmethod
            def count_tokens(**kwargs):
                raise RuntimeError("network down")

    counter = TokenCounter("claude-opus-5", client=Broken(), cache_path=tmp_path / "c.json")
    assert counter.count("hello world, this is a fallback path") > 0


def test_calibration_is_persisted_and_used(tmp_path):
    cache = tmp_path / "c.json"
    text = "x" * 400
    # Ground truth: 400 chars -> 100 tokens, i.e. exactly 4.0 chars/token.
    counter = TokenCounter("claude-opus-5", client=FakeClient(100), cache_path=cache)
    assert counter.calibrate([text]) == pytest.approx(4.0)
    assert json.loads(cache.read_text())["__calibration__:claude-opus-5"] == 4000

    fresh = TokenCounter("claude-opus-5", client=FakeClient(1), cache_path=cache, offline=True)
    est = fresh.estimate("y" * 800)
    assert est.method == "calibrated"
    assert est.tokens == 200


def test_calibrate_rejects_empty_input(tmp_path):
    counter = TokenCounter("claude-opus-5", client=FakeClient(), cache_path=tmp_path / "c.json")
    with pytest.raises(ValueError):
        counter.calibrate(["   "])


def test_count_file(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nSome content.\n")
    counter = TokenCounter("claude-opus-5", client=FakeClient(11), cache_path=tmp_path / "c.json")
    assert counter.count_file(path) == 11
