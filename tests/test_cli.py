import json

import pytest

from finops.cli import _parse_since, main
from finops.ledger import Ledger
from finops.usage import TokenUsage


def run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr()


def test_models_listing(capsys):
    code, out = run(capsys, "models")
    assert code == 0
    assert "claude-opus-5" in out.out
    assert "1,000,000" in out.out


def test_models_json_is_machine_readable(capsys):
    code, out = run(capsys, "models", "--json")
    assert code == 0
    ids = {m["id"] for m in json.loads(out.out)}
    assert "claude-opus-5" in ids


def test_price_reports_totals_and_projection(capsys):
    code, out = run(capsys, "price", "-i", "1000000", "-o", "0", "--per-day", "10")
    assert code == 0
    assert "$5.000000" in out.out
    assert "/month" in out.out


def test_price_json(capsys):
    code, out = run(capsys, "price", "-i", "1000000", "--json")
    payload = json.loads(out.out)
    assert code == 0
    assert payload["total"] == pytest.approx(5.0)


def test_price_flags_a_losing_cache_strategy(capsys):
    code, out = run(capsys, "price", "--cache-write", "100000")
    assert code == 0
    assert "LOST" in out.out


def test_unknown_model_exits_nonzero_with_a_clear_message(capsys):
    code, out = run(capsys, "price", "-m", "gpt-4o", "-i", "10")
    assert code == 2
    assert "no pricing registered" in out.err


def test_cache_recommends_caching_above_breakeven(capsys):
    code, out = run(capsys, "cache", "-t", "50000", "-r", "10")
    assert code == 0
    assert "=> cache it" in out.out


def test_cache_advises_against_caching_a_single_use_prefix(capsys):
    code, out = run(capsys, "cache", "-t", "50000", "-r", "1")
    assert code == 0
    assert "=> do not cache" in out.out


def test_cache_warns_below_the_minimum_cacheable_prefix(capsys):
    code, out = run(capsys, "cache", "-t", "500", "-r", "50")
    assert "below the minimum cacheable" in out.out


def test_count_offline_reports_context_share(capsys, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("hello world " * 500)
    code, out = run(capsys, "count", str(doc), "--offline")
    assert code == 0
    assert "of the claude-opus-5 context window" in out.out


def test_report_on_a_populated_ledger(capsys, tmp_path):
    path = tmp_path / "usage.jsonl"
    ledger = Ledger(path)
    ledger.record("claude-opus-5", TokenUsage(input_tokens=1_000_000), tags={"f": "a"})
    ledger.record("claude-haiku-4-5", TokenUsage(input_tokens=1_000_000), tags={"f": "b"})

    code, out = run(capsys, "report", "--ledger", str(path), "-g", "tag:f")
    assert code == 0
    assert "TOTAL" in out.out and "a" in out.out


def test_report_on_a_missing_ledger_exits_nonzero(capsys, tmp_path):
    code, out = run(capsys, "report", "--ledger", str(tmp_path / "none.jsonl"))
    assert code == 1
    assert "no ledger" in out.err


def test_report_flags_unprofitable_caching(capsys, tmp_path):
    path = tmp_path / "usage.jsonl"
    Ledger(path).record("claude-opus-5", TokenUsage(cache_write_5m_tokens=100_000))
    code, out = run(capsys, "report", "--ledger", str(path))
    assert "silent invalidator" in out.out


def test_bare_invocation_prints_help(capsys):
    code, out = run(capsys)
    assert code == 1
    assert "usage" in out.out.lower()


def test_version(capsys):
    code, out = run(capsys, "--version")
    assert code == 0
    assert out.out.strip()


@pytest.mark.parametrize("value,expected_none", [(None, True), ("", True)])
def test_parse_since_passthrough(value, expected_none):
    assert (_parse_since(value) is None) is expected_none


def test_parse_since_relative_and_absolute():
    import datetime as dt

    now = dt.datetime.now(dt.timezone.utc)
    assert (now - _parse_since("7d")).total_seconds() == pytest.approx(7 * 86400, abs=5)
    assert _parse_since("2026-01-01").year == 2026
    assert _parse_since("2026-01-01T00:00:00+00:00").tzinfo is not None
