import datetime as dt

import pytest

from finops.ledger import Ledger, UsageRecord
from finops.usage import TokenUsage


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "usage.jsonl")


def test_record_round_trips_through_jsonl(ledger):
    ledger.record(
        "claude-opus-5",
        TokenUsage(input_tokens=100, output_tokens=50, cache_read_tokens=900),
        tags={"feature": "search"},
        request_id="msg_abc",
    )
    (record,) = list(ledger.read())
    assert record.model == "claude-opus-5"
    assert record.usage.input_tokens == 100
    assert record.tags == {"feature": "search"}
    assert record.request_id == "msg_abc"
    assert record.cost.total > 0


def test_reading_an_absent_ledger_is_empty_not_an_error(tmp_path):
    assert list(Ledger(tmp_path / "nope.jsonl").read()) == []


def test_malformed_lines_are_skipped(ledger):
    ledger.record("claude-opus-5", TokenUsage(input_tokens=10))
    with ledger.path.open("a") as fh:
        fh.write("this is not json\n\n")
    ledger.record("claude-opus-5", TokenUsage(input_tokens=20))
    assert len(list(ledger.read())) == 2


def test_costs_are_recomputed_on_read_so_price_fixes_apply_retroactively(ledger):
    ledger.record("claude-opus-5", TokenUsage(input_tokens=1_000_000))
    line = ledger.path.read_text()
    tampered = line.replace('"cost":5', '"cost":999999')
    ledger.path.write_text(tampered)
    (record,) = list(ledger.read())
    assert record.cost.total == pytest.approx(5.00)


def test_filters_by_model_tag_and_time(ledger):
    ledger.record("claude-opus-5", TokenUsage(input_tokens=1), tags={"env": "prod"})
    ledger.record("claude-haiku-4-5", TokenUsage(input_tokens=1), tags={"env": "dev"})

    assert len(list(ledger.read(model="claude-opus-5"))) == 1
    assert len(list(ledger.read(tags={"env": "dev"}))) == 1
    assert len(list(ledger.read(tags={"env": "nope"}))) == 0

    future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
    assert list(ledger.read(since=future)) == []


def test_report_groups_and_totals(ledger):
    ledger.record("claude-opus-5", TokenUsage(input_tokens=1_000_000), tags={"f": "a"})
    ledger.record("claude-opus-5", TokenUsage(input_tokens=1_000_000), tags={"f": "b"})
    ledger.record("claude-haiku-4-5", TokenUsage(input_tokens=1_000_000), tags={"f": "a"})

    by_model = ledger.report(group_by="model")
    assert by_model.total.calls == 3
    assert by_model.total.cost == pytest.approx(11.00)  # 5 + 5 + 1
    assert by_model.groups[0].key == "claude-opus-5"  # sorted by spend
    assert by_model.groups[0].cost == pytest.approx(10.00)

    by_tag = ledger.report(group_by="tag:f")
    assert {g.key: g.calls for g in by_tag.groups} == {"a": 2, "b": 1}


def test_report_marks_untagged_records(ledger):
    ledger.record("claude-opus-5", TokenUsage(input_tokens=1))
    assert ledger.report(group_by="tag:missing").groups[0].key == "(untagged)"


def test_report_rejects_unknown_grouping(ledger):
    ledger.record("claude-opus-5", TokenUsage(input_tokens=1))
    with pytest.raises(ValueError):
        ledger.report(group_by="colour")


def test_report_surfaces_cache_savings(ledger):
    ledger.record(
        "claude-opus-5",
        TokenUsage(input_tokens=0, cache_read_tokens=1_000_000),
    )
    total = ledger.report().total
    assert total.cache_savings == pytest.approx(4.50)  # $5 uncached vs $0.50 cached
    assert total.cache_hit_rate == 1.0


def test_render_is_plain_text_with_a_total_row(ledger):
    ledger.record("claude-opus-5", TokenUsage(input_tokens=1000, output_tokens=10))
    rendered = ledger.report().render()
    assert "claude-opus-5" in rendered
    assert "TOTAL" in rendered


def test_record_message_reads_the_sdk_response_shape(ledger):
    message = type(
        "Msg",
        (),
        {
            "id": "msg_xyz",
            "model": "claude-opus-5",
            "usage": {"input_tokens": 10, "output_tokens": 20, "speed": "fast"},
        },
    )()
    record = ledger.record_message(message, tags={"job": "nightly"})
    assert record.request_id == "msg_xyz"
    assert record.speed == "fast"
    # Fast mode bills at the premium rate, not the standard one.
    assert record.cost.input_cost == pytest.approx(10 * 10.00 / 1_000_000)


def test_record_message_without_a_model_is_an_error(ledger):
    with pytest.raises(ValueError):
        ledger.record_message(type("M", (), {"usage": {}})())


def test_concurrent_appends_do_not_interleave(ledger):
    from concurrent.futures import ThreadPoolExecutor

    def write(n):
        ledger.record("claude-opus-5", TokenUsage(input_tokens=n), tags={"n": str(n)})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(100)))

    records = list(ledger.read())
    assert len(records) == 100
    assert {r.tags["n"] for r in records} == {str(n) for n in range(100)}
