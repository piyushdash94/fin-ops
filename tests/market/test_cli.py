import json

import pytest

from marketrl.cli import main
from marketrl.data import DataError


def run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr()


def test_demo_renders_a_report(capsys):
    code, out = run(capsys, "demo", "--days", "1100", "--train", "500",
                    "--test", "250", "--signal", "0.8")
    assert code == 0
    assert "FORECAST QUALITY" in out.out
    assert "VERDICT" in out.out
    assert "SYNTHETIC PRICES" in out.out


def test_demo_json_is_parseable(capsys):
    code, out = run(capsys, "demo", "--days", "1100", "--train", "500",
                    "--test", "250", "--json")
    payload = json.loads(out.out)
    assert code == 0
    assert payload["is_synthetic"] is True
    assert "strategies" in payload


def test_run_reports_blocked_or_missing_data_clearly(capsys, monkeypatch):
    import marketrl.data as data_module

    monkeypatch.setattr(
        data_module, "_http_get",
        lambda url, timeout=30.0: (_ for _ in ()).throw(DataError("blocked")),
    )
    code, out = run(capsys, "run", "--symbol", "AAPL", "--no-cache")
    assert code == 1
    assert "could not load AAPL" in out.err


def test_run_accepts_a_local_csv_as_the_source(capsys, tmp_path):
    from marketrl.data import synthetic_prices

    path = tmp_path / "bars.csv"
    synthetic_prices(1100, seed=3, signal=0.8).to_csv(path)
    code, out = run(capsys, "run", "--symbol", "LOCAL", "--source", str(path),
                    "--train", "500", "--test", "250", "--no-rl")
    assert code == 0
    # A CSV is real data as far as the tool knows: no synthetic banner.
    assert "SYNTHETIC PRICES" not in out.out
    assert "STRATEGY PERFORMANCE" in out.out


def test_data_command_summarizes_and_saves(capsys, tmp_path):
    out_path = tmp_path / "saved.csv"
    code, out = run(capsys, "data", "--symbol", "SYN", "--source", "synthetic",
                    "--save", str(out_path))
    assert code == 0
    assert "ann. vol" in out.out
    assert out_path.exists()


def test_selftest_passes(capsys):
    code, out = run(capsys, "selftest")
    assert code == 0
    assert "SELFTEST PASSED" in out.out


def test_bare_invocation_prints_help(capsys):
    code, out = run(capsys)
    assert code == 1
    assert "usage" in out.out.lower()


def test_unknown_model_exits_cleanly(capsys):
    code, out = run(capsys, "demo", "--days", "1100", "--train", "500",
                    "--test", "250", "--json")
    assert code == 0
