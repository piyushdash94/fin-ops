import numpy as np
import pytest

from marketrl.scan import ScanReport, SymbolResult, benjamini_hochberg, scan_symbols

pytestmark = pytest.mark.filterwarnings("ignore")


# ------------------------------------------------------- FDR correction
def test_fdr_finds_nothing_in_pure_noise():
    rng = np.random.default_rng(0)
    # Uniform p-values are exactly what a true null hypothesis produces.
    assert benjamini_hochberg(rng.uniform(size=500)).sum() == 0


def test_fdr_recovers_real_signals():
    rng = np.random.default_rng(0)
    p_values = np.concatenate([np.full(10, 1e-8), rng.uniform(size=190)])
    discoveries = benjamini_hochberg(p_values)
    assert discoveries[:10].all()          # every planted signal found
    assert discoveries.sum() <= 15         # and few extras


def test_fdr_is_less_strict_than_bonferroni():
    p_values = np.array([0.001, 0.006, 0.02, 0.5, 0.8])
    n_fdr = benjamini_hochberg(p_values, alpha=0.05).sum()
    n_bonferroni = (p_values < 0.05 / len(p_values)).sum()
    assert n_fdr >= n_bonferroni


def test_fdr_handles_empty_and_single():
    assert benjamini_hochberg(np.array([])).shape == (0,)
    assert benjamini_hochberg(np.array([0.001])).tolist() == [True]
    assert benjamini_hochberg(np.array([0.9])).tolist() == [False]


# ------------------------------------------------------------- reporting
def _result(symbol, p, acc=0.51, sharpe=0.1, bench=0.5, days=506):
    return SymbolResult(
        symbol=symbol, n_test_days=days, directional_accuracy=acc, p_value=p,
        information_coefficient=0.01, strategy_sharpe=sharpe, benchmark_sharpe=bench,
        strategy_return=0.1, benchmark_return=0.5, annual_turnover=140.0,
    )


def test_chance_expectation_is_alpha_times_n():
    report = ScanReport("ridge", "sp500", 10.0, 0.05,
                        results=[_result(f"S{i}", 0.5) for i in range(100)])
    assert report.n_expected_by_chance == pytest.approx(5.0)


def test_chance_sd_matches_the_binomial_formula():
    report = ScanReport("ridge", "sp500", 10.0, 0.05,
                        results=[_result("A", 0.5, days=506)])
    assert report.chance_sd == pytest.approx(np.sqrt(0.25 / 506))


def test_nominal_significance_does_not_survive_correction():
    """5 hits out of 100 at alpha=0.05 is exactly chance, and must not count."""
    results = [_result(f"S{i}", 0.04 if i < 5 else 0.6) for i in range(100)]
    report = ScanReport("ridge", "sp500", 10.0, 0.05, results=results)
    assert report.n_nominally_significant == 5
    assert report.discoveries() == []
    assert report.bonferroni_survivors() == []
    assert "no" in report.render().lower()


def test_a_genuinely_strong_result_does_survive():
    results = [_result(f"S{i}", 1e-9 if i == 0 else 0.6) for i in range(100)]
    report = ScanReport("ridge", "sp500", 10.0, 0.05, results=results)
    assert [r.symbol for r in report.discoveries()] == ["S0"]
    assert [r.symbol for r in report.bonferroni_survivors()] == ["S0"]
    assert "survive FDR correction" in report.render()


def test_excess_sharpe_is_strategy_minus_benchmark():
    assert _result("A", 0.5, sharpe=1.5, bench=0.5).excess_sharpe == pytest.approx(1.0)


def test_empty_report_renders_without_crashing():
    report = ScanReport("ridge", "sp500", 10.0, 0.05, failures={"AAA": "too short"})
    assert "no symbols completed" in report.render()
    assert report.discoveries() == []


def test_report_serializes_to_json():
    import json

    report = ScanReport("ridge", "sp500", 10.0, 0.05,
                        results=[_result("AAA", 0.5), _result("BBB", 0.001)])
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["n_symbols"] == 2
    assert "chance_sd" in payload


# ------------------------------------------------------ scanning mechanics
# These four fetch the public S&P 500 dataset. They are excluded from the
# default run so `pytest` stays offline and deterministic; CI runs them
# separately with `-m network`.
@pytest.mark.network
def test_scan_uses_full_history_not_the_default_start_date():
    """Regression: load()'s default start silently shrank every test set to 28 days."""
    report = scan_symbols(["AAPL", "MSFT"], source="sp500", model="ridge",
                          train_size=500, test_size=126)
    assert report.n == 2
    for result in report.results:
        assert result.n_test_days > 400, "history was truncated"


@pytest.mark.network
def test_scan_drops_symbols_with_too_short_a_test_window():
    report = scan_symbols(["AAPL"], source="sp500", model="ridge",
                          train_size=500, test_size=126, min_test_days=10_000)
    assert report.n == 0
    assert "out-of-sample days" in report.failures["AAPL"]


@pytest.mark.network
def test_scan_records_failures_instead_of_raising():
    report = scan_symbols(["AAPL", "NOT_A_TICKER"], source="sp500", model="ridge",
                          train_size=500, test_size=126)
    assert report.n == 1
    assert "NOT_A_TICKER" in report.failures


@pytest.mark.network
def test_scan_on_real_data_finds_no_edge():
    """The headline empirical result, in miniature."""
    report = scan_symbols(
        ["AAPL", "MSFT", "JPM", "XOM", "PG", "KO", "T", "WMT"],
        source="sp500", model="ridge", train_size=500, test_size=126,
    )
    assert report.n == 8
    accuracy = np.array([r.directional_accuracy for r in report.results])
    # Every symbol sits within a few chance standard deviations of a coin flip.
    assert abs(accuracy.mean() - 0.5) < 4 * report.chance_sd
    assert report.discoveries() == []
