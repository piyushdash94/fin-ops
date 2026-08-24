"""Command line interface: ``marketrl <command>``.

    marketrl run --symbol AAPL          load real prices and run everything
    marketrl demo --signal 0.8          same pipeline on simulated prices
    marketrl data --symbol MSFT         fetch and summarize price history
    marketrl scan --limit 120           run one model across many symbols and
                                        correct for having tested them all
    marketrl selftest                   prove the pipeline detects signal
                                        and, more importantly, its absence
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

from . import data as _data
from .data import DataError

warnings.filterwarnings("ignore", category=UserWarning)


def _emit(report, as_json: bool) -> int:
    print(json.dumps(report.as_dict(), indent=2, default=str) if as_json else report.render())
    return 0


def cmd_run(args) -> int:
    from .pipeline import run_pipeline

    try:
        prices = _data.load(args.symbol, args.start, args.end, source=args.source,
                            cache=not args.no_cache)
    except DataError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    if not args.json:
        print(f"loaded {len(prices):,} bars for {args.symbol} "
              f"({prices.index[0].date()} to {prices.index[-1].date()})\n")

    report = run_pipeline(
        prices,
        train_size=args.train, test_size=args.test, cost_bps=args.cost_bps,
        threshold=args.threshold, allow_short=not args.long_only,
        agents=() if args.no_rl else tuple(args.agents), seed=args.seed,
    )
    return _emit(report, args.json)


def cmd_demo(args) -> int:
    from .pipeline import run_pipeline

    prices = _data.synthetic_prices(args.days, seed=args.seed, signal=args.signal)
    report = run_pipeline(
        prices, train_size=args.train, test_size=args.test,
        cost_bps=args.cost_bps, seed=args.seed,
    )
    return _emit(report, args.json)


def cmd_data(args) -> int:
    import numpy as np

    try:
        prices = _data.load(args.symbol, args.start, args.end, source=args.source,
                            cache=not args.no_cache)
    except DataError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    returns = np.log(prices["adj_close"]).diff().dropna()
    print(f"symbol      {prices.attrs['symbol']}  (source: {prices.attrs['source']})")
    print(f"period      {prices.index[0].date()} to {prices.index[-1].date()}  "
          f"({len(prices):,} bars)")
    print(f"last close  {prices['adj_close'].iloc[-1]:,.2f}")
    print(f"ann. return {returns.mean() * 252:>7.1%}")
    print(f"ann. vol    {returns.std() * np.sqrt(252):>7.1%}")
    print(f"worst day   {returns.min():>7.1%}   best day {returns.max():>6.1%}")
    if args.save:
        prices.to_csv(args.save)
        print(f"\nsaved to {args.save}")
    return 0


def cmd_scan(args) -> int:
    from .scan import scan_symbols

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.source == "sp500":
        from .data import sp500_symbols

        try:
            symbols = sp500_symbols()
        except DataError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        if args.limit:
            symbols = symbols[: args.limit]
    else:
        print("error: pass --symbols, or use --source sp500 for the bundled list",
              file=sys.stderr)
        return 2

    if not args.json:
        print(f"scanning {len(symbols)} symbols with {args.model} ...", flush=True)

    report = scan_symbols(
        symbols, source=args.source, model=args.model,
        start=args.start, end=args.end,
        train_size=args.train, test_size=args.test, cost_bps=args.cost_bps,
        threshold=args.threshold, allow_short=not args.long_only,
        alpha=args.alpha, min_test_days=args.min_test_days,
        progress=not args.json,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, default=str))
        return 0
    print()
    print(report.render(top=args.top))
    return 0


def cmd_selftest(args) -> int:
    """Verify the pipeline finds signal when present and none when absent.

    The second half is the one that matters: a backtester that cannot return a
    null result will eventually tell you to trade on noise.
    """
    from .pipeline import run_pipeline

    failures = []

    print("1/2  efficient market (signal=0) -- expecting NO significant edge")
    null = run_pipeline(
        _data.synthetic_prices(1600, seed=101, signal=0.0),
        train_size=600, test_size=250, agents=("q_learning",),
    )
    found = null.significant_forecasts()
    print(f"     significant models: {found or 'none'}")
    if found:
        failures.append(f"found a spurious edge in an efficient market: {found}")

    print("2/2  inefficient market (signal=0.8) -- expecting a detected edge")
    real = run_pipeline(
        _data.synthetic_prices(1600, seed=101, signal=0.8),
        train_size=600, test_size=250, agents=("q_learning",),
    )
    found = real.significant_forecasts()
    print(f"     significant models: {found or 'none'}")
    if not found:
        failures.append("failed to detect a planted edge")

    if failures:
        print("\nSELFTEST FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nSELFTEST PASSED: the pipeline detects real signal and rejects noise.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marketrl",
        description="Next-day return prediction with ML and RL, evaluated honestly.",
        epilog="Research tooling. Not investment advice, and not a trading system.",
    )
    sub = parser.add_subparsers(dest="command")

    def add_common(p, *, with_symbol=True):
        if with_symbol:
            p.add_argument("-s", "--symbol", default="AAPL")
            p.add_argument("--start", default="2010-01-01")
            p.add_argument("--end", default=None)
            p.add_argument(
                "--source", default="auto",
                help="auto, yahoo, stooq, synthetic, or a path to a CSV file",
            )
            p.add_argument("--no-cache", action="store_true")
        p.add_argument("--train", type=int, default=750, help="training window, in days")
        p.add_argument("--test", type=int, default=126, help="test block, in days")
        p.add_argument("--cost-bps", type=float, default=10.0,
                       help="cost in bps per unit of position change")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--json", action="store_true")

    p = sub.add_parser("run", help="run the full pipeline on real price data")
    add_common(p)
    p.add_argument("--threshold", type=float, default=0.0,
                   help="minimum |predicted return| before taking a position")
    p.add_argument("--long-only", action="store_true", help="disallow short positions")
    p.add_argument("--no-rl", action="store_true", help="skip the RL agents (faster)")
    p.add_argument("--agents", nargs="*", default=["q_learning", "reinforce"])
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("demo", help="run the pipeline on simulated prices")
    add_common(p, with_symbol=False)
    p.add_argument("--days", type=int, default=2400)
    p.add_argument("--signal", type=float, default=0.5,
                   help="planted predictability; 0.0 is an efficient market")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("data", help="fetch and summarize price history")
    add_common(p)
    p.add_argument("--save", help="write the bars to this CSV path")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser(
        "scan",
        help="run one model across many symbols, with multiple-comparisons correction",
    )
    p.add_argument("--symbols", nargs="*", help="explicit tickers; omit to use the dataset list")
    p.add_argument("--source", default="sp500",
                   help="sp500 (bundled real 2013-2018 data), yahoo, stooq, or auto")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--limit", type=int, default=120, help="cap how many symbols to scan")
    p.add_argument("--model", default="ridge")
    p.add_argument("--train", type=int, default=500)
    p.add_argument("--test", type=int, default=126)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-test-days", type=int, default=250)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("selftest", help="verify the pipeline detects signal and noise")
    p.set_defaults(func=cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except (ValueError, DataError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
