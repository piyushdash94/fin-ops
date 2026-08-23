"""Command line interface: ``finops <command>``.

    finops models                       list models, limits and rates
    finops count FILE...                token counts for files or stdin
    finops price --model M -i N -o N    cost a hypothetical request
    finops cache --tokens N --requests N  should this prefix be cached?
    finops report [--group-by ...]      aggregate the usage ledger
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from . import models as _models
from . import usage as _usage
from .ledger import DEFAULT_LEDGER_PATH, Ledger
from .tokens import TokenCounter
from .usage import TokenUsage


def _parse_since(value: str | None) -> dt.datetime | None:
    """Accept an ISO date/timestamp or a relative window like ``7d`` / ``12h``."""
    if not value:
        return None
    text = value.strip()
    units = {"d": "days", "h": "hours", "m": "minutes", "w": "weeks"}
    if text[-1:] in units and text[:-1].replace(".", "", 1).isdigit():
        delta = dt.timedelta(**{units[text[-1]]: float(text[:-1])})
        return dt.datetime.now(dt.timezone.utc) - delta
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def cmd_models(args) -> int:
    specs = _models.all_models()
    if args.json:
        print(json.dumps([
            {
                "id": s.id,
                "context_window": s.context_window,
                "max_output_tokens": s.max_output_tokens,
                "input_per_mtok": s.rates_on().input,
                "output_per_mtok": s.rates_on().output,
            }
            for s in specs
        ], indent=2))
        return 0

    print(f"{'model':<22}{'context':>10}{'max out':>10}{'in $/M':>9}{'out $/M':>9}")
    print("-" * 60)
    for spec in specs:
        rates = spec.rates_on()
        note = ""
        if spec.promo_until and rates is spec.promo_rates:
            note = f"  (promo through {spec.promo_until})"
        print(
            f"{spec.id:<22}{spec.context_window:>10,}{spec.max_output_tokens:>10,}"
            f"{rates.input:>9.2f}{rates.output:>9.2f}{note}"
        )
    print(f"\nPrices verified {_models.SNAPSHOT_DATE}; cache reads bill at "
          f"{_models.CACHE_READ_MULTIPLIER:g}x input, writes at "
          f"{_models.CACHE_WRITE_MULTIPLIER['5m']:g}x (5m) / "
          f"{_models.CACHE_WRITE_MULTIPLIER['1h']:g}x (1h).")
    return 0


def cmd_count(args) -> int:
    counter = TokenCounter(args.model, offline=args.offline)
    total = 0
    with counter:
        if not args.paths:
            text = sys.stdin.read()
            total = counter.count(text)
            print(f"{total:>10,}  (stdin)")
        else:
            for path in args.paths:
                try:
                    tokens = counter.count_file(path)
                except OSError as err:
                    print(f"{'-':>10}  {path}  [{err.strerror}]", file=sys.stderr)
                    continue
                total += tokens
                print(f"{tokens:>10,}  {path}")

    if len(args.paths) > 1:
        print(f"{total:>10,}  TOTAL")
    spec = _models.get(args.model)
    cost = _usage.price(TokenUsage(input_tokens=total), args.model).total
    print(
        f"\n{total:,} tokens = {total / spec.context_window:.1%} of the "
        f"{args.model} context window; ${cost:,.4f} to send once as fresh input."
    )
    return 0


def cmd_price(args) -> int:
    usage = TokenUsage(
        input_tokens=args.input,
        output_tokens=args.output,
        cache_read_tokens=args.cache_read,
        cache_write_5m_tokens=args.cache_write,
        cache_write_1h_tokens=args.cache_write_1h,
    )
    cost = _usage.price(usage, args.model, speed=args.speed, batch=args.batch)
    if args.json:
        print(json.dumps({"usage": usage.as_dict(), **cost.as_dict()}, indent=2))
        return 0

    print(f"model              {cost.model}")
    print(f"input              ${cost.input_cost:,.6f}  ({usage.input_tokens:,} tokens)")
    print(f"output             ${cost.output_cost:,.6f}  ({usage.output_tokens:,} tokens)")
    if usage.cache_read_tokens:
        print(f"cache read         ${cost.cache_read_cost:,.6f}  ({usage.cache_read_tokens:,} tokens)")
    if usage.cache_write_tokens:
        print(f"cache write        ${cost.cache_write_cost:,.6f}  ({usage.cache_write_tokens:,} tokens)")
    print(f"total              ${cost.total:,.6f}")
    if usage.cache_read_tokens or usage.cache_write_tokens:
        print(f"without caching    ${cost.uncached_equivalent:,.6f}")
        verb = "saved" if cost.cache_savings >= 0 else "LOST"
        print(f"cache {verb:<12} ${abs(cost.cache_savings):,.6f}")
    if args.per_day:
        print(f"\nat {args.per_day:,} calls/day: ${cost.total * args.per_day:,.2f}/day, "
              f"${cost.total * args.per_day * 30:,.2f}/month")
    return 0


def cmd_cache(args) -> int:
    projection = _usage.cache_projection(
        args.tokens, args.requests, args.model, ttl=args.ttl, writes=args.writes
    )
    if args.json:
        print(json.dumps(projection.as_dict(), indent=2))
        return 0

    print(f"prefix             {projection.prefix_tokens:,} tokens on {projection.model}")
    print(f"traffic            {projection.requests:,} requests, {args.writes} cache write(s), {args.ttl} TTL")
    print(f"uncached           ${projection.uncached_cost:,.4f}")
    print(f"cached             ${projection.cached_cost:,.4f}")
    print(f"break-even         {projection.breakeven_requests} requests per write at {args.ttl}")
    if projection.worth_it:
        pct = projection.savings / projection.uncached_cost if projection.uncached_cost else 0
        print(f"\n=> cache it: saves ${projection.savings:,.4f} ({pct:.0%} of prefix cost)")
    else:
        print(f"\n=> do not cache: costs ${-projection.savings:,.4f} more than sending it fresh")
    if projection.prefix_tokens < 1024:
        print("   note: prefixes under ~1024 tokens are below the minimum cacheable "
              "size and will silently not cache.")
    return 0


def cmd_report(args) -> int:
    ledger = Ledger(args.ledger)
    if not ledger.path.exists():
        print(f"no ledger at {ledger.path}", file=sys.stderr)
        return 1
    report = ledger.report(
        group_by=args.group_by,
        since=_parse_since(args.since),
        model=args.model,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, default=str))
        return 0
    if not report.groups:
        print("no records matched")
        return 0
    print(report.render(limit=args.limit))
    total = report.total
    if total.cache_savings > 0:
        print(f"\nPrompt caching saved ${total.cache_savings:,.4f} "
              f"({total.cache_savings / (total.cost + total.cache_savings):.0%} of "
              f"what this traffic would otherwise have cost).")
    elif total.usage.cache_write_tokens and total.cache_savings <= 0:
        print("\nCache writes are not paying for themselves. Check for a silent "
              "invalidator in the cached prefix (a timestamp, unsorted JSON, a "
              "varying tool list).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finops",
        description="Token accounting and cost control for Claude applications.",
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("models", help="list known models, limits and rates")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("count", help="count tokens in files (or stdin)")
    p.add_argument("paths", nargs="*", type=Path)
    p.add_argument("-m", "--model", default="claude-opus-5")
    p.add_argument("--offline", action="store_true", help="estimate without calling the API")
    p.set_defaults(func=cmd_count)

    p = sub.add_parser("price", help="cost a hypothetical request")
    p.add_argument("-m", "--model", default="claude-opus-5")
    p.add_argument("-i", "--input", type=int, default=0)
    p.add_argument("-o", "--output", type=int, default=0)
    p.add_argument("--cache-read", type=int, default=0)
    p.add_argument("--cache-write", type=int, default=0, help="5m TTL cache writes")
    p.add_argument("--cache-write-1h", type=int, default=0)
    p.add_argument("--speed", choices=["standard", "fast"], default="standard")
    p.add_argument("--batch", action="store_true", help="Message Batches pricing (50%%)")
    p.add_argument("--per-day", type=int, help="project a daily/monthly bill at this call volume")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_price)

    p = sub.add_parser("cache", help="decide whether to cache a shared prefix")
    p.add_argument("-m", "--model", default="claude-opus-5")
    p.add_argument("-t", "--tokens", type=int, required=True, help="size of the shared prefix")
    p.add_argument("-r", "--requests", type=int, required=True, help="requests reusing it")
    p.add_argument("--writes", type=int, default=1, help="times the prefix must be re-written")
    p.add_argument("--ttl", choices=["5m", "1h"], default="5m")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_cache)

    p = sub.add_parser("report", help="aggregate the usage ledger")
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    p.add_argument("-g", "--group-by", default="model",
                   help="model, day, hour, none, or tag:<name>")
    p.add_argument("--since", help="ISO timestamp or a relative window like 7d / 12h")
    p.add_argument("-m", "--model", help="only this model")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from . import __version__

        print(__version__)
        return 0
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except _models.UnknownModel as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except (ValueError, BrokenPipeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
