"""An append-only usage ledger and the reports you can build from it.

Records are newline-delimited JSON. That format is deliberate: appends of a
single short line are atomic on POSIX, so many processes can write the same
ledger without locking, and the file stays greppable and trivially shippable to
any log pipeline. Aggregation happens at read time.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from . import usage as _usage
from .usage import CostBreakdown, TokenUsage

__all__ = ["UsageRecord", "Ledger", "Report", "GroupStats"]

_FINOPS_HOME = Path(os.environ.get("FINOPS_HOME", Path.home() / ".cache" / "finops"))
DEFAULT_LEDGER_PATH = _FINOPS_HOME / "usage.jsonl"


@dataclass
class UsageRecord:
    """One billable API call."""

    model: str
    usage: TokenUsage
    cost: CostBreakdown
    timestamp: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    tags: dict[str, str] = field(default_factory=dict)
    request_id: str | None = None
    duration_ms: float | None = None
    batch: bool = False
    speed: str = "standard"

    def to_json(self) -> str:
        return json.dumps(
            {
                "ts": self.timestamp.isoformat(),
                "model": self.model,
                "usage": self.usage.as_dict(),
                "cost": round(self.cost.total, 8),
                "cost_uncached": round(self.cost.uncached_equivalent, 8),
                "tags": self.tags,
                "request_id": self.request_id,
                "duration_ms": self.duration_ms,
                "batch": self.batch,
                "speed": self.speed,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, line: str) -> "UsageRecord":
        raw = json.loads(line)
        usage = TokenUsage(**raw["usage"])
        model = raw["model"]
        cost = CostBreakdown(
            model=model,
            # Per-category costs are recomputed rather than stored, so a
            # corrected price table retroactively fixes historical reports.
            **{
                k: v
                for k, v in _usage.price(
                    usage,
                    model,
                    speed=raw.get("speed", "standard"),
                    batch=raw.get("batch", False),
                ).as_dict().items()
                if k in {"input_cost", "output_cost", "cache_read_cost",
                         "cache_write_cost", "uncached_equivalent"}
            },
        )
        return cls(
            model=model,
            usage=usage,
            cost=cost,
            timestamp=dt.datetime.fromisoformat(raw["ts"]),
            tags=raw.get("tags") or {},
            request_id=raw.get("request_id"),
            duration_ms=raw.get("duration_ms"),
            batch=raw.get("batch", False),
            speed=raw.get("speed", "standard"),
        )


@dataclass
class GroupStats:
    key: str
    calls: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost: float = 0.0
    uncached_cost: float = 0.0

    @property
    def cache_savings(self) -> float:
        return self.uncached_cost - self.cost

    @property
    def cache_hit_rate(self) -> float:
        return self.usage.cache_hit_rate

    @property
    def cost_per_call(self) -> float:
        return self.cost / self.calls if self.calls else 0.0

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "calls": self.calls,
            "cost": self.cost,
            "cost_per_call": self.cost_per_call,
            "cache_savings": self.cache_savings,
            "cache_hit_rate": self.cache_hit_rate,
            **self.usage.as_dict(),
        }


@dataclass
class Report:
    groups: list[GroupStats]
    total: GroupStats

    def as_dict(self) -> dict:
        return {
            "total": self.total.as_dict(),
            "groups": [g.as_dict() for g in self.groups],
        }

    def render(self, *, limit: int = 20) -> str:
        """A fixed-width table suitable for a terminal or a CI comment."""
        rows = self.groups[:limit]
        width = max([len(g.key) for g in rows] + [len("TOTAL"), 12])
        lines = [
            f"{'group'.ljust(width)}  {'calls':>7}  {'cost':>10}  "
            f"{'$/call':>9}  {'cached':>7}  {'saved':>10}",
            "-" * (width + 50),
        ]
        for g in rows:
            lines.append(
                f"{g.key[:width].ljust(width)}  {g.calls:>7,}  ${g.cost:>9,.4f}  "
                f"${g.cost_per_call:>8,.4f}  {g.cache_hit_rate:>6.0%}  "
                f"${g.cache_savings:>9,.4f}"
            )
        if len(self.groups) > limit:
            lines.append(f"... {len(self.groups) - limit} more groups")
        t = self.total
        lines += [
            "-" * (width + 50),
            f"{'TOTAL'.ljust(width)}  {t.calls:>7,}  ${t.cost:>9,.4f}  "
            f"${t.cost_per_call:>8,.4f}  {t.cache_hit_rate:>6.0%}  "
            f"${t.cache_savings:>9,.4f}",
        ]
        return "\n".join(lines)


class Ledger:
    """Append-only usage log with read-time aggregation."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_LEDGER_PATH

    def record(
        self,
        model: str,
        usage: TokenUsage,
        *,
        tags: dict[str, str] | None = None,
        request_id: str | None = None,
        duration_ms: float | None = None,
        batch: bool = False,
        speed: str = "standard",
        day: dt.date | None = None,
    ) -> UsageRecord:
        """Cost a usage object and append it to the ledger."""
        cost = _usage.price(usage, model, day=day, speed=speed, batch=batch)
        record = UsageRecord(
            model=model,
            usage=usage,
            cost=cost,
            tags=dict(tags or {}),
            request_id=request_id,
            duration_ms=duration_ms,
            batch=batch,
            speed=speed,
        )
        self.append(record)
        return record

    def record_message(self, message, **kw) -> UsageRecord:
        """Record straight from an SDK ``Message`` response object."""
        model = kw.pop("model", None) or getattr(message, "model", None)
        if model is None:
            raise ValueError("message has no model; pass model=...")
        speed = kw.pop("speed", None) or _usage._get(
            getattr(message, "usage", None), "speed", "standard"
        )
        kw.setdefault("request_id", getattr(message, "id", None))
        return self.record(
            model,
            TokenUsage.from_api(getattr(message, "usage", None)),
            speed=speed or "standard",
            **kw,
        )

    def append(self, record: UsageRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One short line, opened O_APPEND: atomic across concurrent writers.
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")

    def __iter__(self) -> Iterator[UsageRecord]:
        return self.read()

    def read(
        self,
        *,
        since: dt.datetime | None = None,
        until: dt.datetime | None = None,
        model: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Iterator[UsageRecord]:
        """Stream matching records. Malformed lines are skipped, not fatal."""
        try:
            handle = self.path.open("r", encoding="utf-8")
        except FileNotFoundError:
            return
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = UsageRecord.from_json(line)
                except Exception:
                    continue
                if since and record.timestamp < since:
                    continue
                if until and record.timestamp > until:
                    continue
                if model and record.model != model:
                    continue
                if tags and any(record.tags.get(k) != v for k, v in tags.items()):
                    continue
                yield record

    def report(self, *, group_by: str = "model", **filters) -> Report:
        """Aggregate the ledger.

        ``group_by`` accepts ``model``, ``day``, ``hour``, ``none``, or
        ``tag:<name>`` to group by one of your own tags.
        """
        groups: dict[str, GroupStats] = {}
        total = GroupStats("TOTAL")

        for record in self.read(**filters):
            key = _group_key(record, group_by)
            stats = groups.get(key)
            if stats is None:
                stats = groups[key] = GroupStats(key)
            for target in (stats, total):
                target.calls += 1
                target.usage = target.usage + record.usage
                target.cost += record.cost.total
                target.uncached_cost += record.cost.uncached_equivalent

        ordered = sorted(groups.values(), key=lambda g: (-g.cost, g.key))
        return Report(ordered, total)


def _group_key(record: UsageRecord, group_by: str) -> str:
    if group_by == "model":
        return record.model
    if group_by == "day":
        return record.timestamp.date().isoformat()
    if group_by == "hour":
        return record.timestamp.strftime("%Y-%m-%d %H:00")
    if group_by == "none":
        return "all"
    if group_by.startswith("tag:"):
        return record.tags.get(group_by[4:], "(untagged)")
    raise ValueError(
        f"unknown group_by {group_by!r}; expected model, day, hour, none, or tag:<name>"
    )
