# fin-ops

Token accounting and cost control for Claude-powered applications.

LLM spend is unusual: the unit price is public, but the quantity is decided at
runtime by a model, and the single largest lever — prompt caching — is invisible
unless you go looking for it. `finops` makes the quantity, the price, and the
caching lever all observable, in about four lines of setup.

Stdlib-only for pricing, budgeting and reporting. The Anthropic SDK is needed
only for exact token counts and the tracking client.

```bash
pip install -e ".[dev]"
```

## Instrument once

```python
from anthropic import Anthropic
from finops import TrackedClient

client = TrackedClient(Anthropic(), tags={"service": "support-bot"})

msg = client.messages.create(
    model="claude-opus-5",
    max_tokens=4096,
    messages=[{"role": "user", "content": "..."}],
)
```

The wrapper is transparent — unknown attributes pass through, and you get the
SDK's own objects back. Every completed call is priced and appended to a ledger.
Use `client.tagged(feature="search")` to attribute spend to a subsystem.

If accounting ever fails, it warns and the request still returns. A cost tool
that can take down a request path is not worth having.

```console
$ finops report --group-by tag:feature --since 7d
group           calls        cost     $/call   cached       saved
------------------------------------------------------------------
search          8,204  $ 241.5533  $  0.0294     91%  $1,884.2210
summarize       1,190  $  88.0412  $  0.0740      0%  $    0.0000
classify       19,553  $  31.2044  $  0.0016     44%  $   61.9832
------------------------------------------------------------------
TOTAL          28,947  $ 360.7989  $  0.0125     78%  $1,946.2042

Prompt caching saved $1,946.2042 (84% of what this traffic would otherwise
have cost).
```

That "saved" column is the point. It is computed per record as *what this exact
request would have cost with no caching* minus what it did cost — so it goes
**negative** when a prefix is written and never reused, and the report says so
rather than quietly showing a smaller number.

## Decide whether to cache, before you build it

Cache reads bill at 0.1x the input rate; writes at 1.25x (5m TTL) or 2x (1h).
So a cached prefix pays for itself from the 2nd request at 5m, the 3rd at 1h.

```console
$ finops cache --tokens 30000 --requests 5 --ttl 1h
prefix             30,000 tokens on claude-opus-5
traffic            5 requests, 1 cache write(s), 1h TTL
uncached           $0.7500
cached             $0.3600
break-even         3 requests per write at 1h

=> cache it: saves $0.3900 (52% of prefix cost)
```

`--writes N` models bursty traffic where entries expire between bursts. Below
~1024 tokens the tool warns you: prefixes that short silently do not cache at
all.

## Count tokens honestly

```python
from finops import TokenCounter

with TokenCounter("claude-opus-5") as counter:
    tokens = counter.count_file("CLAUDE.md")   # exact, via the API
```

Counts come from Anthropic's `count_tokens` endpoint — the only exact source —
and are memoized on disk by content hash, so re-counting an unchanged file
costs nothing. **`tiktoken` is deliberately not used anywhere**: it is OpenAI's
tokenizer and undercounts Claude by 15–20% on prose, more on code.

With no API access, `estimate()` returns a range rather than a fake-precise
integer:

```python
>>> counter.estimate(source_code)
~4,120 tokens (±618, heuristic)
```

`counter.calibrate([...samples])` fits the estimator to your own corpus against
real counts and persists the ratio, after which estimates report as
`calibrated`.

## Fit the context window, and order it for cache hits

```python
from finops import ContextBudget, Item, order_for_cache

budget = ContextBudget("claude-opus-5", reserve_output=8000, headroom=0.05)
result = budget.pack([
    Item("system",    content=SYSTEM,  required=True),
    Item("tools",     content=TOOLS,   required=True),
    Item("history",   content=HISTORY, priority=3.0, stable=False),
    *[Item(doc.name, content=doc.text, priority=doc.score) for doc in retrieved],
    Item("question",  content=question, required=True, stable=False),
])

print(result.summary())
# 14 items, 924,103/941,436 tokens (98% of budget); dropped 3: doc_41, doc_9, doc_18
```

Required items are guaranteed; the rest are chosen greedily by value density
(priority per token), which is the right approximation for the 0/1 knapsack this
actually is. Content is **dropped whole, never truncated mid-document**, and
included items keep your original ordering. An impossible budget raises at
construction rather than yielding a silent zero.

`order_for_cache()` moves byte-stable content ahead of volatile content, so the
cacheable prefix is as long as it can be. Caching is a strict prefix match — one
byte changing early invalidates everything after it.

## Price anything, including what you haven't built yet

```console
$ finops price -i 10000 -o 2000 --cache-read 50000 --cache-write 10000 --per-day 5000
total              $0.187500
without caching    $0.400000
cache saved        $0.212500

at 5,000 calls/day: $937.50/day, $28,125.00/month
```

`--batch` applies Message Batches pricing (50%); `--speed fast` prices fast mode
on the models that offer it.

## On the price table

Rates are a **cached snapshot** (`models.SNAPSHOT_DATE`), covering Anthropic
first-party API pricing. Two deliberate choices:

- **An unknown model raises `UnknownModel`, it does not guess.** A cost tool
  that invents a plausible price is worse than one that refuses to answer.
  Register partner or custom rates yourself with `finops.models.register(...)`.
- **Pricing is never auto-refreshed.** `refresh_limits_from_api()` updates
  context and output *limits* from the Models API, which does not publish
  pricing; rates stay an explicit human decision.

Lookups tolerate provider prefixes and dated snapshots
(`us.anthropic.claude-opus-5`, `claude-opus-4-5@20251101`), so one ledger can
hold traffic from several deployments. Per-category costs are recomputed at read
time, so correcting the table retroactively fixes historical reports.

Bedrock and Vertex are partner-operated with separate pricing — register those
rates rather than trusting the bundled ones.

## Ledger format

Newline-delimited JSON, appended. Single short lines opened `O_APPEND` are
atomic on POSIX, so many processes can share a ledger without locking, and the
file stays greppable and shippable to any log pipeline. Aggregation happens at
read time; malformed lines are skipped, never fatal.

```bash
finops report --group-by day --since 30d --json | jq '.total.cost'
```

## Tests

```bash
python -m pytest        # 77 tests, no network required
```

The suite pins the pricing arithmetic against hand-computed values, so a
mistyped rate fails a test rather than quietly mis-billing a report.
