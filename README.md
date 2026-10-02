# ccr-py

`ccr-py` (import name `ccr`) is a pure-Python library for shared decisions without a
central decision maker.

Many **nodes** (devices) share **pairs**: a belief plus its evidence. Each node keeps
its own inbox and takes a frozen **snapshot** of it. It then runs a pluggable
**aggregator** and produces its own **decision**. Each decision has a **lineage**: the
exact pairs that produced it.

- Python 3.10+, standard library only, fully typed (`mypy --strict`).
- Deterministic: the same snapshot and the same aggregator give the same decision.
- Immutable: all models are frozen, and their payloads are deep-frozen.

The package handles **data and calculation only**. To connect real devices, implement
`ccr.BaseBus` for your transport (Wi-Fi, MQTT, internet, ...). The built-in
`ExchangeBus` is in-process only, for tests and for running several nodes in one program.

## Quick start

```python
from ccr import Belief, Evidence, ExchangeBus, Node

bus = ExchangeBus()
a = Node("drone-a", bus)
b = Node("drone-b", bus)
c = Node("ground-c", bus)

a.observe(Belief("bridge.status", "open"),   Evidence(kind="camera", dominance=0.9))
b.observe(Belief("bridge.status", "closed"), Evidence(kind="radar",  dominance=0.6))
c.observe(Belief("bridge.status", "open"),   Evidence(kind="report", dominance=0.4))

d = c.decide()
d.result["bridge.status"]
# {"value": "open", "support": 1.3, "total": 1.9, "score": 0.684..., "n_pairs": 3, "n_abstain": 0}
d.lineage.by_origin()        # which node's pairs produced it
d.verify(c.snapshot())       # True
bus.export_audit("audit.jsonl")
```

## Concepts

| Thing | What it is |
|---|---|
| `Belief(claim, value, abstain=False)` | A statement about a claim. `abstain=True` means "no opinion". |
| `Evidence(kind, payload, source, dominance)` | Support for a belief. `dominance >= 0` is the **only** thing that influences aggregation. |
| `Pair(origin, belief, evidence)` | What nodes exchange. Its `content_hash` ignores `id` and `created_at`, so re-observing the same content gives a duplicate. |
| `Inbox` | Thread-safe and id-sorted. The first observation of some content wins, and the first-seen pair wins an id conflict. |
| `Snapshot` | A frozen, id-sorted copy of an inbox, with a `digest`. |
| `Decision` | A result, a lineage, the aggregator name and the snapshot digest. Use `verify()` / `trace()` against a snapshot. |

## Aggregators

Built-ins: `"weighted_vote"` (the default) and `"mean"`.

```python
Node("n", aggregator="weighted_vote", aggregator_params={"min_dominance": 0.2})
Node("n", aggregator="mean", aggregator_params={"strict": False})
```

When a claim cannot be decided, its result has `value: None` and a `reason`: `"tie"`
(with `tied_values`), `"no_support"` or `"abstain"`.

**Rule for every aggregator: aggregation is always per claim.** Group the pairs with
`snapshot.by_claim()` and decide each claim independently. There are three ways to add
your own aggregator:

```python
import ccr
from ccr import AggregationResult, BaseAggregator

# 1. A plain function (takes no parameters)
@ccr.aggregator("veto_closed")
def veto_closed(snapshot):
    result, used = {}, []
    for claim, pairs in snapshot.by_claim().items():
        vetoes = [p for p in pairs
                  if p.belief.value == "closed" and p.evidence.dominance >= 0.5]
        result[claim] = "closed" if vetoes else "open"
        used += vetoes or pairs
    return AggregationResult(result, used_pair_ids=tuple(p.id for p in used))

# 2. A class with validated parameters
class Quorum(BaseAggregator):
    name = "quorum"
    PARAMS = {"min_pairs": 2}
    def aggregate(self, snapshot): ...

ccr.register_aggregator("quorum", Quorum)

# 3. A plugin package: entry-point group "ccr.aggregators", value = a factory or class
```

Results and metadata must be JSON-serializable. `used_pair_ids=None` puts every
snapshot pair into the lineage.

A name that is already taken (a built-in, an earlier registration, or an entry-point
plugin) raises `AggregatorError`. To replace it on purpose, pass `replace=True` to
`register_aggregator(...)` or `@ccr.aggregator(...)`.

## Values: exact comparison and frozen containers

- **Values are compared exactly, by their canonical JSON.** `1`, `1.0` and `True` are
  three different answers in `weighted_vote` and in `content_hash`. They are not merged,
  so normalise values yourself before observing them if they should count as the same.
  `mean` is unaffected because it works on the numbers themselves.
- **Python `==` is looser than `content_hash`.** `Belief("c", 1) == Belief("c", 1.0)` is
  `True` and the two hash equally, but their pairs have different content hashes. The
  inbox and the snapshot digest use `content_hash`.
- **Lists come back as tuples.** Values, payloads and decision results are deep-frozen:
  dicts become read-only mappings and lists become tuples. For example,
  `d.result["c"]["tied_values"] == ("x", "y")`. `to_dict()` and `canonical_json()` give
  plain dicts and lists again, and hashes are unaffected.

## Development

```bash
pip install -e ".[dev]"
pytest
mypy --strict src
ruff check
```

> On exFAT/external drives, macOS writes `._*` metadata files. These break a virtualenv
> created on that drive, so keep the venv on an internal disk. `._*` files are excluded
> from ruff and from the build.

## Status

- **Abstain** behaviour is pending client confirmation. Under the current rule, an
  abstain does **not** withdraw a node's earlier opinion.
- Network transport, persistence and the real CCR aggregator are out of scope for V1.0.
- Release is on hold. The package name and license are not decided yet.
