# Dashboard

An optional single-page dashboard for one running `ninfer-serve`. It answers, at a glance, what
the engine is doing: current prefill and decode rates, whether requests are queueing for a lane,
where prompts are being served from, how the VRAM budget is spent, and whether the board is the
limit. It also loads a `--request-log-jsonl` file to analyze a past session offline.

The application lives in [`apps/web/`](../apps/web/) and is built with Bun, Vite, and React.

## Build and run

The Docker image builds the dashboard in its own stage and serves it from `/opt/ninfer/web`, so
nothing below is needed to use it there — `http://127.0.0.1:8080/` is the dashboard and
`http://127.0.0.1:8080/v1` is the API, on one port.

To build it outside Docker:

```bash
cd apps/web
bun install
bun run build
```

Then serve it from the engine itself, same-origin with the API:

```bash
./build-sm89/apps/ninfer-serve models/qwen3_8_27b.ninfer --web-dir apps/web/dist
```

Open `http://127.0.0.1:8080/`. Registered API routes are matched before the static mount, so the
dashboard cannot shadow an endpoint; any other path resolves to the application shell.

For development against a running engine:

```bash
cd apps/web
NINFER_BASE_URL=http://127.0.0.1:8080 bun run dev
```

The dev server proxies `/telemetry`, `/events`, `/metrics`, `/slots`, `/health`, and `/v1` to that
origin, so `--cors` is not required. `NINFER_BASE_URL` is validated as an origin and defaults to
`http://127.0.0.1:8080`.

| Command | Purpose |
|---|---|
| `bun run dev` | development server on `127.0.0.1:5180` |
| `bun run build` | typecheck and emit `dist/` |
| `bun run typecheck` | types only |
| `bun test` | derivation-layer tests |
| `bun run format` | Prettier write over the app sources |
| `bun run format:check` | Prettier check |

## Adapters

Registered LoRA adapters are reported by the engine, and usage is derived from completed request
records, so the panel distinguishes three states that a traffic-only view would conflate: an
adapter serving requests, an adapter resident but idle (loaded, occupying VRAM, carrying no
traffic), and an adapter that appears in a replayed log but is not registered on the engine now
reporting. When a source reports no inventory at all — a pre-schema-15 log, or an older engine —
the panel says residency is unknown rather than claiming an adapter is absent.

Rows are keyed on the resolved `request.adapter`, never on the requested model id: the Anthropic
route passes the client's own model string through and silently falls back to the base weights
when it does not resolve, so only the resolved name identifies what actually ran.

The bank is one device arena committed at startup, outside the weights arena and before KV
capacity is resolved. It is reported as its own segment in the VRAM panel; without that it would
be visible only as a reduction in free memory.

## Data sources

The dashboard reads two channels because they answer different questions.

| Channel | Kind | Carries |
|---|---|---|
| `GET /telemetry` | polled at 1 Hz | levels: board sensors, scheduler occupancy, VRAM, cache fill, adapter inventory |
| `GET /events` | SSE | history: throughput samples and completed requests |

Levels cannot be reconstructed by replaying deltas, and the event stream is bounded and lossy
under backpressure, so the poll is always authoritative for current state. `/metrics` is not read
by the dashboard: every counter it needs is already in the two channels above, in a form that does
not require differencing scrapes.

## Reading the charts

Two sampling semantics are deliberately drawn differently, because conflating them would assert
measurements that were never taken.

- **Throughput** is an average over each reporting interval, so a sample fills its whole interval
  as a band. The reporter skips intervals with no activity and folds the skipped time into the
  next sample's `interval_seconds`, so a wide band is idle time folded forward, not a long
  sustained rate.
- **Scheduler occupancy** (`running`, `waiting`) is a snapshot taken when the report was emitted,
  not an interval average, so it is drawn as points at their own timestamps.

Decode throughput is engine-wide: the engine sums committed tokens over every lane in a round,
so the figure rises with concurrency even when no individual request gets faster. The panel pairs
it with the mean decode batch and the per-sequence rate (aggregate divided by batch), which is what
a single client experiences. Both decode means are taken over intervals that ran a decode round
rather than the whole window, so idle time does not deflate them. Per-request rates in the Requests
table are not a partition of the aggregate — every lane in a round is charged the full round wall
time — so they do not sum to it.

There is no per-lane throughput breakdown, because the engine keeps no per-lane token or round
counter; `SlotState` publishes occupancy only.

Prefill and decode are plotted on separate rate scales over one shared time axis. On this target
prefill runs roughly an order of magnitude faster than decode, so a single linear axis renders
decode as a flat sliver; each series is therefore read against its own axis, and the interval
tooltip reports both rates so they remain directly comparable at any instant.

Every reading carries an explanatory tooltip. Panel headings, stat captions, legend entries,
stacked-bar segments, and table headers are hover and focus targets that define the term, name the
units, and say what it means when the value moves. The wording lives in one glossary module rather
than beside each panel, so a reading and its explanation cannot drift apart.

The execution-thread wall-clock split is the panel that explains contention: one mutex serializes
decode, prefill, admission, publish, and upkeep, so a second spent in any of them is a second in
which no other resident lane advances. The admission decomposition (plan / restore / commit over
the calls that entered the attempt) separates continuation import cost from scheduling cost.

TTFT is decomposed into queue, restore, and prefill. A queue-dominated TTFT means requests are
waiting for a lane rather than being computed, which more lanes or shorter generations address
before any kernel work does.

## Cache churn

The occupancy panel answers what the cache is holding. The churn panel answers whether holding it
is still paying, which is a different question and is not answered by the hit rate: a working set
that has outgrown L1 keeps hitting and simply starts paying a host or disk import on every turn.

Eviction on its own is a cache working normally, so no single count here is pathological. The
readings that carry meaning are:

- **imported reuse** — share of restores served from L2 or L3 rather than a resident L1 lane.
  Drift down the tiers is the churn signal, and the restore chart shows it directly as the tier
  mix per interval.
- **lost sessions** — retained lanes evicted with no publication ticket (`l1_evictions` minus
  `l1_demotions`). Unlike a demotion, the session survives in no tier, so the next turn of that
  conversation has no state to import and prefills from zero.
- **recomputed coverage** — prefill spent on prefix the cache demonstrably held. Preflight reports
  how deep a candidate agreed with the prompt; when a request prefilled from zero anyway,
  everything beyond what the lane already reused was recomputed for nothing. Requests where
  nothing was preflighted are excluded rather than counted as clean, because they are not evidence
  either way, and agreement the resident frontier already covered is not counted either.
- **deferrals** — restores refused for shared-KV capacity with the candidate left live for a
  retry. Reported apart from restore failures because a deferral is recoverable and a failure is
  not.
- **superseded** — publications that completed and were then discarded because the session alias
  had already advanced. The export work was paid for and nothing can ever restore from it.

Host and disk evictions are counted only when a tier exceeded its byte budget. A TTL expiry is
deliberately not counted as one: reclaiming state that went cold is the cache working, while a
capacity eviction means the tier is too small for what is actually in use. A promotion that cannot
find room is refused rather than admitted and then evicted, and a refusal is not churn either.

Both charts are interval counts, so they are drawn as bands over the window each sample covers,
and the tier series are stacked because they partition one total.

## Replay

`load jsonl` reads a `--request-log-jsonl` file and renders it through the same components. When a
file contains records from more than one server instance — it is opened in append mode — only the
last instance is kept, because mixing two configurations on one axis would misattribute every
derived figure.

A log carries request and throughput history plus the `server_start` configuration, so throughput,
latency, cache occupancy against configured capacity, and per-request analysis are all available
offline. Board telemetry and live lane occupancy are sampled, never recorded, and those panels say
so rather than showing a stale or zero reading.

## Derived analytics

The request aggregations — percentiles, prompt-source distribution, miss-reason and
restore-failure partitioning, TTFT decomposition, and restore statistics — follow the definitions
in the maintainer script `cache_health.py`, including its truncating nearest-rank percentile rule.
`src/lib/derive.test.ts` pins the agreement on a fixture whose expected values were produced by
that script, so the dashboard and the script cannot report different numbers for the same log.

Churn and recomputed coverage are summed from the interval deltas the throughput record already
carries, rather than by differencing the cumulative endpoints of the window, so a server restart
inside the window cannot turn a counter reset into a negative or absurd reading. A counter that a
replayed log predates reads as zero rather than as `NaN` in a total.
