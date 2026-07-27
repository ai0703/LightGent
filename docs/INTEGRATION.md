# Integrating with LightGent

For a service that wants to call LightGent. Written 2026-07-26 against measured
behaviour, not intentions. Numbers here were measured on that date; re-measure
before trusting them in a month.

## 1. Pick the right endpoint

| Endpoint | Use it when | Notes |
|---|---|---|
| `POST /enrich-company` | You want decision-makers for a company | Opinionated wrapper. Returns `employees[]` already normalised. **Start here for lead enrichment.** |
| `POST /research/batch` | You have many rows and want throughput | Queue-aware, never fails on lane exhaustion, results returned in input order. **Best for bulk.** |
| `POST /research/queued` | One row, must complete, can wait | Waits for a lane rather than erroring. |
| `POST /research` | One row, want to fail fast | No queue. Errors if the LLM backend is down. |
| `GET /health`, `GET /queue/status` | Before and during load | Check `lanes_available` before sending a batch. |

Do **not** build your own prompt against `/research` for owner-finding. Use
`/enrich-company`, whose task text and output contract are already tuned.

### `/enrich-company`

```json
POST /enrich-company
{"company": "Bakkerij Holtkamp", "domain": "holtkamp.nl",
 "city": "Amsterdam", "limit": 3,
 "roles": ["Owner", "CEO", "Founder", "Managing Director"]}
```

Returns:

```json
{"company": "...", "domain": "...", "status": "success",
 "employees": [{"name": "...", "title": "...", "linkedin_url": null,
                "source": "https://...", "confidence": "high"}]}
```

### `/research` and `/research/batch`

```json
{"task": "What to find, plain English",
 "context": {"company": "X", "domain": "x.nl"},
 "output_fields": {"owner_name": "full name of the owner, or null"},
 "max_iterations": 12}
```

`output_fields` is the contract: name each field and describe exactly what goes
in it, including what to do when it is not found. Ambiguity here is the single
biggest source of unusable answers.

## 2. Handle the three statuses correctly

`status` is `success` | `parse_error` | `error`.

- `error` — infrastructure. Retry is reasonable.
- `parse_error` — **the agent ran but produced nothing usable.** This is a normal
  outcome, not a crash. `/enrich-company` also returns `parse_error` when it
  succeeded but found zero people, so treat it as "no answer for this row".
- `success` — you got data. **It is not necessarily correct.** Measured accuracy on
  the tuned model was 65 percent as of 2026-07-26 (13 of 20), so downstream code
  must treat every field as a claim, not a fact. Keep the `source` URL and the
  `confidence`. **That 65 is n=20 on the dev set: the Wilson 95 percent interval
  is 43 to 82 percent.** Do not size a business process on the point estimate as
  though it were +/- 2 points. A sealed 60-domain test set is reserved and unspent,
  and is the number to quote once it is run.

**Do not immediately retry a `parse_error` with identical input.** It is usually
deterministic and you will pay full cost for the same non-answer.

## 3. Concurrency: what the service can actually take

Measured 2026-07-26:

| Lane | Sustained capacity |
|---|---|
| Search | 267/min |
| Fetch | 2,393/min |
| What one GPU needs to stay busy | 250 searches + 100 fetches/min |

So search is the tight lane and fetch has ~24x headroom. `MAX_CONCURRENT` gates
**LLM calls only** (default 3); tool calls are ungated, so your request
concurrency passes straight through to search and fetch.

**Recommendation:** send batches at concurrency **16 to 32**. The GPU benchmark
showed C64 buys only 5 percent more for worse latency. Use `/research/batch` and
let the internal queue do the pacing rather than firing 200 parallel HTTP calls.

## 4. Timeouts: budget minutes, not seconds

A company runs a multi-step agent loop: up to `MAX_ITERATIONS` (12) turns, each
with an LLM call (`LLM_TIMEOUT` 180s) plus tool calls (`TOOL_TIMEOUT` 45s).

- **Client timeout: allow 5 minutes per company**, longer for `/research/queued`
  which may legitimately wait for a lane.
- The **first** request after startup pays ~0.85s of proxy pool construction. Fire
  one warm-up call before a batch.
- **The warm-up is a CORRECTNESS requirement, not just a latency one.** The SearXNG
  shards scale to zero, and a cold shard answers **HTTP 200 with an empty
  `results` array and no `unresponsive_engines`** — indistinguishable from "nothing
  found". Measured 2026-07-26: all three shards returned 0 results cold, then
  21-40 results each once warm. A batch fired at cold shards produces confident
  nulls, not errors. Warm every shard and assert a non-zero result count before
  starting, and treat a zero-result *first* call as "not ready", never as "no data".
- Per-fetch p50 is 1.30s and worst observed 2.71s, so a company that takes many
  minutes is stuck in the LLM loop, not in the network.

## 5. Things that will bite you

**Serper is the primary search lane and it is metered.** `web_search` tries Serper
(Google) first because result *quality* was the binding constraint. The free tier
is 2,500 searches. Budget it at **6.35 searches per COMPANY** (127 tool calls over
20 companies), which is roughly **390 companies**. Do not budget with the 9.8
figure: that is searches per *correct* answer, and you pay for searches on the
failures too, so it is the wrong denominator for capacity and under-counts by
about a third. After the free tier it silently falls back to the SearXNG shards,
which are good but Google-limited. **Monitor your Serper balance; do not discover
this at scale.**

**Google suspension is per SearXNG instance and asymmetric.** A 429 benches Google
for 60s on our shards, but a CAPTCHA benches it for **seven days** by default.
Never hammer search to "warm it up".

**THE ENGINE IS NAMED `google cse`, NOT `google`. Fixed 2026-07-27.** Google was
returning zero results on every shard, costing about 60 pct of the evidence per
query. Six configuration attempts changed nothing because they all targeted
`- name: google`, and **no engine by that name exists** in this SearXNG build: of
279 registered engines the Google family is `google news`, `google cse`,
`google cse images`, `google scholar`. An unmatched engine name is a SILENT no-op,
with no warning in the logs. The same mistake means the shard's `disabled: true`
entries for `brave`, `startpage` and `wikipedia` never applied either, so roughly
80 engines are enabled instead of the intended three.

Three things were required together, and any one alone measures as no change:

1. `- name: "google cse"` — the real engine name, quoted for the space.
2. `retries: 3` plus `retry_on_http_error: [429, 403, 302]`. Without the latter a
   429 is **terminal**: the engine is marked failed and suspended. With it the
   refusal is retried, and SearXNG's docs state that *"on each retry, SearXNG uses
   an different proxy and source ip"*. The 302 matters because Google signals bot
   detection as a redirect to `/sorry/index`.
3. `max_keepalive_connections: 0` and `keepalive_expiry: 0.0`, because **Proxiware
   rotates its residential exit IP per TCP CONNECTION, not per request**. Proven:
   one reused connection returned the same IP 9 times out of 9, while 10 fresh
   connections returned 8 distinct IPs. With keepalive on, SearXNG pinned one
   residential IP and Google burned it within a few queries.

Also give that engine `request_timeout: 18.0` (and raise the global
`max_request_timeout`), or three retries at ~3s each through a residential proxy
overrun the 10s budget and fail as `Suspended: timeout` instead of succeeding on
retry 2.

Route ONLY Google through residential. Bing and DuckDuckGo measure 20/20 on the
cheap datacenter pool and are about 2x faster there, and Jina 451s more often
through residential exits.

Measured effect across all three shards: Google availability **0 -> 80 pct
(24/30 queries)** and median results per query **12.5 -> 33.0**. Search latency
rises from ~1.5s to ~3-4s, which is roughly +8 pct on per-company wall clock for
2.6x the evidence.

**Operational trap: a Modal redeploy does not recycle a warm container.** Shard c
kept serving the old build after deploy, and the giveaway was `suspended_time=60`
in its logs where the new config says 5. Confirm a config change is live by
grepping the logs for a value only the new build has, then `modal app stop` to
force a cold start.

**Silent quality degradation is the real failure mode, not errors.** On
2026-07-26 a throttled search layer returned Jeffrey Epstein articles for a Dutch
agri company, with HTTP 200 and a full result count. If you log anything, log
result counts and which engines answered, and alert when average results per query
drops. Zero errors is not proof of health.

**A `linkedin_url` may be null and that is correct.** The agent used to fabricate
LinkedIn slugs; fabricated primary person URLs are now stripped. Do not treat
`null` as failure, and never synthesise the URL yourself.

## 6. Configuration the operator must set

Required: `LLM_BASE_URL`, `LLM_MODEL` (both endpoints return an `error` status
without them). `SERPER_API_KEY` for the quality search lane. `SEARXNG_URLS` as a
comma-separated shard list, since suspension is per instance and sharding is what
keeps Google available. `proxies.txt` (gitignored) of `ip:port:user:pass` lines
for the fetch pool.

Recommended for enrichment runs: **`MIN_TOOL_CALLS=2`**. It refuses an answer that
was not actually researched, because an eager `null` is the expensive failure.

## 7. Minimal correct client

```python
import httpx

async def enrich(company, domain="", city=""):
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post(f"{BASE}/enrich-company",
                         json={"company": company, "domain": domain,
                               "city": city, "limit": 3})
        r.raise_for_status()
        out = r.json()
    if out["status"] != "success":
        return []                      # parse_error is a normal no-answer
    return [e for e in out["employees"] if e.get("source")]
```

Check `GET /health` first, keep `source` and `confidence` on every record, and
never let a `success` status alone promote a claim to a verified fact.
