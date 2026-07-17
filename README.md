# LightGent — a lightweight Claygent clone on a free Colab GPU

A Clay-style AI research agent that web-searches, reasons, and answers
questions to enrich rows — powered by a **free open-source LLM served from
Google Colab** (16 GB T4) behind a Cloudflare tunnel, with **Jina AI** for
web scraping, instead of a paid API. Give it a prompt and output fields (or a
JSON Schema); it plans, searches, reads pages, and returns verified structured
data. The self-hosted alternative to Clay's "Claygent" — only cost is an
optional proxy.

```
CSV / any caller
      │
      ▼
Orchestration layer (runs anywhere — your PC or a server)
  A. lightgent_agent.py   — OpenAI Agents SDK (primary)
  B. lightgent_service.py — FastAPI custom loop (fallback: has tag-mode
      │                     tool-calling for servers without a tool parser)
      │  OpenAI-compatible chat + tools          │ tools execute locally
      ▼                                          ▼
Cloudflare quick tunnel                   SearXNG (self-hosted on Colab)
      ▼                                   Jina reader (r.jina.ai, hosted API)
vLLM + model on Colab GPU  ←  colab_llm_server.ipynb
(the ONLY thing Colab exposes: one API-key-protected /v1 endpoint)
```

## Files

| File | What it is |
|---|---|
| `colab_llm_server.ipynb` | Run in Colab: installs vLLM, serves a model, opens the tunnel, prints `LLM_BASE_URL` |
| `colab_tpu_llm_server.ipynb` | EXPERIMENTAL: same thing on Colab's TPU v5e-1 runtime via `vllm-tpu` |
| `lightgent_agent.py` | Orchestration on the **OpenAI Agents SDK** — CLI + importable `run_research()` |
| `lightgent_service.py` | FastAPI custom-loop service — same job, keeps the tag-mode fallback for models without native tool calling |
| `batch_enrich.py` | Clay-table feel: CSV in → enriched CSV out ({{column}} prompt, fields or schema, multi-endpoint) |
| `benchmark.py` | Measures tok/s and req/s of the served model through the tunnel |
| `search_test.py` | Isolated SearXNG diagnostic (no LLM) — per-engine probe |
| `.env.example` | Copy to `.env`, paste the notebook's printed values |

## Batch enrichment (the Clay "Use AI" column)

`batch_enrich.py` runs each CSV row through the agent and writes answers back
as columns. It mirrors Clay's column UI:

- **Variable prompt** — `{{Column}}` placeholders are filled per row (case-
  insensitive on the column name):
  ```
  python batch_enrich.py leads.csv \
    --prompt "Research {{Company}} at {{Domain}} and say what they sell." \
    --field what_they_sell:"one sentence" --field business_model:"B2B or B2C"
  ```
- **Output as Fields** (`--field NAME:DESC`, repeatable) → adds those columns
  plus `lg_confidence` / `lg_sources`, or **JSON Schema** (`--schema file.json`)
  → adds one column per top-level schema property.
- Runs **in-process** (no server needed) so Claude Code can drive it directly.

## Scale to 10× — run many Colab notebooks

Each free Colab GPU handles ~3 concurrent agents. To go faster, run the
notebook in **N Colab sessions** (separate tabs, or separate Google accounts)
and register them all — the batch runner load-balances rows across every
session and pools their SearXNG backends, so N notebooks ≈ N× throughput.

The easy way is a **notebook registry**, `endpoints.json` (gitignored — copy
`endpoints.example.json`). Each session's cell 5 prints a ready-to-paste line:

```json
[
  {"llm": "https://session-1.trycloudflare.com/v1", "searxng": "https://s1.trycloudflare.com"},
  {"llm": "https://session-2.trycloudflare.com/v1", "searxng": "https://s2.trycloudflare.com"}
]
```

Then just run `batch_enrich.py` — it auto-loads `endpoints.json`,
**health-checks every session and drops dead tunnels** (Colab sessions are
ephemeral; some will drop), and spreads the CSV across the survivors. Use the
**same** `LIGHTGENT_API_KEY` Colab secret in every session. Falls back to
`.env` (`LLM_BASE_URLS` / `SEARXNG_URLS`) when there's no registry file.

## Quickstart

1. **Colab:** upload `colab_llm_server.ipynb` to [colab.research.google.com](https://colab.research.google.com), set Runtime → GPU, Run All. Change `API_KEY` in cell 1 first. Copy the three printed `LLM_*` lines.
2. **Local:** in this folder:
   ```
   pip install -r requirements.txt
   copy .env.example .env       (paste the LLM_* values)
   uvicorn lightgent_service:app --port 8100
   ```
3. **Test one row:**
   ```
   curl -X POST http://127.0.0.1:8100/research -H "Content-Type: application/json" -d "{\"task\": \"Find the owner or CEO of this company\", \"context\": {\"company\": \"Bakkerij Holtkamp\", \"city\": \"Amsterdam\"}, \"output_fields\": {\"owner_name\": \"full name\", \"owner_title\": \"job title\"}}"
   ```
4. **Batch a CSV:** see the docstring in `batch_enrich.py`.

## Using it from your own code

The Colab box is a standard OpenAI-compatible endpoint, so any existing tool
that takes a base URL can point at it — set `LLM_BASE_URL`, `LLM_API_KEY`,
`LLM_MODEL` to the values the notebook prints. Keep in-flight requests low
(~3 per Colab GPU); a single free T4 will queue-explode under heavy concurrency.

## Model choice for the 16 GB T4 (free Colab)

Requirements: fits 16 GB, reliable tool calling, non-thinking (thinking
models burn tokens = slow on T4), fast enough to leave throughput headroom.
T4 caveats that drive this: no bf16 (SM75), and AWQ/GPTQ run on slow
fallback kernels (the fast Marlin kernels need newer GPUs) — so a small
fp16 model often BEATS a bigger quantized one on speed.

| Model | Weights | Tool calls | Est. speed (single stream) | Verdict |
|---|---|---|---|---|
| **Qwen3-4B-Instruct-2507 (fp16)** | ~8 GB | native (hermes) | ~35–45 tok/s | **DEFAULT** — best brains-per-GB, ~7 GB left for KV cache = real concurrency headroom |
| Qwen2.5-7B-Instruct-AWQ | ~5.5 GB | native | ~25–35 tok/s | quality bump option, slower per token |
| Llama-3.1-8B-Instruct-AWQ | ~5.8 GB | OK | ~25–35 tok/s | fallback if Qwen misbehaves |
| Qwen3-8B / 14B (thinking) | 6.5 GB+ | native | slow (thinks first) | avoid on T4 |

## Throughput — what to expect and how to measure it

**Measured 2026-07-16** (Qwen3-4B fp16, free-tier T4, through the tunnel):

- **Single stream:** ~20 tok/s.
- **Sweet spot: concurrency 4** — 0.50 req/s, ~88 tok/s aggregate;
  at concurrency 8 latency doubles and req/s DROPS. Keep `MAX_CONCURRENT=3`.
- **Native tool calling: works** (hermes parser) — 1.5 s round trip.
- **Full enrichment, end-to-end:** ~20 s for an easy 2-turn task (CEO from
  the company site, 4 pages fetched in parallel); expect 1–2 min for hard
  multi-search tasks → roughly **60–120 leads/hour** per Colab session.

Measure the real numbers after the notebook boots:

```
python benchmark.py --url https://<tunnel>.trycloudflare.com/v1 --key <API_KEY> --model <MODEL>
```

It reports single-stream tok/s, then req/s + aggregate tok/s at
concurrency 2, 4, 8 — raise `MAX_CONCURRENT` in `.env` until req/s stops
improving, then back off one step (that's your headroom setting).

## TPU variant (experimental)

Colab's TPU runtime is now a **v5e-1** (1 chip, ~16 GB HBM, ~3× a T4's
compute) and vLLM ships an official [`vllm-tpu`](https://pypi.org/project/vllm-tpu/)
package that supports v5e — same `vllm serve` CLI, same OpenAI API.
`colab_tpu_llm_server.ipynb` tries exactly that. Known differences vs GPU:

- No AWQ/GPTQ quantization on TPU → unquantized bf16 model, so the model
  ceiling on 16 GB HBM is ~4–7B (default: Qwen3-4B-Instruct-2507).
- First startup XLA-compiles the model: +10–20 min before ready.
- vllm-tpu targets Cloud TPU VMs; Colab is close but unproven — if it fails,
  the GPU notebook is the fallback, and the debugging hints are in cell 4.
- Order of preference stays: GPU notebook (proven) → TPU (if GPU quota is
  exhausted and you want to experiment) → free API tiers (Groq / Gemini /
  OpenRouter — zero infra, stable URL, just change `LLM_BASE_URL`).

## Caveats (read before relying on this)

- **Tunnel URL rotates** on every notebook restart → update `.env` /
  `endpoints.json` each time. A stable URL needs a *named* Cloudflare tunnel on
  a domain you control — one-time setup in the Cloudflare dashboard, then
  replace the quick-tunnel cell with `cloudflared tunnel run --token <token>`.
- **Colab sessions die**: free tier disconnects after idle/≈12 h and free-tier
  terms frown on long-running background services — fine for dev/testing and
  batch runs you babysit, not a 24/7 production backend.
- **Concurrency**: one Colab GPU ≈ 2–4 concurrent agents max (`MAX_CONCURRENT=3`);
  scale out with more notebooks (see "Scale to 10×") rather than up.
- **Model quality**: the T4's 4B model is weaker than a big model — expect more
  `parse_error`/low-confidence rows. On Colab Pro (L4/A100) the auto-selected
  bigger models close most of the gap.
- **Secrets**: the tunnel is public — anyone with URL + API key uses your GPU.
  Set a real `LIGHTGENT_API_KEY` (Colab secret), never commit `.env` /
  `endpoints.json`.

## Proxy (recommended)

SearXNG scrapes Google/Bing/DuckDuckGo/Brave, and Colab's shared IPs get
CAPTCHA'd fast — a residential proxy fixes it (set `SEARXNG_PROXY` as a Colab
secret). A cheap option that's been verified working here:
[Proxiware](https://proxiware.com) **Eco Residential — ~$3 per 10 GB**
(`socks5h://user-...-network-eco:pass@proxy.proxiware.com:1337`). Use the
**rotating** (`-network-eco`, no `-session-…`) credential so each request gets
a fresh IP. Search traffic is light, so 10 GB goes a long way. Without a proxy,
only a weak Google wrapper survives and `site:`/exact-match results degrade.
