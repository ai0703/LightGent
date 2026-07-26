# LightGent Enrichment SFT Dataset, Full Design

Status: DESIGN, approved sizes pending Abdul. 2026-07-26.
Owner task: given a company (name, domain, country), return the owner or top
decision maker as strict JSON, grounded in fetched evidence, null when the
evidence does not support an answer.

## 1. Why the v1 dataset failed (measured, not guessed)

The v1 set (81 train / 8 val, banked 2026-07-26 under a SATURATED search
layer) taught the model to over-search and never conclude:

- The answer was visible at tool result 2 (median) of 15, so **71 percent of
  all tool calls in the training data were unnecessary**. The model learned
  the dithering faithfully.
- The final answer was **4.65 percent of training characters**. The stop
  signal was diluted 20:1 by the search signal.
- Only 4 of 89 rows were negatives. Nothing taught "conclude null after a
  real search" as a distinct skill from "keep searching".
- n=20 production benchmark of the v1 model: 17/20 surfaced the right person,
  1/20 emitted the JSON. It hunts the OPTIONAL linkedin_url forever.

## 2. Terminology (used everywhere below)

| Term | Meaning |
|---|---|
| **Rung** | A research layer the agent can climb to. R0 own-site, R1 registry, R2 social-search, R3 third-party web, R4 news/archive. |
| **Discovery step** | The tool-result index where the correct owner surname first appears in evidence. |
| **Confirm window** | Research groups kept AFTER discovery (default 1) so the model learns one verification step, not zero and not ten. |
| **Archetype** | The shape of a trajectory (T1 to T6 below). The dataset is a controlled mix of archetypes, not whatever the teacher happened to do. |
| **Conclude nudge** | The runtime user message injected near the iteration cap ("output your FINAL JSON now"). Must appear in training so the model obeys it. |
| **Regrounding** | After truncation, stripping any URL from the gold answer that only appeared in a deleted step. Zero tolerance for citing unseen pages. |
| **Gold** | An answer is gold only if (a) the surname appears in kept evidence and (b) an independent labeller (Haiku) extracting from the same evidence agrees with the teacher. |
| **Sealed test** | The frozen 60-domain holdout (seed 42, hashes in holdout_manifest.json). Used once, at the end. Never mined for training. |
| **Guard metrics** | premature rate (answered with <3 tool results), null rate, accuracy, searches per correct answer. All four reported together, always. |

## 3. What the evidence ladder actually is (measured on 84 solved rows)

Attribution of the winning source, joined through tool_call_id (the earlier
"86 percent other-web" number was an attribution bug):

| Rung | Share of answers | Discovery step (median) |
|---|---|---|
| R0 own-site (team/about/contact page) | **67 pct** | 1 |
| R3 third-party web | 20 pct | 2 |
| R1 registry (KVK, drimble, companyinfo) | 7 pct | 2 |
| R2 social (LinkedIn SEARCH snippets, never fetches) | 5 pct | 2 |
| R4 news/archive | 1 pct | 8 |

Depth spread: 57 pct solve at step 1, p90 is 4 steps, max 8. This spread is
why no fixed script works and why the model must learn the climb decision:
"answer now, or climb one rung" is the entire learned skill.

## 4. Target composition (350 rows)

| Archetype | Rows | Share | What it teaches |
|---|---|---|---|
| **T1 instant-hit**: R0 answers it, 1-2 steps, conclude | 115 | 33 pct | Answer when the evidence is in hand. Kills the dithering. |
| **T2 short-climb**: R0 fails, one escalation (R1/R2/R3), 2-3 steps | 90 | 26 pct | The single-rung climb decision. |
| **T3 deep-climb**: 4-8 steps, multiple rungs, includes the conclude nudge | 50 | 14 pct | Persistence WITH a terminus. The tail that v1 barely had (12 rows). |
| **T4 recovery**: a wrong lead surfaces first (departing CEO, staff member), trajectory corrects to the right person | 25 | 7 pct | Not trusting the first name seen. Recency. Research says recovery successes are as valuable as clean ones. |
| **T5 exhausted-negative**: real full search, owner not findable, conclude owner_name null after the nudge | 45 | 13 pct | An honest null is a valid TERMINAL action, reached by searching, not by laziness. |
| **T6 distractor-negative**: evidence names people who are NOT the owner (staff lists, Apollo-style contacts), correct output is null or the actual owner | 25 | 7 pct | The anti-eagerness contrast. Names in evidence are not automatically answers. |

Negatives total 20 percent (v1: 4.5 percent). Positives keep the measured
depth spread rather than a uniform one, because the deployment distribution
IS 57 percent step-1 companies.

Field completeness stays as measured in curated v1 (owner_name 96 pct filled,
title 80 pct, linkedin_url 27 pct): the data itself demonstrates that
linkedin_url null is normal and acceptable. No gold row may fill a field the
kept evidence does not support.

## 5. Where each slice comes from

| Source | Supplies | Cost |
|---|---|---|
| Curated v1 (89 rows, truncate at discovery+1, regrounded, 0 unsupported) | ~70 T1/T2, ~8 T3 | $0, done |
| Re-bank with teacher + WORKING search (Serper tier 0), new domains from the 62.7k NL stock, sector-mixed (agri, marketing, law, accounting, real estate) | T1-T3 topup, most of T4 | see lanes below |
| **Ablation of positives** (see 5b) | all of T5 and T6 | $0, no teacher |
| n=20 benchmark failures (model had the name, would not conclude) | rewritten as T1/T2 with correct terminus | $0, on disk |

### 5b. Negatives come from ABLATION, not from the negatives file

**Correction, 2026-07-26.** `rescued-evals/negatives_to_verify.json` is NOT a
negatives list. Checked against gold: **12 of the first 12 entries are cases
where OUR enrichment was right and the Apollo ground-truth column was wrong**
(abbreeding.nl found Gadret, gold says gadret, Apollo said briga/groot/
pantophlet). It is a list of Apollo disagreements, i.e. mostly positives.
This matches the standing rule never to score against Apollo columns, which
list scraped staff rather than owners.

Real "owner genuinely not findable" cases occur at only ~4.5 percent
naturally (4 of 89 in v1), so harvesting 70 of them would mean banking ~1,500
companies. Instead, synthesise them from positives, which is standard
augmentation for teaching abstention and costs nothing:

- **T5 exhausted-negative**: take a positive trajectory, replace the ONE tool
  result that carried the owner name with a realistic barren result (empty
  list, or a page with no names). Keep every other step and the full length.
  Rewrite the final answer to `owner_name: null`. The trajectory now honestly
  shows a full search that found nothing, and the correct terminal action is
  an honest null.
- **T6 distractor-negative**: same ablation, but choose positives whose
  surviving evidence still names OTHER people (staff, authors, unrelated
  profiles). Correct answer is still null. This is the direct antidote to
  "a name is present, therefore answer it".

Both are grounded by construction: the answer is null and the evidence
genuinely lacks the owner, so no gate can be violated. Ablation targets are
picked only from rows where the discovery step is unambiguous, so we never
accidentally leave the name in a second result.

### 5c. Teacher lanes: CLI shim, $0 marginal (SELECTED 2026-07-26)

Abdul ruled out another GLM run on cost. Both affordable teachers are CLIs,
not APIs, so `finetune/cli_teacher_shim.py` serves them over the OpenAI
protocol the pipeline already speaks:

| Lane | Teacher | Bills to | Shim concurrency |
|---|---|---|---|
| sol | `codex exec` gpt-5.6-sol | ChatGPT Plus flat rate | 4 |
| haiku | `claude -p --model haiku` | Claude subscription | 8 |

Running BOTH lanes in parallel roughly doubles throughput and gives teacher
diversity, which the literature treats as a benefit rather than a defect.

Three failure modes are already handled in the shim, all found by smoke test:

1. **Evidence-free answers (the poison lane).** Sol recognises real companies
   and answers from world knowledge, producing a trajectory with ZERO tool
   results. Training on that teaches fabrication directly. The shim
   mechanically refuses any non-tool-call reply when the transcript has no
   tool results, and retries with an explicit rejection notice. Verified: the
   guard converted an instant evidence-free null on vermeerboomkwekerij.nl
   into a genuine 3-iteration, 8-tool-call deep-climb trajectory that
   concluded correctly (Aad Vermeer).
2. **Sloppy JSON.** Sol occasionally emits unbalanced braces; one correction
   round trip fixes it.
3. **Windows temp handles.** `mkstemp` keeps the file open so codex cannot
   write to it; the shim hands codex a path inside a scratch dir instead.

Codex's own `web_search` is disabled (`-c tools.web_search=false`) so every
fact must arrive through OUR tools and therefore into the trajectory.

Both lanes verified against gold on the same company (3dagroadvies.nl ->
Diana Zoer, correct via sol and via haiku).

Serper budget check: ~6 searches/company x 350 companies = ~2,100 of the
2,500 free tier. Feasible ONLY if the teacher fetches own-site first (it
does, 74/81 openings) and some searches route through recovered SearXNG.
If it runs out mid-bank, pause and continue after reset, or Abdul approves
the $50 top-up. Never silently fall back to broken search: banking under
garbage search is exactly what poisoned v1.

## 6. Quality gates (every row passes all, order matters)

1. **Surname-in-evidence**: gold owner surname must appear in a KEPT tool
   result. Fails = row dropped (teaches fabrication otherwise).
2. **Dual label**: Haiku independently extracts from the same kept evidence.
   Disagreement with the teacher = row to the audit pile, never to train.
3. **Truncate at discovery + confirm window (1)**, negatives kept full
   length. Positives median 2 steps, spread preserved 1-8.
4. **Reground**: strip URLs not present in kept evidence. Zero unsupported
   answers (verified property of the curated v1: 0 of 88).
5. **Conclude nudge inserted** in all T5 and half of T3 rows, exactly the
   runtime string, so obeying it is in-distribution.
6. **Length cap**: 14k tokens per row post-truncation (count from rendered
   text, never len() of a BatchEncoding).
7. **Domain-level dedupe and split**: a domain appears in exactly one of
   train/val. Val is a stratified 10 percent (every archetype represented).
8. **Sealed test untouched**: the 60 frozen holdout domains are excluded
   from banking lists by manifest hash, not by convention.
9. **50-row human audit** (plan rev 4 gate): stratified over archetypes,
   Abdul signs before training.

## 7. Training recipe changes that ride along

- **Assistant-only loss masking** (standard practice): loss on assistant
  turns only, not on tool results and system prompt. With the curated mix the
  final answer rises from 4.65 to ~8 percent of characters, and masking the
  tool-result tokens roughly triples its gradient share again. No
  final-turn-only masking: tool-call turns still carry the query-phrasing
  skill the v1 model visibly learned.
- **Loop guards ship regardless of training** (they make the eager-null
  failure impossible): floor = no accepted answer before 2 tool results
  unless evidence names the owner; ceiling = tool_choice "none" on the
  conclude turn (the current nudge still OFFERS tools, which is why it has
  never once worked).
- Same base (Qwen3-4B-Instruct-2507), same QLoRA config, same 2-tool schema
  as runtime. Interface consistency between train and serve is a feature: the
  shortcut-learning risk in the literature applies to CHANGED interfaces,
  and ours is fixed.

## 8. Evaluation (before/after, same harness)

Dev set: the 20-company benchmark plus a 30-domain Friesland dev slice with
ground truth (never the sealed test). Report the four guard metrics side by
side for stock Qwen, v1 adapter, v2 adapter. Ship gate: v2 beats v1 on
correct answers per search AND premature rate stays under 10 percent AND
accuracy is at least 55 percent on dev. Only then spend the sealed test.

## 9. Cost summary

| Item | Cost |
|---|---|
| Banking via CLI shim (sol + haiku) | **$0 marginal**, subscription flat rates |
| Negatives via ablation | $0, no teacher call |
| Training (A100-80GB, ~25 min) | ~$1.40 |
| Eval on L40S | ~$0.50 |
| **Total Modal spend** | **~$2 of the $11.72 remaining** |

No GLM run, no new API keys required. Serper stays the search tier-0 for
banking; at ~6 searches per company the remaining free allowance covers a few
hundred companies, and the run pauses rather than degrading if it runs out.
Banking under broken search is what poisoned v1 and must never be repeated.
