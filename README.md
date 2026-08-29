# ThinkSpark-v2-350M — a tiny full-duplex *floor controller*

> A 270M-parameter "conversation referee" (Gemma-3-270M + Mimi audio tokens) that watches
> the caller every **80 ms** and decides **when** to listen, hold, interrupt, or
> back-channel — turning any `STT → LLM → TTS` cascade into a **full-duplex** voice agent
> *without* timestamps, agent audio, or vendor lock, while keeping your own Indic LLM.

The referee never writes the answer (your LLM still does that). It only emits (a) internal
**control flags** and (b) plain speakable **back-channels** ("haan", "right", "ek minute…"),
which is exactly the one thing a normal half-duplex pipeline cannot do.

> **Naming.** The base weights are **Gemma-3-270M** (chosen for native Devanagari/Gujarati
> tokenization — see Architecture). This project is shipped as **`thinkspark-v2-350m`** per
> the product name; the base model id stays `google/gemma-3-270m` everywhere in code.

---

## Why (the 8 behaviours)

A plain cascade is a walkie-talkie: one side holds the floor, hand-off gated by a dumb
silence timer. Humans are full-duplex. ThinkSpark learns all 8 behaviours while keeping
your LLM:

`B1` barge-in · `B2` back-channel · `B3` overlap/cross-talk · `B4` dynamic turn-taking ·
`B5` mid-sentence correction · `B6` thinking-sounds on incomplete turns ·
`B7` dead-air breaker · `B8` forceful interrupt.

The external **VAD is deleted** — the referee does VAD, endpointing, barge logic and
back-channel timing itself; STT runs *in parallel*, not as a gate.

---

## Architecture (two heads, one small model)

```
User mic  ─ Mimi encoder (cb0 @12.5Hz + energy,f0) ┐
Agent-state flag (IDLE/LLM_GEN/TTS_SPEAKING/DONE) ──┤
Agent text (rolling, no timing) ────────────────────┼─► Gemma-3-270M ─┬─► control head  <LISTEN> <BARGE_*> <TURN_END> …
STT partials (optional) ────────────────────────────┤   (KV-cache      │    (internal, never spoken)
System prompt (persona/domain) ─────────────────────┘    streaming)    └─► spoken head   "haan", "ek minute", "right"
                                                                              (plain text → any TTS)
```

- **Control head** → one of 11 flags per 80 ms frame (internal).
- **Spoken head** → plain multilingual text sent to your TTS (no bracket tags → no vendor lock).
- **Agent-state channel** is a live model *input* that replaces TTS char-timestamps — the
  thing that killed vendor lock (works with any TTS: ElevenLabs, Google, Azure, Sarvam…).

Base model is **Gemma-3-270M**: its 256K-token embedding table reads Devanagari/Gujarati
natively while the active transformer stack is small (~100M), so per-frame decode is a
single incremental step — single-digit ms on GPU, ≤ 25 ms on CPU, inside the 80 ms frame.

The code layout mirrors the pipeline (see `thinkspark/__init__.py` for the full map):
`vocab · schema · config · distribution · prompts · llm_client · tts_soniox · mimi_codec ·
frames · validators · dataset · model · losses · metrics · trainer · inference`.

---

## Full docs (Mintlify)

A complete two-tab docs site lives in [`docs/`](docs/) — **Literature** (theory, diagrams,
architecture, math) and **Commands** (every run command, step by step, plus a one-page
cheatsheet). Preview it locally:

```bash
cd docs && mint dev
```

## Setup

```bash
conda activate llms
pip install -r requirements.txt
cp .env.example .env      # fill in OPENAI_API_KEY / SONIOX_API_KEY / HF_TOKEN
```

- **OPENAI_API_KEY** — the scenario/label writer uses the OpenAI SDK. Set `OPENAI_BASE_URL`
  to target any OpenAI-compatible endpoint (Sarvam v2, vLLM, Together…).
- **SONIOX_API_KEY** — Soniox TTS renders the *user* side only (agent side stays text).
- **HF_TOKEN** — pulls the gated Gemma-3-270M weights (and Mimi).

Everything is built to run on **Kaggle 2×T4** (`conda activate llms`); single-GPU and
CPU/MPS also work (slower). Nothing runs automatically — you drive each stage by hand.

---

## The pipeline (run stage-by-stage)

`scripts/run_all.sh` lists every command in order; run them one at a time to respect
Kaggle's 9 h/session budget. Each stage is **resumable**.

### 0 · Sample data first (offline, no API)
```bash
python scripts/make_samples.py          # writes data/samples/*  (read data/samples/README.md)
```

### 1 · Generate scenarios with the LLM — just run it (Section 8.1–8.4)
```bash
python scripts/02_generate_scripts.py --config configs/data_gen.yaml
```
That's the whole command, same simplicity as `kupe-thinkspark`/`kupe-tts`'s generators:
the balanced plan (Section 8.1–8.3 — gender-balanced, behaviour/language-balanced,
~9k scenarios ≈ 55 h, split into `num_parts` shards) is **auto-built on first run** — no
separate step required — and every re-run **resumes automatically** toward the same
target. The back-channel bucket is split so ~35 % of it is the deliberate **silent** case
(say nothing — the natural move mid-sentence).

Want to inspect the plan before spending any budget, or just prefer an explicit step?
```bash
python scripts/01_plan_distribution.py --config configs/data_gen.yaml   # optional, same numbers
```
Both call the exact same `thinkspark.distribution.write_plan()` — there is only one
plan-building code path, so the two can never disagree.

Each LLM call requests `batch_size` (config, default **12**) DISTINCT scenarios in one
completion — much faster than one-at-a-time at a given concurrency. `llm_concurrency`
(config, default **30**) is how many worker threads issue batched calls in parallel.
Splitting a big run across Kaggle's 9 h/session budget by hand still works:
```bash
python scripts/02_generate_scripts.py --config configs/data_gen.yaml --part 0
```

**Running in a range.** Prefer your own chunk sizes over the fixed 8 `--part` shards?
`--range START:END` selects a slice of the global scenario index (0 up to the plan's
total — 9009 by default) and only generates that slice — run these in any order, any
number of sessions:
```bash
python scripts/02_generate_scripts.py --config configs/data_gen.yaml --range 0:3000
python scripts/02_generate_scripts.py --config configs/data_gen.yaml --range 3000:6000
python scripts/02_generate_scripts.py --config configs/data_gen.yaml --range 6000:9009
```
All three write into the same `scenarios_all.jsonl`. Resumability is governed by the same
per-job count as always, so overlapping or out-of-order ranges still converge to exactly
the right total with zero duplicates — see the docs for the one caveat (global-index
labeling precision, not data correctness).

**SQLite tracks what was actually generated, not just cost.** Alongside `llm_calls` /
`tts_calls` (cost) and `unit_evals` (pass/fail per scenario), a `scenario_registry` table
records every scenario_id that made it into the shard, tagged with its `global_index` —
so `--range` progress is queryable straight from SQL:
```sql
SELECT COUNT(*) FROM scenario_registry WHERE global_index >= 3000 AND global_index < 6000;
```

**Does batching confuse the model?** It can, if you push it too far. Asking a small/fast
model (DeepSeek-flash-class, Gemma-3-27B-class — see "Choosing a generation model" below)
for many structurally-identical JSON objects in one response risks near-duplicate items,
language/behaviour bleeding between items, or a truncated array if `max_tokens` wasn't
scaled up. Three mitigations are built in (`thinkspark/prompts.py`,
`scripts/02_generate_scripts.py`):
1. The batch prompt explicitly demands **distinct** items and forbids paraphrase-only variation.
2. `max_tokens` is scaled by `batch_size` so the array is never cut off mid-response.
3. Every item is schema-validated **independently**; only the scenarios still missing are
   re-requested on the next pass — one bad item never discards a whole good batch.

If you see repeats or drift in practice, lower `batch_size` in `configs/data_gen.yaml`
(`1` = fully single-scenario calls — slowest, most reliable). Keeping it ≤ 15 is a good
rule of thumb regardless of model.

**Resumability.** The JSONL shard is the source of truth: on restart, the script counts
how many scenarios already exist per job (by an embedded `_job_key`) and only tops up the
remainder — a killed or corrupted run always continues from exactly where it stopped,
with no re-generation of work already done.

**Cost + audit trail.** Every LLM call (tokens in/out, cost, batch outcome, latency) and
every TTS call (script 3) is logged to a SQLite DB at `data/thinkspark_runs.db`
(`thinkspark.db`, WAL mode, thread-safe) — independent of the shard file, so you always
have a full cost history even across many resumed sessions. Export it any time:
```bash
python scripts/10_export_costs.py       # prints a summary + writes reports/cost_report.csv
```
Fill in `llm_price_in_per_1m_usd` / `llm_price_out_per_1m_usd` in `configs/data_gen.yaml`
for your chosen provider — pricing varies and isn't hard-coded (0.0 means "unknown", not
"free"). `soniox_price_per_hour_usd` defaults to the Section 13 figure.

**Live monitor.** While generation (or TTS rendering) is running, open a *second*
terminal and watch progress/cost/latency update in real time:
```bash
python scripts/11_monitor.py --config configs/data_gen.yaml         # live, refreshes every 5s
python scripts/11_monitor.py --once                                  # single snapshot, exit
```
It reads only from the SQLite DB (never writes, safe to run alongside the generator) and
shows: scenarios done vs. target, **unit-level Section 8.5 eval** (every scenario checked
the instant it's produced — pass/fail counts, job-scoped `fail1`/`fail2`/… retry tags,
worst-retries seen), LLM + TTS latency (mean/p50/p95, and an estimated seconds-per-scenario),
throughput and ETA, and the budget roll-up — **cost spent so far**, **projected total cost**
(extrapolated from the observed cost-per-scenario rate against the plan's target), and
**how much budget is left** against `budget_inr_target` (config, default ₹5000, Section 13).
Uses `rich` for a live dashboard if installed, else a plain-text refresh loop.

**Chart report.** For a shareable, self-contained HTML view (same dark-theme + Chart.js
convention as `kupe-tts`'s cost-EDA reports) — stat cards, cumulative-spend line chart, and
**actual-vs-planned-target bar charts by behaviour and by language** (the fastest way to see
the corpus is still exactly the balanced distribution Section 8.1–8.2 planned, not drifting):
```bash
python scripts/12_build_report.py --config configs/data_gen.yaml    # writes reports/generation_report.html
```
Safe to run anytime, including mid-generation — it only reads the DB + `plan_summary.json`,
never the distribution/validation logic itself, so the report can never disagree with what
was actually planned and produced.

**Choosing a generation model.** `llm_model` + `llm_base_url` in `configs/data_gen.yaml`
point at any OpenAI-compatible endpoint — e.g. a DeepSeek V3/V4-flash-class or
Gemma-3-27B-class host. Both are solid, cheap choices for this JSON-scenario-writing task;
just keep `batch_size` conservative (8-12) with either, since smaller/faster models are
exactly the ones most prone to the batching confusion described above — validate a small
`--limit` run first (`--limit 20`) before committing the full budget.

### 2 · Validate before you train on junk (Section 8.5)
```bash
python scripts/05_validate_data.py --in data/scenarios/scenarios_all.jsonl --judge --judge-n 200
```
Checks schema ≥ 99 %, vocab = 100 %, script ≥ 98 %, balance ± 2 %, LLM-judge naturalness ≥ 4.2.
The report also prints a **`user_text words`** line (mean/median word count + implied
seconds) so you can see at a glance how long your turns actually are.

### 2.25 · Longer user turns (~3-8 s extended, ~12-25 s long) — additive, merged before rendering
The default corpus produces short user turns (~1-2 s of audio). Because the frame-window
length is derived from the real audio and floored at ~1.3 s
([`frames.py`](thinkspark/frames.py) `_MIN_WINDOW`), an all-short corpus pins nearly every
window at that floor — so the model barely sees the temporal dynamics that endpointing,
mid-utterance pauses (`barge_lookalike`), and backchannel-during-speech actually depend
on, and it will endpoint too eagerly on the real 3-25 s turns it meets at inference.
**Duration matters here** — not because Mimi can't encode short clips (it can), but
because the *labels* the floor-controller learns are only meaningful over realistic spans.

Fix, without disturbing anything you've already generated: two additive length bands.
`utterance_length: extended` (~10-24 word / ~3-8 s, natural mid-utterance pauses) and
`utterance_length: long` (~25-55 word / ~12-25 s multi-sentence turns — the efficient way
to add many **hours**: ~3× fewer clips than extended for the same hours, so far fewer
LLM calls and Soniox stream starts; best mixed in, not used alone, since it isn't natural
for terse behaviours). Everything flows through the *same* render/encode/frame pipeline
(length-agnostic — longer text simply yields longer audio and longer windows):
```bash
# each writes to its OWN scenario files — your existing short corpus is untouched:
python scripts/02_generate_scripts.py --config configs/data_gen_extended.yaml \
    --plan-dir data/plan_extended --out-dir data/scenarios_extended
python scripts/02_generate_scripts.py --config configs/data_gen_long.yaml \
    --plan-dir data/plan_long --out-dir data/scenarios_long
```

**Merge straight into the main corpus before rendering** — extended/long share identical
Soniox/pricing settings with the main corpus, so there's no reason to track them
separately at all; everything renders as ONE corpus, through ONE meter, from ONE place.
Verify no `scenario_id` / wav-filename collisions first, then fold in:
```bash
python3 -c "
import json
a = {json.loads(l)['scenario_id'] for l in open('data/scenarios/scenarios_all.jsonl')}
b = {json.loads(l)['scenario_id'] for l in open('data/scenarios_extended/scenarios_all.jsonl')}
c = {json.loads(l)['scenario_id'] for l in open('data/scenarios_long/scenarios_all.jsonl')}
print('collisions:', len((a&b)|(a&c)|(b&c)))"   # must print 0 before merging

cat data/scenarios_extended/scenarios_all.jsonl data/scenarios_long/scenarios_all.jsonl \
    >> data/scenarios/scenarios_all.jsonl
```
That's it — `data/scenarios/scenarios_all.jsonl` now holds all three bands. Render with
the **main config**, same as step 3 below, nothing extended-specific needed anymore:
```bash
python scripts/05_validate_data.py --in data/scenarios/scenarios_all.jsonl   # check "user_text words"
python scripts/03_render_user_audio.py --config configs/data_gen.yaml \
    --in data/scenarios/scenarios_all.jsonl --audio-dir data/audio
```
Resumable exactly as always: whatever scenario already has a `.wav` (from ANY of the
original three corpora) is skipped; everything else renders. The dashboard's audio-hours
meter now reflects the TRUE whole-corpus total from the first line of output — no
per-corpus splitting, no meter that resets when you switch configs.

**Render, unattended, once — and sleep:**
```bash
bash scripts/18_render_all.sh                                          # stay attached
nohup bash scripts/18_render_all.sh --quiet > render_all.log 2>&1 &     # run it, sleep
python scripts/17_audio_summary.py                                     # real hours by band, anytime
```
`total_hours` in `configs/data_gen.yaml` (55.0) is now the TRUE overall target for the
whole merged corpus — no separate per-band total to keep in sync. Short/extended/long
scenarios all carry distinct SQLite keys (via `length_band`), so re-running generation
for any one band never mixes up resumability with the others, merged or not.

### 2.5 · Clone your own voices — required before rendering (no Soniox catalog is used)
By design, `thinkspark/tts_soniox.py::resolve_voice()` uses **only your own cloned
voices** — it never reaches for Soniox's built-in catalog. Drop reference clips into
`data/voice_refs/` named `female_<name>.wav` / `male_<name>.wav` (see
`data/voice_refs/README.md`; at least 1 clip per gender you plan to render), then:
```bash
python scripts/15_create_voice_profiles.py --config configs/data_gen.yaml --dry-run
python scripts/15_create_voice_profiles.py --config configs/data_gen.yaml
```
Clones each clip via Soniox's real voice-cloning API (`POST /v1/voices`, verified
against `kupe-backend`'s working implementation — not guessed), waits for Soniox to
prepare it for the TTS model (`recompute` + poll, same as `kupe-backend`), and writes
`data/voice_refs/voice_profiles.json`. `03_render_user_audio.py` reads this
automatically — no flags, no config changes — and **fails fast, before any API calls
or spend**, if a gender it needs to render has zero voices yet. Resumable:
already-cloned clips (tracked by content hash) are skipped on re-run.

**Submitted more clips than Soniox will clone?** Soniox caps how many voices *one
account* can have cloned at once (`soniox_max_cloned_voices`, default **20 total across
both genders**). If you submit more than that (e.g. 15 female + 15 male against a cap of
20), the script clones as many as the cap allows — split proportionally per gender
(10/10 in that example) — and fills the rest with named Soniox **stock** voices of the
matching gender (`thinkspark/soniox_default_voices.py`) so your final per-gender voice
*count* still matches what you submitted. Both kinds land in the same
`voice_profiles.json` and are used identically. `--dry-run` shows the exact split before
anything is cloned. `--cleanup` deletes every cloned voice from your Soniox account too
(best-effort) and resets the file, if you need to rebuild from scratch.

Each render deterministically rotates across your pool by a hash of the scenario text
(same text -> same voice, different scenarios -> spread across your voices) — see
`data/voice_refs/README.md` for the full breakdown.

### 3 · Render user audio with Soniox TTS (Section 8.4)
```bash
python scripts/03_render_user_audio.py --config configs/data_gen.yaml \
    --in data/scenarios/scenarios_all.jsonl
```
Synthesises only the user line + **character-level timestamps** from Soniox's TTS
response itself (`return_timestamps=true` — no separate STT call needed). Timestamps are
used **only to build training data** (frame calibration, Section 8.4 → `04_build_frames.py`)
— inference never needs them; the live agent-state channel (Section 4.3) replaces timing
entirely. The protocol (WS URL, message format, response field names) is verified against
a real working client, not guessed — see `thinkspark/tts_soniox.py`'s module docstring.

<details>
<summary><b>If you rendered audio before 2026-08-28, check it — an earlier version pointed at the wrong endpoint</b></summary>

An earlier version of this script connected to Soniox's **STT** endpoint by mistake
(`stt-rt.soniox.com`) instead of the **TTS** endpoint (`tts-rt.soniox.com`), and used the
wrong message protocol. It didn't error — it silently wrote empty (0-duration, no
timestamps) wav files while reporting "rendered ok". If your `data/audio/` predates this
fix, clean it up first:
```bash
python scripts/14_cleanup_corrupt_audio.py --config configs/data_gen.yaml --dry-run
python scripts/14_cleanup_corrupt_audio.py --config configs/data_gen.yaml
```
It scans for near-empty wavs / missing-or-zero timestamps, deletes the corrupt pairs, and
corrects their historical SQLite rows — re-running `03_render_user_audio.py` then
naturally re-renders exactly those scenarios with the fixed client. The dashboard now
also refuses to record 0-duration audio as a success at all, so this class of bug can't
recur silently.

**A second, separate bug** (found right after the endpoint fix, on the first real run
against it): the `voice` field was a made-up `"language-gender"` string (e.g. `"hi-female"`)
— Soniox voice IDs are actual character names (`"Priya"`, `"Arjun"`, `"Maya"`, ...), so
every real call failed with `Soniox error 400 (invalid_request)`. Fixed at the time by
resolving from Soniox's live `GET /v1/tts-models` catalog by gender. **That catalog path
has since been removed entirely** (see 2.5 above) — the project now uses only your own
cloned voices, never the catalog, by explicit choice.
</details>

The startup log prints `voice profiles: N of your own cloned voice(s) loaded from ...`
confirming which of your voices are in the rotation for this run.

Same live dashboard treatment as scenario generation (rendered/skipped/failed, audio
hours done vs. `total_hours` target, cost spent/projected/**gap** vs. budget,
throughput+ETA, and now an explicit **input-file total** so `target` — remaining work
this run — is never mistaken for "how many scenarios exist") — renders `soniox_concurrency`
(config, default **3**, Soniox's real default account-wide concurrent-stream limit) clips
in parallel, and retries a failure with exponential backoff, backing off harder and
tagging it distinctly if it looks like a rate limit (`SonioxRateLimitError` — see
`thinkspark.tts_soniox`). Lower concurrency (`-j 2`) if you see rate-limit hits. Cost is
computed per-request with Soniox's real token-based pricing (`soniox_cost_usd()` — $4/1M
input text tokens + $21.50/1M output audio tokens, the same formula kupe-tts verified),
not a flat $/hour guess, so **projected cost updates progressively** as real clips render.

**Speed, without exceeding your rate limits.** Two changes make a long render notably
faster while still respecting Soniox's real documented limits
(soniox.com/docs/tts/rt/limits-and-quotas), not looser than before:
- Each worker thread **reuses one persistent WebSocket connection** across every
  scenario it renders (`thinkspark.tts_soniox.SonioxTTS._get_ws`), instead of
  reconnecting per scenario — a real TCP+TLS handshake saved on every request but the
  first per thread. Concurrency (still `soniox_concurrency`, still one stream per
  thread at a time) is unchanged, so the account's 3-concurrent-stream cap is respected
  exactly as before; this only removes reconnect overhead, verified offline (mocked
  transport: same-thread calls reuse one connection, a broken connection self-heals by
  reconnecting on the next call, different threads never share a connection).
- New streams are now **proactively paced** under `soniox_max_stream_starts_per_min`
  (config, default **90** — a safety margin under Soniox's documented 100/minute
  account-wide cap) instead of only reacting to a 429 after the fact. A small, evenly
  spread wait here is cheaper than bursting past the real limit and paying the
  exponential backoff (starts at 3s, doubles) that a 429 triggers — verified offline
  (a tight test cap of 5/min correctly blocks the 6th call for close to a full window).

**Cleanup.** Delete every rendered wav + its `tts_calls`/`hf_sync` SQLite rows for a given
`--audio-dir`, with an "are you sure?" confirmation first (same pattern as script 02's
own `--cleanup`):
```bash
python scripts/03_render_user_audio.py --config configs/data_gen.yaml --cleanup
```
Scoped deliberately — leaves `llm_calls`/`unit_evals`/`scenario_registry` (scenario
*generation*, not rendering) completely untouched; use script 02's `--cleanup` for that
instead.

### 4 · Get the Phase-1 free corpus (Section 7)

**Recommended — run this locally (e.g. your Mac), not on Kaggle:** one command
downloads every language+source concurrently, encodes + builds frames for each language
the instant its downloads finish (while the others keep downloading), and continuously
uploads finished languages to Hugging Face in the background — all with live, tagged,
timestamped logs across every stage:
```bash
pip install datasets soundfile huggingface_hub
export HF_TOKEN=hf_...   # WRITE access — only for the upload step, downloads need none

python scripts/P1_00_pipeline.py --config configs/phase1_corpus.yaml \
    --hf-repo <your-hf-username>/kupe-thinkspark-270m-phase1-data --dry-run
python scripts/P1_00_pipeline.py --config configs/phase1_corpus.yaml \
    --hf-repo <your-hf-username>/kupe-thinkspark-270m-phase1-data
```
Fully resumable (Ctrl+C, re-run, every stage picks up exactly where it left off — the
manifest for downloads, existing `.npz` for encoding, a `hf_sync` SQLite table for
uploads). `--no-upload` runs it purely locally, no HF repo needed. Then, on your
training machine (Kaggle), fetch the ready-to-train result in one command — see
[Fetching training data onto Kaggle](#fetching-training-data-onto-kaggle-phase-1--phase-2)
below.

<details>
<summary>Or the older, single-threaded, no-upload CLI (<code>scripts/P1_01_fetch_corpus.py</code>)</summary>

```bash
pip install datasets soundfile

python scripts/P1_01_fetch_corpus.py --config configs/phase1_corpus.yaml --dry-run  # see the plan first
python scripts/P1_01_fetch_corpus.py --config configs/phase1_corpus.yaml            # fetch (resumable)
```
Same underlying fetch logic (`thinkspark.phase1_corpus.fetch_source`), just one source
at a time and no HF upload — simpler for a quick one-off local fetch. Then encode +
build frames yourself, per language:
```bash
for lang in en hi gu; do
  python scripts/00_encode_audio.py --wav-dir "data/phase1_raw/$lang" --out-dir data/encoded
  python scripts/P1_02_build_frames.py --lang "$lang"
done
```
</details>

Streams (never fully downloads) the real, verified sources — per language, stops the
moment its target hours are reached, so disk/bandwidth stay bounded regardless of how
huge the upstream dataset is. **No `HF_TOKEN` needed to download at all** — every source
below is public/ungated (only the upload step needs one):

<details>
<summary><b>Common Voice was removed from this mix on 2026-08-29 — Mozilla pulled it off Hugging Face entirely</b></summary>

`mozilla-foundation/common_voice_17_0` (and every other Common Voice version) was used
here for en/hi/gu. As of this writing, the HF repo itself is genuinely empty — "the
Dataset Viewer is disabled pending file uploads" and Hugging Face's own notice reads:
"Effective October 2025, Mozilla Common Voice datasets are now exclusively available
through Mozilla Data Collective." This is **not** a gating/token/permissions issue — a
correct `HF_TOKEN` plus clicking "Agree" won't fix it, because there's no data left in
the repo to agree into. If you hit `failed to open mozilla-foundation/common_voice_.../
en: The directory ... doesn't contain any data files`, that confirms it.

Fixed by removing Common Voice everywhere and either substituting or redistributing its
share:
- **en**: replaced with `openslr/librispeech_asr` (public, ~1000h of read English
  speech, no gating) at the same 60% weight.
- **hi** / **gu**: Common Voice's share was redistributed proportionally across the
  remaining sources (Kathbath/Shrutilipi/FLEURS for hi; Kathbath/Shrutilipi/IndicTTS-
  Gujarati for gu) — no replacement needed, those sources are already large enough to
  absorb it.
</details>

| Lang | Sources (share of target hours) | Target |
|---|---|---|
| en | LibriSpeech (60%) + FLEURS (40%) | 150h |
| hi | Kathbath (45%) + Shrutilipi (40%) + FLEURS (15%) | 150h |
| gu | Kathbath (45%) + Shrutilipi (35%) + IndicTTS-Gujarati (20%) | 130h |

<sup>[openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) · [ai4bharat/Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath) · [ai4bharat/Shrutilipi](https://huggingface.co/datasets/ai4bharat/Shrutilipi) · [google/fleurs](https://huggingface.co/datasets/google/fleurs) · [SPRINGLab/IndicTTS_Gujarati](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Gujarati)</sup>

All free (CC0 or a one-click research license), none gated. Writes wavs + a resumable
`data/phase1_raw/manifest.jsonl` (transcript + gender where the source provides it —
LibriSpeech doesn't carry a gender field, which the fetcher handles gracefully, same as
Kathbath/Shrutilipi already did).

### 5 · Phase 0 — encode audio to Mimi (Section 4.2)
```bash
python scripts/00_encode_audio.py --audio-dir data/audio --out-dir data/encoded         # Phase-2 user audio
python scripts/00_encode_audio.py --wav-dir data/phase1_raw/hi --out-dir data/encoded   # Phase-1 free audio (per lang)
```
Writes `cb0` + `energy` + `f0` shards (`.npz`) at 12.5 Hz.

### 6 · Build per-frame training records (Section 5.3)
```bash
python scripts/04_build_frames.py --in data/scenarios/scenarios_all.jsonl \
    --frames-out data/frames/frames_all.jsonl                       # Phase 2
python scripts/P1_02_build_frames.py --lang hi                       # Phase 1, per language
```
Phase-2 calibrates the LLM's event offsets to the real Soniox timings and emits the
per-frame flag/agent-state/VAP/spoken-span labels. Phase-1 pairs each clip's Mimi shard
with its transcript (the alignment target) — see `thinkspark.losses.Phase1Loss`, which
only reads `align_labels` + `vap`, never `flags`.

### 7 · Train (Section 9)
```bash
# Phase 1 — modality alignment on ~400–450 h free audio (LoRA); needs frames from the open corpora
python scripts/06_train_phase1.py --config configs/train_phase1.yaml --frames "data/frames_phase1/*.jsonl"

# Phase 2 — referee fine-tune on the ~55 h synthetic corpus (focal + text + VAP)
python scripts/07_train_phase2.py --config configs/train_phase2.yaml --frames "data/frames/*.jsonl" \
    --init artifacts/thinkspark-v2-350m/phase1/final/model.pt

# 2×T4 DDP:
torchrun --nproc_per_node=2 scripts/07_train_phase2.py --config configs/train_phase2.yaml --frames "data/frames/*.jsonl"
```
Losses (exact, Section 9.1): Phase 1 `L = L_align + 0.3·L_vap`; Phase 2
`L = 1.0·L_ctrl(focal) + 0.5·L_txt + 0.2·L_vap`. Full bf16 fine-tune fits one T4 (~4.3 GB
weights+opt) with gradient checkpointing; DDP doubles throughput.

### 8 · Evaluate against the targets (Section 10)
```bash
python scripts/08_evaluate.py --config configs/train_phase2.yaml \
    --ckpt artifacts/thinkspark-v2-350m/phase2/final --frames "data/frames_val/*.jsonl"
```
Reports per-flag F1, barge F1 + false-barge rate, VAD-F1, endpoint latency, referee decode
p50/p95 vs. the Section 10 bars (barge F1 ≥ 0.85, false-barge ≤ 5 %, cutoff ≤ 3 %,
decode p95 ≤ 40 ms, …).

### 9 · Run the live referee (Section 11)
```bash
python scripts/09_infer_demo.py --config configs/train_phase2.yaml \
    --ckpt artifacts/thinkspark-v2-350m/phase2/final \
    --encoded data/encoded/<scenario_id>.npz --agent-text "aapka EMI due hai"
```
Drives `StreamingReferee` + `ReferenceOrchestrator` over a simulated frame stream and prints
the `agent-state → flag (→ spoken)` decision log. Wire the callbacks to your SDK / LiveKit /
Pipecat layer in production.

---

## Publish the corpus to Hugging Face (optional)

```bash
pip install huggingface_hub pyarrow
export HF_TOKEN=hf_...     # needs WRITE access

python scripts/13_upload_hf.py --repo <your-hf-username>/Thinkspark-v2-270m-training-data --dry-run
python scripts/13_upload_hf.py --repo <your-hf-username>/Thinkspark-v2-270m-training-data
```
`--repo` is required — no default namespace is guessed. It must be a namespace your
token can actually create repos under: your own username always works
(`anuj-inavlabs/...`); an org namespace (e.g. `kupe/...`) only works if you're a member
of that org with repo-create rights — otherwise you'll hit `403 Forbidden: You don't
have the rights to create a dataset under the namespace "..."`, which is a
membership/permission issue, **not** a bad token (the script now tells you exactly that
if it happens, and suggests your own username instead).

Uploads scenarios + rendered audio + Soniox timestamps to that HF dataset repo, sharded into resumable commits with
progress/rate/ETA logged every commit — same bulk-upload pattern as `kupe-tts`'s
`hf_bulk_upload.py`. Only rows with **both** real audio and non-empty text are included
(run [script 14](#3--render-user-audio-with-soniox-tts-section-84) first if your local
audio might be corrupt). On top of the raw files it also builds a **Dataset Viewer**
parquet — a real `Audio` feature column paired with `user_text` and every scenario field
(behaviour, language, domain, gender, prosody) — so the Hub UI shows a playable audio
player next to the transcript for every row, plus a dataset card README. **Also uploads
the raw `scenarios/scenarios_all.jsonl`** (full schema — the parquet alone is Viewer-
oriented and doesn't carry `target`/`event_char`, so it can't rebuild real training
frames on its own; the raw file is what makes the repo actually self-sufficient for
training reconstruction elsewhere). Resumable via a `hf_sync` table in the same SQLite DB
(`thinkspark.db`), so re-running only uploads what's still pending. Never deletes local
files.

### Fetching training data onto Kaggle (Phase 1 + Phase 2)

On your training machine, pull both phases straight into the local `data/` layout the
training scripts expect — no re-encoding for Phase 1, no reconstruction guesswork for
Phase 2:
```bash
pip install huggingface_hub

python scripts/19_fetch_training_data.py \
    --phase1-repo <your-hf-username>/kupe-thinkspark-270m-phase1-data \
    --phase2-repo <your-hf-username>/Thinkspark-v2-270m-training-data --dry-run
python scripts/19_fetch_training_data.py \
    --phase1-repo <your-hf-username>/kupe-thinkspark-270m-phase1-data \
    --phase2-repo <your-hf-username>/Thinkspark-v2-270m-training-data
```
- **Phase 1** lands as `data/encoded/*.npz` + `data/frames_phase1/*.jsonl` — already
  encoded and frame-built by `scripts/P1_00_pipeline.py`, ready to train immediately:
  `python scripts/06_train_phase1.py --config configs/train_phase1.yaml --frames "data/frames_phase1/*.jsonl"`.
- **Phase 2** lands as `data/scenarios/scenarios_all.jsonl` (the real full-schema file,
  not the Viewer parquet) + `data/audio/*.wav` + `data/audio/*.words.json`. Pre-encoded
  Phase-2 data isn't uploaded (only raw audio + text) — encode + build frames locally
  once, it's fast and free:
  ```bash
  python scripts/00_encode_audio.py --audio-dir data/audio --out-dir data/encoded
  python scripts/04_build_frames.py --in data/scenarios/scenarios_all.jsonl --frames-out data/frames/frames_all.jsonl
  ```

Either `--phase1-repo` or `--phase2-repo` can be omitted to fetch just the one you need.
Files are **moved** (not copied) out of the download snapshot into their final flattened
locations, so this never doubles your disk usage — resumable, an existing local file is
left alone and only what's missing gets fetched.

---

## Latency (why it feels instant)

Naïve cascade on turn-end: `VAD_wait(600) + LLM(400–800) + TTS(200) ≈ 1.2–1.6 s`.
ThinkSpark fires `PREFETCH_LLM` ~300–500 ms *before* the user finishes, so at `TURN_END`
the reply is already (partly) ready ⇒ perceived reply `≈ referee(≤40) + TTS_first_chunk(≤200)
≈ 0.2–0.3 s`. Back-channels are ≤ 120 ms (no LLM at all).

## Budget (INR 5000, Section 13)

~55 h Soniox user-audio (~₹3050) + Gemma-4B/OpenAI script+label gen (~₹500) + LLM-judge
(~₹300) + GPU buffer (~₹1150). Phase-1 open audio costs ₹0 in data (only free Kaggle GPU).

## Directory layout

```
thinkspark/            the package (one module per pipeline stage; db.py = SQLite log;
                        phase1_corpus.py = shared fetch logic; hf_upload.py = shared HF helpers)
scripts/               00–19 + P1_00/P1_01/P1_02 CLI stages + make_samples.py + run_all.sh
configs/               data_gen.yaml · phase1_corpus.yaml · train_phase1.yaml · train_phase2.yaml
data/samples/          committed, offline sample scenarios + frame records (read its README)
data/voice_refs/        your own reference clips (read its README) + voice_profiles.json (cloned IDs)
data/plan/              auto-built balanced job plan (Section 8.1-8.3; git-ignored)
data/phase1_raw/         fetched free-audio wavs + manifest.jsonl (Section 7; git-ignored)
data/{scenarios,audio,encoded,frames,frames_phase1}/   generated artifacts (git-ignored)
data/thinkspark_runs.db  SQLite audit/cost log for every LLM + TTS call (git-ignored)
data/thinkspark_phase1.db  Phase-1 pipeline's HF-upload sync log (scripts/P1_00_pipeline.py; git-ignored)
reports/cost_report.csv        flat CSV export of the DB (scripts/10_export_costs.py)
reports/generation_report.html  chart report — cost, progress, unit-eval (scripts/12_build_report.py)
artifacts/             checkpoints (git-ignored)
```

## Not included / bring your own

- ~~Phase-1 free corpora~~ — **no longer bring-your-own.** `scripts/P1_01_fetch_corpus.py`
  streams the real sources (LibriSpeech, AI4Bharat Kathbath + Shrutilipi, Google FLEURS,
  IndicTTS-Gujarati — see step 4 above) and stops per-source once its target hours are
  reached, so bandwidth/disk stay bounded on Kaggle regardless of how huge the upstream
  dataset is. **No manual step needed** — every default source is public/ungated (Common
  Voice was removed from Hugging Face entirely and dropped from this mix; see step 4's
  note). The referee never trains on agent audio, in either phase.
- **Exact Soniox TTS field names.** The transport is isolated in
  `thinkspark/tts_soniox.py::_synthesize_ws`; verify the current route/fields against
  soniox.com/docs before a large paid run. Everything downstream depends only on the
  `TTSResult` contract (audio + word timestamps).
