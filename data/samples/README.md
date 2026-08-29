# Sample data — read this to understand the on-disk format

These files are **hand-authored, offline samples** so you can see exactly what the
pipeline produces before spending any budget. Regenerate them (and 5 more) any time with:

```bash
conda activate llms
python scripts/make_samples.py
```

`make_samples.py` uses only the pure-python parts of `thinkspark` (schema, frames,
validators) — no OpenAI, Soniox, torch or transformers — so it runs anywhere.

## `scenarios_sample.jsonl` — the LLM output (Section 8.4)

One JSON object per line, exactly what `scripts/02_generate_scripts.py` writes. Fields
match `thinkspark.schema.Scenario`:

| field | meaning |
|---|---|
| `behaviour` | one of the 12 generation buckets (barge_real, backchannel, …) |
| `language` | `hi` / `en` / `gu` / `hi_en_native` / `gu_en_native` (native script) |
| `domain` | `bfsi_collections` / `support` / `sales` |
| `agent_text` | what the agent is saying (may be `""`) — never synthesised to audio |
| `agent_state` | `IDLE` / `LLM_GEN` / `TTS_SPEAKING` / `TTS_DONE` at window start |
| `user_text` | the exact line Soniox TTS speaks (≤ 25 words) |
| `prosody` | `falling` / `rising` / `held` / `flat` / `distressed` / `neutral` |
| `event_char` | char index in `user_text` where the key event happens |
| `target` | ordered `[{frame_offset, flag, spoken_text}]` timeline |
| `notes` | one line: why the case is hard |

The 8 samples deliberately include the **negative / "say nothing" back-channel**
(`sample_backchannel_silent_hi`): the user is mid-sentence, so the correct behaviour is
to stay silent (`LISTEN`, empty `spoken_text`). Always emitting "haan"/"right" would be
unnatural — the corpus keeps ~35 % of the back-channel bucket as this silent case
(see `thinkspark.distribution.BACKCHANNEL_SILENCE_SHARE`). It also includes the
`barge_lookalike` hard-negative that must resolve to `CONTINUE`, not a real barge.

## `frames_sample.jsonl` — the per-frame training record (Section 5.3)

One JSON object per line, what `scripts/04_build_frames.py` writes and
`thinkspark.dataset.ThinkSparkDataset` consumes. 3 representative records are committed
here (run `make_samples.py` for all 8). Each frame is **80 ms** (12.5 Hz Mimi grid).

| field | meaning |
|---|---|
| `num_frames` | window length `T` in frames |
| `audio_frames` | how many frames actually carry user audio |
| `encoded_path` | path to the Mimi `.npz` (cb0/energy/f0); `null` in these dry samples |
| `flags` | `int[T]` — control-flag id per frame (control-head target) |
| `agent_state` | `int[T]` — agent-state channel (a model **input**) |
| `speaking_mask` | `int[T]` — 1 where a spoken back-channel is emitted |
| `spoken_spans` | `[{frame, text}]` — plain words for the spoken head |

Flag ids are the index into `thinkspark.vocab.CONTROL_FLAGS`
(`LISTEN=0, HOLD=1, INCOMPLETE=2, TURN_END=3, BARGE_SOFT=4, BARGE_HARD=5, CONTINUE=6,
PREFETCH_LLM=7, COMMIT_LLM=8, CANCEL_LLM=9, SILENCE_BREAK=10`). Agent-state ids index
`AGENT_STATES` (`IDLE=0, LLM_GEN=1, TTS_SPEAKING=2, TTS_DONE=3`).

**Read `sample_endpoint_en`**: `LISTEN` across the turn, then `PREFETCH_LLM` at frame 18
(agent-state flips to `LLM_GEN`), `TURN_END` at 24, `COMMIT_LLM` at 25 — the model starts
the LLM speculatively *before* the user finishes so the reply feels instant (Section 11).

> These committed samples are **dry** (audio-free): `encoded_path` is null and offsets
> come straight from the LLM. In the real run, `frames.build_frames` calibrates every
> offset to the Soniox word timestamps and fills `cb0`/`energy`/`f0` from the Mimi shard.
