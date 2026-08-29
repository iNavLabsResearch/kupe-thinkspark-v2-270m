# Your voice reference clips go here

This folder is intentionally empty except for this file. Drop your own reference audio
clips straight into it — one file per person/voice you want cloned — then run:

```bash
conda activate llms
python scripts/15_create_voice_profiles.py --config configs/data_gen.yaml --dry-run   # preview the plan
python scripts/15_create_voice_profiles.py --config configs/data_gen.yaml             # clone for real
```

## Naming convention (required)

The script derives **gender** and **display name** from the filename — no separate
manifest needed. Name each file:

```
<gender>_<name>.<ext>
```

- `<gender>` — `female` or `male` (case-insensitive; also accepts `f`/`m` as a short form)
- `<name>` — anything you like, becomes the cloned voice's display name on Soniox
  (letters/digits/`-`/`_` only; anything else gets stripped)
- `<ext>` — `wav`, `mp3`, `m4a`, `flac`, or `ogg`

Examples:
```
female_anita.wav
female_priyanka_2.wav
male_rahul.mp3
male_dev.wav
```

Files that don't match `<gender>_<name>.<ext>` are skipped with a warning — they won't
silently get cloned under a guessed gender.

## How many do I need?

**At least 1 clip per gender is required.** `thinkspark/tts_soniox.py::resolve_voice()`
uses ONLY what's in `voice_profiles.json` — no live Soniox catalog fetch.
`03_render_user_audio.py` refuses to start (fails fast, before spending anything) if a
gender it needs to render has zero voices.

Beyond that minimum, more clips = more speaker diversity — each render picks a voice
from your pool, deterministically rotated by a hash of the scenario text.

## Soniox's account-wide cloning cap — what happens if you submit more clips than it allows

Soniox caps how many voices **one account** can have cloned at once
(`soniox_max_cloned_voices` in `configs/data_gen.yaml`, default **20 total across both
genders** — reported against a real account; override if yours differs). If you submit
more reference clips than that:

1. The clone budget is split across genders **proportional to how many clips you
   submitted per gender** (largest-remainder rounding — e.g. 15 female + 15 male clips
   against a cap of 20 → 10 cloned per gender).
2. The clips that don't fit the budget are **not cloned at all** — instead, that many
   named Soniox **stock voices** of the matching gender
   (`thinkspark/soniox_default_voices.py` — a fixed, explicitly-chosen list, not a live
   catalog fetch) are added to `voice_profiles.json` to fill the shortfall. Your final
   per-gender voice *count* still matches what you submitted; the overflow just comes
   from stock voices instead of your own clips.

Both kinds land in the same `voice_profiles.json` and are used identically by
`resolve_voice()` — nothing to configure differently. Re-running the script after adding
more clips (or raising the cap) tops up the difference; it never re-clones or re-adds
what's already there.

Clips should be clean, single-speaker, a few seconds to ~1 minute long, in any of the
formats above. One clip per person is enough — Soniox clones from a single sample.

## What the script does

1. Scans this folder for files matching the naming convention above.
2. Figures out the cap-aware clone/default-fill split (see above) and prints the full
   plan — nothing is cloned yet.
3. Calls Soniox's real voice-cloning API (`POST /v1/voices`, multipart upload) for each
   clip the plan assigns to cloning — skips any file already cloned (resumable, tracked
   by content hash, so renaming doesn't cause a re-clone but replacing the audio does).
4. "Recomputes" and polls each new voice so it's actually usable with `tts-rt-v2`
   (Soniox prepares a voice for a specific TTS model asynchronously; this step waits for
   that to finish, matching what `kupe-backend` does for its own voice cloning).
5. Writes every voice — cloned or default — into `voice_profiles.json` in this folder.

`03_render_user_audio.py` then reads `voice_profiles.json` and rotates deterministically
across your voice pool per gender (hash of the scenario text — same text always picks
the same voice, different scenarios spread across the pool) — nothing else to wire up.
It never reaches for a live Soniox catalog; if a gender it needs has zero voices yet
(cloned or default), it exits immediately with instructions, before any API calls or spend.

## Cleaning up

```bash
python scripts/15_create_voice_profiles.py --config configs/data_gen.yaml --cleanup
```
Deletes every **cloned** voice from your Soniox account too (best-effort — a failure
there doesn't block removing it locally) and resets `voice_profiles.json` to empty, with
an "are you sure?" confirmation first. Use this before rebuilding from scratch — e.g.
after changing which clips you're using, or changing the cap.
