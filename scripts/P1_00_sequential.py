#!/usr/bin/env python
"""
Phase-1 corpus, STRICTLY SEQUENTIAL, one language per run — the opposite architecture
from scripts/P1_00_pipeline.py's concurrent pipelining. Use this when you want total
control over one language at a time instead of everything overlapping in the background:

    STAGE 1  DOWNLOAD  — every source for this language, fully, to completion
        |                (own tqdm bar: hours so far / target)
        v
    STAGE 2  ENCODE    — every wav just downloaded -> Mimi cb0/energy/f0, fully
        |                (own tqdm bar: clips encoded / pending; split across
        |                 --devices in parallel if you pass more than one GPU)
        v
    STAGE 3  UPLOAD     — pack ALL of this language's encoded clips into a handful of
        |                self-contained Parquet shards (cb0/energy/f0 embedded
        |                directly as columns) and upload them in one commit to
        |                --hf-repo, as data/<lang>/<lang>-shard-NNNNN.parquet
        |                (own tqdm bar: shards uploaded)
        v
    STAGE 4  CLEANUP    — delete the now-uploaded local .npz (and raw wavs, already
                          deleted right after encoding) to free local/Kaggle disk
                          before you start the NEXT language
                          (own tqdm bar: files deleted / eligible)

Nothing starts until the stage before it is FULLY done — no overlap, no background
threads guessing at each other's progress. Each stage prints its own separate,
self-contained tqdm progress bar instead of one shared live dashboard
(see scripts/P1_00_pipeline.py if you want the concurrent/pipelined version instead —
same underlying fetch/encode logic, different scheduling; that one still uploads loose
`.npz` files rather than Parquet shards).

Runs are fully resumable at every stage (same manifest/`.npz`-existence checks as
P1_00_pipeline.py) — safe to Ctrl+C and re-run the same command. Upload is a full
re-pack + re-upload of the language's CURRENT local state every run, not incremental —
see thinkspark.phase1_parquet's module docstring for why (real observed failure with
per-clip incremental upload: HF/git hard-limits any directory to ~10,000 files, and a
single language's flat encoded/<lang>/ directory blew past that at real scale).

Output layout matches the TRAINING format exactly — no extra step needed before
scripts/06_train_phase1.py can use it:
    data/encoded/<lang>/<clip_id>.npz        Mimi cb0 + energy + f0 (actual training input)
    data/frames_phase1/frames_<lang>.jsonl   frame records referencing the .npz above
Uploaded to --hf-repo as Parquet shards (data/<lang>/*.parquet) instead, since that's
what stays clean at scale — scripts/19_fetch_training_data.py unpacks those shards
straight back into the exact local layout above, so the trainer needs zero changes.

    conda activate llms
    pip install datasets soundfile huggingface_hub tqdm
    # in the SAME cell/process (NOT a separate `!export` cell on Kaggle/Jupyter — see
    # docs/commands/phase1-corpus.mdx for why that silently doesn't work):
    python -c "import os; os.environ['HF_TOKEN']='hf_...'"   # or just set it before this script

    # wipe everything already on the HF repo first, so you can push clean data (asks to
    # confirm; only touches the REMOTE repo + local hf_sync tracking, never local files):
    python scripts/P1_00_pipeline.py --hf-repo anuj-inavlabs/kupe-thinkspark-270m-phase1-data --cleanup

    # then run one language fully, start to finish:
    python scripts/P1_00_sequential.py --config configs/phase1_corpus.yaml \\
        --hf-repo anuj-inavlabs/kupe-thinkspark-270m-phase1-data --lang en
    python scripts/P1_00_sequential.py --config configs/phase1_corpus.yaml \\
        --hf-repo anuj-inavlabs/kupe-thinkspark-270m-phase1-data --lang hi
    python scripts/P1_00_sequential.py --config configs/phase1_corpus.yaml \\
        --hf-repo anuj-inavlabs/kupe-thinkspark-270m-phase1-data --lang gu
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import env
from thinkspark.hf_upload import create_commit_with_backoff, ensure_repo
from thinkspark.phase1_corpus import (
    Phase1CorpusConfig, build_frame_record, existing_written, fetch_source, manifest_path,
)

DATASET_CARD = """\
---
license: cc0-1.0
language:
  - en
  - hi
  - gu
tags:
  - thinkspark-v2-350m
  - phase1
  - mimi
  - audio-tokens
  - speech
pretty_name: ThinkSpark-v2-350M Phase-1 Free-Audio Corpus
---

# ThinkSpark-v2-350M — Phase-1 free-audio training data

Pre-encoded [Mimi](https://huggingface.co/kyutai/mimi) `cb0` (12.5Hz semantic) tokens +
per-frame `energy`/`f0` prosody, packaged for Phase 1 of ThinkSpark-v2-350M — teaching a
270M-parameter Gemma-3-based full-duplex floor-controller that a stream of Mimi audio
tokens carries language + prosody, before Phase 2 teaches it to referee turn-taking.

Sourced from free/open corpora (LibriSpeech, AI4Bharat Kathbath/Shrutilipi, IndicTTS,
Google FLEURS) via `scripts/P1_00_sequential.py` / `scripts/P1_00_pipeline.py` in the
[kupe-thinkspark-v2-270m](https://github.com/iNavLabsResearch/kupe-thinkspark-v2-270m)
repo — see that repo's `configs/phase1_corpus.yaml` for the exact per-language source
mix, weights, and citations.

## Layout

Each language has its own handful of self-contained Parquet shards:

```
data/en/en-shard-00000.parquet
data/en/en-shard-00001.parquet
...
data/hi/hi-shard-00000.parquet
...
data/gu/gu-shard-00000.parquet
...
```

Every row is one clip's full frame record — audio tokens included directly as columns,
no separate file to join against:

| column | type | meaning |
|---|---|---|
| `scenario_id` | string | stable clip id |
| `behaviour` | string | always `"phase1_free_audio"` here |
| `language` | string | `en` / `hi` / `gu` |
| `domain` | string | source dataset id (e.g. `librispeech`, `kathbath`) |
| `agent_text` | string | always empty for Phase-1 (no agent turn) |
| `user_text` | string | the clip's transcript |
| `num_frames` / `audio_frames` | int | frame count (also `len(cb0)`) |
| `cb0` | list<int64> | Mimi codebook-0 token id, one per 80ms frame |
| `energy` | list<float32> | log-RMS energy per frame |
| `f0` | list<float32> | fundamental frequency (Hz), 0 = unvoiced, per frame |
| `flags` | list<int32> | control-flag id per frame (vocab.CONTROL_FLAG_TO_ID) |
| `agent_state` | list<int32> | agent-state id per frame (vocab.AGENT_STATE_TO_ID) |
| `speaking_mask` | list<int32> | 1 = user speaking that frame |
| `spoken_spans` | list<string> | JSON-encoded spans (empty for Phase-1) |

## Loading

Straight into a training machine's local layout (recommended — no extra code needed,
matches what `scripts/06_train_phase1.py` in the source repo reads directly):
```bash
python scripts/19_fetch_training_data.py --phase1-repo {hf_repo}
```

Or directly via `datasets`/`pandas`/`pyarrow` if you just want to inspect it:
```python
from datasets import load_dataset
ds = load_dataset("{hf_repo}", data_files="data/en/*.parquet", split="train")
```

## License

CC0 — free/open source audio only (see the source repo's `configs/phase1_corpus.yaml`
for each source's own license/citation).
"""


# --------------------------------------------------------------------------- #
def resolve_devices(args) -> list[str]:
    """Same auto-detect as P1_00_pipeline.py: --devices explicit list > --device single
    override > every visible CUDA device > cpu."""
    if getattr(args, "devices", None):
        return [d.strip() for d in args.devices.split(",") if d.strip()]
    if args.device:
        return [args.device]
    try:
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n = 0
    return [f"cuda:{i}" for i in range(n)] if n > 0 else ["cpu"]


# --------------------------------------------------------------------------- #
# STAGE 1 — DOWNLOAD, fully, one language, own tqdm bar
# --------------------------------------------------------------------------- #
def stage_download(lang: str, cfg: Phase1CorpusConfig, args, out_dir: Path,
                   manifest_fh, already: dict, hf_token: str | None) -> None:
    all_specs = cfg.sources.get(lang, [])
    skip_ids = getattr(args, "skip_source_ids", None) or set()
    specs = [s for s in all_specs if s.id not in skip_ids]
    skipped = [s.id for s in all_specs if s.id in skip_ids]

    lang_target_h = cfg.target_hours.get(lang, 0.0)
    # Bar total reflects only the sources ACTUALLY being fetched this run, not the
    # language's full target — otherwise a skipped source's share (e.g. fleurs' 40% of
    # en's 150h) would permanently cap the bar around 60%, reading as unfinished when
    # everything this run was actually asked to do is in fact done.
    target_h = sum(lang_target_h * s.weight for s in specs)
    have_h = sum(already.get((lang, s.id), {}).get("hours", 0.0) for s in specs)

    print(f"\n[STAGE 1/4] DOWNLOAD  {lang}  (target {target_h:.0f}h across {len(specs)} source(s))")
    if skipped:
        print(f"  skipping this run (--skip-source): {skipped}")
    pbar = tqdm(total=round(target_h, 2), initial=round(min(have_h, target_h), 2),
               desc=f"download {lang}", unit="h", colour="cyan",
               bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}h [{elapsed}, {rate_fmt}]")
    try:
        for spec in specs:
            tqdm.write(f"  -> {spec.id}: {spec.hf_dataset}/{spec.hf_config or spec.split} "
                      f"(weight={spec.weight:.2f})")
            r = fetch_source(cfg, lang, spec, out_dir, ROOT, manifest_fh, already, hf_token, False,
                             log_fn=lambda m: tqdm.write(f"     [{spec.id}] {m}"),
                             progress_fn=lambda h: pbar.update(h))
            if r["status"] == "already_done":
                tqdm.write(f"     [{spec.id}] already done: {r['have_hours']:.2f}h "
                          f">= {r['target_hours']:.2f}h target")
            else:
                tqdm.write(f"     [{spec.id}] wrote {r.get('written', 0)} clips -> "
                          f"{r['have_hours']:.2f}h (target {r['target_hours']:.2f}h)")
    finally:
        pbar.close()
    print(f"[STAGE 1/4] DOWNLOAD  {lang}  done.")


# --------------------------------------------------------------------------- #
# STAGE 2 — ENCODE, fully, own tqdm bar, split across --devices if more than one
# --------------------------------------------------------------------------- #
_encoders: dict[str, object] = {}
_encoders_lock = threading.Lock()


def _get_encoder(device: str, cfg: Phase1CorpusConfig):
    with _encoders_lock:
        enc = _encoders.get(device)
        if enc is None:
            from thinkspark.mimi_codec import MimiEncoder
            tqdm.write(f"  loading Mimi encoder on {device} "
                      f"({getattr(cfg, 'mimi_repo', 'kyutai/mimi')})...")
            enc = MimiEncoder(repo=getattr(cfg, "mimi_repo", "kyutai/mimi"), device=device)
            cb_size = enc.codebook_size   # triggers load
            tqdm.write(f"  Mimi encoder ready (device={enc._device}, codebook_size={cb_size})")
            _encoders[device] = enc
        return enc


def _maybe_delete_wav(wav: Path, out_path: Path, args) -> None:
    if args.keep_raw_audio:
        return
    import numpy as np
    if len(np.load(out_path)["cb0"]) > 0:
        wav.unlink()
    else:
        tqdm.write(f"  ! {wav.name} encoded to 0 frames — keeping wav for inspection")


def _encode_shard(wavs: list[Path], device: str, encoded_dir: Path, cfg, args,
                  pbar: tqdm, pbar_lock: threading.Lock) -> int:
    """Encodes `wavs` in batches of `args.encode_batch_size` — the real fix for slow GPU
    throughput (real observed case: an L4 doing ~2.6 clips/sec one-at-a-time is almost
    entirely Python/CUDA-launch overhead per call, not compute; batching amortizes that
    across many clips per forward pass). Falls back to one-at-a-time ONLY for a batch
    that raised (so one corrupt/unreadable clip can't waste an entire batch's otherwise-
    good clips — they'd just get silently re-attempted next run instead of being lost)."""
    import numpy as np
    import soundfile as sf

    encoder = _get_encoder(device, cfg)
    new_count = 0
    batch_size = max(1, args.encode_batch_size)

    i = 0
    while i < len(wavs):
        batch = wavs[i:i + batch_size]
        i += batch_size

        todo = []
        for wav in batch:
            out_path = encoded_dir / f"{wav.stem}.npz"
            if out_path.exists():
                with pbar_lock:
                    pbar.update(1)
            else:
                todo.append(wav)
        if not todo:
            continue

        try:
            waveforms, sample_rates = [], []
            for wav in todo:
                arr, sr = sf.read(str(wav), dtype="float32", always_2d=False)
                waveforms.append(np.asarray(arr, dtype=np.float32))
                sample_rates.append(sr)
            encoded_list = encoder.encode_batch(waveforms, sample_rates)
            for wav, enc in zip(todo, encoded_list):
                out_path = encoded_dir / f"{wav.stem}.npz"
                enc.save(out_path)
                new_count += 1
                _maybe_delete_wav(wav, out_path, args)
                with pbar_lock:
                    pbar.update(1)
        except Exception as e:
            tqdm.write(f"  ! batch of {len(todo)} failed ({e}) — retrying one at a time "
                      f"so one bad clip doesn't cost the rest of the batch")
            for wav in todo:
                out_path = encoded_dir / f"{wav.stem}.npz"
                try:
                    enc = encoder.encode_wav_file(str(wav))
                    enc.save(out_path)
                    new_count += 1
                    _maybe_delete_wav(wav, out_path, args)
                except Exception as e2:
                    tqdm.write(f"  ! failed {wav.name}: {e2}")
                with pbar_lock:
                    pbar.update(1)

        # Once per BATCH now, not once per clip — batching already means far fewer GIL-
        # holding forward-pass calls per second than the old one-clip-at-a-time loop, so
        # this yield is needed far less often to still give any other thread a fair
        # window (relevant only when len(devices) > 1, i.e. these shards run concurrently
        # with each other — irrelevant, but harmless, for a single-device run).
        time.sleep(args.encode_yield_ms / 1000.0)
    return new_count


def stage_encode(lang: str, cfg: Phase1CorpusConfig, args) -> None:
    wav_dir = ROOT / args.out_dir / lang
    encoded_dir = ROOT / args.encoded_dir
    encoded_dir.mkdir(parents=True, exist_ok=True)

    all_wavs = sorted(wav_dir.glob("*.wav"))
    pending = [w for w in all_wavs if not (encoded_dir / f"{w.stem}.npz").exists()]
    devices = args.devices_list

    print(f"\n[STAGE 2/4] ENCODE  {lang}  ({len(pending)} pending / {len(all_wavs)} total wavs, "
         f"devices={devices})")
    if not pending:
        print(f"[STAGE 2/4] ENCODE  {lang}  nothing to do.")
    else:
        pbar = tqdm(total=len(pending), desc=f"encode {lang}", unit="clip", colour="yellow")
        pbar_lock = threading.Lock()
        try:
            if len(devices) == 1:
                _encode_shard(pending, devices[0], encoded_dir, cfg, args, pbar, pbar_lock)
            else:
                # Round-robin split so every device gets an even mix of clip sizes.
                shards = {d: pending[i::len(devices)] for i, d in enumerate(devices)}
                with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                    futs = [pool.submit(_encode_shard, shard, d, encoded_dir, cfg, args, pbar, pbar_lock)
                           for d, shard in shards.items() if shard]
                    for fut in as_completed(futs):
                        fut.result()
        finally:
            pbar.close()
        print(f"[STAGE 2/4] ENCODE  {lang}  done.")

    # BUILD FRAMES — rebuild fully from the manifest, filtered to this language.
    manifest_file = manifest_path(ROOT / args.out_dir)
    recs = []
    for line in manifest_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("lang") == lang:
            recs.append(rec)

    frames_dir = ROOT / args.frames_out_dir
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_path = frames_dir / f"frames_{lang}.jsonl"
    written = missing = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for rec in recs:
            stem = Path(rec["wav_path"]).stem
            frame = build_frame_record(rec, encoded_dir / f"{stem}.npz", ROOT)
            if frame is None:
                missing += 1
                continue
            fout.write(json.dumps(frame, ensure_ascii=False) + "\n")
            written += 1
    print(f"[STAGE 2/4] FRAMES  wrote {written} frame records -> {out_path} "
         f"({missing} skipped, not encoded)")


# --------------------------------------------------------------------------- #
# STAGE 3 — UPLOAD, fully, own tqdm bar. Runs on the MAIN thread (unlike
# P1_00_pipeline.py's background upload worker) — any failure here (missing HF_TOKEN,
# bad repo, network) raises immediately and visibly instead of ever being able to die
# silently in a background thread. This IS "make sure we upload it properly".
# --------------------------------------------------------------------------- #
def _upload_dataset_card(api, args) -> None:
    """Writes and uploads the Phase-1 dataset card (README.md) — see DATASET_CARD below
    for the actual content. Small, always safe to re-upload/overwrite."""
    from huggingface_hub import CommitOperationAdd
    card_path = ROOT / "data" / "phase1_parquet" / "README.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(DATASET_CARD.replace("{hf_repo}", args.hf_repo), encoding="utf-8")
    create_commit_with_backoff(
        api, repo=args.hf_repo,
        operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card_path))],
        commit_message="phase1: update dataset card",
        max_retries=args.max_retries, base_backoff=args.backoff, max_backoff=args.max_backoff,
        log_fn=print,
    )


def stage_upload(lang: str, args) -> None:
    """Packs ALL of this language's currently-encoded local data into a handful of
    self-contained Parquet shards (cb0/energy/f0 embedded directly as columns — no
    separate .npz needed in the repo at all) and uploads them in ONE commit, replacing
    whatever shards existed for this language before.

    Deliberately a FULL re-pack + re-upload every run, not an incremental delta against
    individually-tracked clip ids (the old per-.npz upload path did that, and hit HF/
    git's ~10,000-files-per-directory limit at scale — see thinkspark.hf_upload's
    module docstring). Repacking is cheap (local disk only, seconds even for tens of
    thousands of clips) so there's no real cost to always uploading the exact current
    full state instead of tracking deltas — simpler and can't drift out of sync."""
    from huggingface_hub import CommitOperationAdd, HfApi
    from thinkspark.phase1_parquet import pack_lang_to_parquet

    print(f"\n[STAGE 3/4] UPLOAD  {lang}  -> {args.hf_repo}")
    token = env("HF_TOKEN", required=True)   # raises loudly HERE, on the main thread, if unset
    api = HfApi(token=token)
    ensure_repo(api, args.hf_repo, args.private)   # raises RuntimeError with a clear message on failure

    frames_path = ROOT / args.frames_out_dir / f"frames_{lang}.jsonl"
    if not frames_path.exists():
        raise SystemExit(f"no {frames_path} — run stage 2 (encode) first for {lang}")

    print(f"  packing {lang} into Parquet shards...")
    shard_dir = ROOT / "data" / "phase1_parquet"
    shards = pack_lang_to_parquet(lang, frames_path, ROOT, shard_dir,
                                  rows_per_shard=args.rows_per_shard, log_fn=print)
    if not shards:
        print(f"[STAGE 3/4] UPLOAD  {lang}  nothing to upload (no encoded clips found).")
        return

    pbar = tqdm(total=len(shards), desc=f"upload {lang}", unit="shard", colour="green")
    try:
        ops = [CommitOperationAdd(path_in_repo=f"data/{lang}/{s.name}", path_or_fileobj=str(s))
              for s in shards]
        create_commit_with_backoff(
            api, repo=args.hf_repo, operations=ops,
            commit_message=f"phase1: {lang} — {len(shards)} parquet shard(s), full re-upload",
            max_retries=args.max_retries, base_backoff=args.backoff, max_backoff=args.max_backoff,
            log_fn=tqdm.write,
        )
        pbar.update(len(shards))
    finally:
        pbar.close()
    print(f"[STAGE 3/4] UPLOAD  {lang}  done — {len(shards)} shard(s) uploaded.")

    _upload_dataset_card(api, args)
    print(f"[STAGE 3/4] UPLOAD  {lang}  -> https://huggingface.co/datasets/{args.hf_repo}")


# --------------------------------------------------------------------------- #
# STAGE 4 — delete local .npz now that they're safely in the just-uploaded Parquet
# shards (raw wavs are already deleted right after encoding in stage 2, unless
# --keep-raw-audio) — frees local/Kaggle disk before the next language, own tqdm bar.
# Only reached if stage_upload returned WITHOUT raising, so every locally-encoded clip
# for this language is confirmed already embedded in the shards just committed.
# --------------------------------------------------------------------------- #
def stage_cleanup_local(lang: str, args) -> None:
    if args.keep_local_encoded:
        print(f"\n[STAGE 4/4] CLEANUP  {lang}  skipped (--keep-local-encoded)")
        return

    print(f"\n[STAGE 4/4] CLEANUP  {lang}  deleting local .npz now embedded in the uploaded shards...")
    frames_path = ROOT / args.frames_out_dir / f"frames_{lang}.jsonl"
    if not frames_path.exists():
        print(f"[STAGE 4/4] CLEANUP  {lang}  no frames file — nothing to check.")
        return

    to_delete: list[Path] = []
    with frames_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = ROOT / rec["encoded_path"]
            if p.exists():
                to_delete.append(p)

    if not to_delete:
        print(f"[STAGE 4/4] CLEANUP  {lang}  nothing eligible yet.")
        return

    pbar = tqdm(total=len(to_delete), desc=f"delete {lang}", unit="file", colour="red")
    try:
        for p in to_delete:
            p.unlink()
            pbar.update(1)
    finally:
        pbar.close()
    print(f"[STAGE 4/4] CLEANUP  {lang}  deleted {len(to_delete)} local .npz "
         f"(safely uploaded to {args.hf_repo}).")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_corpus.yaml")
    ap.add_argument("--lang", required=True, help="exactly ONE language to run, "
                    "start to finish, e.g. en / hi / gu — this script is strictly "
                    "sequential and single-language by design; run it again for the "
                    "next one")
    ap.add_argument("--skip-source", default=None,
                    help="comma-separated source id(s) to skip entirely this run, e.g. "
                        "'fleurs' — useful to hold back a problematic source (still "
                        "counted toward the language's total target hours, just not "
                        "fetched) while running the rest of that language's sources "
                        "normally; fetch the skipped one later with its own command/script")
    ap.add_argument("--out-dir", default="data/phase1_raw")
    ap.add_argument("--encoded-dir", default="data/encoded")
    ap.add_argument("--frames-out-dir", default="data/frames_phase1")
    ap.add_argument("--rows-per-shard", type=int, default=2000,
                    help="frame records packed into each uploaded Parquet shard "
                        "(default 2000) — keeps HF repo file counts small (a handful "
                        "of shards per language instead of tens of thousands of loose "
                        ".npz files, which hit HF/git's ~10,000-files-per-directory "
                        "limit at real scale)")
    ap.add_argument("--hf-repo", default=None,
                    help="HF dataset repo, e.g. anuj-inavlabs/kupe-thinkspark-270m-phase1-data "
                        "— required unless --no-upload")
    ap.add_argument("--no-upload", action="store_true", help="skip stages 3+4 entirely, "
                    "download+encode only")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--device", default=None, help="single Mimi encoder device override "
                    "(cpu/cuda/mps)")
    ap.add_argument("--devices", default=None, help="comma-separated devices to encode on "
                    "in parallel, e.g. 'cuda:0,cuda:1' for both of Kaggle's T4s. Default: "
                    "auto-detect every visible CUDA device.")
    ap.add_argument("--keep-raw-audio", action="store_true",
                    help="don't delete a wav right after it's encoded (stage 2)")
    ap.add_argument("--keep-local-encoded", action="store_true",
                    help="don't delete local .npz after they're confirmed uploaded (stage 4)")
    ap.add_argument("--min-free-disk-gb", type=float, default=None,
                    help="pause downloads (stage 1) below this many GB free (default: "
                        "configs/phase1_corpus.yaml's min_free_disk_gb, 5.0 if unset)")
    ap.add_argument("--encode-batch-size", type=int, default=16,
                    help="clips encoded in ONE forward pass (default 16) — the main "
                        "throughput lever on a GPU: one-at-a-time encoding leaves the "
                        "GPU mostly idle behind per-call Python/CUDA-launch overhead "
                        "(real observed case: an L4 doing ~2.6 clips/sec one-at-a-time). "
                        "Raise it (e.g. 32-64) on a bigger GPU if VRAM allows; lower it "
                        "(down to 1 for the old one-at-a-time behavior) if you hit an "
                        "out-of-memory error. A batch that fails falls back to encoding "
                        "its clips one at a time automatically, so a single bad clip "
                        "never costs the rest of a batch.")
    ap.add_argument("--encode-yield-ms", type=float, default=0.0,
                    help="ms slept after every BATCH to release the GIL (default 0 — "
                        "this script's download stage is a separate, already-finished "
                        "phase by the time encoding starts, unlike P1_00_pipeline.py's "
                        "concurrent architecture, so there's no download thread to "
                        "protect from GIL starvation here; only matters if you pass "
                        "multiple --devices, where shards run concurrently with each "
                        "other)")
    ap.add_argument("--backoff", type=float, default=20.0)
    ap.add_argument("--max-backoff", type=float, default=300.0)
    ap.add_argument("--max-retries", type=int, default=10)
    args = ap.parse_args()

    if not args.no_upload and not args.hf_repo:
        raise SystemExit("--hf-repo is required unless --no-upload is set")

    cfg = Phase1CorpusConfig.from_yaml(ROOT / args.config)
    if args.min_free_disk_gb is not None:
        cfg.min_free_disk_gb = args.min_free_disk_gb
    args.devices_list = resolve_devices(args)
    args.skip_source_ids = {s.strip() for s in args.skip_source.split(",") if s.strip()} \
        if args.skip_source else set()

    lang = args.lang
    if lang not in cfg.sources:
        raise SystemExit(f"--lang {lang!r} not in {args.config}'s languages "
                         f"({list(cfg.sources.keys())})")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_path(out_dir)
    already = existing_written(manifest_file)
    hf_token = env("HF_TOKEN")

    print("=" * 72)
    print(f"ThinkSpark-v2-350M — Phase-1 SEQUENTIAL pipeline — lang={lang}")
    print("download (fully) -> encode (fully) -> upload (fully) -> delete local (fully)")
    print("=" * 72)
    print(f"target: {cfg.target_hours.get(lang, 0.0):.0f}h  "
         f"sources={[s.id for s in cfg.sources.get(lang, [])]}")
    print(f"encode devices: {args.devices_list}")
    print(f"min free disk: {cfg.min_free_disk_gb:.1f}GB")
    print(f"HF repo: {args.hf_repo or '(uploads disabled, --no-upload)'}")

    manifest_fh = manifest_file.open("a", encoding="utf-8")
    try:
        stage_download(lang, cfg, args, out_dir, manifest_fh, already, hf_token)
    finally:
        manifest_fh.close()

    stage_encode(lang, cfg, args)

    if not args.no_upload:
        stage_upload(lang, args)
        stage_cleanup_local(lang, args)

    print("\n" + "=" * 72)
    print(f"Phase-1 sequential run complete for lang={lang}.")
    if not args.no_upload:
        print(f"  uploaded -> https://huggingface.co/datasets/{args.hf_repo}")
        print(f"  next language: python scripts/P1_00_sequential.py --config {args.config} "
             f"--hf-repo {args.hf_repo} --lang <next-lang>")
    print("=" * 72)


if __name__ == "__main__":
    main()
