#!/usr/bin/env python
"""
Phase-1 corpus pipeline, clustered and continuous — the fast path to a ready-to-train
Phase-1 corpus, run locally (built for a Mac, no Kaggle session needed for this stage).

Pipeline (all stages run CONCURRENTLY, not one after another):

    DOWNLOAD (N workers, one per language+source "cluster", e.g. en/librispeech,
              hi/kathbath, gu/indictts — see configs/phase1_corpus.yaml)
        |
        v  the INSTANT every source for a language finishes downloading
    ENCODE + BUILD FRAMES (one worker — wav -> Mimi cb0/energy/f0 -> frame records)
        |
        v  the INSTANT a language's frames are (re)built
    UPLOAD (one worker, background, continuous — pushes ready .npz + frames to HF)

So while en/fleurs is still downloading in the background, hi (already fully
downloaded) is already being encoded; while hi is encoding, gu (already encoded
earlier) is already uploading. Nothing waits for everything else to finish — that's
the "cluster" pipelining: one cluster done downloading gets processed immediately,
while the other clusters keep downloading in the background.

Every stage logs its own timestamped, tagged lines (thread-safe — safe to read while
several stages are active at once), plus a periodic one-line status summary across all
of them. Uploads to Hugging Face are resumable via `hf_sync` in a dedicated SQLite DB
(same table/pattern as scripts/13_upload_hf.py) — killing this and restarting picks up
exactly where every stage left off: downloads via the manifest (thinkspark.phase1_corpus),
encodes via existing .npz files, uploads via the sync DB.

Repo layout uploaded (dataset repo, e.g. anuj-inavlabs/kupe-thinkspark-270m-phase1-data):
    encoded/<lang>/<clip_id>.npz         Mimi cb0 + energy + f0 (the actual training input)
    frames_phase1/frames_<lang>.jsonl    frame records referencing the .npz above
    manifest.jsonl                        full provenance manifest (uploaded once, at the end)
    README.md                             dataset card

This is genuinely "ready for training" data — scripts/19_fetch_training_data.py (Kaggle
side) downloads this repo straight into a local data/ layout that
scripts/06_train_phase1.py can train on immediately, no re-encoding needed.

    conda activate llms
    pip install datasets soundfile huggingface_hub
    # no HF_TOKEN needed for the DOWNLOAD sources (all public/ungated); HF_TOKEN with
    # WRITE access IS needed for the upload:
    export HF_TOKEN=hf_...

    # see the plan, no downloads/uploads:
    python scripts/P1_00_pipeline.py --config configs/phase1_corpus.yaml \\
        --hf-repo anuj-inavlabs/kupe-thinkspark-270m-phase1-data --dry-run

    # run for real (resumable — safe to Ctrl+C and re-run):
    python scripts/P1_00_pipeline.py --config configs/phase1_corpus.yaml \\
        --hf-repo anuj-inavlabs/kupe-thinkspark-270m-phase1-data

    # local only, no HF upload:
    python scripts/P1_00_pipeline.py --config configs/phase1_corpus.yaml --no-upload
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import env
from thinkspark.db import RunDB
from thinkspark.hf_upload import create_commit_with_backoff, ensure_repo, log as _hf_log
from thinkspark.phase1_corpus import (
    Phase1CorpusConfig, build_frame_record, existing_written, fetch_source, manifest_path,
)

_ = _hf_log  # imported for parity/reference; this script uses its own tagged Logger below

STATUS_INTERVAL_S = 30.0


# --------------------------------------------------------------------------- #
class Logger:
    """Thread-safe, tagged, timestamped logging — many workers print concurrently."""

    def __init__(self):
        self._lock = threading.Lock()

    def __call__(self, stage: str, tag: str, msg: str) -> None:
        with self._lock:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] [{stage:<8}] {tag:<22} {msg}", flush=True)


class LockedWriter:
    """Wraps a file handle so concurrent DOWNLOAD workers can share ONE manifest file
    safely — thinkspark.phase1_corpus.fetch_source() only ever calls .write()/.flush()."""

    def __init__(self, fh):
        self._fh = fh
        self._lock = threading.Lock()

    def write(self, s: str) -> None:
        with self._lock:
            self._fh.write(s)

    def flush(self) -> None:
        with self._lock:
            self._fh.flush()


# --------------------------------------------------------------------------- #
class PipelineStatus:
    """Shared, lock-guarded counters the periodic summary thread reads from."""

    def __init__(self, langs: list[str], cfg: Phase1CorpusConfig):
        self._lock = threading.Lock()
        self.langs = langs
        self.cfg = cfg
        self.download_hours: dict[str, float] = {lang: 0.0 for lang in langs}
        self.download_targets: dict[str, float] = {lang: cfg.target_hours.get(lang, 0.0) for lang in langs}
        self.lang_downloaded: set[str] = set()     # all sources done for this language
        self.lang_encoded: set[str] = set()
        self.lang_uploaded_frames: set[str] = set()
        self.npz_encoded_total = 0
        self.npz_uploaded_total = 0
        self.upload_queue_depth = 0
        self.encode_queue_depth = 0
        self.done = False

    def add_download_hours(self, lang: str, hours: float) -> None:
        with self._lock:
            self.download_hours[lang] = self.download_hours.get(lang, 0.0) + hours

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "download_hours": dict(self.download_hours),
                "download_targets": dict(self.download_targets),
                "lang_downloaded": set(self.lang_downloaded),
                "lang_encoded": set(self.lang_encoded),
                "lang_uploaded_frames": set(self.lang_uploaded_frames),
                "npz_encoded_total": self.npz_encoded_total,
                "npz_uploaded_total": self.npz_uploaded_total,
                "encode_queue_depth": self.encode_queue_depth,
                "upload_queue_depth": self.upload_queue_depth,
            }


def status_summary_loop(status: PipelineStatus, log: Logger, stop_event: threading.Event) -> None:
    while not stop_event.wait(STATUS_INTERVAL_S):
        s = status.snapshot()
        dl_bits = ", ".join(
            f"{lang}={s['download_hours'].get(lang, 0.0):.1f}/{s['download_targets'].get(lang, 0.0):.0f}h"
            for lang in status.langs
        )
        log("STATUS", "overall", f"download[{dl_bits}]  "
           f"encoded_langs={sorted(s['lang_encoded'])}  "
           f"uploaded_langs={sorted(s['lang_uploaded_frames'])}  "
           f"npz_encoded={s['npz_encoded_total']}  npz_uploaded={s['npz_uploaded_total']}  "
           f"queues[encode={s['encode_queue_depth']} upload={s['upload_queue_depth']}]")


# --------------------------------------------------------------------------- #
# DOWNLOAD stage
# --------------------------------------------------------------------------- #
def download_worker(cfg, lang, spec, out_dir, manifest_w, already, hf_token, log, status):
    tag = f"{lang}/{spec.id}"
    log("DOWNLOAD", tag, f"starting -> {spec.hf_dataset}/{spec.hf_config or spec.split} "
       f"(weight={spec.weight:.2f})")
    r = fetch_source(cfg, lang, spec, out_dir, ROOT, manifest_w, already, hf_token, False,
                     log_fn=lambda m: log("DOWNLOAD", tag, m))
    if r["status"] == "already_done":
        log("DOWNLOAD", tag, f"already done: {r['have_hours']:.2f}h >= {r['target_hours']:.2f}h target")
    else:
        log("DOWNLOAD", tag, f"done: wrote {r.get('written', 0)} clips -> "
           f"{r['have_hours']:.2f}h (target {r['target_hours']:.2f}h)")
    status.add_download_hours(lang, r["have_hours"])
    return lang, spec.id, r


# --------------------------------------------------------------------------- #
# ENCODE + FRAMES stage — single worker, one shared Mimi model instance
# --------------------------------------------------------------------------- #
def encode_and_build_frames(lang: str, cfg, args, log, status) -> int:
    """Encodes any not-yet-encoded wavs for `lang`, then rebuilds its frames_<lang>.jsonl
    from the manifest. Returns the number of NEWLY encoded clips (for upload targeting)."""
    from thinkspark.mimi_codec import MimiEncoder

    global _encoder
    if _encoder is None:
        log("ENCODE", lang, f"loading Mimi encoder ({cfg.mimi_repo if hasattr(cfg, 'mimi_repo') else 'kyutai/mimi'})...")
        _encoder = MimiEncoder(repo=getattr(cfg, "mimi_repo", "kyutai/mimi"), device=args.device)
        log("ENCODE", lang, f"Mimi encoder ready (codebook_size={_encoder.codebook_size})")

    wav_dir = ROOT / args.out_dir / lang
    encoded_dir = ROOT / args.encoded_dir
    encoded_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(wav_dir.glob("*.wav"))
    new_count = 0
    for wav in wavs:
        out_path = encoded_dir / f"{wav.stem}.npz"
        if out_path.exists():
            continue
        try:
            enc = _encoder.encode_wav_file(str(wav))
            enc.save(out_path)
            new_count += 1
            status.npz_encoded_total += 1
            if new_count % 25 == 0:
                log("ENCODE", lang, f"...{new_count} new clips encoded so far")
        except Exception as e:
            log("ENCODE", lang, f"! failed {wav.name}: {e}")

    log("ENCODE", lang, f"encoding done: {new_count} new / {len(wavs)} total wavs -> {encoded_dir}")

    # BUILD FRAMES — reads the whole manifest, filters to this language (cheap: manifest
    # rows are small JSON lines; re-filtering per language on every rebuild is simpler
    # and safer than maintaining a separate per-language index file).
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
    log("FRAMES", lang, f"wrote {written} frame records -> {out_path} ({missing} skipped, not yet encoded)")
    return new_count


_encoder = None   # shared MimiEncoder instance, lazily loaded once by the encode worker


def encode_worker_loop(encode_q: "queue.Queue[str | None]", upload_q: "queue.Queue[str | None]",
                       cfg, args, log, status, stop_event: threading.Event) -> None:
    while True:
        try:
            lang = encode_q.get(timeout=1.0)
        except queue.Empty:
            if stop_event.is_set():
                return
            continue
        if lang is None:   # sentinel — all downloads done, no more languages coming
            upload_q.put(None)
            return
        status.encode_queue_depth = max(0, encode_q.qsize())
        try:
            encode_and_build_frames(lang, cfg, args, log, status)
            status.lang_encoded.add(lang)
            if not args.no_upload:
                upload_q.put(lang)
        except Exception as e:
            log("ENCODE", lang, f"! stage failed, skipping upload for this language: {e}")
        encode_q.task_done()


# --------------------------------------------------------------------------- #
# UPLOAD stage — single worker, background, continuous
# --------------------------------------------------------------------------- #
def upload_worker_loop(upload_q: "queue.Queue[str | None]", args, db: RunDB, log,
                       status, stop_event: threading.Event) -> None:
    if args.no_upload:
        return
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        log("UPLOAD", "-", "huggingface_hub not installed — `pip install huggingface_hub`. "
           "Uploads disabled for this run; local data is still complete.")
        return

    token = env("HF_TOKEN", required=True)
    api = HfApi(token=token)
    try:
        ensure_repo(api, args.hf_repo, args.private)
    except RuntimeError as e:
        log("UPLOAD", "-", f"! {e}")
        log("UPLOAD", "-", "uploads disabled for this run — local data is still being "
           "produced normally, fix the repo/token and re-run to upload what's pending.")
        return
    log("UPLOAD", "-", f"repo ready: https://huggingface.co/datasets/{args.hf_repo}")

    while True:
        try:
            lang = upload_q.get(timeout=1.0)
        except queue.Empty:
            if stop_event.is_set():
                return
            continue
        status.upload_queue_depth = max(0, upload_q.qsize())
        if lang is None:
            break
        try:
            _upload_language(api, args, db, lang, log, status, CommitOperationAdd)
            status.lang_uploaded_frames.add(lang)
        except Exception as e:
            log("UPLOAD", lang, f"! upload failed, will retry on next pipeline run: {e}")
        upload_q.task_done()

    log("UPLOAD", "-", "all languages processed — uploading final manifest + README")
    try:
        _upload_final_manifest_and_readme(api, args, log)
    except Exception as e:
        log("UPLOAD", "-", f"! final manifest/README upload failed (data itself is already "
           f"uploaded; safe to re-run): {e}")


def _upload_language(api, args, db: RunDB, lang: str, log, status, CommitOperationAdd) -> None:
    frames_dir = ROOT / args.frames_out_dir
    frames_path = frames_dir / f"frames_{lang}.jsonl"
    if not frames_path.exists():
        log("UPLOAD", lang, "no frames file yet, skipping")
        return

    already_synced = db.hf_synced_ids(args.hf_repo)

    # gather referenced .npz files not yet uploaded
    to_upload: list[tuple[str, Path]] = []   # (clip_id, local_path)
    with frames_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec["scenario_id"]
            if cid in already_synced:
                continue
            npz_path = ROOT / rec["encoded_path"]
            if npz_path.exists():
                to_upload.append((cid, npz_path))

    log("UPLOAD", lang, f"{len(to_upload)} new .npz to upload, plus frames_{lang}.jsonl")

    commits = []
    cur, cur_ids = [], []
    for cid, path in to_upload:
        cur.append(CommitOperationAdd(path_in_repo=f"encoded/{lang}/{path.name}", path_or_fileobj=str(path)))
        cur_ids.append(cid)
        if len(cur) >= args.files_per_commit:
            commits.append((cur, cur_ids))
            cur, cur_ids = [], []
    if cur:
        commits.append((cur, cur_ids))

    for i, (ops, ids) in enumerate(commits):
        log("UPLOAD", lang, f"commit {i + 1}/{len(commits)}: {len(ops)} .npz files...")
        create_commit_with_backoff(
            api, repo=args.hf_repo, operations=ops,
            commit_message=f"phase1: {lang} encoded batch ({len(ops)} clips)",
            max_retries=args.max_retries, base_backoff=args.backoff, max_backoff=args.max_backoff,
            log_fn=lambda m: log("UPLOAD", lang, m),
        )
        for cid in ids:
            db.mark_hf_synced(cid, args.hf_repo)
        status.npz_uploaded_total += len(ops)

    # frames file itself — small, always re-upload the current version (it's rebuilt
    # fully on every encode pass for this language, so a stale remote copy is just wrong)
    create_commit_with_backoff(
        api, repo=args.hf_repo,
        operations=[CommitOperationAdd(path_in_repo=f"frames_phase1/frames_{lang}.jsonl",
                                       path_or_fileobj=str(frames_path))],
        commit_message=f"phase1: {lang} frames ({sum(1 for _ in frames_path.open()) } records)",
        max_retries=args.max_retries, base_backoff=args.backoff, max_backoff=args.max_backoff,
        log_fn=lambda m: log("UPLOAD", lang, m),
    )
    log("UPLOAD", lang, f"done — {len(to_upload)} .npz + frames_{lang}.jsonl synced")


def _upload_final_manifest_and_readme(api, args, log) -> None:
    from huggingface_hub import CommitOperationAdd

    manifest_file = manifest_path(ROOT / args.out_dir)
    ops = []
    if manifest_file.exists():
        ops.append(CommitOperationAdd(path_in_repo="manifest.jsonl", path_or_fileobj=str(manifest_file)))

    readme = (
        f"---\n"
        f"license: cc0-1.0\n"
        f"language: [en, hi, gu]\n"
        f"tags: [thinkspark-v2-350m, phase1, mimi, audio]\n"
        f"---\n\n"
        f"# ThinkSpark-v2-350M — Phase-1 free-audio training data\n\n"
        f"Pre-encoded Mimi cb0/energy/f0 tokens + frame records, ready to train Phase 1\n"
        f"(modality alignment) directly — no re-encoding needed. Produced by\n"
        f"`scripts/P1_00_pipeline.py`; fetch onto a training machine (e.g. Kaggle) with\n"
        f"`scripts/19_fetch_training_data.py --phase1-repo {args.hf_repo}`.\n\n"
        f"## Layout\n\n"
        f"```\n"
        f"encoded/<lang>/<clip_id>.npz         Mimi cb0 + energy + f0\n"
        f"frames_phase1/frames_<lang>.jsonl    frame records (encoded_path is ROOT-relative)\n"
        f"manifest.jsonl                        full provenance (source, transcript, gender, duration)\n"
        f"```\n\n"
        f"See the main project README for the source mix (LibriSpeech / Kathbath / "
        f"Shrutilipi / FLEURS / IndicTTS-Gujarati) and licensing per source.\n"
    )
    tmp_readme = ROOT / args.out_dir / "_README_upload_tmp.md"
    tmp_readme.write_text(readme, encoding="utf-8")
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(tmp_readme)))

    if ops:
        create_commit_with_backoff(
            api, repo=args.hf_repo, operations=ops, commit_message="phase1: manifest + README",
            max_retries=args.max_retries, base_backoff=args.backoff, max_backoff=args.max_backoff,
            log_fn=lambda m: log("UPLOAD", "-", m),
        )
        log("UPLOAD", "-", f"manifest + README uploaded -> https://huggingface.co/datasets/{args.hf_repo}")
    tmp_readme.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_corpus.yaml")
    ap.add_argument("--out-dir", default="data/phase1_raw")
    ap.add_argument("--encoded-dir", default="data/encoded")
    ap.add_argument("--frames-out-dir", default="data/frames_phase1")
    ap.add_argument("--db", default="data/thinkspark_phase1.db")
    ap.add_argument("--hf-repo", default=None,
                    help="HF dataset repo, e.g. anuj-inavlabs/kupe-thinkspark-270m-phase1-data "
                        "— required unless --no-upload")
    ap.add_argument("--no-upload", action="store_true", help="run locally only, skip HF upload")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--download-concurrency", type=int, default=3,
                    help="concurrent (language, source) download workers (default 3)")
    ap.add_argument("--device", default=None, help="Mimi encoder device override (cpu/cuda/mps)")
    ap.add_argument("--files-per-commit", type=int, default=200,
                    help="max .npz files packed into one HF commit (default 200)")
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--backoff", type=float, default=20.0)
    ap.add_argument("--max-backoff", type=float, default=300.0)
    ap.add_argument("--max-retries", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, touch nothing")
    args = ap.parse_args()

    if not args.no_upload and not args.hf_repo:
        raise SystemExit("--hf-repo is required unless --no-upload is set "
                         "(e.g. --hf-repo anuj-inavlabs/kupe-thinkspark-270m-phase1-data)")

    cfg = Phase1CorpusConfig.from_yaml(ROOT / args.config)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_path(out_dir)
    already = existing_written(manifest_file)
    hf_token = env("HF_TOKEN")

    log = Logger()
    langs = list(cfg.sources.keys())
    jobs = [(lang, spec) for lang in langs for spec in cfg.sources.get(lang, [])]

    print("=" * 72)
    print("ThinkSpark-v2-350M — Phase-1 pipeline (download -> encode -> frames -> upload)")
    print("=" * 72)
    for lang in langs:
        print(f"[{lang}] target={cfg.target_hours.get(lang, 0.0):.0f}h  "
             f"sources={[s.id for s in cfg.sources.get(lang, [])]}")
    print(f"download concurrency: {args.download_concurrency}")
    print(f"HF repo: {args.hf_repo or '(uploads disabled)'}")
    print()

    if args.dry_run:
        manifest_fh_dry = None
        for lang, spec in jobs:
            r = fetch_source(cfg, lang, spec, out_dir, ROOT, manifest_fh_dry, already,
                             hf_token, True)
            print(f"  [{lang}/{spec.id}] have={r['have_hours']:.2f}h target={r['target_hours']:.2f}h "
                 f"remaining={r['remaining_hours']:.2f}h")
        print("\nplan only — nothing downloaded/encoded/uploaded. Re-run without --dry-run.")
        return

    manifest_fh = manifest_file.open("a", encoding="utf-8")
    manifest_w = LockedWriter(manifest_fh)

    status = PipelineStatus(langs, cfg)
    stop_event = threading.Event()
    encode_q: "queue.Queue[str | None]" = queue.Queue()
    upload_q: "queue.Queue[str | None]" = queue.Queue()

    db = RunDB(ROOT / args.db)

    status_thread = threading.Thread(target=status_summary_loop, args=(status, log, stop_event), daemon=True)
    status_thread.start()

    encode_thread = threading.Thread(
        target=encode_worker_loop, args=(encode_q, upload_q, cfg, args, log, status, stop_event), daemon=True)
    encode_thread.start()

    upload_thread = threading.Thread(
        target=upload_worker_loop, args=(upload_q, args, db, log, status, stop_event), daemon=True)
    upload_thread.start()

    # DOWNLOAD: bounded thread pool, one task per (lang, source). As each (lang, source)
    # completes, check if that language's ENTIRE source set is now done -> enqueue it.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pending_sources: dict[str, set[str]] = {
        lang: {s.id for s in cfg.sources.get(lang, [])} for lang in langs
    }
    langs_enqueued: set[str] = set()

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.download_concurrency)) as pool:
            futures = {
                pool.submit(download_worker, cfg, lang, spec, out_dir, manifest_w, already,
                           hf_token, log, status): (lang, spec.id)
                for lang, spec in jobs
            }
            for fut in as_completed(futures):
                lang, source_id = futures[fut]
                try:
                    fut.result()
                except SystemExit as e:
                    log("DOWNLOAD", f"{lang}/{source_id}", f"! FAILED (not retryable): {e}")
                except Exception as e:
                    log("DOWNLOAD", f"{lang}/{source_id}", f"! unexpected error: {e}")
                pending_sources[lang].discard(source_id)
                if not pending_sources[lang] and lang not in langs_enqueued:
                    langs_enqueued.add(lang)
                    status.lang_downloaded.add(lang)
                    log("DOWNLOAD", lang, "ALL sources done for this language -> encoding now")
                    encode_q.put(lang)
    finally:
        manifest_fh.close()

    log("MAIN", "-", "all downloads finished — waiting for encode/upload to drain...")
    encode_q.put(None)   # sentinel: no more languages, propagates to upload_q when encode drains
    encode_thread.join()
    upload_thread.join()
    stop_event.set()

    print("\n" + "=" * 72)
    print("Phase-1 pipeline complete.")
    print(f"  encoded  -> {ROOT / args.encoded_dir}")
    print(f"  frames   -> {ROOT / args.frames_out_dir}")
    if not args.no_upload:
        print(f"  uploaded -> https://huggingface.co/datasets/{args.hf_repo}")
    print("next (Kaggle): python scripts/19_fetch_training_data.py --phase1-repo "
         f"{args.hf_repo or '<repo>'}")
    db.close()


if __name__ == "__main__":
    main()
