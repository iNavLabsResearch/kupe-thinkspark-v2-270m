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

    # delete everything already uploaded to the HF repo (encoded/, frames_phase1/,
    # manifest.jsonl, README.md) AND the matching local hf_sync records, so a following
    # real run re-uploads cleanly instead of thinking it's all already there. Asks for
    # confirmation first; local files (data/encoded, data/frames_phase1, manifest) are
    # NEVER touched by this — only the remote repo + the local sync-tracking DB:
    python scripts/P1_00_pipeline.py --hf-repo anuj-inavlabs/kupe-thinkspark-270m-phase1-data --cleanup
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
from thinkspark.hf_upload import create_commit_with_backoff, ensure_repo, npz_repo_path, log as _hf_log
from thinkspark.phase1_corpus import (
    Phase1CorpusConfig, build_frame_record, existing_written, fetch_source, manifest_path,
)

_ = _hf_log  # imported for parity/reference; this script uses its own tagged Logger below

STATUS_INTERVAL_S = 30.0
DASHBOARD_REFRESH_S = 2.0   # how often the rich Live table redraws (independent of STATUS_INTERVAL_S)

# `rich` is an existing optional dependency in this project (see requirements.txt,
# already used by scripts/11_monitor.py with the same plain-text-fallback pattern) — no
# new dependency introduced here, just reused for a colorful/tqdm-style dashboard instead
# of flat log lines. Every rich object below funnels through ONE shared Console so the
# Live table and the tagged log lines interleave correctly instead of corrupting each
# other's redraws (both write to the same stdout).
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None

_STAGE_STYLE = {
    "DOWNLOAD": "cyan", "ENCODE": "yellow", "FRAMES": "blue",
    "UPLOAD": "green", "STATUS": "magenta", "MAIN": "white", "CLEANUP": "red",
}


# --------------------------------------------------------------------------- #
class Logger:
    """Thread-safe, tagged, timestamped logging — many workers print concurrently.
    Colorized via `rich` when available (one color per stage, errors in bold red),
    identical plain-text output otherwise — same fallback pattern as scripts/11_monitor.py."""

    def __init__(self):
        self._lock = threading.Lock()

    def __call__(self, stage: str, tag: str, msg: str) -> None:
        with self._lock:
            ts = time.strftime("%H:%M:%S")
            if _RICH:
                is_err = msg.lstrip().startswith("!")
                style = "bold red" if is_err else _STAGE_STYLE.get(stage, "white")
                _console.print(
                    f"[dim]\\[{ts}][/dim] [{style}]\\[{stage:<8}][/{style}] "
                    f"[bold]{tag:<22}[/bold] {msg}"
                )
            else:
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
        self.deleted_wavs_total = 0
        # Per-language breakdown — what the colorful dashboard actually renders per row
        # ("how many encoded", "how many uploaded [of which db/repo]", "what deleted").
        self.lang_npz_encoded: dict[str, int] = {lang: 0 for lang in langs}
        self.lang_npz_uploaded: dict[str, int] = {lang: 0 for lang in langs}
        self.lang_deleted: dict[str, int] = {lang: 0 for lang in langs}
        self.upload_queue_depth = 0
        self.encode_queue_depth = 0
        self.upload_disabled_reason: str | None = None
        self.done = False

    def add_download_hours(self, lang: str, hours: float) -> None:
        with self._lock:
            self.download_hours[lang] = self.download_hours.get(lang, 0.0) + hours

    def add_encoded(self, lang: str, n: int = 1) -> None:
        with self._lock:
            self.npz_encoded_total += n
            self.lang_npz_encoded[lang] = self.lang_npz_encoded.get(lang, 0) + n

    def add_uploaded(self, lang: str, n: int) -> None:
        with self._lock:
            self.npz_uploaded_total += n
            self.lang_npz_uploaded[lang] = self.lang_npz_uploaded.get(lang, 0) + n

    def add_deleted(self, lang: str, n: int = 1) -> None:
        with self._lock:
            self.deleted_wavs_total += n
            self.lang_deleted[lang] = self.lang_deleted.get(lang, 0) + n

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
                "deleted_wavs_total": self.deleted_wavs_total,
                "lang_npz_encoded": dict(self.lang_npz_encoded),
                "lang_npz_uploaded": dict(self.lang_npz_uploaded),
                "lang_deleted": dict(self.lang_deleted),
                "encode_queue_depth": self.encode_queue_depth,
                "upload_queue_depth": self.upload_queue_depth,
                "upload_disabled_reason": self.upload_disabled_reason,
            }


def _bar(frac: float, width: int = 20) -> str:
    """tqdm-style block bar, e.g. '████████░░░░░░░░ 41%' — used inside the rich table."""
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return f"{'█' * filled}{'░' * (width - filled)} {frac * 100:4.0f}%"


def _render_dashboard(status: PipelineStatus, hf_repo: str | None) -> "Table":
    from rich.table import Table

    s = status.snapshot()
    table = Table(title=f"ThinkSpark-v2-350M — Phase-1 pipeline"
                        f"{'  ->  ' + hf_repo if hf_repo else '  (local only, no upload)'}",
                 title_style="bold white", header_style="bold white", expand=True)
    table.add_column("Lang", style="bold")
    table.add_column("Downloaded", ratio=3)
    table.add_column("Encoded", justify="right", style="yellow")
    table.add_column("Uploaded", justify="right", style="green")
    table.add_column("Deleted (freed)", justify="right", style="red")

    for lang in status.langs:
        have = s["download_hours"].get(lang, 0.0)
        target = s["download_targets"].get(lang, 0.0) or 1.0
        bar = _bar(have / target)
        dl_style = "cyan" if lang not in s["lang_downloaded"] else "bold cyan"
        table.add_row(
            lang,
            f"[{dl_style}]{bar}[/{dl_style}] {have:.1f}/{target:.0f}h",
            str(s["lang_npz_encoded"].get(lang, 0)),
            str(s["lang_npz_uploaded"].get(lang, 0)),
            str(s["lang_deleted"].get(lang, 0)),
        )

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]", "",
        f"[bold yellow]{s['npz_encoded_total']}[/bold yellow]",
        f"[bold green]{s['npz_uploaded_total']}[/bold green]",
        f"[bold red]{s['deleted_wavs_total']}[/bold red]",
    )
    caption = (f"queues: encode={s['encode_queue_depth']} upload={s['upload_queue_depth']}   "
              f"encoded_langs={sorted(s['lang_encoded']) or '-'}   "
              f"uploaded_langs={sorted(s['lang_uploaded_frames']) or '-'}")
    if s["upload_disabled_reason"]:
        # Impossible to miss: red, on every single redraw of the pinned table, not just
        # a one-time log line that scrolls away in a long Kaggle run.
        caption += f"\n[bold red]⚠ UPLOADS DISABLED: {s['upload_disabled_reason']}[/bold red]"
    table.caption = caption
    return table


def status_summary_loop(status: PipelineStatus, log: Logger, stop_event: threading.Event,
                        hf_repo: str | None = None) -> None:
    if not _RICH:
        # Plain-text fallback — identical to the pre-dashboard behavior, so a machine
        # without `rich` installed still gets a full status line, just not colorful.
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
               f"deleted={s['deleted_wavs_total']}  "
               f"queues[encode={s['encode_queue_depth']} upload={s['upload_queue_depth']}]")
            if s["upload_disabled_reason"]:
                log("STATUS", "overall", f"! UPLOADS DISABLED: {s['upload_disabled_reason']}")
        return

    # Rich path: a small Live table pinned at the bottom of the terminal, redrawn every
    # DASHBOARD_REFRESH_S. Tagged log lines (Logger, above) keep printing normally through
    # the SAME Console — Live is explicitly designed to support interleaved console.print
    # calls from other threads while it's active, so the scrolling event log and the
    # pinned live summary coexist without corrupting each other's redraws.
    with Live(_render_dashboard(status, hf_repo), console=_console,
             refresh_per_second=1.0 / DASHBOARD_REFRESH_S, transient=False) as live:
        while not stop_event.wait(DASHBOARD_REFRESH_S):
            live.update(_render_dashboard(status, hf_repo))


def periodic_encode_sweep(encode_q: "queue.Queue[str | None]", langs: list[str],
                          interval_s: float, sweep_stop_event: threading.Event, log: Logger) -> None:
    """
    Enqueues every language onto the encode queue on a fixed timer, REGARDLESS of
    whether any individual source or language has actually finished downloading yet.

    Why this exists: encoding used to only get triggered once ALL of a language's
    sources finished — fine when sources are small, but a single big source (e.g.
    LibriSpeech's 90h English slice) can download for a long time before that ever
    happens, during which raw wav accumulates completely unbounded (no cleanup — see
    `encode_and_build_frames`'s wav-delete-after-encode) with nothing to stop it. This
    sweep decouples encoding from any completion event entirely: every `interval_s`,
    whatever's downloaded so far — even mid-source, even the very first clip — gets
    encoded and its wav freed, so disk usage stays bounded by "how much downloads in
    one interval" instead of "the single biggest source's entire target". Enqueuing an
    already-fully-encoded language is a cheap no-op (`encode_and_build_frames` skips
    wavs that already have a matching .npz), so over-sweeping costs almost nothing.
    """
    while not sweep_stop_event.wait(interval_s):
        for lang in langs:
            encode_q.put(lang)


# --------------------------------------------------------------------------- #
# DOWNLOAD stage
# --------------------------------------------------------------------------- #
def download_worker(cfg, lang, spec, out_dir, manifest_w, already, hf_token, log, status):
    tag = f"{lang}/{spec.id}"
    log("DOWNLOAD", tag, f"starting -> {spec.hf_dataset}/{spec.hf_config or spec.split} "
       f"(weight={spec.weight:.2f})")

    # Credit whatever's already on disk from a PRIOR run immediately, then stream live
    # progress via progress_fn as NEW clips are written this run — otherwise a big
    # source (e.g. 90h) shows 0.0h in the [STATUS] line the whole time it's actively
    # downloading, only updating once the entire source finishes, which reads as
    # "stuck" even though it never was.
    baseline_h = already.get((lang, spec.id), {}).get("hours", 0.0)
    if baseline_h:
        status.add_download_hours(lang, baseline_h)

    r = fetch_source(cfg, lang, spec, out_dir, ROOT, manifest_w, already, hf_token, False,
                     log_fn=lambda m: log("DOWNLOAD", tag, m),
                     progress_fn=lambda h: status.add_download_hours(lang, h))
    if r["status"] == "already_done":
        log("DOWNLOAD", tag, f"already done: {r['have_hours']:.2f}h >= {r['target_hours']:.2f}h target")
    else:
        log("DOWNLOAD", tag, f"done: wrote {r.get('written', 0)} clips -> "
           f"{r['have_hours']:.2f}h (target {r['target_hours']:.2f}h)")
    return lang, spec.id, r


# --------------------------------------------------------------------------- #
# ENCODE + FRAMES stage — one Mimi encoder instance PER DEVICE, reused across calls
# (not reloaded every sweep), so `--devices cuda:0,cuda:1` genuinely uses both GPUs.
# --------------------------------------------------------------------------- #
_encoders: dict[str, "object"] = {}   # device string -> MimiEncoder, lazily created once
_encoders_lock = threading.Lock()


def resolve_devices(args) -> list[str]:
    """--devices explicit list > --device single override > auto-detect ALL visible
    CUDA devices (so `--num-gpu 2` isn't even needed, both T4s are used automatically)
    > cpu. Resolved ONCE in main() and reused for the whole run — avoids re-probing
    torch.cuda on every encode sweep."""
    if getattr(args, "devices", None):
        return [d.strip() for d in args.devices.split(",") if d.strip()]
    if args.device:
        return [args.device]
    try:
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n = 0
    if n > 0:
        return [f"cuda:{i}" for i in range(n)]
    return ["cpu"]


def _get_encoder(device: str, cfg, log):
    """One MimiEncoder per device, created once and cached — NOT reloaded on every
    encode sweep, and never shared across two threads at once (callers only ever
    dispatch one shard per device concurrently, see encode_and_build_frames)."""
    with _encoders_lock:
        enc = _encoders.get(device)
        if enc is None:
            from thinkspark.mimi_codec import MimiEncoder
            log("ENCODE", device, f"loading Mimi encoder on {device} "
               f"({getattr(cfg, 'mimi_repo', 'kyutai/mimi')})...")
            enc = MimiEncoder(repo=getattr(cfg, "mimi_repo", "kyutai/mimi"), device=device)
            cb_size = enc.codebook_size   # triggers _ensure_loaded()
            log("ENCODE", device, f"Mimi encoder ready (device={enc._device}, codebook_size={cb_size})")
            _encoders[device] = enc
        return enc


def _encode_shard(wavs: list[Path], device: str, encoded_dir: Path, cfg, args, log,
                  status, lang: str) -> tuple[int, int]:
    """Encode one shard of `wavs` on `device`. Returns (new_count, deleted_count)."""
    encoder = _get_encoder(device, cfg, log)
    new_count = 0
    deleted_count = 0
    for wav in wavs:
        out_path = encoded_dir / f"{wav.stem}.npz"
        if out_path.exists():
            continue
        try:
            enc = encoder.encode_wav_file(str(wav))
            enc.save(out_path)
            new_count += 1
            status.add_encoded(lang)
            if new_count % 25 == 0:
                log("ENCODE", f"{lang}@{device}", f"...{new_count} new clips encoded so far")
            # Raw wav is only needed to PRODUCE the .npz — once that's saved (and
            # verified non-empty), keep the .npz (the actual training input) and delete
            # the wav. This is what stops disk from filling: at ~48 KB/s of raw 24kHz
            # audio, a full multi-hundred-hour target needs tens of GB of raw wav that
            # nothing downstream ever reads again once encoded. `--keep-raw-audio` opts
            # out if you want the raw wavs kept for some other reason.
            if not getattr(args, "keep_raw_audio", False):
                try:
                    import numpy as np
                    if len(np.load(out_path)["cb0"]) > 0:
                        wav.unlink()
                        deleted_count += 1
                        status.add_deleted(lang)
                    else:
                        log("ENCODE", f"{lang}@{device}", f"! {wav.name} encoded to 0 frames — keeping wav for inspection")
                except Exception as ve:
                    log("ENCODE", f"{lang}@{device}", f"! couldn't verify {out_path.name}, keeping its wav: {ve}")
        except Exception as e:
            log("ENCODE", f"{lang}@{device}", f"! failed {wav.name}: {e}")

        # Explicitly yield the GIL after every clip. Real, confirmed issue (not
        # theoretical): a continuous run of torch/numpy C-extension calls back-to-back
        # can hold the GIL for the vast majority of wall-clock time even though each
        # individual call is short — CPython's normal bytecode-level GIL check happens
        # BETWEEN Python instructions, not mid-C-call, so a tight loop dominated by C
        # extension work can nearly starve other threads of GIL time entirely (a
        # well-documented CPython "convoy effect": a CPU-bound thread that's always
        # immediately ready to run tends to keep winning the GIL against I/O-bound
        # threads that only intermittently become ready). Measured on this exact
        # pipeline: download threads made ZERO progress for 10+ minutes while this loop
        # ran continuously, confirmed via `ps -M` showing one thread pinned at ~85% CPU
        # and all download threads at ~0%. `time.sleep()` unconditionally releases the
        # GIL for its duration, giving the download threads a guaranteed window every
        # single clip — a small, deliberate cost (default 30ms/clip) worth paying so
        # downloading and encoding actually run concurrently instead of one starving
        # the other on a single machine.
        time.sleep(args.encode_yield_ms / 1000.0)

    return new_count, deleted_count


def encode_and_build_frames(lang: str, cfg, args, log, status) -> int:
    """Encodes any not-yet-encoded wavs for `lang` — split across ALL of `args.devices_list`
    in parallel when there's more than one (e.g. `--devices cuda:0,cuda:1` uses both of
    Kaggle's T4s at once instead of one sitting idle) — then rebuilds its
    frames_<lang>.jsonl from the manifest. Returns the number of NEWLY encoded clips."""
    wav_dir = ROOT / args.out_dir / lang
    encoded_dir = ROOT / args.encoded_dir
    encoded_dir.mkdir(parents=True, exist_ok=True)

    all_wavs = sorted(wav_dir.glob("*.wav"))
    pending = [w for w in all_wavs if not (encoded_dir / f"{w.stem}.npz").exists()]

    devices = getattr(args, "devices_list", None) or ["cpu"]
    new_count = deleted_count = 0
    if pending:
        if len(devices) == 1 or len(pending) == 1:
            new_count, deleted_count = _encode_shard(pending, devices[0], encoded_dir, cfg, args, log, status, lang)
        else:
            # Round-robin split (not contiguous chunks) so every device gets a similar mix
            # of clip sizes instead of one device getting a run of unusually long/short
            # clips back-to-back purely by file-sort-order luck.
            shards = {d: pending[i::len(devices)] for i, d in enumerate(devices)}
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                futs = {
                    pool.submit(_encode_shard, shard, d, encoded_dir, cfg, args, log, status, lang): d
                    for d, shard in shards.items() if shard
                }
                for fut in as_completed(futs):
                    n, d_ = fut.result()
                    new_count += n
                    deleted_count += d_

    if deleted_count:
        log("ENCODE", lang, f"freed disk: deleted {deleted_count} raw wav(s) now that they're encoded")

    log("ENCODE", lang, f"encoding done: {new_count} new / {len(all_wavs)} total wavs "
       f"-> {encoded_dir} (devices={devices})")

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
        status.upload_disabled_reason = "disabled via --no-upload"
        return

    # REAL BUG THIS GUARDS AGAINST: `env("HF_TOKEN", required=True)` used to be called
    # bare here — if HF_TOKEN was missing, it raised RuntimeError with NOTHING catching
    # it, which silently killed this whole thread the instant the run started. Python
    # doesn't crash the process or print through our tagged Logger when a background
    # thread dies uncaught — it just vanishes, so "Uploaded" stayed frozen at 0 for the
    # ENTIRE run with no visible reason (confirmed: this is exactly what happened on a
    # real Kaggle run — `!export HF_TOKEN=...` in one notebook cell does NOT persist into
    # a later `!python ...` cell's subprocess environment; each `!` cell is its own shell).
    # Now every failure mode is caught, logged LOUDLY and repeatedly through the same
    # Logger everything else uses (so it can't scroll past unnoticed in a long Kaggle
    # log), and surfaced on the live dashboard too — never silent again.
    setup_error: str | None = None
    api = None
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as e:
        setup_error = f"huggingface_hub not installed — `pip install huggingface_hub` ({e})"

    if setup_error is None:
        try:
            token = env("HF_TOKEN", required=True)
        except Exception as e:
            setup_error = (
                f"HF_TOKEN not set in this process's environment ({e}). If you're on "
                f"Kaggle/Jupyter: `!export HF_TOKEN=...` in one cell does NOT persist to "
                f"a later `!python ...` cell — each `!` line runs in its own throwaway "
                f"shell. Set it BEFORE launching this script instead, in the SAME cell "
                f"or process: `import os; os.environ['HF_TOKEN'] = '...'` (or a Kaggle "
                f"Secret loaded the same way), then re-run."
            )
        else:
            api = HfApi(token=token)
            try:
                ensure_repo(api, args.hf_repo, args.private)
            except RuntimeError as e:
                setup_error = str(e)
            except Exception as e:
                setup_error = f"unexpected error reaching HF: {e}"

    if setup_error:
        status.upload_disabled_reason = setup_error
        log("UPLOAD", "-", f"! UPLOADS DISABLED FOR THIS RUN: {setup_error}")
        log("UPLOAD", "-", "! local data (download/encode) is unaffected and keeps "
           "running normally — fix this, then re-run the SAME command; everything "
           "encoded so far is still there and will upload on the next run.")
        # Keep nagging periodically instead of just dying — a single line at the top of
        # a run that logs for hours (Kaggle's 12h session) is trivially lost to scrollback.
        while not stop_event.wait(300.0):
            log("UPLOAD", "-", f"! still disabled: {setup_error}")
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

    if not to_upload:
        # Nothing new since last upload for this language — with the periodic encode
        # sweep (encoding + this function both get invoked repeatedly, independent of
        # any source/language actually finishing), this is the common case most passes.
        # Skip entirely rather than re-uploading an unchanged frames_<lang>.jsonl on
        # every sweep, which would otherwise spam the repo with a commit every
        # `--encode-sweep-interval` seconds for the whole run even when idle.
        log("UPLOAD", lang, "nothing new since last upload, skipping")
        return

    log("UPLOAD", lang, f"{len(to_upload)} new .npz to upload, plus frames_{lang}.jsonl")

    commits = []
    cur, cur_ids = [], []
    for cid, path in to_upload:
        cur.append(CommitOperationAdd(path_in_repo=npz_repo_path(lang, path.name), path_or_fileobj=str(path)))
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
        status.add_uploaded(lang, len(ops))

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
def _run_cleanup(args) -> None:
    """Deletes everything currently uploaded to args.hf_repo — real content, listed
    from the repo itself (not assumed) — and clears this repo's hf_sync rows in the
    local DB so a subsequent real run re-uploads cleanly instead of thinking it's all
    already there. Never touches local files (data/encoded, data/frames_phase1,
    data/phase1_raw/manifest.jsonl are all left exactly as they are)."""
    try:
        from huggingface_hub import CommitOperationDelete, HfApi
    except ImportError:
        raise SystemExit("`huggingface_hub` not installed. `pip install huggingface_hub`.")

    token = env("HF_TOKEN", required=True)
    api = HfApi(token=token)

    print("=" * 68)
    print(f"ThinkSpark-v2-350M — Phase-1 HF cleanup: {args.hf_repo}")
    print("=" * 68)

    try:
        files = api.list_repo_files(repo_id=args.hf_repo, repo_type="dataset")
    except Exception as e:
        msg = str(e).lower()
        if "404" in msg or "not found" in msg:
            print(f"repo {args.hf_repo} doesn't exist (or has nothing in it) — nothing to clean up.")
            return
        raise SystemExit(f"couldn't list files in {args.hf_repo}: {e}")

    if not files:
        print("repo exists but is empty — nothing to clean up.")
        return

    # group into top-level entries (folders vs bare files) purely from what's REALLY
    # there, not an assumed fixed layout — so this also cleans up anything left over
    # from an older/different run shape
    top_level: dict[str, bool] = {}   # name -> is_folder
    for f in files:
        if "/" in f:
            top_level[f.split("/", 1)[0]] = True
        else:
            top_level.setdefault(f, False)

    print(f"found {len(files)} file(s) under {len(top_level)} top-level entr{'y' if len(top_level)==1 else 'ies'}:")
    for name, is_folder in sorted(top_level.items()):
        n = sum(1 for f in files if f == name or f.startswith(name + "/"))
        print(f"  {'[dir] ' if is_folder else '[file]'} {name}  ({n} file{'s' if n != 1 else ''})")

    db = RunDB(ROOT / args.db)
    n_synced = len(db.hf_synced_ids(args.hf_repo))
    print(f"\nlocal hf_sync records for this repo: {n_synced} (will also be cleared)")
    print("local files (data/encoded, data/frames_phase1, manifest.jsonl) are NOT touched.")

    try:
        answer = input(f"\nDelete all {len(files)} file(s) from {args.hf_repo} and clear "
                       f"{n_synced} local sync record(s)? Type 'yes' to confirm: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\ncancelled.")
        db.close()
        return
    if answer.lower() != "yes":
        print("cancelled — nothing deleted.")
        db.close()
        return

    ops = [CommitOperationDelete(path_in_repo=name, is_folder=is_folder)
          for name, is_folder in top_level.items()]
    print(f"\ndeleting {len(ops)} top-level entr{'y' if len(ops)==1 else 'ies'} in one commit...")
    create_commit_with_backoff(
        api, repo=args.hf_repo, operations=ops,
        commit_message="cleanup: remove all phase1 pipeline content",
        max_retries=args.max_retries, base_backoff=args.backoff, max_backoff=args.max_backoff,
        log_fn=print,
    )

    n_cleared = db.clear_hf_sync(args.hf_repo)
    db.close()
    print(f"\ndone — {args.hf_repo} is now empty, {n_cleared} local sync record(s) cleared.")
    print("re-run scripts/P1_00_pipeline.py normally — it will re-upload from your local data/ "
         "encoded+frames files (no re-download or re-encode needed, they're untouched).")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_corpus.yaml")
    ap.add_argument("--lang", default=None,
                    help="only run this language (e.g. en / hi / gu), comma-separated "
                        "for more than one (e.g. en,hi). Default: all languages in "
                        "--config at once. Run one language at a time (three separate "
                        "commands) instead of all together when disk/bandwidth is tight "
                        "(e.g. Kaggle) — each command still downloads -> encodes -> "
                        "uploads -> deletes-old-wavs gradually AS IT GOES for that one "
                        "language, it's fully resumable, so `en` finishing before you "
                        "start `hi` costs nothing extra.")
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
    ap.add_argument("--device", default=None, help="Mimi encoder device override (cpu/cuda/mps) "
                    "— forces a SINGLE device; use --devices instead to spread encoding "
                    "across more than one GPU")
    ap.add_argument("--devices", default=None,
                    help="comma-separated devices to encode on IN PARALLEL, e.g. "
                        "'cuda:0,cuda:1' to use both of Kaggle's T4s at once. Default: "
                        "auto-detect — uses EVERY visible CUDA device already (so this "
                        "flag is normally optional on a 2-GPU Kaggle session), falls "
                        "back to a single cpu worker if no GPU is visible. Each device "
                        "gets its own persistent Mimi encoder instance + an even round-"
                        "robin share of whatever's pending for a language.")
    ap.add_argument("--keep-raw-audio", action="store_true",
                    help="don't delete a wav after encoding it (default: delete once its "
                        ".npz is verified — a full target's raw audio alone can be tens "
                        "of GB that nothing downstream ever reads again)")
    ap.add_argument("--encode-yield-ms", type=float, default=30.0,
                    help="milliseconds to sleep (explicitly releasing the GIL) after "
                        "EVERY encoded clip (default 30) — without this, continuous "
                        "back-to-back torch/numpy encoding can nearly starve the "
                        "download threads of CPU/GIL time entirely (measured: zero "
                        "download progress for 10+ minutes on this exact pipeline). "
                        "Set to 0 to disable if you ever run with a single download "
                        "source (nothing to starve) and want maximum encode throughput.")
    ap.add_argument("--encode-sweep-interval", type=float, default=90.0,
                    help="seconds between periodic encode passes over whatever's "
                        "downloaded SO FAR, independent of any source/language finishing "
                        "(default 90) — without this, disk usage tracks whichever source "
                        "has the BIGGEST target, not the smallest, since encoding used to "
                        "only start once an entire language's full download finished")
    ap.add_argument("--files-per-commit", type=int, default=200,
                    help="max .npz files packed into one HF commit (default 200)")
    ap.add_argument("--min-free-disk-gb", type=float, default=None,
                    help="pause downloads (polling until the encoder frees space via "
                        "wav-delete-after-encode) whenever free disk drops below this "
                        "many GB (default: configs/phase1_corpus.yaml's min_free_disk_gb, "
                        "itself 3.0 if unset) — the fix for a fast connection (e.g. "
                        "Kaggle) outrunning the encoder and hitting ENOSPC. Raise this on "
                        "a small disk (e.g. Kaggle's 57.6GiB) if it still fills.")
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--backoff", type=float, default=20.0)
    ap.add_argument("--max-backoff", type=float, default=300.0)
    ap.add_argument("--max-retries", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, touch nothing")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete everything already uploaded to --hf-repo (encoded/, "
                        "frames_phase1/, manifest.jsonl, README.md) + the matching local "
                        "hf_sync records, so a following run re-uploads cleanly; asks for "
                        "confirmation first. Local files (data/encoded etc.) are untouched.")
    args = ap.parse_args()

    if args.cleanup:
        if not args.hf_repo:
            raise SystemExit("--cleanup needs --hf-repo (which repo to clean up)")
        _run_cleanup(args)
        return

    if not args.no_upload and not args.hf_repo:
        raise SystemExit("--hf-repo is required unless --no-upload is set "
                         "(e.g. --hf-repo anuj-inavlabs/kupe-thinkspark-270m-phase1-data)")

    cfg = Phase1CorpusConfig.from_yaml(ROOT / args.config)
    if args.min_free_disk_gb is not None:
        cfg.min_free_disk_gb = args.min_free_disk_gb
    args.devices_list = resolve_devices(args)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_path(out_dir)
    already = existing_written(manifest_file)
    hf_token = env("HF_TOKEN")

    log = Logger()
    langs = list(cfg.sources.keys())
    if args.lang:
        wanted = {s.strip() for s in args.lang.split(",") if s.strip()}
        unknown = wanted - set(langs)
        if unknown:
            raise SystemExit(f"--lang {sorted(unknown)} not in {args.config}'s languages "
                             f"({langs})")
        langs = [l for l in langs if l in wanted]
    jobs = [(lang, spec) for lang in langs for spec in cfg.sources.get(lang, [])]

    print("=" * 72)
    print("ThinkSpark-v2-350M — Phase-1 pipeline (download -> encode -> frames -> upload)")
    if args.lang:
        print(f"(--lang filter active: only {langs} this run)")
    print("=" * 72)
    for lang in langs:
        print(f"[{lang}] target={cfg.target_hours.get(lang, 0.0):.0f}h  "
             f"sources={[s.id for s in cfg.sources.get(lang, [])]}")
    print(f"download concurrency: {args.download_concurrency}")
    print(f"encode devices: {args.devices_list}"
         f"{'  (parallel across ' + str(len(args.devices_list)) + ' devices)' if len(args.devices_list) > 1 else ''}")
    print(f"min free disk: {cfg.min_free_disk_gb:.1f}GB (downloads pause below this)")
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

    status_thread = threading.Thread(
        target=status_summary_loop, args=(status, log, stop_event, args.hf_repo), daemon=True)
    status_thread.start()

    encode_thread = threading.Thread(
        target=encode_worker_loop, args=(encode_q, upload_q, cfg, args, log, status, stop_event), daemon=True)
    encode_thread.start()

    upload_thread = threading.Thread(
        target=upload_worker_loop, args=(upload_q, args, db, log, status, stop_event), daemon=True)
    upload_thread.start()

    # Encode+free-disk on a timer too, not just when a whole language's downloads
    # finish — see periodic_encode_sweep's docstring for why this matters (disk usage
    # would otherwise track the single BIGGEST source's entire target).
    sweep_stop_event = threading.Event()
    sweep_thread = threading.Thread(
        target=periodic_encode_sweep,
        args=(encode_q, langs, args.encode_sweep_interval, sweep_stop_event, log), daemon=True)
    sweep_thread.start()

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

    sweep_stop_event.set()   # stop the periodic sweep — the sentinel below guarantees one final pass
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
