"""
Phase-1 free-audio corpus config + streaming fetch — shared by scripts/P1_01_fetch_corpus.py
(single-source-at-a-time CLI) and scripts/P1_00_pipeline.py (concurrent download/encode/
upload orchestrator). Moved here so both entry points share the exact same, hardened
fetch logic rather than risking it drifting between two copies.

See scripts/P1_01_fetch_corpus.py's module docstring for the full source list and the
Common Voice removal story; configs/phase1_corpus.yaml for the per-language source mix.

Audio is ALWAYS raw bytes/path + `soundfile`. This project does not use torchcodec.

`datasets` 4.x wants to decode Audio columns via torchcodec, which needs torch>=2.5
and a matching FFmpeg .so — neither is true on the boxes this pipeline actually
runs on (real crash: torch 2.4.1+cu124 + torchcodec 0.10.0 -> libavutil missing +
`Aborted (core dumped)`). English already finished without torchcodec; hi/gu must
use the same path.

Do NOT `cast_column(..., Audio(decode=False))` to "disable" decoding — on datasets
4.x that still runs `Audio.encode_example()`, which imports torchcodec UNCONDITIONALLY
(before it even looks at `decode`). Disable decode by mutating the existing feature
flag in place, and decode the `{bytes, path}` dict ourselves with soundfile.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

# Column-name candidates tried, in order, when a source doesn't set an explicit
# text_col/gender_col override — different HF datasets name these differently and we'd
# rather auto-detect than hard-crash on a guess that turns out wrong for one source.
_TEXT_COL_CANDIDATES = ["sentence", "text", "transcription", "transcript", "normalized_text"]
_GENDER_COL_CANDIDATES = ["gender", "sex", "speaker_gender"]
_AUDIO_COL_CANDIDATES = ["audio", "audio_filepath", "audio_path", "wav", "speech"]


# --------------------------------------------------------------------------- #
@dataclass
class SourceSpec:
    id: str
    hf_dataset: str
    hf_config: str | None
    split: str
    gated: bool = False
    weight: float = 1.0
    note: str = ""
    audio_col: str | None = None    # override; else auto-detect the Audio-typed column
    text_col: str | None = None     # override; else try _TEXT_COL_CANDIDATES
    gender_col: str | None = None   # override; else try _GENDER_COL_CANDIDATES

    @staticmethod
    def from_dict(d: dict) -> "SourceSpec":
        known = {f for f in SourceSpec.__dataclass_fields__}
        return SourceSpec(**{k: v for k, v in d.items() if k in known})


@dataclass
class Phase1CorpusConfig:
    target_hours: dict[str, float]
    sources: dict[str, list[SourceSpec]]
    gender_balance: bool = True
    sample_rate: int = 24000
    min_clip_seconds: float = 1.5
    max_clip_seconds: float = 20.0
    # Real, observed failure this guards against: on a fast connection (e.g. Kaggle),
    # downloads can outrun a single Mimi encoder by 50-60x — periodic encode+delete
    # sweeps alone can't drain a backlog that grows faster than they empty it, and the
    # whole run dies with ENOSPC. This makes every WRITE check real free disk space
    # first and BLOCK (not crash) once it's low, giving the encode side time to catch
    # up and free space via wav-delete-after-encode — turning "crash when disk fills"
    # into "downloads pace themselves to encoding speed", regardless of how fast the
    # network or how slow the encoder is on a given machine.
    min_free_disk_gb: float = 3.0

    @staticmethod
    def from_yaml(path: str) -> "Phase1CorpusConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        sources = {
            lang: [SourceSpec.from_dict(s) for s in specs]
            for lang, specs in (raw.get("sources") or {}).items()
        }
        return Phase1CorpusConfig(
            target_hours=raw.get("target_hours", {}),
            sources=sources,
            gender_balance=bool(raw.get("gender_balance", True)),
            sample_rate=int(raw.get("sample_rate", 24000)),
            min_clip_seconds=float(raw.get("min_clip_seconds", 1.5)),
            max_clip_seconds=float(raw.get("max_clip_seconds", 20.0)),
            min_free_disk_gb=float(raw.get("min_free_disk_gb", 3.0)),
        )


# --------------------------------------------------------------------------- #
def manifest_path(out_dir: Path) -> Path:
    return out_dir / "manifest.jsonl"


def existing_written(manifest_file: Path) -> dict[tuple[str, str], dict]:
    """
    {(lang, source_id): {"count": N, "hours": H, "female": F, "male": M}} from what's
    already on disk — the resume checkpoint. File-based, same pattern as script 02.
    """
    stats: dict[tuple[str, str], dict] = {}
    if not manifest_file.exists():
        return stats
    for line in manifest_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (rec.get("lang"), rec.get("source"))
        s = stats.setdefault(key, {"count": 0, "hours": 0.0, "female": 0, "male": 0})
        s["count"] += 1
        s["hours"] += rec.get("duration_s", 0.0) / 3600.0
        g = rec.get("gender")
        if g in ("female", "male"):
            s[g] += 1
    return stats


def clip_id(lang: str, source_id: str, row_index: int) -> str:
    raw = f"{lang}|{source_id}|{row_index}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def pick_column(candidates: list[str], feature_names: list[str]) -> str | None:
    for c in candidates:
        if c in feature_names:
            return c
    return None


def relative_or_absolute(path: Path, root: Path) -> str:
    """Store paths relative to `root` when possible (portable manifest/frame records —
    the whole point is these are meant to be re-downloaded onto a DIFFERENT machine and
    still resolve correctly); fall back to absolute if `path` isn't under `root`."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


_AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".pcm")


def _looks_like_audio_path(v: str) -> bool:
    lower = v.lower()
    return any(lower.endswith(s) for s in _AUDIO_SUFFIXES)


def _is_audio_value(v) -> bool:
    """True for raw audio shapes this project decodes with soundfile — never
    torchcodec `AudioDecoder` (that path imports libtorchcodec and crashes on
    torch<2.5 / missing FFmpeg). Strings are only audio if they look like a
    media path — a transcript/speaker-id string must not win column detection."""
    if isinstance(v, dict):
        if "array" in v and "sampling_rate" in v:
            return True
        if "bytes" in v or "path" in v:
            return True
        return False
    if isinstance(v, (bytes, bytearray)):
        return True
    if isinstance(v, str) and v:
        return _looks_like_audio_path(v)
    return hasattr(v, "_hf_encoded")


def detect_audio_column(row: dict) -> str | None:
    for name in _AUDIO_COL_CANDIDATES:
        if name in row and row[name] is not None:
            return name
    for k, v in row.items():
        if _is_audio_value(v):
            return k
    return None


def _read_with_soundfile(raw, path) -> tuple:
    """Decode wav/flac/ogg bytes or a local/remote path. Returns (None, None) if
    neither is usable — never falls through to torchcodec."""
    import io

    import numpy as np
    import soundfile as sf

    if raw:
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        return np.asarray(arr, dtype=np.float32), int(sr)
    if not path:
        return None, None
    if os.path.isfile(path):
        arr, sr = sf.read(path, dtype="float32", always_2d=False)
        return np.asarray(arr, dtype=np.float32), int(sr)
    if path.startswith(("hf://", "https://", "http://")) or "::" in path:
        try:
            from datasets.utils.file_utils import xopen
            with xopen(path, "rb") as fh:
                raw = fh.read()
        except Exception:
            return None, None
        if not raw:
            return None, None
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        return np.asarray(arr, dtype=np.float32), int(sr)
    return None, None


def extract_waveform(audio_value) -> tuple:
    """Returns (numpy_array, sample_rate) from raw bytes/path/array. Never calls
    torchcodec (`get_all_samples` is intentionally not used). (None, None) if
    the value isn't a known raw-audio shape."""
    import numpy as np

    if audio_value is None:
        return None, None
    if hasattr(audio_value, "_hf_encoded") and audio_value._hf_encoded:
        return extract_waveform(audio_value._hf_encoded)
    if isinstance(audio_value, (bytes, bytearray)):
        return _read_with_soundfile(bytes(audio_value), None)
    if isinstance(audio_value, str):
        return _read_with_soundfile(None, audio_value)
    if not isinstance(audio_value, dict):
        return None, None
    if "array" in audio_value and "sampling_rate" in audio_value:
        return np.asarray(audio_value["array"], dtype=np.float32), int(audio_value["sampling_rate"])
    return _read_with_soundfile(audio_value.get("bytes"), audio_value.get("path"))


def _set_decode_false(feat) -> None:
    """Walk a `datasets` feature tree and flip every media `decode` flag off."""
    if getattr(feat, "decode", None) is True:
        feat.decode = False
    inner = getattr(feat, "feature", None)
    if inner is not None:
        _set_decode_false(inner)
    if isinstance(feat, dict):
        for v in feat.values():
            _set_decode_false(v)


def _disable_media_decode(ds):
    """Stop `datasets` from decoding Audio/Image/Video columns.

    Mutates existing features IN PLACE. Do not `cast_column` / `cast` to
    `Audio(decode=False)` — datasets 4.x implements that via `Audio.encode_example()`,
    which does `from torchcodec.encoders import AudioEncoder` before it even
    inspects `self.decode`. That is the exact crash on hi/gu (Kathbath) after
    English already succeeded without torchcodec.
    """
    seen_ids: set[int] = set()
    for obj in (ds, getattr(ds, "info", None), getattr(ds, "_info", None)):
        if obj is None:
            continue
        feats = getattr(obj, "features", None)
        if feats is None or id(feats) in seen_ids:
            continue
        seen_ids.add(id(feats))
        values = feats.values() if hasattr(feats, "values") else []
        for feat in values:
            _set_decode_false(feat)
    return ds


def _patch_datasets_audio_no_torchcodec() -> None:
    """Make `datasets.Audio` encode/decode a no-op so a later accidental
    `cast_column` / `decode_example` cannot import torchcodec."""
    try:
        from datasets.features.audio import Audio
    except ImportError:
        return

    def _identity_decode(self, value, token_per_repo_id=None):
        return value

    def _identity_encode(self, value):
        if value is None:
            return {"bytes": None, "path": None}
        if isinstance(value, (bytes, bytearray)):
            return {"bytes": bytes(value), "path": None}
        if isinstance(value, str):
            return {"bytes": None, "path": value}
        if isinstance(value, dict):
            if "bytes" in value or "path" in value:
                return {"bytes": value.get("bytes"), "path": value.get("path")}
            return value
        encoded = getattr(value, "_hf_encoded", None)
        if isinstance(encoded, dict):
            return encoded
        return value

    Audio.decode_example = _identity_decode
    Audio.encode_example = _identity_encode


# --------------------------------------------------------------------------- #
# Transient-network resilience for HF streaming (Kaggle's network layer can drop a
# parquet GET mid-stream — real observed failure: '[Errno 9] Bad file descriptor' while
# resolving a shard, which also corrupts whatever `row` was mid-flight, making audio/
# text column auto-detection fail on that ONE row even though the dataset is fine). Two
# layers of defense, matching the retry-classification pattern already used in
# thinkspark.tts_soniox / thinkspark.hf_upload:
#   1. column auto-detection tolerates a few bad/incomplete rows before giving up — a
#      truly missing column fails EVERY row, a transient glitch fails only one.
#   2. the whole per-source stream is retried (fresh `load_dataset` call) a few times if
#      the iterator itself dies mid-stream, instead of crashing the whole run.
#
# Phrase markers (safe as plain substrings — no false-positive risk).
_TRANSIENT_PHRASES = ("bad file descriptor", "errno 9", "connection", "timeout",
                     "temporarily", "reset", "broken pipe")
# Bare HTTP status codes need a WORD-BOUNDARY regex, not a substring check — a real
# stack trace's line number (e.g. "line 1503") contains "503" as a substring and would
# otherwise misfire as a transient network error. Confirmed real bug: a permanent
# torchcodec/FFmpeg load failure was retried 4 times uselessly because its traceback
# happened to include a "...1503..." line number.
_TRANSIENT_STATUS_RE = re.compile(r"\b(502|503|504)\b")
COLUMN_DETECT_ROW_BUDGET = 5     # rows to try before concluding a column truly doesn't exist
STREAM_RETRY_ATTEMPTS = 4        # fresh-iterator retries on a transient mid-stream failure


def looks_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _TRANSIENT_PHRASES) or bool(_TRANSIENT_STATUS_RE.search(msg))


# --------------------------------------------------------------------------- #
# Proactive disk-space backpressure — real observed failure on Kaggle: download
# throughput (~92 clips/sec across concurrent sources) vastly outpaced the single Mimi
# encoder's encode+delete rate (~2 clips/sec), so the periodic 90s encode sweep alone
# could never drain the backlog fast enough, and the run died with
# `[Errno 28] No space left on device` — which then cascaded into every OTHER error in
# that log (download "System error"s, encode-stage ENOSPC, ZipFile.__del__ tracebacks):
# all downstream symptoms of this one root cause, not separate bugs. This makes every
# write CHECK real free space first and BLOCK until the encode/delete side frees enough
# room, instead of writing blind and letting the OS crash the process once it's too late.
DISK_CHECK_EVERY_N_WRITES = 20       # stat() the filesystem every N clips, not every clip
DISK_WAIT_POLL_S = 5.0               # how often to recheck while blocked and waiting
DISK_WAIT_LOG_EVERY_S = 30.0         # don't spam the log every 5s while blocked


def _free_disk_gb(path: Path) -> float:
    import shutil as _shutil
    return _shutil.disk_usage(str(path)).free / (1024 ** 3)


def wait_for_disk_headroom(path: Path, min_free_gb: float, log_fn: Callable[[str], None] = print) -> None:
    """Block (polling) until `path`'s filesystem has at least `min_free_gb` free.
    Relies on something else in the process (scripts/P1_00_pipeline.py's encode sweep,
    which deletes each wav right after it's encoded) actually freeing space while this
    blocks — a download thread parked here isn't burning CPU, it's just polling stat()
    every few seconds, so it doesn't fight the encoder for the machine's real resources
    (unlike the earlier GIL-starvation issue, which was about CPU, not disk)."""
    free_gb = _free_disk_gb(path)
    if free_gb >= min_free_gb:
        return
    log_fn(f"low disk space ({free_gb:.2f}GB free < {min_free_gb:.2f}GB threshold) — "
           f"pausing downloads until the encoder frees more space")
    last_log = time.monotonic()
    while free_gb < min_free_gb:
        time.sleep(DISK_WAIT_POLL_S)
        free_gb = _free_disk_gb(path)
        # Re-check the exit condition BEFORE logging "still waiting" — without this, the
        # exact poll that recovers enough space could still print a "still waiting ...
        # 6.00GB free < 3.00GB threshold" message (true a moment ago, false and
        # self-contradictory by the time it's printed). Confirmed via an offline test
        # with a mocked disk_usage sequence that recovers on the periodic-log tick.
        if free_gb >= min_free_gb:
            break
        now = time.monotonic()
        if now - last_log >= DISK_WAIT_LOG_EVERY_S:
            log_fn(f"still waiting for disk space ({free_gb:.2f}GB free < "
                   f"{min_free_gb:.2f}GB threshold)")
            last_log = now
    log_fn(f"disk space recovered ({free_gb:.2f}GB free) — resuming downloads")


# --------------------------------------------------------------------------- #
# Real observed hang (not theoretical): `load_dataset(...)` for `google/fleurs` froze
# indefinitely — every OTHER source in the same run opened fine, this one printed its
# "starting" line and then produced ZERO further output, and Ctrl+C didn't respond. The
# `datasets` library, for datasets that ship a loading SCRIPT (fleurs is one — several
# other legacy/community HF datasets are too) rather than a plain parquet layout, can
# block on a hidden `input()` prompt asking to confirm running that script's custom code
# when `trust_remote_code` isn't passed explicitly — a blocking stdin read is exactly
# what produces a silent, Ctrl+C-resistant hang with no exception and no timeout,
# matching what was observed. Two independent fixes, since the exact cause can't be
# confirmed without reproducing it live: (1) pass `trust_remote_code=True` explicitly so
# that prompt never fires for fleurs (an official Google dataset — safe to trust); (2) a
# hard watchdog timeout around the call regardless of cause, so ANY hang (this one, a
# genuine network stall, HF being slow to index a dataset) times out with a clear error
# and gets picked up by the existing transient-retry loop, instead of blocking forever
# with zero feedback. A timed-out background thread can't be force-killed in Python — it
# leaks as a harmless zombie thread rather than actually stopping, which is an accepted
# trade-off for turning "hangs forever, silently" into "fails loudly and retries".
LOAD_DATASET_TIMEOUT_S = 120.0


# Real observed hang, still unresolved by the trust_remote_code fix above: `google/
# fleurs` itself — not a network blip, not a version mismatch — its OWN loading path
# (resolving/enumerating its ~100 per-language configs even when only one is requested,
# apparently) stalls on some networks even though the underlying data is plain Parquet.
# Confirmed via HF's file-listing API (not guessed): the real layout is exactly
# `parquet-data/<config>/{train,test,validation}-00000-of-00001.parquet`, and this
# project's configured `hf_config` values (en_us, hi_in, gu_in) already match those
# folder names exactly. Loading those parquet files DIRECTLY via the generic "parquet"
# builder — same bytes, same schema — bypasses fleurs' own loading machinery (and
# whatever in it is slow/hanging) entirely.
_FLEURS_PARQUET_GLOB = "hf://datasets/google/fleurs/parquet-data/{config}/{split}-*.parquet"


def _dataset_load_spec(spec: "SourceSpec") -> tuple[tuple, dict]:
    """(positional args, kwargs) for `_load_dataset_with_timeout`/`load_dataset`,
    special-cased for google/fleurs to load its Parquet files directly (see above)."""
    if spec.hf_dataset == "google/fleurs" and spec.hf_config:
        glob = _FLEURS_PARQUET_GLOB.format(config=spec.hf_config, split=spec.split)
        # A single (non-dict) `data_files` glob is assigned the default split name
        # "train" by `datasets` regardless of what the file is named on disk — the glob
        # above already selects the CORRECT file (test/train/validation) by filename
        # prefix, so `split="train"` here is that fixed default name, not `spec.split`.
        return ("parquet",), {"data_files": glob, "split": "train"}
    return (spec.hf_dataset, spec.hf_config), {"split": spec.split}


def _load_dataset_with_timeout(*args, timeout: float = LOAD_DATASET_TIMEOUT_S, **kwargs):
    # A `concurrent.futures.ThreadPoolExecutor` was tried here first and is WRONG: its
    # worker threads are non-daemon, so if the call genuinely never returns (a true
    # network stall, not just this one bug), that thread leaks forever and the whole
    # Python PROCESS then hangs at exit waiting to join it — CPython won't shut down
    # until every non-daemon thread finishes. Confirmed by hanging an actual test run on
    # this exact code before switching to `threading.Thread(daemon=True)` below: a daemon
    # thread is killed automatically when the process exits, so a still-hung load stays
    # contained to "this one attempt returned a timeout error" and never blocks exit.
    import threading

    from datasets import load_dataset

    def _attempt(call_kwargs: dict):
        result: dict = {}

        def _run():
            try:
                result["value"] = load_dataset(*args, **call_kwargs)
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            raise RuntimeError(
                f"load_dataset() timed out after {timeout:.0f}s opening "
                f"{args[0] if args else call_kwargs.get('path')} — likely a network "
                f"stall or a blocked confirmation prompt; will retry with a fresh "
                f"attempt (timeout)"
            )
        if "error" in result:
            raise result["error"]
        return result["value"]

    # Try WITHOUT trust_remote_code first — this is the common/modern case: every
    # default source in this project's config (LibriSpeech, Kathbath, Shrutilipi,
    # FLEURS, IndicTTS) is Parquet-backed on HF now ("Auto-converted to Parquet" on
    # their dataset pages — confirmed for google/fleurs specifically), no script
    # involved, so no trust prompt of any kind ever fires. An EARLIER version of this
    # fix instead injected `trust_remote_code=True` on the FIRST attempt (to dodge an
    # older-datasets hanging prompt) — that was backwards: on a NEWER `datasets`
    # version (4.0+) the kwarg was removed entirely and merely PASSING it — even
    # `=True` — raises immediately: "trust_remote_code is not supported anymore.
    # Please check that ... isn't based on a loading script and remove
    # `trust_remote_code`." That's a `RuntimeError` on that version, NOT `TypeError`/
    # `ValueError` (confirmed via the actual upstream report, not guessed) — the
    # original except clause here didn't catch it, so the raw error passed straight
    # through unchanged on a real run. Fixed by (a) not passing the kwarg unless the
    # first attempt's error says it's actually needed, and (b) catching broadly by
    # message content instead of a specific exception type, since the exact type isn't
    # stable across `datasets` versions.
    try:
        return _attempt(dict(kwargs))
    except Exception as e:
        msg = str(e)
        if "trust_remote_code" in msg and "not supported" not in msg.lower():
            # Older `datasets` + a genuinely script-based dataset asking permission to
            # run it — retry once WITH the kwarg (safe: only fires for datasets this
            # project's config explicitly lists, e.g. an official Google/AI4Bharat repo).
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["trust_remote_code"] = True
            return _attempt(fallback_kwargs)
        raise


# --------------------------------------------------------------------------- #
def fetch_source(
    cfg: Phase1CorpusConfig,
    lang: str,
    spec: SourceSpec,
    out_dir: Path,
    root: Path,
    manifest_fh,
    already: dict,
    hf_token: str | None,
    dry_run: bool,
    log_fn: Callable[[str], None] = print,
    progress_fn: Callable[[float], None] | None = None,
) -> dict:
    """
    Stream one source, writing wavs + manifest rows until this source's share of
    target_hours is reached (or the stream runs out). Returns a small result summary.

    `progress_fn`, if given, is called with the INCREMENTAL hours (not running total)
    added by each clip as it's written — lets a caller (scripts/P1_00_pipeline.py) show
    live progress instead of only learning the total once this whole source finishes,
    which for a big source (e.g. 90h) could otherwise look like nothing is happening
    for a long time even while clips are actively streaming in.

    `manifest_fh` must be safe to call `.write()`/`.flush()` on from whatever thread
    calls this (the CLI passes a plain file handle from a single-threaded run;
    scripts/P1_00_pipeline.py passes a lock-guarded wrapper since multiple sources can
    fetch concurrently). `log_fn` receives one-line progress/retry messages — the CLI
    defaults to plain `print`; the pipeline passes a tagged, thread-safe logger.
    """
    target_h = cfg.target_hours.get(lang, 0.0) * spec.weight
    have_h = already.get((lang, spec.id), {}).get("hours", 0.0)
    have_n = already.get((lang, spec.id), {}).get("count", 0)

    if have_h >= target_h:
        return {"lang": lang, "source": spec.id, "status": "already_done",
               "have_hours": have_h, "target_hours": target_h}

    if dry_run:
        return {"lang": lang, "source": spec.id, "status": "dry_run",
               "have_hours": have_h, "target_hours": target_h,
               "remaining_hours": max(0.0, target_h - have_h),
               "hf_dataset": spec.hf_dataset, "hf_config": spec.hf_config,
               "gated": spec.gated}

    try:
        import datasets  # noqa: F401  — import must succeed; we never use Audio()
    except ImportError:
        raise SystemExit("`datasets` not installed. `pip install datasets soundfile`.")

    try:
        import soundfile as sf
    except ImportError:
        raise SystemExit("`soundfile` not installed. `pip install soundfile`.")

    _patch_datasets_audio_no_torchcodec()

    # FLAT per-language dir (not lang/source/) — scripts/00_encode_audio.py globs
    # "*.wav" non-recursively, so this must match exactly what it expects. The source
    # id is folded into the filename instead, so provenance is still visible.
    lang_dir = out_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    audio_col = spec.audio_col
    text_col = spec.text_col
    gender_col = spec.gender_col
    audio_col_fail_rows = 0    # rows where column auto-detect failed — see COLUMN_DETECT_ROW_BUDGET
    text_col_fail_rows = 0
    last_err: Exception | None = None

    # Retries the WHOLE stream from a fresh `load_dataset` call if it dies mid-iteration
    # on a transient error (real observed case: '[Errno 9] Bad file descriptor' from a
    # Kaggle network blip while GETting a parquet shard) — restarting is safe because the
    # skipped_for_resume counter below re-skips already-written rows from the top, same
    # as a normal script restart would.
    load_args, load_kwargs = _dataset_load_spec(spec)
    for attempt in range(STREAM_RETRY_ATTEMPTS):
        try:
            ds = _load_dataset_with_timeout(
                *load_args, streaming=True, token=hf_token,
                **load_kwargs,
            )
        except Exception as e:
            msg = str(e)
            if spec.gated and ("gated" in msg.lower() or "401" in msg or "403" in msg):
                raise SystemExit(
                    f"'{spec.hf_dataset}' is gated. Visit "
                    f"https://huggingface.co/datasets/{spec.hf_dataset}, log in, and click "
                    f"'Agree and access repository' — then re-run with HF_TOKEN set."
                )
            if looks_transient(e) and attempt < STREAM_RETRY_ATTEMPTS - 1:
                wait = 2.0 * (attempt + 1)
                log_fn(f"transient error opening stream (attempt {attempt + 1}/"
                      f"{STREAM_RETRY_ATTEMPTS}): {e} — retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            raise SystemExit(f"failed to open {spec.hf_dataset}/{spec.hf_config}: {e}")

        # Flip Audio.decode=False on the existing features. Do NOT cast_column to
        # Audio(...) — that runs encode_example and imports torchcodec (datasets 4.x).
        ds = _disable_media_decode(ds)
        # `written` (not just `have_n`) must be included here: a fresh stream restarts
        # from row 0, so it must also re-skip whatever THIS call already wrote in an
        # earlier attempt before the failure — otherwise a mid-stream restart duplicates
        # those rows under new clip_ids. Relies on the streaming order being stable
        # across repeated `load_dataset` calls for the same split (true for HF's default
        # unshuffled parquet-backed streaming).
        skipped_for_resume = have_n + written
        row_index = -1

        try:
            for row in ds:
                row_index += 1

                if audio_col is None:
                    detected = spec.audio_col or detect_audio_column(row)
                    if detected is None:
                        audio_col_fail_rows += 1
                        if audio_col_fail_rows >= COLUMN_DETECT_ROW_BUDGET:
                            raise SystemExit(
                                f"couldn't find an audio column in {spec.hf_dataset}/"
                                f"{spec.hf_config} after inspecting "
                                f"{COLUMN_DETECT_ROW_BUDGET} rows — set `audio_col:` "
                                f"explicitly in configs/phase1_corpus.yaml"
                            )
                        continue   # might be one transiently-corrupted row — try the next
                    audio_col = detected

                if text_col is None:
                    detected = spec.text_col or pick_column(_TEXT_COL_CANDIDATES, list(row.keys()))
                    if detected is None:
                        text_col_fail_rows += 1
                        if text_col_fail_rows >= COLUMN_DETECT_ROW_BUDGET:
                            raise SystemExit(
                                f"couldn't find a transcript column in {spec.hf_dataset}/"
                                f"{spec.hf_config} after inspecting "
                                f"{COLUMN_DETECT_ROW_BUDGET} rows (tried "
                                f"{_TEXT_COL_CANDIDATES}) — set `text_col:` explicitly in "
                                f"configs/phase1_corpus.yaml"
                            )
                        continue
                    text_col = detected

                if gender_col is None and cfg.gender_balance:
                    gender_col = spec.gender_col or pick_column(_GENDER_COL_CANDIDATES, list(row.keys()))
                    # may legitimately stay None (many sources don't carry gender) — fine

                if skipped_for_resume > 0:
                    skipped_for_resume -= 1
                    continue  # already saved from a previous run/attempt; keep streaming

                try:
                    audio = row.get(audio_col)
                    if audio is None:
                        continue
                    arr, sr = extract_waveform(audio)
                    if arr is None:
                        continue
                    duration_s = len(arr) / float(sr)
                    if duration_s < cfg.min_clip_seconds or duration_s > cfg.max_clip_seconds:
                        continue

                    transcript = str(row.get(text_col, "") or "").strip()
                    if not transcript:
                        continue

                    gender = None
                    if gender_col:
                        raw_g = str(row.get(gender_col, "") or "").lower()
                        if raw_g.startswith("f"):
                            gender = "female"
                        elif raw_g.startswith("m"):
                            gender = "male"

                    # Proactive backpressure: check real free disk space BEFORE writing,
                    # not just periodically after the fact. Checked every N writes (not
                    # every single one) since stat() on every clip would be wasteful, but
                    # frequently enough that a fast source can't blow through the whole
                    # threshold between checks (~1.5-20s of audio per clip at ~48KB/s raw
                    # is small — 20 clips is still well inside the safety margin below).
                    if written % DISK_CHECK_EVERY_N_WRITES == 0:
                        wait_for_disk_headroom(out_dir, cfg.min_free_disk_gb, log_fn)

                    cid = clip_id(lang, spec.id, row_index)
                    wav_path = lang_dir / f"{spec.id}_{cid}.wav"
                    # Write to a temp name in the SAME directory, then atomically rename
                    # into place (os.replace is atomic on the same filesystem). Matters
                    # once anything else can be reading this directory concurrently
                    # while downloads are still writing to it — scripts/P1_00_pipeline.py's
                    # periodic encode sweep now does exactly that, so a plain sf.write()
                    # straight to wav_path could let the encoder glob/open a file that's
                    # only partially flushed. A reader only ever sees "doesn't exist yet"
                    # or "fully written", never a partial file, this way.
                    # NOTE: the temp name must still end in ".wav" — soundfile.write()
                    # infers the output format from the file extension by default, so a
                    # suffix like ".tmp12345" (no trailing ".wav") makes it raise "No
                    # format specified". Passing format="WAV" explicitly removes that
                    # extension-guessing dependency entirely (belt-and-suspenders: this
                    # bug is exactly why the explicit format arg is worth the one word).
                    tmp_path = wav_path.with_name(f"{wav_path.stem}.tmp{os.getpid()}.wav")
                    sf.write(str(tmp_path), arr, sr, format="WAV")
                    os.replace(str(tmp_path), str(wav_path))
                except Exception as e:
                    # one corrupted row (same class of transient glitch as a dead stream,
                    # just caught at the single-row level) — skip it, keep going, don't
                    # let it take down the whole multi-hour fetch
                    if looks_transient(e):
                        continue
                    raise

                manifest_fh.write(json.dumps({
                    "id": cid, "lang": lang, "source": spec.id,
                    "wav_path": relative_or_absolute(wav_path, root),
                    "transcript": transcript, "gender": gender,
                    "duration_s": round(duration_s, 3),
                }, ensure_ascii=False) + "\n")
                if hasattr(manifest_fh, "flush"):
                    manifest_fh.flush()

                written += 1
                have_h += duration_s / 3600.0
                if progress_fn:
                    progress_fn(duration_s / 3600.0)
                if have_h >= target_h:
                    break

            break   # completed (or hit target) without a fatal stream error — done retrying

        except SystemExit:
            raise   # a real "column doesn't exist" / gated / etc. — not retryable, surface it
        except Exception as e:
            last_err = e
            # Real observed secondary crash (not theoretical): raising straight out of
            # this `for row in ds:` loop can leave the streaming iterator's own internal
            # resources (torchcodec/aiohttp worker threads, in-flight async I/O) still
            # alive when the process exits right after — CPython's interpreter shutdown
            # then races those resources' own C-extension threads, producing a `Fatal
            # Python error: PyGILState_Release: thread state ... must be current` crash
            # and `Aborted (core dumped)`, seen on a real run right after a torchcodec
            # ImportError. Explicitly dropping the reference and forcing a collection
            # HERE — while the interpreter is still fully alive, not mid-finalization —
            # gives those resources a normal chance to clean up instead of leaking into
            # shutdown. Not a 100% guarantee against every possible native-extension
            # shutdown race, but removes the specific one observed.
            del ds
            gc.collect()
            if "torchcodec" in str(e).lower():
                # Not transient, and installing torchcodec is the WRONG fix on this
                # stack (torch 2.4.x + no matching FFmpeg). English already streams
                # via soundfile; this means decode was still left on for this source.
                raise SystemExit(
                    f"{spec.hf_dataset}/{spec.hf_config}: datasets still tried torchcodec "
                    f"(do NOT pip install it — English already runs without it). "
                    f"{written} clip(s) written this attempt are kept. Original error: {e}"
                ) from last_err
            if looks_transient(e) and attempt < STREAM_RETRY_ATTEMPTS - 1:
                wait = 2.0 * (attempt + 1)
                log_fn(f"transient error mid-stream (attempt {attempt + 1}/"
                      f"{STREAM_RETRY_ATTEMPTS}): {e} — {written} clip(s) written so far, "
                      f"restarting stream in {wait:.0f}s")
                time.sleep(wait)
                continue
            raise SystemExit(
                f"fetch failed for {spec.hf_dataset}/{spec.hf_config} after "
                f"{attempt + 1} attempt(s), {written} clip(s) written before the failure "
                f"(kept — safe to just re-run, it resumes): {e}"
            ) from last_err

    return {"lang": lang, "source": spec.id, "status": "ok",
           "written": written, "have_hours": round(have_h, 3), "target_hours": round(target_h, 3)}


# --------------------------------------------------------------------------- #
# Frame-record building (shared with scripts/P1_02_build_frames.py) — see that
# script's module docstring for what fields Phase1Loss actually reads.
def build_frame_record(rec: dict, encoded_path: Path, root: Path) -> dict | None:
    """
    `encoded_path` in the returned record is stored ROOT-RELATIVE (e.g.
    "data/encoded/xyz.npz"), NOT absolute — this is what makes a frames_<lang>.jsonl
    file portable across machines (generated on your Mac, downloaded and used as-is on
    Kaggle) as long as the training script is run from the project root, same convention
    every other path in this project already follows. An earlier version of this
    function stored the absolute path, which silently only worked on the machine that
    generated it — fixed here, not just papered over downstream.
    """
    if not encoded_path.exists():
        return None
    import numpy as np

    from thinkspark import vocab

    d = np.load(encoded_path)
    T = len(d["cb0"])
    if T <= 0:
        return None

    default_flag = vocab.CONTROL_FLAG_TO_ID[vocab.DEFAULT_FLAG]  # LISTEN
    idle_state = vocab.AGENT_STATE_TO_ID["IDLE"]

    return {
        "scenario_id": rec["id"],
        "behaviour": "phase1_free_audio",
        "language": rec["lang"],
        "domain": rec["source"],
        "agent_text": "",
        "user_text": rec["transcript"],
        "num_frames": T,
        "audio_frames": T,
        "encoded_path": relative_or_absolute(encoded_path, root),
        "flags": [default_flag] * T,
        "agent_state": [idle_state] * T,
        "speaking_mask": [1] * T,
        "spoken_spans": [],
    }
