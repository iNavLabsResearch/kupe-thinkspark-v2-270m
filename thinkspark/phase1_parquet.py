"""
Phase-1 Parquet packing — the HF-repo storage/transport format for encoded Phase-1 data.

Real problem this replaces: uploading one loose `.npz` file per clip. HF/git hard-limits
any single directory to ~10,000 files (real observed failure: `encoded/en/` hit that cap
mid-upload and every further commit was rejected with "too many files per directory"),
and even bucketing around that limit (see `thinkspark.hf_upload.npz_repo_path`) still
leaves a repo with tens of thousands of tiny files scattered across hundreds of folders —
messy to browse, slow for HF's own Dataset Viewer to index. Packing many clips' full
frame record — INCLUDING the cb0/energy/f0 arrays, embedded directly as Parquet list
columns, no separate `.npz` needed at all in the repo — into a handful of self-contained
shard files sidesteps all of that permanently, and is the standard, idiomatic way HF
datasets are actually organized (enables the Dataset Viewer natively too).

Parquet is ONLY the storage/transport format for the HF repo. Locally, and for training,
nothing changes: scripts/19_fetch_training_data.py unpacks each shard straight back into
the exact same data/encoded/<clip_id>.npz + data/frames_phase1/frames_<lang>.jsonl
layout scripts/06_train_phase1.py already reads — the trainer needs zero changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

# Scalar (non-array) fields copied straight from a frames_<lang>.jsonl row.
_SCALAR_FIELDS = ("scenario_id", "behaviour", "language", "domain", "agent_text",
                  "user_text", "num_frames", "audio_frames")
# Per-frame integer list fields, also copied straight across (already lists in the
# source jsonl row — vocab.CONTROL_FLAG_TO_ID / AGENT_STATE_TO_ID ids, one per frame).
_LIST_FIELDS = ("flags", "agent_state", "speaking_mask")


def pack_lang_to_parquet(lang: str, frames_path: Path, root: Path, out_dir: Path,
                         rows_per_shard: int = 2000,
                         log_fn: Callable[[str], None] = print) -> list[Path]:
    """Reads every row of `frames_path` (frames_<lang>.jsonl), loads each row's
    referenced `.npz` (cb0/energy/f0), and writes self-contained Parquet shards to
    `out_dir` (named `<lang>-shard-NNNNN.parquet`). Returns the shard paths written.

    Rebuilds from scratch every call — cheap (local disk only, no network) and the
    simplest way to guarantee the shards always exactly match current local state. This
    project always re-uploads a language's FULL current data on each run rather than
    tracking incremental deltas against these shards — see P1_00_sequential.py's
    stage_upload — so any stale shards from a previous run are deleted first.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{lang}-shard-*.parquet"):
        old.unlink()

    schema = pa.schema([
        pa.field("scenario_id", pa.string()),
        pa.field("behaviour", pa.string()),
        pa.field("language", pa.string()),
        pa.field("domain", pa.string()),
        pa.field("agent_text", pa.string()),
        pa.field("user_text", pa.string()),
        pa.field("num_frames", pa.int32()),
        pa.field("audio_frames", pa.int32()),
        pa.field("cb0", pa.list_(pa.int64())),
        pa.field("energy", pa.list_(pa.float32())),
        pa.field("f0", pa.list_(pa.float32())),
        pa.field("flags", pa.list_(pa.int32())),
        pa.field("agent_state", pa.list_(pa.int32())),
        pa.field("speaking_mask", pa.list_(pa.int32())),
        # spoken_spans is always [] for phase1_free_audio today; stored as JSON strings
        # per span for forward-compat with future behaviours that DO have real spans,
        # without needing a struct type for what's currently always empty.
        pa.field("spoken_spans", pa.list_(pa.string())),
    ])

    rows: list[dict] = []
    shard_paths: list[Path] = []
    shard_idx = 0
    missing = 0
    total_rows = 0

    def flush():
        nonlocal rows, shard_idx
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=schema)
        shard_path = out_dir / f"{lang}-shard-{shard_idx:05d}.parquet"
        pq.write_table(table, shard_path, compression="zstd")
        shard_paths.append(shard_path)
        log_fn(f"  wrote {shard_path.name}: {len(rows)} rows, "
              f"{shard_path.stat().st_size / 1e6:.1f}MB")
        rows = []
        shard_idx += 1

    with frames_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            npz_path = root / rec["encoded_path"]
            if not npz_path.exists():
                missing += 1
                continue
            d = np.load(npz_path)
            row = {k: rec[k] for k in _SCALAR_FIELDS}
            row["cb0"] = d["cb0"].astype(np.int64).tolist()
            row["energy"] = d["energy"].astype(np.float32).tolist()
            row["f0"] = d["f0"].astype(np.float32).tolist()
            for k in _LIST_FIELDS:
                row[k] = rec[k]
            row["spoken_spans"] = [json.dumps(s) for s in rec.get("spoken_spans", [])]
            rows.append(row)
            total_rows += 1
            if len(rows) >= rows_per_shard:
                flush()
    flush()

    if missing:
        log_fn(f"  ! {missing} frame record(s) skipped — referenced .npz missing locally")
    log_fn(f"  packed {total_rows} rows into {len(shard_paths)} shard(s) for {lang}")
    return shard_paths


def unpack_shard_to_local(shard_path: Path, encoded_dir: Path, frames_fh, root: Path) -> int:
    """Reads one Parquet shard and reconstructs the LOCAL layout scripts/06_train_phase1.py
    already expects: one `.npz` per clip in `encoded_dir`, plus a frame record appended
    to `frames_fh` (an open frames_<lang>.jsonl file handle) with `encoded_path` pointing
    at it. Returns the number of NEW `.npz` files written (0 for a clip whose `.npz`
    already exists locally — resumable, safe to re-run across shards/sessions)."""
    import numpy as np
    import pyarrow.parquet as pq

    encoded_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(shard_path)
    written = 0
    for row in table.to_pylist():
        cid = row["scenario_id"]
        npz_path = encoded_dir / f"{cid}.npz"
        if not npz_path.exists():
            np.savez_compressed(
                npz_path,
                cb0=np.asarray(row["cb0"], dtype=np.int64),
                energy=np.asarray(row["energy"], dtype=np.float32),
                f0=np.asarray(row["f0"], dtype=np.float32),
            )
            written += 1
        try:
            encoded_rel = str(npz_path.relative_to(root))
        except ValueError:
            encoded_rel = str(npz_path)
        frame = {
            "scenario_id": cid,
            "behaviour": row["behaviour"],
            "language": row["language"],
            "domain": row["domain"],
            "agent_text": row["agent_text"],
            "user_text": row["user_text"],
            "num_frames": row["num_frames"],
            "audio_frames": row["audio_frames"],
            "encoded_path": encoded_rel,
            "flags": row["flags"],
            "agent_state": row["agent_state"],
            "speaking_mask": row["speaking_mask"],
            "spoken_spans": [json.loads(s) for s in row.get("spoken_spans", [])],
        }
        frames_fh.write(json.dumps(frame, ensure_ascii=False) + "\n")
    return written
