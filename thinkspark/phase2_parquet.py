"""
Phase-2 Parquet repacking — same rationale as thinkspark.phase1_parquet (see its module
docstring for the full story): a Phase-2 HF repo uploaded via scripts/13_upload_hf.py has
one `.wav` + one `.json` per clip, bucketed into 200 folders (`audio/<000-199>/<id>.wav`,
`timestamps/<000-199>/<id>.json`) — real, observed cost of that layout: fetching it is
thousands of individual small-file HTTP requests, dominated by per-request round-trip
latency rather than bandwidth (minutes per folder on a real run). Packing many clips'
audio bytes + timestamps + full scenario record into a handful of self-contained Parquet
shards (`data/phase2-shard-NNNNN.parquet`) fixes both the fetch-speed problem and HF/
git's ~10,000-files-per-directory limit.

Used by scripts/23_repack_phase2_to_parquet.py (one-time repo conversion, done in
folder-count-bounded batches so it fits a small-RAM box) and
scripts/19_fetch_training_data.py's fetch_phase2 (reads the new layout back into the
exact local data/audio/<id>.wav + <id>.words.json + data/scenarios/scenarios_all.jsonl
layout scripts/00_encode_audio.py / 04_build_frames.py already expect — those scripts
need zero changes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

_SCHEMA_FIELDS = ("scenario_id", "wav_bytes", "timestamps_json", "scenario_json")


_PARQUET_SCHEMA = None


def _schema():
    import pyarrow as pa
    global _PARQUET_SCHEMA
    if _PARQUET_SCHEMA is None:
        _PARQUET_SCHEMA = pa.schema([
            pa.field("scenario_id", pa.string()),
            pa.field("wav_bytes", pa.binary()),
            pa.field("timestamps_json", pa.string()),
            pa.field("scenario_json", pa.string()),
        ])
    return _PARQUET_SCHEMA


def pack_batch_to_parquet(rows: list[dict], out_path: Path,
                          log_fn: Callable[[str], None] = print) -> None:
    """rows: [{"scenario_id": str, "wav_bytes": bytes, "timestamps_json": str (may be
    ""), "scenario_json": str}] — `timestamps_json`/`scenario_json` are the RAW original
    file contents, carried through verbatim (not re-parsed/re-serialized), so this is a
    lossless repack, not a reinterpretation of the schema.

    Kept for callers that already hold a small, fully-materialized row list (e.g. tests).
    For a large batch on a small-RAM box, use pack_rows_streaming instead — this function
    builds one full pa.Table in memory (a second copy on top of `rows` itself)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_schema())
    pq.write_table(table, out_path, compression="zstd")
    log_fn(f"  wrote {out_path.name}: {len(rows)} rows, "
          f"{out_path.stat().st_size / 1e6:.1f}MB")


def pack_rows_streaming(row_iter, out_path: Path, chunk_size: int = 100,
                        log_fn: Callable[[str], None] = print) -> int:
    """Same output as pack_batch_to_parquet, but never holds more than `chunk_size` rows'
    wav bytes in memory at once — `row_iter` is any iterable/generator yielding row dicts
    one at a time (the caller reads each wav file's bytes lazily, right before it's
    appended to the current chunk). Writes each chunk as its own row-group via
    ParquetWriter, so peak memory is O(chunk_size), not O(batch size) — the fix for the
    real OOM-kill observed packing ~1800 rows into one in-memory pa.Table on a 4GB box.
    Returns the total row count written."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = _schema()
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    total = 0
    chunk: list[dict] = []
    try:
        for row in row_iter:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                writer.write_table(pa.Table.from_pylist(chunk, schema=schema))
                total += len(chunk)
                chunk = []
        if chunk:
            writer.write_table(pa.Table.from_pylist(chunk, schema=schema))
            total += len(chunk)
    finally:
        writer.close()
    log_fn(f"  wrote {out_path.name}: {total} rows, "
          f"{out_path.stat().st_size / 1e6:.1f}MB")
    return total


def unpack_shard_to_local(shard_path: Path, audio_dir: Path, scenarios_fh,
                          seen_ids: set) -> int:
    """Reconstructs `audio_dir/<id>.wav` + `<id>.words.json`, and appends any scenario
    record not already in `seen_ids` to `scenarios_fh` (an open scenarios_all.jsonl
    handle) — `seen_ids` is the caller's running set across all shards/fetches, so
    overlapping shards or a re-run never duplicate a scenario line. Returns the number of
    NEW wav files written (0 for ones already on disk — resumable, safe to re-run)."""
    import pyarrow.parquet as pq

    audio_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(shard_path)
    written = 0
    for row in table.to_pylist():
        sid = row["scenario_id"]
        wav_path = audio_dir / f"{sid}.wav"
        if not wav_path.exists():
            wav_path.write_bytes(row["wav_bytes"])
            written += 1
        meta_path = audio_dir / f"{sid}.words.json"
        if not meta_path.exists() and row.get("timestamps_json"):
            meta_path.write_text(row["timestamps_json"], encoding="utf-8")
        if sid not in seen_ids:
            seen_ids.add(sid)
            scenarios_fh.write(row["scenario_json"].rstrip("\n") + "\n")
    return written
