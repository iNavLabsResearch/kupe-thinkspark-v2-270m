#!/usr/bin/env python
"""
Upload the whole ThinkSpark-v2-350M training corpus to Hugging Face — scenarios, audio,
and Soniox timestamps, all paired for the Dataset Viewer (audio player + text side by
side), same pattern as kupe-tts's hf_bulk_upload.py + hf_metadata.py (bulk sharded
commits with progress/rate/ETA, then a parquet with a real `Audio` feature column).

Only rows with BOTH real audio (a non-corrupt wav) AND text are uploaded — run
`scripts/14_cleanup_corrupt_audio.py` first if you're not sure your local data is clean.

What gets uploaded:
    audio/<shard>/<scenario_id>.wav        the rendered user audio
    timestamps/<shard>/<scenario_id>.json  Soniox character-level timestamps (TRAINING
                                            use only — see thinkspark.tts_soniox; never
                                            needed at inference)
    data/train-*.parquet                   Dataset Viewer source: audio (Audio feature)
                                            + user_text + a few flat metadata columns —
                                            browsable, NOT enough on its own to rebuild
                                            training frames (no `target`/`event_char`)
    scenarios/scenarios_all.jsonl          the RAW scenario file, full schema (target
                                            control-flag timeline, event_char, everything)
                                            — this is what makes the repo actually
                                            self-sufficient for training reconstruction
                                            elsewhere; see scripts/19_fetch_training_data.py
    metadata.jsonl / metadata.csv          AudioFolder-style side index
    README.md                              dataset card (Viewer config, column docs)

Resumable: every uploaded scenario_id is marked in SQLite (`hf_sync` table) — re-running
only uploads what's still pending. Nothing local is ever deleted.

    conda activate llms
    pip install huggingface_hub pyarrow
    export HF_TOKEN=hf_...   # needs WRITE access — https://huggingface.co/settings/tokens

    python scripts/13_upload_hf.py --dry-run          # plan only, no auth/upload
    python scripts/13_upload_hf.py                    # upload (resumable)
    python scripts/13_upload_hf.py --repo <your-hf-username>/Thinkspark-v2-270m-training-data
    python scripts/13_upload_hf.py --max-scenarios 500  # cap this run, resume later

--repo must be a namespace your HF_TOKEN can actually create repos under — your own
username (e.g. "anuj-inavlabs/...") always works; an org namespace (e.g. "kupe/...")
needs you to actually be a member of that org with repo-create rights, or create_repo
fails with 403 "You don't have the rights to create a dataset under the namespace ..."
even with a fully-scoped write token — that error is a namespace/membership issue, not
a token-validity one.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import DataGenConfig, env
from thinkspark.db import RunDB
from thinkspark.hf_upload import (
    create_commit_with_backoff, ensure_repo, log, pack_by_file_count, sleep_with_jitter, utc_now,
)

# No safe default namespace to guess — always pass --repo explicitly (your own HF
# username, e.g. "anuj-inavlabs/Thinkspark-v2-270m-training-data", unless you're
# actually a member of whatever org you're targeting with repo-create rights there).
DEFAULT_REPO = None
ROWS_PER_PARQUET_SHARD = 10_000
HF_DIR_SHARD_SIZE = 1000    # HF rejects >10k files per directory; shard well under that


def _shard_dir(scenario_id: str) -> str:
    """Stable shard folder from a hash of the id (ids aren't sequential integers here,
    unlike kupe-tts's index-based sharding, so hash instead of a numeric // )."""
    import hashlib
    h = int(hashlib.md5(scenario_id.encode("utf-8")).hexdigest(), 16)
    return f"{h % 200:03d}"   # 200 shard dirs is plenty for tens of thousands of files


def wav_repo_path(scenario_id: str) -> str:
    return f"audio/{_shard_dir(scenario_id)}/{scenario_id}.wav"


def ts_repo_path(scenario_id: str) -> str:
    return f"timestamps/{_shard_dir(scenario_id)}/{scenario_id}.json"


# --------------------------------------------------------------------------- #
def load_scenarios(scenarios_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in scenarios_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = rec.get("scenario_id")
        if sid:
            out[sid] = rec
    return out


def pending_rows(
    scenarios: dict[str, dict], audio_dir: Path, already_synced: set[str],
    max_scenarios: int | None,
) -> list[dict[str, Any]]:
    """Scenarios with BOTH real audio + text, not yet uploaded. Corrupt/empty audio
    (duration_s<=0, same check as the render script's own defensive guard) is skipped —
    run scripts/14_cleanup_corrupt_audio.py first to also fix it locally."""
    out: list[dict[str, Any]] = []
    for sid, rec in scenarios.items():
        if sid in already_synced:
            continue
        wav_path = audio_dir / f"{sid}.wav"
        meta_path = audio_dir / f"{sid}.words.json"
        if not (wav_path.exists() and meta_path.exists()):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        duration_s = meta.get("duration_s", 0.0)
        if not duration_s or duration_s <= 0.0:
            continue
        text = (rec.get("user_text") or "").strip()
        if not text:
            continue
        out.append({
            "scenario_id": sid, "wav_path": wav_path, "meta_path": meta_path,
            "duration_s": duration_s, "text": text, "scenario": rec, "words_meta": meta,
        })
        if max_scenarios is not None and len(out) >= max_scenarios:
            break
    return out


def pack_commits(pending: list[dict], *, files_per_commit: int) -> list[list[dict]]:
    """Greedy-pack rows (2 files each: wav + timestamps json) into commits of up to
    `files_per_commit` files. Thin wrapper over the generic packer in thinkspark.hf_upload,
    shared with scripts/P1_00_pipeline.py."""
    return pack_by_file_count(pending, files_per_item=2, files_per_commit=files_per_commit)


# --------------------------------------------------------------------------- #
# Dataset Viewer parquet (audio Audio-feature + text + full scenario metadata) —
# same technique as kupe-tts/tts_scripts/soniox/hf_metadata.py::write_audio_text_parquet
def write_parquet_shards(rows: list[dict], out_dir: Path, *,
                        sample_rate: int, rows_per_shard: int) -> list[Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("train-*.parquet"):
        old.unlink(missing_ok=True)

    audio_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    hf_features = {"info": {"features": {
        "audio": {"sampling_rate": sample_rate, "_type": "Audio"},
        "user_text": {"dtype": "string", "_type": "Value"},
        "scenario_id": {"dtype": "string", "_type": "Value"},
        "behaviour": {"dtype": "string", "_type": "Value"},
        "language": {"dtype": "string", "_type": "Value"},
        "domain": {"dtype": "string", "_type": "Value"},
        "gender": {"dtype": "string", "_type": "Value"},
        "prosody": {"dtype": "string", "_type": "Value"},
        "agent_text": {"dtype": "string", "_type": "Value"},
        "duration_s": {"dtype": "float64", "_type": "Value"},
        "num_words": {"dtype": "int64", "_type": "Value"},
    }}}
    meta = {"huggingface": json.dumps(hf_features)}

    paths: list[Path] = []
    n = len(rows)
    n_shards = max(1, (n + rows_per_shard - 1) // rows_per_shard) if n else 1
    for s in range(n_shards):
        chunk = rows[s * rows_per_shard: (s + 1) * rows_per_shard]
        if not chunk and s > 0:
            break
        audio_col = [{"bytes": None, "path": wav_repo_path(r["scenario_id"])} for r in chunk]
        table = pa.table({
            "audio": pa.array(audio_col, type=audio_type),
            "user_text": pa.array([r["text"] for r in chunk], type=pa.string()),
            "scenario_id": pa.array([r["scenario_id"] for r in chunk], type=pa.string()),
            "behaviour": pa.array([r["scenario"].get("behaviour", "") for r in chunk], type=pa.string()),
            "language": pa.array([r["scenario"].get("language", "") for r in chunk], type=pa.string()),
            "domain": pa.array([r["scenario"].get("domain", "") for r in chunk], type=pa.string()),
            "gender": pa.array([r["scenario"].get("gender", "") for r in chunk], type=pa.string()),
            "prosody": pa.array([r["scenario"].get("prosody", "") for r in chunk], type=pa.string()),
            "agent_text": pa.array([r["scenario"].get("agent_text", "") or "" for r in chunk], type=pa.string()),
            "duration_s": pa.array([float(r["duration_s"]) for r in chunk], type=pa.float64()),
            "num_words": pa.array([len(r["words_meta"].get("words") or []) for r in chunk], type=pa.int64()),
        }).replace_schema_metadata(meta)
        out = out_dir / f"train-{s:05d}-of-{n_shards:05d}.parquet"
        pq.write_table(table, out, compression="zstd")
        paths.append(out)
    return paths


def write_metadata_files(rows: list[dict], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "metadata.jsonl"
    csv_path = out_dir / "metadata.csv"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "file_name": wav_repo_path(r["scenario_id"]),
                "text": r["text"], "scenario_id": r["scenario_id"],
            }, ensure_ascii=False) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "text", "scenario_id"])
        w.writeheader()
        for r in rows:
            w.writerow({"file_name": wav_repo_path(r["scenario_id"]),
                       "text": r["text"], "scenario_id": r["scenario_id"]})
    return {"jsonl": jsonl_path, "csv": csv_path}


def dataset_card_readme(*, n_paired: int, repo: str, languages: list[str]) -> str:
    lang_yaml = "\n".join(f"  - {l.split('_')[0]}" for l in sorted(set(languages)))
    return f"""---
license: apache-2.0
task_categories:
  - text-to-speech
  - automatic-speech-recognition
language:
{lang_yaml}
tags:
  - hindi
  - gujarati
  - english
  - tts
  - soniox
  - kupe
  - thinkspark
pretty_name: ThinkSpark-v2-350M training data
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
---

# ThinkSpark-v2-350M — training data

Full-duplex floor-controller (Section 8) training corpus: **playable audio** + **text**,
paired for the Dataset Viewer, plus every scenario field (behaviour, language, domain,
gender, prosody, agent text) and Soniox character-level timestamps.

## Dataset Viewer

Default split is parquet with a real **Audio** feature — a player renders inline next to
the text in the Hub UI:

| column | type | description |
|---|---|---|
| `audio` | Audio | playable wav (already stored under `audio/`) |
| `user_text` | string | the exact line synthesised (Section 8.4) |
| `scenario_id` | string | stable id, joins to `timestamps/<shard>/<id>.json` |
| `behaviour` | string | one of the 12 generation buckets (Section 8.1) |
| `language` | string | hi / en / gu / hi_en_native / gu_en_native |
| `domain` | string | bfsi_collections / support / sales |
| `gender` | string | requested TTS voice gender |
| `prosody` | string | falling / rising / held / flat / distressed / neutral |
| `agent_text` | string | what the agent was saying (may be empty) |
| `duration_s` | float64 | audio duration in seconds |
| `num_words` | int64 | word count in the Soniox timestamp alignment |

Paired rows: **{n_paired:,}** (only scenarios with both real audio and non-empty text).

```python
from datasets import load_dataset
ds = load_dataset("{repo}", split="train")
print(ds[0]["user_text"], ds[0]["behaviour"])
# ds[0]["audio"] -> array / sampling_rate / path
```

## Layout

- `data/train-*.parquet` — Viewer source (audio + text + full scenario metadata)
- `audio/<shard>/<scenario_id>.wav` — rendered user audio (sharded, ≤1000 files/dir)
- `timestamps/<shard>/<scenario_id>.json` — Soniox character-level timestamps.
  **Training use only** (Section 8.4 frame calibration) — never needed at inference,
  where a live agent-state flag replaces timing entirely (Section 4.3).
- `metadata.jsonl` / `metadata.csv` — AudioFolder-style side index

See [ThinkSpark-v2-350M](https://github.com) for the full pipeline this data feeds
(Phase 1 modality alignment + Phase 2 referee fine-tune on Gemma-3-270M).
"""


# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    cfg = DataGenConfig.from_yaml(ROOT / args.config)
    scenarios_path = ROOT / args.scenarios
    audio_dir = ROOT / args.audio_dir

    log(f"loading scenarios from {scenarios_path}")
    scenarios = load_scenarios(scenarios_path)
    log(f"loaded {len(scenarios)} scenarios")

    db = RunDB(ROOT / cfg.db_path)
    already_synced = db.hf_synced_ids(args.repo)
    log(f"already synced to {args.repo}: {len(already_synced)}")

    pending = pending_rows(scenarios, audio_dir, already_synced, args.max_scenarios)
    if not pending:
        log("nothing pending — all paired (audio+text) scenarios already synced")
        db.close()
        return 0

    log(f"pending: {len(pending)} scenarios (audio ∩ text), "
       f"total audio = {sum(r['duration_s'] for r in pending) / 3600.0:.3f}h")

    commits = pack_commits(pending, files_per_commit=args.files_per_commit)
    est_files = sum(len(c) * 2 for c in commits)
    log(f"plan: {len(commits)} commit(s) · ~{est_files} files "
       f"(wav+timestamps) · up to {args.files_per_commit} files/commit")

    if args.dry_run:
        log("DRY RUN: no auth, no upload, nothing marked synced")
        db.close()
        return 0

    token = env("HF_TOKEN", required=True)
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=token)
    log(f"auth ok -> repo={args.repo}")
    ensure_repo(api, args.repo, private=args.private)

    uploaded_total = 0
    t0 = time.monotonic()

    for c_i, batch in enumerate(commits):
        ops = []
        for row in batch:
            ops.append(CommitOperationAdd(
                path_in_repo=wav_repo_path(row["scenario_id"]),
                path_or_fileobj=str(row["wav_path"]),
            ))
            ops.append(CommitOperationAdd(
                path_in_repo=ts_repo_path(row["scenario_id"]),
                path_or_fileobj=str(row["meta_path"]),
            ))
        msg = (f"thinkspark-v2-350m bulk {utc_now()} commit {c_i + 1}/{len(commits)} "
              f"(+{len(ops)} files / {len(batch)} scenarios)")
        log(f"commit {c_i + 1}/{len(commits)}: uploading {len(ops)} files "
           f"({len(batch)} scenarios, sharded audio/NNN/ + timestamps/NNN/) "
           f"in ONE create_commit…")
        create_commit_with_backoff(
            api, repo=args.repo, operations=ops, commit_message=msg,
            max_retries=args.max_retries, base_backoff=args.backoff, max_backoff=args.max_backoff,
        )

        for row in batch:
            db.mark_hf_synced(row["scenario_id"], args.repo)
        uploaded_total += len(batch)
        elapsed = time.monotonic() - t0
        rate = uploaded_total / elapsed if elapsed > 0 else 0.0
        remaining = len(pending) - uploaded_total
        eta = remaining / rate if rate > 0 else 0.0
        log(f"  ok · progress={uploaded_total}/{len(pending)} · "
           f"{rate:.2f} scenarios/s · eta≈{eta / 60:.1f} min")

        if c_i + 1 < len(commits):
            log(f"  sleeping {args.sleep:.1f}s before next commit (rate-limit buffer)")
            sleep_with_jitter(args.sleep)

    # ---- Dataset Viewer: parquet (audio Audio-feature + text + metadata) ----
    log("building Dataset Viewer parquet (audio ∩ text, all synced scenarios)…")
    all_synced = db.hf_synced_ids(args.repo)
    viewer_rows = pending_rows(scenarios, audio_dir, already_synced=set(), max_scenarios=None)
    viewer_rows = [r for r in viewer_rows if r["scenario_id"] in all_synced]
    log(f"  paired rows for viewer: {len(viewer_rows)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        parquet_paths = write_parquet_shards(
            viewer_rows, tmp_path, sample_rate=cfg.soniox_sample_rate,
            rows_per_shard=ROWS_PER_PARQUET_SHARD,
        )
        meta_paths = write_metadata_files(viewer_rows, tmp_path)
        log(f"  parquet shards: {len(parquet_paths)}")

        ops = [
            CommitOperationAdd(path_in_repo="metadata.jsonl", path_or_fileobj=str(meta_paths["jsonl"])),
            CommitOperationAdd(path_in_repo="metadata.csv", path_or_fileobj=str(meta_paths["csv"])),
        ]
        for i, pq_path in enumerate(parquet_paths):
            ops.append(CommitOperationAdd(
                path_in_repo=f"data/train-{i:05d}-of-{len(parquet_paths):05d}.parquet",
                path_or_fileobj=str(pq_path),
            ))
        readme = dataset_card_readme(n_paired=len(viewer_rows), repo=args.repo,
                                     languages=[r["scenario"].get("language", "") for r in viewer_rows])
        readme_path = tmp_path / "README.md"
        readme_path.write_text(readme, encoding="utf-8")
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme_path)))

        # The parquet/metadata above are Viewer-oriented (audio+text+a few flat fields) —
        # they deliberately DON'T carry `target` (the control-flag timeline) or
        # `event_char`, both of which scripts/04_build_frames.py needs to build real
        # training frames. Upload the raw scenarios file too so this repo is actually
        # SELF-SUFFICIENT for full training reconstruction on another machine (Kaggle),
        # not just browsable — see scripts/19_fetch_training_data.py, which reads this
        # verbatim into data/scenarios/scenarios_all.jsonl.
        ops.append(CommitOperationAdd(path_in_repo="scenarios/scenarios_all.jsonl",
                                      path_or_fileobj=str(scenarios_path)))

        log(f"  uploading playable audio+text parquet + full scenarios.jsonl + README -> {args.repo}…")
        create_commit_with_backoff(
            api, repo=args.repo, operations=ops,
            commit_message=f"add Dataset Viewer parquet ({len(viewer_rows)} paired rows)",
            max_retries=args.max_retries, base_backoff=args.backoff, max_backoff=args.max_backoff,
        )

    db.close()
    elapsed = time.monotonic() - t0
    log(f"DONE · uploaded={uploaded_total} scenarios · {elapsed / 60:.1f} min · "
       f"repo=https://huggingface.co/datasets/{args.repo} · local files kept")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("--config", default="configs/data_gen.yaml")
    p.add_argument("--scenarios", default="data/scenarios/scenarios_all.jsonl")
    p.add_argument("--audio-dir", default="data/audio")
    p.add_argument("--repo", default=DEFAULT_REPO, required=DEFAULT_REPO is None,
                   help="HF dataset repo id, e.g. <your-hf-username>/Thinkspark-v2-270m-"
                        "training-data — must be a namespace your token can create "
                        "repos under (your own username, or an org you're a member of "
                        "with repo-create rights)")
    p.add_argument("--private", action="store_true", help="create the repo private (default: public)")
    p.add_argument("--files-per-commit", type=int, default=2000,
                   help="max files (wav+timestamps) packed into ONE HF commit (default: 2000)")
    p.add_argument("--sleep", type=float, default=5.0, help="seconds between commits (default: 5)")
    p.add_argument("--backoff", type=float, default=20.0, help="initial backoff on 429/transient (default: 20)")
    p.add_argument("--max-backoff", type=float, default=300.0, help="backoff cap (default: 300)")
    p.add_argument("--max-retries", type=int, default=10, help="retries per commit (default: 10)")
    p.add_argument("--max-scenarios", type=int, default=None,
                   help="cap how many pending scenarios to upload this run (resume later)")
    p.add_argument("--dry-run", action="store_true", help="plan only; no auth/upload/mark")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.files_per_commit < 2:
        log("--files-per-commit must be >= 2 (each scenario is 2 files)")
        return 2
    try:
        return run(args)
    except KeyboardInterrupt:
        log("interrupted — re-run the same command to resume (local files intact)")
        return 130
    except Exception:
        import traceback
        traceback.print_exc()
        log("failed — local files were NOT touched; fix and re-run to resume")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
