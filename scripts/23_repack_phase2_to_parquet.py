#!/usr/bin/env python
"""
One-time conversion: repack an existing Phase-2 HF dataset repo's loose per-clip files
(`audio/<000-199>/<id>.wav` + `timestamps/<000-199>/<id>.json`, from
scripts/13_upload_hf.py) into a handful of self-contained Parquet shards
(`data/phase2-shard-NNNNN.parquet`) — real problem this fixes: fetching thousands of
tiny files is dominated by per-request round-trip latency, not bandwidth (minutes per
folder on a real run, ~200 folders total).

Built for a SMALL-RAM box (default assumption: a 4GB DigitalOcean CPU droplet — no GPU,
no Mimi model needed, this never touches audio content, just repackages files): the
200 upload folders are processed in bounded BATCHES (`--folders-per-shard`, default 20 —
10 batches total), one Parquet shard per batch. Only one batch's worth of files is ever
downloaded/held on disk or in memory at a time — downloaded, packed, uploaded, deleted,
then the next batch starts. Resumable: a batch whose shard already exists on the repo is
skipped without re-downloading anything.

`scenarios/scenarios_all.jsonl` (the full-schema scenario records) is downloaded ONCE up
front and kept in memory as a lookup dict — same pattern scripts/13_upload_hf.py already
uses, sized for tens of thousands of scenarios comfortably within 4GB.

Data is NEVER modified — wav bytes, timestamps JSON, and scenario JSON are carried
through byte-for-byte (verified offline with a round-trip test, see
thinkspark.phase2_parquet). This only changes STORAGE FORMAT on the HF repo.

    conda activate llms   # or just: pip install huggingface_hub pyarrow
    export HF_TOKEN=hf_...   # WRITE access

    # see the plan (how many folders/shards, no download):
    python scripts/23_repack_phase2_to_parquet.py --dry-run

    # convert (resumable — safe to stop and re-run):
    python scripts/23_repack_phase2_to_parquet.py

    # after verifying the new data/phase2-shard-*.parquet files (e.g. re-fetch with
    # scripts/21_fetch_phase2.py and spot-check), reclaim repo space by deleting the OLD
    # loose audio/ + timestamps/ folders (asks to confirm; leaves scenarios/, the new
    # data/*.parquet, metadata, and README untouched):
    python scripts/23_repack_phase2_to_parquet.py --delete-old
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import env
from thinkspark.hf_upload import create_commit_with_backoff, ensure_repo
from thinkspark.phase2_parquet import pack_batch_to_parquet

DEFAULT_PHASE2_REPO = "anuj-inavlabs/Thinkspark-v2-270m-training-data"


def _load_scenarios(api, repo: str, token: str | None, tmp_dir: Path) -> dict[str, str]:
    """Downloads scenarios/scenarios_all.jsonl once and returns {scenario_id: raw JSON
    line (str, no trailing newline)} — kept as raw strings (not re-parsed into dicts) so
    packing is a pure byte-passthrough and this dict is cheaper in memory.

    Falls back to the Dataset Viewer parquet (`data/train-*.parquet` — produced
    automatically by scripts/13_upload_hf.py's original upload, exists even when the
    full-schema scenarios_all.jsonl was never uploaded) if the full file isn't there.
    That parquet has real per-clip metadata (scenario_id, user_text, behaviour,
    language, domain, gender, prosody, agent_text, duration_s, num_words) but NOT
    `target`/`event_char` (the focal-loss control-flag timeline) — fine to repack
    now and fill in later if that full schema ever turns up."""
    from huggingface_hub import hf_hub_download

    print("downloading scenarios/scenarios_all.jsonl (full schema, needed once)...")
    try:
        path = hf_hub_download(repo_id=repo, repo_type="dataset", token=token,
                               filename="scenarios/scenarios_all.jsonl",
                               local_dir=str(tmp_dir))
    except Exception as e:
        print(f"  ! no scenarios/scenarios_all.jsonl on {repo} ({e}) — falling back to "
             f"the Dataset Viewer parquet (data/train-*.parquet) for per-clip metadata "
             f"instead. This repacks scenario_id/user_text/behaviour/... but NOT "
             f"target/event_char.")
        return _load_scenarios_from_viewer(api, repo, token, tmp_dir)
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sid = json.loads(line).get("scenario_id")
            except json.JSONDecodeError:
                continue
            if sid:
                out[sid] = line
    print(f"  {len(out)} scenario records loaded")
    return out


def _load_scenarios_from_viewer(api, repo: str, token: str | None, tmp_dir: Path) -> dict[str, str]:
    """Reads scenario METADATA ONLY from the Dataset Viewer parquet — never the "audio"
    column, which embeds full audio bytes and can be gigabytes per shard file (real
    numbers on this repo: ~19k rows, ~7GB of audio). Parquet is columnar, so requesting
    only the small text columns skips reading the audio column's data pages entirely,
    keeping this safe on a 4GB-RAM box even though the source file itself is large on
    disk (streamed there by the download, never held whole in memory)."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    files = api.list_repo_files(repo_id=repo, repo_type="dataset", token=token)
    viewer_files = sorted(f for f in files if f.startswith("data/train-") and f.endswith(".parquet"))
    if not viewer_files:
        print("  ! no data/train-*.parquet Viewer files either — no scenario metadata "
             "available at all; every wav will be skipped as orphaned.")
        return {}

    print(f"  downloading {len(viewer_files)} Dataset Viewer parquet file(s) for metadata "
         f"(audio bytes inside them are NOT read into memory)...")
    viewer_dir = tmp_dir / "viewer_parquet"
    snapshot_download(repo_id=repo, repo_type="dataset", token=token,
                      allow_patterns=viewer_files, local_dir=str(viewer_dir),
                      max_workers=int(os.environ.get("HF_FETCH_WORKERS", "32")))

    meta_cols = ["scenario_id", "user_text", "behaviour", "language", "domain",
                "gender", "prosody", "agent_text", "duration_s", "num_words"]
    out: dict[str, str] = {}
    for vf in viewer_files:
        local_path = viewer_dir / vf
        table = pq.read_table(local_path, columns=meta_cols)   # column-pruned — no audio bytes read
        for row in table.to_pylist():
            sid = row.get("scenario_id")
            if sid:
                out[sid] = json.dumps(row, ensure_ascii=False)
        local_path.unlink(missing_ok=True)   # free disk immediately, one file at a time
    print(f"  {len(out)} scenario metadata rows loaded from the Viewer parquet "
         f"(no target/event_char — audio+text+prosody metadata only)")
    return out


def _list_buckets(api, repo: str, token: str | None) -> list[str]:
    """One lightweight repo_info/list_repo_files call (not per-file) — the real folder
    layout (`audio/<bucket>/...`), not assumed."""
    files = api.list_repo_files(repo_id=repo, repo_type="dataset", token=token)
    buckets = sorted({f.split("/")[1] for f in files
                      if f.startswith("audio/") and f.count("/") == 2})
    return buckets


def _existing_shard_indices(api, repo: str, token: str | None) -> set[int]:
    files = api.list_repo_files(repo_id=repo, repo_type="dataset", token=token)
    out = set()
    for f in files:
        if f.startswith("data/phase2-shard-") and f.endswith(".parquet"):
            try:
                out.add(int(f.rsplit("-", 1)[-1].split(".")[0]))
            except ValueError:
                pass
    return out


def repack(repo: str, args) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download

    token = env("HF_TOKEN", required=True)
    api = HfApi(token=token)
    ensure_repo(api, repo, args.private)

    buckets = _list_buckets(api, repo, token)
    if not buckets:
        print(f"no audio/<bucket>/ files found in {repo} — nothing to repack "
             f"(already converted, or repo is empty).")
        return
    batches = [buckets[i:i + args.folders_per_shard]
              for i in range(0, len(buckets), args.folders_per_shard)]
    print(f"found {len(buckets)} bucket folder(s) -> {len(batches)} batch(es) of up to "
         f"{args.folders_per_shard} folders each -> {len(batches)} Parquet shard(s)")

    if args.dry_run:
        for i, b in enumerate(batches):
            print(f"  batch {i}: buckets {b[0]}..{b[-1]} ({len(b)} folders) "
                 f"-> data/phase2-shard-{i:05d}.parquet")
        print("\ndry run — nothing downloaded/uploaded. Re-run without --dry-run to convert.")
        return

    done = _existing_shard_indices(api, repo, token)
    if done:
        print(f"{len(done)} shard(s) already on the repo — those batches will be skipped")

    tmp_dir = ROOT / args.tmp_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    scenarios = _load_scenarios(api, repo, token, tmp_dir)

    max_workers = int(os.environ.get("HF_FETCH_WORKERS", "32"))
    missing_scenario = 0
    missing_ts = 0

    for i, batch in enumerate(batches):
        if i in done:
            print(f"\nbatch {i}: shard already uploaded, skipping (no download needed)")
            continue

        print(f"\nbatch {i}/{len(batches) - 1}: downloading buckets {batch[0]}..{batch[-1]} "
             f"({len(batch)} folders)...")
        patterns = [f"audio/{b}/*.wav" for b in batch] + [f"timestamps/{b}/*.json" for b in batch]
        batch_dir = tmp_dir / f"batch{i:05d}"
        snapshot_download(repo_id=repo, repo_type="dataset", token=token,
                          allow_patterns=patterns, local_dir=str(batch_dir),
                          max_workers=max_workers)

        wav_files = sorted(batch_dir.glob("audio/*/*.wav"))
        print(f"  {len(wav_files)} wav file(s) downloaded, packing...")
        rows = []
        for wav_path in wav_files:
            sid = wav_path.stem
            scen_line = scenarios.get(sid)
            if scen_line is None:
                missing_scenario += 1
                continue   # audio with no matching scenario record is unusable, skip
            ts_path = wav_path.parent.parent.parent / "timestamps" / wav_path.parent.name / f"{sid}.json"
            ts_text = ts_path.read_text(encoding="utf-8") if ts_path.exists() else ""
            if not ts_text:
                missing_ts += 1
            rows.append({
                "scenario_id": sid,
                "wav_bytes": wav_path.read_bytes(),
                "timestamps_json": ts_text,
                "scenario_json": scen_line,
            })

        if not rows:
            print(f"  ! batch {i}: no usable rows (all missing scenario records?) — skipping upload")
            shutil.rmtree(batch_dir, ignore_errors=True)
            continue

        shard_path = tmp_dir / f"phase2-shard-{i:05d}.parquet"
        pack_batch_to_parquet(rows, shard_path)

        print(f"  uploading data/{shard_path.name} ...")
        create_commit_with_backoff(
            api, repo=repo, operations=[
                CommitOperationAdd(path_in_repo=f"data/{shard_path.name}",
                                   path_or_fileobj=str(shard_path))],
            commit_message=f"phase2: repack batch {i} ({len(rows)} clips) to parquet",
            max_retries=args.max_retries, base_backoff=args.backoff, max_backoff=args.max_backoff,
        )

        # free disk before the next batch — the whole point of batching on a small box
        shutil.rmtree(batch_dir, ignore_errors=True)
        shard_path.unlink(missing_ok=True)
        print(f"  batch {i} done, local files freed")

    if missing_scenario:
        print(f"\n! {missing_scenario} wav file(s) had no matching scenario record — "
             f"skipped (orphaned audio, not usable for training anyway)")
    if missing_ts:
        print(f"! {missing_ts} clip(s) had no timestamps JSON — packed with empty "
             f"timestamps_json (still usable for training, just no word-level timing)")

    print(f"\ndone — {len(batches)} shard(s) at data/phase2-shard-*.parquet on {repo}")
    print(f"verify with: python scripts/21_fetch_phase2.py")
    print(f"then reclaim space: python scripts/23_repack_phase2_to_parquet.py --delete-old")


def delete_old(repo: str, args) -> None:
    from huggingface_hub import CommitOperationDelete, HfApi

    token = env("HF_TOKEN", required=True)
    api = HfApi(token=token)

    files = api.list_repo_files(repo_id=repo, repo_type="dataset", token=token)
    has_parquet = any(f.startswith("data/phase2-shard-") for f in files)
    if not has_parquet:
        raise SystemExit("no data/phase2-shard-*.parquet found on the repo yet — run "
                         "this script WITHOUT --delete-old first to convert, and verify "
                         "the new shards before deleting the originals.")

    targets = [f for f in {"audio", "timestamps"} if any(x == f or x.startswith(f + "/") for x in files)]
    if not targets:
        print("no audio/ or timestamps/ folders found — already cleaned up.")
        return

    print(f"will delete these top-level folders from {repo}: {targets}")
    print("(scenarios/, data/*.parquet, metadata.*, README.md are NOT touched)")
    try:
        answer = input("Type 'yes' to confirm: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\ncancelled.")
        return
    if answer.lower() != "yes":
        print("cancelled — nothing deleted.")
        return

    ops = [CommitOperationDelete(path_in_repo=t, is_folder=True) for t in targets]
    create_commit_with_backoff(
        api, repo=repo, operations=ops,
        commit_message="phase2: remove old loose audio/timestamps (superseded by parquet shards)",
        max_retries=args.max_retries, base_backoff=args.backoff, max_backoff=args.max_backoff,
    )
    print(f"done — {targets} removed from {repo}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_PHASE2_REPO)
    ap.add_argument("--folders-per-shard", type=int, default=20,
                    help="bucket folders per batch/shard (default 20 -> ~10 shards from "
                        "the 200 upload buckets). Lower this on a very small box if disk "
                        "or memory is still tight; raise it for fewer, bigger shards.")
    ap.add_argument("--tmp-dir", default="data/.phase2_repack_tmp",
                    help="scratch dir — only ever holds ONE batch's files at a time")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--delete-old", action="store_true",
                    help="delete the OLD audio/ + timestamps/ folders (asks to confirm) "
                        "instead of converting — run this AFTER converting + verifying")
    ap.add_argument("--backoff", type=float, default=20.0)
    ap.add_argument("--max-backoff", type=float, default=300.0)
    ap.add_argument("--max-retries", type=int, default=10)
    args = ap.parse_args()

    print("=" * 68)
    print(f"ThinkSpark-v2-350M — Phase-2 repo repack to Parquet: {args.repo}")
    print("=" * 68)

    if args.delete_old:
        delete_old(args.repo, args)
    else:
        repack(args.repo, args)


if __name__ == "__main__":
    main()
