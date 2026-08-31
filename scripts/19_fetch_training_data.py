#!/usr/bin/env python
"""
Fetch BOTH Phase-1 and Phase-2 training data from Hugging Face straight into the local
`data/` layout scripts/06_train_phase1.py and scripts/07_train_phase2.py expect — the
Kaggle-side counterpart to scripts/P1_00_pipeline.py (Phase-1, local Mac) and
scripts/13_upload_hf.py (Phase-2). No re-encoding needed; this is genuinely ready-to-train
data, just moved onto whatever machine you're training on.

Phase-1 repo layout (from scripts/P1_00_pipeline.py) -> local layout:
    encoded/<lang>/<clip_id>.npz          -> data/encoded/<clip_id>.npz          (flattened)
    frames_phase1/frames_<lang>.jsonl     -> data/frames_phase1/frames_<lang>.jsonl
    manifest.jsonl                         -> data/phase1_raw/manifest.jsonl      (provenance only)

Phase-2 repo layout (from scripts/13_upload_hf.py) -> local layout:
    scenarios/scenarios_all.jsonl          -> data/scenarios/scenarios_all.jsonl  (full schema —
                                              NOT the Viewer parquet, which is missing `target`/
                                              `event_char` and can't rebuild real training frames)
    audio/<shard>/<scenario_id>.wav        -> data/audio/<scenario_id>.wav        (flattened)
    timestamps/<shard>/<scenario_id>.json  -> data/audio/<scenario_id>.words.json (flattened + renamed
                                              to match the local sidecar convention)

After fetching, Phase-2 still needs the (fast, local, no-API) encode + frame-build steps
— those aren't uploaded, only raw audio + full scenario text:
    python scripts/00_encode_audio.py --audio-dir data/audio --out-dir data/encoded
    python scripts/04_build_frames.py --in data/scenarios/scenarios_all.jsonl \\
        --frames-out data/frames/frames_all.jsonl
Phase-1 needs NEITHER of those — it arrives already encoded with frames built.

Files are MOVED (not copied) out of the temporary download snapshot into their final
flattened locations, so this never doubles your disk usage — important on Kaggle's
bounded disk. Safe to re-run: existing local files are left alone (skipped), only
missing ones are fetched, so an interrupted fetch just resumes.

    conda activate llms
    pip install huggingface_hub

    # see what would be fetched, no download:
    python scripts/19_fetch_training_data.py \\
        --phase1-repo anuj-inavlabs/kupe-thinkspark-270m-phase1-data \\
        --phase2-repo anuj-inavlabs/Thinkspark-v2-270m-training-data --dry-run

    # fetch both (either can be omitted to fetch just one phase):
    python scripts/19_fetch_training_data.py \\
        --phase1-repo anuj-inavlabs/kupe-thinkspark-270m-phase1-data \\
        --phase2-repo anuj-inavlabs/Thinkspark-v2-270m-training-data

    # then, Phase-2 only, encode + build frames locally (Phase-1 needs neither):
    python scripts/00_encode_audio.py --audio-dir data/audio --out-dir data/encoded
    python scripts/04_build_frames.py --in data/scenarios/scenarios_all.jsonl \\
        --frames-out data/frames/frames_all.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import env


def utc_log(msg: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _list_repo_files(api, repo: str, patterns: list[str]) -> list[str]:
    from huggingface_hub import list_repo_files
    all_files = list_repo_files(repo_id=repo, repo_type="dataset", token=api.token if hasattr(api, "token") else None)
    import fnmatch
    return [f for f in all_files if any(fnmatch.fnmatch(f, p) for p in patterns)]


def _snapshot_download(repo: str, patterns: list[str], token: str | None, local_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        raise SystemExit("`huggingface_hub` not installed. `pip install huggingface_hub`.")
    import os, time
    # xet's per-file token endpoint (xet-read-token) is what trips HF's 1000-req/5-min
    # limit on big many-file repos — the classic path is slower but doesn't fan out
    # token calls, so it stays under the quota. Fewer workers keeps it there too.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    max_workers = int(os.environ.get("HF_FETCH_WORKERS", "4"))
    delay = 30
    for attempt in range(1, 9):  # up to 8 tries; snapshot_download resumes each time
        try:
            path = snapshot_download(
                repo_id=repo, repo_type="dataset", token=token,
                allow_patterns=patterns, local_dir=str(local_dir),
                max_workers=max_workers,
            )
            return Path(path)
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status not in (429, 500, 502, 503, 504) or attempt == 8:
                raise
            wait = min(delay, 300)
            utc_log(f"[fetch] {status} from HF (attempt {attempt}/8) — "
                   f"resuming in {wait}s (already-fetched files are skipped)")
            time.sleep(wait)
            delay *= 2
    return Path(local_dir)  # unreachable; loop either returns or raises


def _move_flatten(src_root: Path, glob_pattern: str, dest_dir: Path, *, rename=None) -> tuple[int, int]:
    """Moves every file matching `glob_pattern` under `src_root` into `dest_dir`, flat
    (dropping any subdirectory structure). `rename(path) -> str` optionally renames each
    file (e.g. Phase-2's timestamps/<id>.json -> <id>.words.json). Skips a destination
    file that already exists (resumable — re-running never re-fetches/re-moves what's
    already there). Returns (moved, skipped)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = skipped = 0
    for src in src_root.rglob(glob_pattern):
        if not src.is_file():
            continue
        name = rename(src) if rename else src.name
        dst = dest_dir / name
        if dst.exists():
            skipped += 1
            continue
        shutil.move(str(src), str(dst))
        moved += 1
    return moved, skipped


def fetch_phase1(repo: str, token: str | None, args) -> None:
    """Fetches BOTH repo layouts a Phase-1 repo might have, since not every uploader
    script here produces the same one: `scripts/P1_00_sequential.py` uploads Parquet
    shards (`data/<lang>/*.parquet`, cb0/energy/f0 embedded as columns — the current,
    recommended layout, see thinkspark.phase1_parquet's module docstring for why); the
    older `encoded/<lang>/*.npz` + `frames_phase1/*.jsonl` layout (still what
    `scripts/P1_00_pipeline.py`'s concurrent uploader produces) is fetched too if
    present. Both land in the exact same local data/encoded + data/frames_phase1
    layout scripts/06_train_phase1.py reads — mixing languages/sources across the two
    layouts in one repo is fine."""
    utc_log(f"[phase1] fetching from {repo} ...")
    tmp_dir = ROOT / args.tmp_dir / "phase1_snapshot"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    patterns = ["data/*/*.parquet", "encoded/**/*.npz", "frames_phase1/*.jsonl"]
    if args.with_manifest:
        patterns.append("manifest.jsonl")

    snap = _snapshot_download(repo, patterns, token, tmp_dir)

    encoded_dir = ROOT / args.encoded_dir
    frames_dir = ROOT / args.frames_phase1_dir
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ---- new layout: Parquet shards, unpacked into the same local layout -------------
    shard_paths = sorted((snap / "data").glob("*/*.parquet")) if (snap / "data").exists() else []
    if shard_paths:
        from thinkspark.phase1_parquet import unpack_shard_to_local
        by_lang: dict[str, list[Path]] = {}
        for p in shard_paths:
            by_lang.setdefault(p.parent.name, []).append(p)
        for lang, shards in sorted(by_lang.items()):
            frames_path = frames_dir / f"frames_{lang}.jsonl"
            already = {json.loads(l)["scenario_id"] for l in frames_path.read_text().splitlines()} \
                if frames_path.exists() else set()
            n_new = 0
            with frames_path.open("a", encoding="utf-8") as fh:
                for shard in sorted(shards):
                    n_new += unpack_shard_to_local(shard, encoded_dir, fh, ROOT)
            # unpack_shard_to_local always appends a frame record per row even if the
            # .npz already existed locally — that would duplicate rows in frames_<lang>.jsonl
            # on a re-run, so dedupe by scenario_id against what was already there.
            if already:
                lines = frames_path.read_text().splitlines()
                seen: set[str] = set()
                deduped = []
                for l in lines:
                    rec = json.loads(l)
                    if rec["scenario_id"] in seen:
                        continue
                    seen.add(rec["scenario_id"])
                    deduped.append(l)
                frames_path.write_text("\n".join(deduped) + "\n", encoding="utf-8")
            utc_log(f"[phase1] {lang}: {len(shards)} parquet shard(s) unpacked, "
                   f"{n_new} new .npz written -> {encoded_dir}")

    # ---- older layout: loose .npz + frames_phase1/*.jsonl ----------------------------
    n_npz, skip_npz = _move_flatten(snap / "encoded", "*.npz", encoded_dir)
    if n_npz or skip_npz:
        utc_log(f"[phase1] encoded (legacy .npz layout): {n_npz} moved, {skip_npz} already present -> {encoded_dir}")

    n_frames, skip_frames = _move_flatten(snap / "frames_phase1", "*.jsonl", frames_dir)
    if n_frames or skip_frames:
        utc_log(f"[phase1] frames (legacy layout): {n_frames} moved, {skip_frames} already present -> {frames_dir}")

    if not shard_paths and not n_npz and not skip_npz:
        utc_log(f"[phase1] ! nothing found in {repo} matching either the Parquet or "
               f"legacy .npz layout — check the repo actually has Phase-1 data uploaded")

    if args.with_manifest:
        manifest_src = snap / "manifest.jsonl"
        manifest_dst = ROOT / "data/phase1_raw/manifest.jsonl"
        if manifest_src.exists() and not manifest_dst.exists():
            manifest_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(manifest_src), str(manifest_dst))
            utc_log(f"[phase1] manifest -> {manifest_dst}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    utc_log("[phase1] done — ready to train directly, no encode/frame-build step needed:")
    utc_log("[phase1]   python scripts/06_train_phase1.py --config configs/train_phase1.yaml "
           "--frames \"data/frames_phase1/*.jsonl\"")


def fetch_phase2(repo: str, token: str | None, args) -> None:
    utc_log(f"[phase2] fetching from {repo} ...")
    tmp_dir = ROOT / args.tmp_dir / "phase2_snapshot"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    patterns = ["scenarios/scenarios_all.jsonl", "audio/**/*.wav", "timestamps/**/*.json"]
    snap = _snapshot_download(repo, patterns, token, tmp_dir)

    scenarios_src = snap / "scenarios" / "scenarios_all.jsonl"
    scenarios_dst = ROOT / "data/scenarios/scenarios_all.jsonl"
    if scenarios_src.exists():
        scenarios_dst.parent.mkdir(parents=True, exist_ok=True)
        if scenarios_dst.exists():
            utc_log(f"[phase2] {scenarios_dst} already exists — leaving it alone "
                   f"(delete it first if you want the freshly-fetched version instead)")
        else:
            shutil.move(str(scenarios_src), str(scenarios_dst))
            utc_log(f"[phase2] scenarios -> {scenarios_dst}")
    else:
        utc_log(f"[phase2] ! no scenarios/scenarios_all.jsonl in {repo} — this repo predates "
               f"the full-schema upload (scripts/13_upload_hf.py); only the Dataset Viewer "
               f"parquet exists there, which is NOT enough to rebuild real training frames "
               f"(missing `target`/`event_char`). Re-upload with the current 13_upload_hf.py "
               f"to fix this.")

    audio_dir = ROOT / args.audio_dir
    n_wav, skip_wav = _move_flatten(snap / "audio", "*.wav", audio_dir)
    utc_log(f"[phase2] audio: {n_wav} moved, {skip_wav} already present -> {audio_dir}")

    n_ts, skip_ts = _move_flatten(
        snap / "timestamps", "*.json", audio_dir,
        rename=lambda p: p.stem + ".words.json",
    )
    utc_log(f"[phase2] timestamps: {n_ts} moved, {skip_ts} already present -> {audio_dir}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    utc_log("[phase2] done — encode + build frames locally next (not uploaded, fast+free):")
    utc_log("[phase2]   python scripts/00_encode_audio.py --audio-dir data/audio --out-dir data/encoded")
    utc_log("[phase2]   python scripts/04_build_frames.py --in data/scenarios/scenarios_all.jsonl "
           "--frames-out data/frames/frames_all.jsonl")


def _dry_run_report(repo: str, patterns: list[str], token: str | None, label: str) -> None:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        files = _list_repo_files(api, repo, patterns)
    except Exception as e:
        utc_log(f"[{label}] ! couldn't list {repo}: {e}")
        return
    utc_log(f"[{label}] {repo}: {len(files)} matching file(s) would be fetched")
    for f in files[:10]:
        utc_log(f"[{label}]   {f}")
    if len(files) > 10:
        utc_log(f"[{label}]   ... +{len(files) - 10} more")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-repo", default=None,
                    help="e.g. anuj-inavlabs/kupe-thinkspark-270m-phase1-data (omit to skip Phase-1)")
    ap.add_argument("--phase2-repo", default=None,
                    help="e.g. anuj-inavlabs/Thinkspark-v2-270m-training-data (omit to skip Phase-2)")
    ap.add_argument("--encoded-dir", default="data/encoded")
    ap.add_argument("--frames-phase1-dir", default="data/frames_phase1")
    ap.add_argument("--audio-dir", default="data/audio")
    ap.add_argument("--tmp-dir", default="data/.hf_fetch_tmp",
                    help="scratch dir for the download snapshot before flattening (removed after)")
    ap.add_argument("--with-manifest", action="store_true",
                    help="also fetch Phase-1's manifest.jsonl (provenance; not needed to train)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched, download nothing")
    args = ap.parse_args()

    if not args.phase1_repo and not args.phase2_repo:
        raise SystemExit("pass --phase1-repo and/or --phase2-repo")

    token = env("HF_TOKEN")  # only needed if either repo is private

    print("=" * 68)
    print("ThinkSpark-v2-350M — fetch training data from Hugging Face")
    print("=" * 68)

    if args.dry_run:
        if args.phase1_repo:
            patterns = ["encoded/**/*.npz", "frames_phase1/*.jsonl"]
            if args.with_manifest:
                patterns.append("manifest.jsonl")
            _dry_run_report(args.phase1_repo, patterns, token, "phase1")
        if args.phase2_repo:
            _dry_run_report(args.phase2_repo,
                            ["scenarios/scenarios_all.jsonl", "audio/**/*.wav", "timestamps/**/*.json"],
                            token, "phase2")
        print("\ndry run — nothing downloaded. Re-run without --dry-run to fetch.")
        return

    if args.phase1_repo:
        fetch_phase1(args.phase1_repo, token, args)
    if args.phase2_repo:
        fetch_phase2(args.phase2_repo, token, args)

    print("\n" + "=" * 68)
    print("done.")


if __name__ == "__main__":
    main()
