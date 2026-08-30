#!/usr/bin/env python
"""
Fetch ONLY Phase-1 training data from its HF dataset repo into the local layout
scripts/06_train_phase1.py reads — a focused, single-phase wrapper around the shared
fetch logic in scripts/19_fetch_training_data.py (which can fetch both phases at once).

Phase-1 data arrives already encoded (Mimi cb0/energy/f0 + frame records), so there's
NOTHING to encode/build afterward — it's ready to train the moment this finishes:
    data/encoded/<lang>/<clip_id>.npz
    data/frames_phase1/frames_<lang>.jsonl

Handles both HF repo layouts: the current Parquet-shard layout (data/<lang>/*.parquet,
from scripts/P1_00_sequential.py) and the older loose-.npz layout — see fetch_phase1's
docstring in scripts/19_fetch_training_data.py.

    conda activate llms
    pip install huggingface_hub pyarrow
    # HF_TOKEN only needed if the repo is private:
    export HF_TOKEN=hf_...

    python scripts/20_fetch_phase1.py                       # default repo below
    python scripts/20_fetch_phase1.py --repo <user>/<repo>  # a different repo
    python scripts/20_fetch_phase1.py --dry-run             # list, download nothing

Then train directly:
    python scripts/06_train_phase1.py --config configs/train_phase1.yaml \\
        --frames "data/frames_phase1/*.jsonl"
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import env

DEFAULT_PHASE1_REPO = "anuj-inavlabs/kupe-thinkspark-270m-phase1-data"

# Reuse the tested fetch logic from scripts/19_fetch_training_data.py. Its filename
# starts with a digit so it can't be a normal `import`; load it by path instead. It runs
# `setup()` at import (harmless — just re-resolves the same ROOT) and defines the exact
# same fetch_phase1 / _dry_run_report this project already uses in production.
_spec = importlib.util.spec_from_file_location(
    "_fetch19", str(Path(__file__).with_name("19_fetch_training_data.py")))
_fetch19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fetch19)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_PHASE1_REPO,
                    help=f"Phase-1 HF dataset repo (default {DEFAULT_PHASE1_REPO})")
    ap.add_argument("--encoded-dir", default="data/encoded")
    ap.add_argument("--frames-phase1-dir", default="data/frames_phase1")
    ap.add_argument("--tmp-dir", default="data/.hf_fetch_tmp",
                    help="scratch dir for the download snapshot (removed after)")
    ap.add_argument("--with-manifest", action="store_true",
                    help="also fetch manifest.jsonl (provenance; not needed to train)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched, download nothing")
    # fetch_phase2 is never called from here, but fetch_phase1 reads none of these —
    # kept only so the shared args object has every field 19's functions might touch.
    ap.add_argument("--audio-dir", default="data/audio")
    args = ap.parse_args()

    token = env("HF_TOKEN")  # only needed if the repo is private

    print("=" * 68)
    print(f"ThinkSpark-v2-350M — fetch PHASE-1 data from {args.repo}")
    print("=" * 68)

    if args.dry_run:
        patterns = ["data/*/*.parquet", "encoded/**/*.npz", "frames_phase1/*.jsonl"]
        if args.with_manifest:
            patterns.append("manifest.jsonl")
        _fetch19._dry_run_report(args.repo, patterns, token, "phase1")
        print("\ndry run — nothing downloaded. Re-run without --dry-run to fetch.")
        return

    _fetch19.fetch_phase1(args.repo, token, args)
    print("\n" + "=" * 68)
    print("done — Phase-1 data ready. Train with:")
    print('  python scripts/06_train_phase1.py --config configs/train_phase1.yaml '
         '--frames "data/frames_phase1/*.jsonl"')
    print("=" * 68)


if __name__ == "__main__":
    main()
