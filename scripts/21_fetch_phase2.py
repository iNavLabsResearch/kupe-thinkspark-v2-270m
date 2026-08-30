#!/usr/bin/env python
"""
Fetch ONLY Phase-2 training data from its HF dataset repo into the local layout — a
focused, single-phase wrapper around the shared fetch logic in
scripts/19_fetch_training_data.py.

Unlike Phase-1, Phase-2 data is NOT pre-encoded: the repo holds raw audio + the full
scenario schema, so after fetching you still run the (fast, local, no-API) encode +
frame-build steps this script prints at the end:
    data/scenarios/scenarios_all.jsonl   (full schema — target/event_char included)
    data/audio/<scenario_id>.wav
    data/audio/<scenario_id>.words.json  (per-word timestamps sidecar)

    conda activate llms
    pip install huggingface_hub
    export HF_TOKEN=hf_...   # only needed if the repo is private

    python scripts/21_fetch_phase2.py                       # default repo below
    python scripts/21_fetch_phase2.py --repo <user>/<repo>  # a different repo
    python scripts/21_fetch_phase2.py --dry-run             # list, download nothing

Then encode + build frames locally (fast, free), and train:
    python scripts/00_encode_audio.py --audio-dir data/audio --out-dir data/encoded
    python scripts/04_build_frames.py --in data/scenarios/scenarios_all.jsonl \\
        --frames-out data/frames/frames_all.jsonl
    python scripts/07_train_phase2.py --config configs/train_phase2.yaml \\
        --frames "data/frames/*.jsonl" \\
        --init artifacts/thinkspark-v2-350m/phase1/final/model.pt
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import env

DEFAULT_PHASE2_REPO = "anuj-inavlabs/Thinkspark-v2-270m-training-data"

_spec = importlib.util.spec_from_file_location(
    "_fetch19", str(Path(__file__).with_name("19_fetch_training_data.py")))
_fetch19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fetch19)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_PHASE2_REPO,
                    help=f"Phase-2 HF dataset repo (default {DEFAULT_PHASE2_REPO})")
    ap.add_argument("--audio-dir", default="data/audio")
    ap.add_argument("--tmp-dir", default="data/.hf_fetch_tmp",
                    help="scratch dir for the download snapshot (removed after)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched, download nothing")
    args = ap.parse_args()

    token = env("HF_TOKEN")  # only needed if the repo is private

    print("=" * 68)
    print(f"ThinkSpark-v2-350M — fetch PHASE-2 data from {args.repo}")
    print("=" * 68)

    if args.dry_run:
        _fetch19._dry_run_report(
            args.repo,
            ["scenarios/scenarios_all.jsonl", "audio/**/*.wav", "timestamps/**/*.json"],
            token, "phase2")
        print("\ndry run — nothing downloaded. Re-run without --dry-run to fetch.")
        return

    _fetch19.fetch_phase2(args.repo, token, args)
    print("\n" + "=" * 68)
    print("done — Phase-2 raw data fetched. Next: encode + build frames locally, then train:")
    print("  python scripts/00_encode_audio.py --audio-dir data/audio --out-dir data/encoded")
    print("  python scripts/04_build_frames.py --in data/scenarios/scenarios_all.jsonl "
         "--frames-out data/frames/frames_all.jsonl")
    print("=" * 68)


if __name__ == "__main__":
    main()
