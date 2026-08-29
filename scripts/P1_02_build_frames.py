#!/usr/bin/env python
"""
Phase 1 — turn the fetched manifest + encoded Mimi shards into frame records
(`thinkspark.dataset.ThinkSparkDataset(phase=1)` input format).

Phase 1's loss (`thinkspark.losses.Phase1Loss`) only reads `align_labels` (built from
`user_text`, the ASR-style target) and `vap` (built from `speaking_mask`) — it never
reads `flags`/`agent_state` (those are Phase-2-only). So a Phase-1 frame record needs
just enough structure to satisfy the shared record schema:

    num_frames     = length of the clip's Mimi cb0 sequence
    encoded_path   = path to the clip's Mimi .npz (cb0 + energy + f0)
    user_text      = the transcript (this IS the alignment target)
    agent_text     = ""            (no agent side in Phase 1)
    flags          = all LISTEN     (present for schema compatibility; unused by Phase1Loss)
    agent_state    = all IDLE       (same — unused by Phase1Loss)
    speaking_mask  = all 1s         (the whole clip is user speech -> VAP target = "speaking")

    conda activate llms
    python scripts/P1_02_build_frames.py --lang hi
    python scripts/P1_02_build_frames.py --lang en --lang hi --lang gu   # or run 3x
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark import vocab


def build_record(rec: dict, encoded_path: Path) -> dict | None:
    if not encoded_path.exists():
        return None
    import numpy as np
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
        "encoded_path": str(encoded_path),
        "flags": [default_flag] * T,
        "agent_state": [idle_state] * T,
        "speaking_mask": [1] * T,
        "spoken_spans": [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/phase1_raw/manifest.jsonl")
    ap.add_argument("--encoded-dir", default="data/encoded")
    ap.add_argument("--frames-out-dir", default="data/frames_phase1")
    ap.add_argument("--lang", action="append", default=None,
                    help="only build these languages (repeatable); default = all in manifest")
    args = ap.parse_args()

    manifest_path = ROOT / args.manifest
    if not manifest_path.exists():
        raise SystemExit(f"no manifest at {manifest_path} — run scripts/P1_01_fetch_corpus.py first.")

    encoded_dir = ROOT / args.encoded_dir
    out_dir = ROOT / args.frames_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    want_langs = set(args.lang) if args.lang else None

    by_lang: dict[str, list[dict]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if want_langs and rec["lang"] not in want_langs:
            continue
        by_lang.setdefault(rec["lang"], []).append(rec)

    if not by_lang:
        raise SystemExit("no matching manifest rows — check --lang and that fetch/encode ran.")

    for lang, recs in by_lang.items():
        out_path = out_dir / f"frames_{lang}.jsonl"
        written = missing_encoding = 0
        with out_path.open("w", encoding="utf-8") as fout:
            for rec in recs:
                stem = Path(rec["wav_path"]).stem
                encoded_path = encoded_dir / f"{stem}.npz"
                frame = build_record(rec, encoded_path)
                if frame is None:
                    missing_encoding += 1
                    continue
                fout.write(json.dumps(frame, ensure_ascii=False) + "\n")
                written += 1
        print(f"[{lang}] wrote {written} frame records -> {out_path} "
             f"({missing_encoding} skipped — not yet encoded; run scripts/00_encode_audio.py first)")


if __name__ == "__main__":
    main()
