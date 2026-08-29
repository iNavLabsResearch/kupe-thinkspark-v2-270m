#!/usr/bin/env python
"""
Print real Phase-2 audio-hours on disk, broken down by length band (short/extended/
long), from the single consolidated corpus — data/scenarios/scenarios_all.jsonl +
data/audio/*.words.json. (short, extended, and long were originally three separate
corpora with their own scenario/audio dirs; they've since been merged — verified: 0
scenario_id / wav-filename collisions between any of them — into these two paths.)

Real duration where a clip is already rendered; a words/2.2 projection (same formula
thinkspark/frames.py's fallback uses) for what isn't rendered yet, so you get an honest
projected TOTAL for the whole corpus, not just what's done so far.

Read-only: touches no API, spends nothing, and can be run any time.

    python scripts/17_audio_summary.py
    python scripts/17_audio_summary.py --target-hours 55
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

WORDS_PER_SECOND_FALLBACK = 2.2   # thinkspark/frames.py's own fallback rate


def real_duration(audio_dir: Path, scenario_id: str) -> float | None:
    p = audio_dir / f"{scenario_id}.words.json"
    if not p.exists():
        return None
    try:
        dur = float(json.loads(p.read_text(encoding="utf-8")).get("duration_s", 0.0))
        return dur if dur > 0 else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="data/scenarios/scenarios_all.jsonl")
    ap.add_argument("--audio-dir", default="data/audio")
    ap.add_argument("--target-hours", type=float, default=55.0,
                    help="overall Phase-2 target, for the 'still to generate' line")
    args = ap.parse_args()

    scenarios_path = ROOT / args.scenarios
    audio_dir = ROOT / args.audio_dir
    if not scenarios_path.exists():
        raise SystemExit(f"no scenarios file at {scenarios_path}")

    # per-band: [n_total, n_rendered, real_seconds, projected_seconds]
    bands: dict[str, list] = {}
    for line in scenarios_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        band = d.get("length_band") or "short"
        sid = d.get("scenario_id", "")
        row = bands.setdefault(band, [0, 0, 0.0, 0.0])
        row[0] += 1
        dur = real_duration(audio_dir, sid)
        if dur:
            row[1] += 1
            row[2] += dur
            row[3] += dur
        else:
            words = max(1, len((d.get("user_text") or "").split()))
            row[3] += words / WORDS_PER_SECOND_FALLBACK

    print("=" * 68)
    print("ThinkSpark-v2-350M — Phase-2 audio hours (real + projected)")
    print("=" * 68)
    print(f"{'band':<12}{'total':>8}{'rendered':>10}{'real h':>9}{'projected h':>13}")
    print("-" * 55)

    grand_total = grand_rendered = 0
    grand_real_s = grand_proj_s = 0.0
    for band in sorted(bands):
        n_total, n_rendered, real_s, proj_s = bands[band]
        grand_total += n_total
        grand_rendered += n_rendered
        grand_real_s += real_s
        grand_proj_s += proj_s
        print(f"{band:<12}{n_total:>8,}{n_rendered:>10,}{real_s / 3600:>8.2f}h"
             f"{proj_s / 3600:>12.2f}h")

    print("-" * 55)
    print(f"{'TOTAL':<12}{grand_total:>8,}{grand_rendered:>10,}{grand_real_s / 3600:>8.2f}h"
         f"{grand_proj_s / 3600:>12.2f}h")
    print()
    print(f"rendered so far (real):     {grand_real_s / 3600:.2f}h  ({grand_rendered:,} clips)")
    print(f"projected full-corpus total: {grand_proj_s / 3600:.1f}h  "
         f"(real where rendered, words/{WORDS_PER_SECOND_FALLBACK} projection elsewhere)")

    if args.target_hours > 0:
        gap = args.target_hours - grand_proj_s / 3600
        if gap > 0:
            print(f"vs {args.target_hours:.0f}h target -> ~{gap:.1f}h short (projected)")
        else:
            print(f"vs {args.target_hours:.0f}h target -> ~{-gap:.1f}h surplus (projected)")

    note_bands = [b for b in bands if b not in ("short", "extended", "long")]
    if note_bands:
        print(f"\nnote: unrecognised length_band value(s) present: {note_bands}")


if __name__ == "__main__":
    main()
