#!/usr/bin/env python
"""
Data sanity check — run this BEFORE training, on the frame shards you're about to use.

It answers the three questions that silently ruined the previous runs:

  1. VAP targets:  what fraction of frames are actually labelled "user speaking"?
     If this is ~0 (or ~1), the VAP head has nothing to learn, the BCE collapses to a
     tiny number, and evaluation reports VAD-F1 = 0.000 no matter how you weight the
     loss. That is a DATA bug, not a training bug — no lambda fixes it.

  2. Control-flag distribution: how rare are TURN_END / BARGE_SOFT / COMMIT_LLM?
     Flags with a few hundred frames cannot reach F1 0.85 regardless of the recipe;
     you need more of them in the corpus.

  3. Spoken text coverage: how many records actually carry back-channel text?

    python scripts/25_check_data.py --frames "data/frames/*.jsonl"
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter

import numpy as np

from _bootstrap import setup

ROOT = setup()

from thinkspark import vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="glob of frame shards")
    ap.add_argument("--limit", type=int, default=0, help="only read the first N records")
    ap.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                    help="1 or 2 to force the expected schema; 0 (default) auto-detects. "
                        "Phase-1 records legitimately carry all-LISTEN flags, an all-ones "
                        "speaking_mask and their text in `user_text`; grading them against "
                        "Phase-2 expectations produces false alarms.")
    args = ap.parse_args()

    shards = sorted(glob.glob(args.frames if args.frames.startswith("/")
                              else str(ROOT / args.frames)))
    if not shards:
        raise SystemExit(f"no frame shards matched {args.frames}")

    flag_freq: Counter = Counter()
    n_rec = n_frames = 0
    speak_sum = 0.0
    n_missing_speaking = 0
    n_with_text = 0
    per_rec_speak_rate: list[float] = []

    for path in shards:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_rec += 1
                T = int(rec.get("num_frames", 0))
                n_frames += T
                for fid in rec.get("flags", []):
                    flag_freq[vocab.ID_TO_CONTROL_FLAG[int(fid)]] += 1
                sm = rec.get("speaking_mask")
                if sm is None:
                    n_missing_speaking += 1
                else:
                    arr = np.asarray(sm, dtype=np.float32)
                    speak_sum += float(arr.sum())
                    if arr.size:
                        per_rec_speak_rate.append(float(arr.mean()))
                # Phase 1 keeps its alignment target in `user_text`; Phase 2 keeps
                # back-channel text in `spoken_spans` (or spoken_text/target_text).
                spans = rec.get("spoken_spans") or []
                txt = (rec.get("spoken_text") or rec.get("target_text")
                       or rec.get("user_text") or "")
                if isinstance(txt, str) and txt.strip():
                    n_with_text += 1
                elif spans:
                    n_with_text += 1
                if args.limit and n_rec >= args.limit:
                    break
        if args.limit and n_rec >= args.limit:
            break

    non_listen = sum(c for f, c in flag_freq.items() if f != "LISTEN")
    phase = args.phase or (1 if non_listen == 0 else 2)

    print("=" * 66)
    print(f"Data check — {len(shards)} shard(s), {n_rec} records, {n_frames} frames")
    print(f"schema: PHASE {phase}" + ("  (auto-detected)" if not args.phase else ""))
    print("=" * 66)

    # ---- 1. VAP / speaking ----------------------------------------------------------
    print("\n[1] VAP targets (speaking_mask)")
    if n_missing_speaking:
        print(f"  ! {n_missing_speaking}/{n_rec} records have NO 'speaking_mask' field.")
        print("    The dataset defaults those to ALL-ONES, which is also degenerate.")
    if per_rec_speak_rate:
        rate = speak_sum / max(1, n_frames)
        arr = np.array(per_rec_speak_rate)
        print(f"  overall speaking rate : {rate:.4f}")
        print(f"  per-record rate p10/p50/p90 : "
              f"{np.percentile(arr,10):.3f} / {np.percentile(arr,50):.3f} / "
              f"{np.percentile(arr,90):.3f}")
        if rate < 0.02:
            print("  ✗ DEGENERATE: almost no frame is labelled 'speaking'. The VAP head")
            print("    will learn to always output 'not speaking' — BCE goes to ~0 and")
            print("    VAD-F1 is 0.000 by construction. FIX THE FRAME BUILDER, not the loss.")
        elif rate > 0.98:
            if phase == 1:
                print("  ~ EXPECTED for Phase 1: the corpus is continuous speech clips, so")
                print("    scripts/P1_02_build_frames.py sets speaking_mask = all 1s by")
                print("    design. There are no negatives, so VAP cannot be learned here —")
                print("    keep lambda_vap LOW (see configs/train_phase1*.yaml) so a constant")
                print("    target does not bias the head. Phase 2 is where VAP is trained.")
            else:
                print("  ✗ DEGENERATE: almost every frame is 'speaking' — no negatives to learn.")
        else:
            print("  ✓ healthy — VAP has real signal to learn.")
    else:
        print("  ✗ no speaking_mask data found at all.")

    # ---- 2. control flags -----------------------------------------------------------
    print("\n[2] Control-flag distribution")
    if phase == 1:
        print("  (Phase-1 records carry all-LISTEN flags for schema compatibility only —")
        print("   Phase1Loss never reads them. Nothing below is a problem for Phase 1.)")
    total = sum(flag_freq.values()) or 1
    for flag in vocab.CONTROL_FLAG_TO_ID:
        c = flag_freq.get(flag, 0)
        pct = 100.0 * c / total
        warn = ""
        if phase == 1:
            warn = ""            # not supervised in Phase 1 — never warn
        elif c == 0:
            warn = "  ✗ ABSENT — cannot be learned or scored"
        elif c < 500:
            warn = "  ✗ far too rare to reach a useful F1"
        elif pct < 0.5:
            warn = "  ! very rare"
        print(f"  {flag:<14} {c:>9,}  {pct:5.2f}%{warn}")

    # ---- 3. spoken text -------------------------------------------------------------
    print("\n[3] Target text  ("
          + ("Phase 1: user_text / ASR alignment target"
             if phase == 1 else "Phase 2: spoken_spans back-channel text") + ")")
    print(f"  records with non-empty text : {n_with_text}/{n_rec} "
          f"({100.0*n_with_text/max(1,n_rec):.1f}%)")

    print("\n" + "=" * 66)


if __name__ == "__main__":
    main()
