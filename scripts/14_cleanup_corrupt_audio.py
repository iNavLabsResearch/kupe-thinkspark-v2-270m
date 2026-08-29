#!/usr/bin/env python
"""
Clean up corrupted audio from a run made with the OLD (buggy) Soniox client.

That version connected to the wrong WebSocket endpoint entirely (STT instead of TTS)
and used the wrong message protocol — it silently "succeeded" on every scenario while
writing near-empty wav files (0.0s duration, no real timestamps). If you ran
`scripts/03_render_user_audio.py` before this fix, your `data/audio/` and the
`tts_calls` SQLite log are full of these false successes.

This script finds every wav that's actually broken (too small to hold real audio, or
whose sidecar `.words.json` says `duration_s <= 0`), deletes the wav + sidecar, and
marks the matching `tts_calls` rows in SQLite so the render script's resume logic will
naturally re-attempt them next time you run it (against the now-fixed client).

Nothing is deleted without your confirmation (unless --yes). Dry-run by default via
--dry-run to see exactly what would be removed first.

    conda activate llms
    python scripts/14_cleanup_corrupt_audio.py --config configs/data_gen.yaml --dry-run
    python scripts/14_cleanup_corrupt_audio.py --config configs/data_gen.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import DataGenConfig
from thinkspark.db import RunDB

# A valid WAV header alone is 44 bytes; Soniox's real minimum-viable clip (even a very
# short utterance) is comfortably larger than this. Anything at or near 44 bytes is
# the empty-RIFF-shell the old buggy client produced.
MIN_VALID_WAV_BYTES = 200


def find_corrupt(audio_dir: Path) -> list[dict]:
    """Scan data/audio/ for wav+words.json pairs that look corrupted."""
    corrupt: list[dict] = []
    for wav_path in sorted(audio_dir.glob("*.wav")):
        sid = wav_path.stem
        meta_path = audio_dir / f"{sid}.words.json"
        reasons = []

        size = wav_path.stat().st_size if wav_path.exists() else 0
        if size < MIN_VALID_WAV_BYTES:
            reasons.append(f"wav too small ({size} bytes)")

        duration_s = None
        n_words = 0
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                duration_s = meta.get("duration_s")
                n_words = len(meta.get("words") or [])
            except (json.JSONDecodeError, OSError):
                reasons.append("words.json unreadable")
        else:
            reasons.append("missing words.json")

        if duration_s is not None and duration_s <= 0.0:
            reasons.append(f"duration_s={duration_s}")
        if meta_path.exists() and n_words == 0 and duration_s is not None and duration_s > 0:
            # audio exists but Soniox never returned timestamps — the OTHER symptom of
            # the wrong-endpoint bug (audio field name matched but timestamps didn't)
            reasons.append("zero word timestamps despite non-zero duration")

        if reasons:
            corrupt.append({
                "scenario_id": sid, "wav_path": wav_path, "meta_path": meta_path,
                "size_bytes": size, "duration_s": duration_s, "reasons": reasons,
            })
    return corrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_gen.yaml")
    ap.add_argument("--audio-dir", default="data/audio")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be removed, delete nothing")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt")
    args = ap.parse_args()

    cfg = DataGenConfig.from_yaml(ROOT / args.config)
    audio_dir = ROOT / args.audio_dir
    if not audio_dir.exists():
        raise SystemExit(f"no audio dir at {audio_dir}")

    corrupt = find_corrupt(audio_dir)
    total_wavs = sum(1 for _ in audio_dir.glob("*.wav"))

    print("=" * 68)
    print("ThinkSpark-v2-350M — corrupt audio scan")
    print("=" * 68)
    print(f"scanned {total_wavs} wav files in {audio_dir}")
    print(f"found {len(corrupt)} corrupt (empty/near-empty or missing timestamps)")

    if not corrupt:
        print("\nnothing to clean up.")
        return

    print("\nsample (first 10):")
    for item in corrupt[:10]:
        print(f"  {item['scenario_id']:<20} {item['size_bytes']:>6}B  "
             f"duration={item['duration_s']}  {', '.join(item['reasons'])}")
    if len(corrupt) > 10:
        print(f"  … +{len(corrupt) - 10} more")

    if args.dry_run:
        print("\nDRY RUN — nothing deleted. Re-run without --dry-run to actually clean up.")
        return

    if not args.yes:
        try:
            answer = input(f"\nDelete these {len(corrupt)} wav+meta pairs and mark their "
                          f"SQLite rows for re-render? Type 'yes' to confirm: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\ncancelled.")
            return
        if answer.lower() != "yes":
            print("cancelled — nothing deleted.")
            return

    db = RunDB(ROOT / cfg.db_path)
    run_id = db.start_run("cleanup_corrupt_audio", cfg.__dict__, vars(args))

    removed = 0
    for item in corrupt:
        item["wav_path"].unlink(missing_ok=True)
        item["meta_path"].unlink(missing_ok=True)
        # Fix the HISTORICAL rows too, not just log a new marker — old "ok" rows from
        # the broken client would otherwise keep inflating tts_calls' ok-count in every
        # future cost/monitor query even though they never held real audio.
        db.mark_tts_call_corrected(item["scenario_id"], new_status="error",
                                   error=f"corrected by 14_cleanup_corrupt_audio.py: "
                                        f"{', '.join(item['reasons'])}")
        db.log_tts_call(
            run_id, item["scenario_id"], 0, 0.0, 0.0, status="cleaned_up_corrupt",
            error=f"removed by 14_cleanup_corrupt_audio.py: {', '.join(item['reasons'])}",
        )
        removed += 1

    db.close()
    print(f"\nremoved {removed} corrupt wav+meta pairs and corrected their historical "
         f"SQLite rows.")
    print("re-run scripts/03_render_user_audio.py — resumability will naturally re-render "
         "exactly these scenarios (their wav no longer exists) with the fixed client.")


if __name__ == "__main__":
    main()
