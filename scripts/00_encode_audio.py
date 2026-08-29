#!/usr/bin/env python
"""
Phase 0 (offline) — encode wavs to Mimi cb0 tokens + energy + f0 shards (Section 4.2).

Handles BOTH corpora:
  * Phase-1 free audio (Common Voice / IndicSUPERB / Shrutilipi / FLEURS ...) — pass a
    directory of wavs with `--wav-dir`.
  * Phase-2 Soniox user audio — pass `--audio-dir data/audio` (the .wav files rendered
    by script 03); shards are keyed by scenario_id so the frame builder can find them.

Output: one <name>.npz per wav in `--out-dir` containing cb0/energy/f0. Resumable.

    conda activate llms
    export HF_TOKEN=hf_...        # Mimi is public but token avoids rate limits
    python scripts/00_encode_audio.py --audio-dir data/audio --out-dir data/encoded
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import DataGenConfig
from thinkspark.mimi_codec import MimiEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_gen.yaml")
    ap.add_argument("--wav-dir", default=None, help="Phase-1 free-audio directory")
    ap.add_argument("--audio-dir", default=None, help="Phase-2 Soniox audio directory")
    ap.add_argument("--out-dir", default="data/encoded")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = DataGenConfig.from_yaml(ROOT / args.config)
    src = args.wav_dir or args.audio_dir
    if not src:
        raise SystemExit("pass --wav-dir or --audio-dir")
    src_dir = ROOT / src
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(src_dir.glob("*.wav"))
    if args.limit:
        wavs = wavs[:args.limit]
    if not wavs:
        raise SystemExit(f"no wavs found in {src_dir}")

    encoder = MimiEncoder(repo=cfg.mimi_repo, device=args.device)
    print(f"encoding {len(wavs)} wavs from {src_dir} with {cfg.mimi_repo}")

    done = failed = skipped = 0
    for wav in wavs:
        out_path = out_dir / f"{wav.stem}.npz"
        if out_path.exists():
            skipped += 1
            continue
        try:
            enc = encoder.encode_wav_file(str(wav))
            enc.save(out_path)
            done += 1
        except Exception as e:
            print(f"  ! failed {wav.name}: {e}")
            failed += 1
        if done and done % 50 == 0:
            print(f"  ... {done} encoded (codebook_size={encoder.codebook_size})")

    print(f"done: encoded={done} skipped={skipped} failed={failed} -> {out_dir}")


if __name__ == "__main__":
    main()
