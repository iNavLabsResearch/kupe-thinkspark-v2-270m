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
from thinkspark.mimi_codec import MimiEncoder, _read_wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_gen.yaml")
    ap.add_argument("--wav-dir", default=None, help="Phase-1 free-audio directory")
    ap.add_argument("--audio-dir", default=None, help="Phase-2 Soniox audio directory")
    ap.add_argument("--out-dir", default="data/encoded")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="clips per GPU forward pass via encoder.encode_batch — the real "
                        "speed fix (one-at-a-time left the GPU mostly idle between tiny "
                        "launches). Drop this if you OOM on a small GPU.")
    ap.add_argument("--io-workers", type=int, default=8,
                    help="threads reading+resampling wav files off disk while the GPU "
                        "encodes the previous batch (overlaps I/O with compute)")
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

    pending = [w for w in wavs if not (out_dir / f"{w.stem}.npz").exists()]
    skipped = len(wavs) - len(pending)

    encoder = MimiEncoder(repo=cfg.mimi_repo, device=args.device)
    print(f"encoding {len(pending)} wavs ({skipped} already done) from {src_dir} "
         f"with {cfg.mimi_repo}, batch_size={args.batch_size}")

    from concurrent.futures import ThreadPoolExecutor

    def _load(wav_path: Path):
        wav, sr = _read_wav(str(wav_path))
        return wav_path, wav, sr

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.io_workers) as pool:
        for i in range(0, len(pending), args.batch_size):
            chunk = pending[i:i + args.batch_size]
            loaded = list(pool.map(_load, chunk))  # I/O overlapped across the chunk
            paths = [p for p, _, _ in loaded]
            waveforms = [w for _, w, _ in loaded]
            rates = [sr for _, _, sr in loaded]
            try:
                encs = encoder.encode_batch(waveforms, rates)
                for wav_path, enc in zip(paths, encs):
                    enc.save(out_dir / f"{wav_path.stem}.npz")
                done += len(chunk)
            except Exception as e:
                print(f"  ! batch at {i} failed ({e}), falling back to one-at-a-time for it")
                for wav_path, wav, sr in loaded:
                    try:
                        enc = encoder.encode_waveform(wav, sr)
                        enc.save(out_dir / f"{wav_path.stem}.npz")
                        done += 1
                    except Exception as e2:
                        print(f"  ! failed {wav_path.name}: {e2}")
                        failed += 1
            print(f"  ... {done}/{len(pending)} encoded (codebook_size={encoder.codebook_size})")

    print(f"done: encoded={done} skipped={skipped} failed={failed} -> {out_dir}")


if __name__ == "__main__":
    main()
