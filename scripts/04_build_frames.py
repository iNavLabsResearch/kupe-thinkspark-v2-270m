#!/usr/bin/env python
"""
Section 5.3 / 8.4 step 3 — turn scenarios (+ Soniox timings + Mimi shards) into per-frame
training records.

For each scenario:
  * load its Soniox word timings (data/audio/<id>.words.json) if present
  * load its Mimi shard (data/encoded/<id>.npz) if present
  * call thinkspark.frames.build_frames to get the per-frame flag / agent-state / VAP /
    spoken-span labels, calibrated to real audio
  * write one JSON record to the output frame shard

The result shards are exactly what thinkspark.dataset.ThinkSparkDataset consumes.

    conda activate llms
    python scripts/04_build_frames.py --in data/scenarios/scenarios_part00.jsonl \
        --frames-out data/frames/frames_part00.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import TrainConfig, DataGenConfig
from thinkspark.schema import Scenario
from thinkspark.frames import build_frames
from thinkspark.tts_soniox import TTSResult, WordSpan
from thinkspark.mimi_codec import EncodedAudio


def _load_tts(meta_path: Path) -> TTSResult | None:
    if not meta_path.exists():
        return None
    d = json.loads(meta_path.read_text(encoding="utf-8"))
    words = [WordSpan(**w) for w in d.get("words", [])]
    return TTSResult(sample_rate=d["sample_rate"], duration_s=d["duration_s"], words=words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_gen.yaml")
    ap.add_argument("--in", dest="in_path", required=True, help="scenarios JSONL")
    ap.add_argument("--audio-dir", default="data/audio")
    ap.add_argument("--encoded-dir", default="data/encoded")
    ap.add_argument("--frames-out", required=True)
    ap.add_argument("--vap-horizon", type=int, default=25)
    args = ap.parse_args()

    audio_dir = ROOT / args.audio_dir
    enc_dir = ROOT / args.encoded_dir
    out_path = ROOT / args.frames_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [l for l in Path(ROOT / args.in_path).read_text(encoding="utf-8").splitlines() if l.strip()]

    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for line in lines:
            s = Scenario.from_dict(json.loads(line))
            sid = s.scenario_id or "unknown"
            tts = _load_tts(audio_dir / f"{sid}.words.json")
            enc_path = enc_dir / f"{sid}.npz"
            encoded = EncodedAudio.load(enc_path) if enc_path.exists() else None

            fl = build_frames(s, tts=tts, encoded=encoded, vap_horizon=args.vap_horizon)
            rec = fl.to_record(s, encoded_path=str(enc_path) if encoded else None)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"  ... {written} frame records")

    print(f"done: {written} frame records -> {out_path}")


if __name__ == "__main__":
    main()
