#!/usr/bin/env python
"""
Phase 1 — modality alignment (Section 9, Phase 1).

Teaches Gemma-3-270M that a stream of Mimi cb0 tokens carries language + prosody:
ASR-style text prediction from audio + a VAP auxiliary (is-user-speaking). Uses the
large free open corpora (~400-450 h), LoRA / partial unfreeze, loss L_P1.

    conda activate llms
    export HF_TOKEN=hf_...
    python scripts/06_train_phase1.py --config configs/train_phase1.yaml \
        --frames "data/frames_phase1/*.jsonl"

    # 2xT4 DDP:
    torchrun --nproc_per_node=2 scripts/06_train_phase1.py --config configs/train_phase1.yaml \
        --frames "data/frames_phase1/*.jsonl"
"""

from __future__ import annotations

import argparse
import glob

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import TrainConfig
from thinkspark.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_phase1.yaml")
    ap.add_argument("--frames", required=True, help="glob of Phase-1 frame shards")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(ROOT / args.config)
    cfg.phase = 1
    shards = sorted(glob.glob(str(ROOT / args.frames)) if not args.frames.startswith("/")
                    else glob.glob(args.frames))
    if not shards:
        raise SystemExit(f"no frame shards matched {args.frames}")
    print(f"Phase-1 alignment on {len(shards)} shards -> {cfg.out_dir}")
    Trainer(cfg, shards).train()


if __name__ == "__main__":
    main()
