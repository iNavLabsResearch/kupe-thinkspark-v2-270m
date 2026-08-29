#!/usr/bin/env python
"""
Phase 2 — referee fine-tune (Section 9, Phase 2).

Full/LoRA fine-tune on the ~55 h synthetic corpus: per-frame control flag (focal loss)
+ spoken back-channel text (masked CE) + VAP auxiliary, loss L_P2. Start from the
Phase-1 checkpoint with `--init`.

    conda activate llms
    export HF_TOKEN=hf_...
    python scripts/07_train_phase2.py --config configs/train_phase2.yaml \
        --frames "data/frames/*.jsonl" --init artifacts/thinkspark-v2-350m/phase1/final/model.pt

    # 2xT4 DDP:
    torchrun --nproc_per_node=2 scripts/07_train_phase2.py --config configs/train_phase2.yaml \
        --frames "data/frames/*.jsonl"
"""

from __future__ import annotations

import argparse
import glob

import torch

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import TrainConfig
from thinkspark.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_phase2.yaml")
    ap.add_argument("--frames", required=True, help="glob of Phase-2 frame shards")
    ap.add_argument("--init", default=None, help="Phase-1 checkpoint model.pt to warm-start")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(ROOT / args.config)
    cfg.phase = 2
    shards = sorted(glob.glob(args.frames if args.frames.startswith("/")
                              else str(ROOT / args.frames)))
    if not shards:
        raise SystemExit(f"no frame shards matched {args.frames}")

    trainer = Trainer(cfg, shards)
    if args.init:
        state = torch.load(args.init, map_location=trainer.device)
        model = trainer.model.module if trainer.ddp else trainer.model
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"warm-started from {args.init} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    print(f"Phase-2 referee fine-tune on {len(shards)} shards -> {cfg.out_dir}")
    trainer.train()


if __name__ == "__main__":
    main()
