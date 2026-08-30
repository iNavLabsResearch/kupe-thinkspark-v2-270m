#!/usr/bin/env python
"""
Phase 1 — modality alignment (Section 9, Phase 1).

Teaches Gemma-3-270M that a stream of Mimi cb0 tokens carries language + prosody:
ASR-style text prediction from audio + a VAP auxiliary (is-user-speaking). Uses the
large free open corpora (~400-450 h), LoRA / partial unfreeze, loss L_P1.

Each invocation is a RUN (own id + own checkpoint folder). Checkpoints save every
`save_every` steps + at the end, and — unless --no-push — upload live during training to
`phase1/runs/<run-id>/<tag>/` on the model repo, so an interrupted run resumes from HF.

    conda activate llms
    export HF_TOKEN=hf_...

    # fresh run (auto run-id = timestamp), checkpoints uploaded live to the model repo:
    python scripts/06_train_phase1.py --config configs/train_phase1.yaml \
        --frames "data/frames_phase1/*.jsonl"

    # resume a specific run from its latest checkpoint (local, else pulled from HF):
    python scripts/06_train_phase1.py --config configs/train_phase1.yaml \
        --frames "data/frames_phase1/*.jsonl" --run-id 20260830-101500 --resume

    # force a fresh start / train without uploading:
    python scripts/06_train_phase1.py ... --fresh
    python scripts/06_train_phase1.py ... --no-push

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
from thinkspark.train_runs import add_run_args, wire_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_phase1.yaml")
    ap.add_argument("--frames", required=True, help="glob of Phase-1 frame shards")
    add_run_args(ap, default_repo="anuj-inavlabs/kupe-thinkspark-audio-270m")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(ROOT / args.config)
    cfg.phase = 1
    shards = sorted(glob.glob(str(ROOT / args.frames)) if not args.frames.startswith("/")
                    else glob.glob(args.frames))
    if not shards:
        raise SystemExit(f"no frame shards matched {args.frames}")

    trainer = Trainer(cfg, shards)
    run_id = wire_run(trainer, cfg, args, phase="phase1", root=ROOT)
    print(f"Phase-1 alignment on {len(shards)} shards -> {cfg.out_dir} (run {run_id})")
    trainer.train()
    print(f"\nPhase-1 run {run_id} complete.")


if __name__ == "__main__":
    main()
