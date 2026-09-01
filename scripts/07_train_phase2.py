#!/usr/bin/env python
"""
Phase 2 — referee fine-tune (Section 9, Phase 2).

Full/LoRA fine-tune on the ~55 h synthetic corpus: per-frame control flag (focal loss)
+ spoken back-channel text (masked CE) + VAP auxiliary, loss L_P2. Start from the
Phase-1 checkpoint with `--init`.

Same run/resume/live-upload behavior as Phase 1 (checkpoints -> phase2/runs/<run-id>/ on
the model repo). --init warm-starts from a Phase-1 checkpoint, but is ignored when
--resume restores this Phase-2 run's own weights.

    conda activate llms
    export HF_TOKEN=hf_...

    # fresh run, warm-started from Phase-1, checkpoints uploaded live:
    python scripts/07_train_phase2.py --config configs/train_phase2.yaml \
        --frames "data/frames/*.jsonl" \
        --init artifacts/thinkspark-v2-350m/phase1/runs/<run-id>/final/model.pt

    # resume a specific Phase-2 run:
    python scripts/07_train_phase2.py --config configs/train_phase2.yaml \
        --frames "data/frames/*.jsonl" --run-id 20260830-140000 --resume

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
from thinkspark.train_runs import add_run_args, wire_run
from thinkspark.warmstart import remap_phase1_state_dict, report_load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_phase2.yaml")
    ap.add_argument("--frames", required=True, help="glob of Phase-2 frame shards")
    ap.add_argument("--init", default=None, help="Phase-1 checkpoint model.pt to warm-start")
    add_run_args(ap, default_repo="anuj-inavlabs/kupe-thinkspark-audio-270m")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(ROOT / args.config)
    cfg.phase = 2
    shards = sorted(glob.glob(args.frames if args.frames.startswith("/")
                              else str(ROOT / args.frames)))
    if not shards:
        raise SystemExit(f"no frame shards matched {args.frames}")

    trainer = Trainer(cfg, shards)
    run_id = wire_run(trainer, cfg, args, phase="phase2", root=ROOT)

    # --init warm-starts from a PHASE-1 checkpoint — but only when NOT resuming this
    # phase-2 run (a resume already restored the phase-2 weights; re-applying phase-1
    # weights on top would clobber the resumed progress).
    resumed = trainer._start_step > 0 or trainer._start_epoch > 0
    if args.init and not resumed:
        state = torch.load(args.init, map_location=trainer.device)
        model = trainer.model.module if trainer.ddp else trainer.model
        # A LoRA-trained Phase-1 checkpoint has PEFT-wrapped keys that match NOTHING in a
        # full-finetune Phase-2 model; strict=False used to swallow that and train from
        # base Gemma. Remap (merging any LoRA deltas) and then verify the load loudly.
        state = remap_phase1_state_dict(state, lora_alpha=cfg.lora_alpha,
                                        lora_r=cfg.lora_r)
        missing, unexpected = model.load_state_dict(state, strict=False)
        report_load(missing, unexpected, len(model.state_dict()), source=args.init)
    elif args.init and resumed:
        print(f"ignoring --init (resuming run {run_id} from its own checkpoint instead)")

    print(f"Phase-2 referee fine-tune on {len(shards)} shards -> {cfg.out_dir} (run {run_id})")
    trainer.train()
    print(f"\nPhase-2 run {run_id} complete.")


if __name__ == "__main__":
    main()
