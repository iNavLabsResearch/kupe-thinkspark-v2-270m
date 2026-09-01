#!/usr/bin/env python
"""
Audio-ablation check — does the model ACTUALLY use the audio stream?

A low Phase-1 perplexity is necessary but NOT sufficient evidence that modality
alignment worked. For a Phase-1 record the text context is only a system prompt plus
empty agent/STT markers, so a model can reach a respectable perplexity purely by
learning the prior over the corpus's transcript distribution — never once consulting
the Mimi audio tokens. If that is what happened, Phase 1 did not do its job and Phase 2
inherits a backbone that cannot read audio.

This runs validation TWICE on the same batches:
    intact   — normal inputs
    ablated  — cb0 replaced by a constant token, prosody zeroed (audio carries no info)

and reports the gap. Interpretation:
    large gap (ablated perplexity >> intact)  -> the model genuinely relies on audio  ✓
    little/no gap                             -> the audio stream is being IGNORED    ✗

    python scripts/26_ablate_audio.py --config configs/train_phase1_bigGPU.yaml \
        --ckpt artifacts/.../phase1/runs/<run>/best --frames "data/frames_phase1/*.jsonl"
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import torch

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import TrainConfig
from thinkspark.dataset import ThinkSparkDataset, build_tokenizer, make_collate
from thinkspark.losses import spoken_ce_loss
from torch.utils.data import DataLoader

import importlib
_ev = importlib.import_module("08_evaluate") if False else None
from thinkspark.model import ThinkSparkModel
from thinkspark.trainer import _codebook_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_phase1_bigGPU.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--max-batches", type=int, default=60,
                    help="how many batches to average over (default 60 — plenty)")
    args = ap.parse_args()

    import os
    cfg = TrainConfig.from_yaml(ROOT / args.config)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    from pathlib import Path
    ckpt = Path(args.ckpt if args.ckpt.startswith("/") else ROOT / args.ckpt)
    tok = build_tokenizer(cfg.base_model, hf_token=os.environ.get(cfg.hf_token_env))

    model = ThinkSparkModel(base_model=cfg.base_model, codebook_size=_codebook_size(cfg),
                            vap_horizon=cfg.vap_horizon,
                            hf_token=os.environ.get(cfg.hf_token_env),
                            gradient_checkpointing=False)
    model.resize_token_embeddings(len(tok))
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location=device), strict=False)
    model = model.to(device).eval()

    shards = sorted(glob.glob(args.frames if args.frames.startswith("/")
                              else str(ROOT / args.frames)))
    ds = ThinkSparkDataset(shards, tok, phase=cfg.phase, vap_horizon=cfg.vap_horizon)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=make_collate(tok.pad_token_id, phase=cfg.phase))

    keys = ["text_ids", "text_seg", "text_mask", "cb0", "prosody",
            "agent_state", "audio_mask"]
    use_bf16 = device == "cuda"
    intact, ablated = [], []

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for i, batch in enumerate(loader):
            if i >= args.max_batches:
                break
            inp = {k: batch[k].to(device) for k in keys}
            labels = batch["align_labels"].to(device)

            out = model(**inp)
            intact.append(float(spoken_ce_loss(out.lm_logits, labels).item()))

            # kill every bit of information the audio stream carries
            abl = dict(inp)
            abl["cb0"] = torch.zeros_like(inp["cb0"])
            abl["prosody"] = torch.zeros_like(inp["prosody"])
            out2 = model(**abl)
            ablated.append(float(spoken_ce_loss(out2.lm_logits, labels).item()))

    ce_i, ce_a = float(np.mean(intact)), float(np.mean(ablated))
    ppl_i, ppl_a = float(np.exp(ce_i)), float(np.exp(ce_a))
    print("=" * 62)
    print(f"Audio ablation — {ckpt}")
    print("=" * 62)
    print(f"  intact   CE={ce_i:.4f}   perplexity={ppl_i:8.2f}")
    print(f"  ablated  CE={ce_a:.4f}   perplexity={ppl_a:8.2f}")
    print(f"  delta    CE={ce_a-ce_i:+.4f}  ({100*(ppl_a-ppl_i)/max(ppl_i,1e-9):+.1f}% perplexity)")
    print()
    if ce_a - ce_i < 0.05:
        print("  ✗ THE AUDIO IS BEING IGNORED. Removing it barely changes the loss, so the")
        print("    model is scoring on a text prior alone. Phase-1 modality alignment did")
        print("    NOT happen — more epochs will not fix this.")
    elif ce_a - ce_i < 0.3:
        print("  ! WEAK audio reliance. Some signal is used, but far less than expected for")
        print("    an ASR-style alignment objective.")
    else:
        print("  ✓ The model genuinely relies on the audio stream — alignment is working.")
    print("=" * 62)


if __name__ == "__main__":
    main()
