#!/usr/bin/env python
"""
Measure the RELATIVE MAGNITUDE of each input stream entering the backbone.

Gemma scales its token embeddings by sqrt(hidden_size) (HF implements this inside
Gemma3TextScaledWordEmbedding, so calling get_input_embeddings() applies it). The
multi-modal front-end in thinkspark/model.py adds audio_embed / prosody_proj /
state_embed / seg_embed, which are plain nn.Modules initialised at std 0.02 and are NOT
scaled. If that is true, every audio frame enters the residual stream roughly
sqrt(hidden_size) times weaker than a text token — and the model will lean on text.

This prints the mean L2 norm of each stream for a real checkpoint.
    ratio ~1     -> streams are balanced
    ratio >> 1   -> audio is being drowned out by text; that is the bug
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import torch
from _bootstrap import setup
ROOT = setup()
from thinkspark.config import TrainConfig
from thinkspark.dataset import build_tokenizer
from thinkspark.model import ThinkSparkModel
from thinkspark.trainer import _codebook_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_phase1_bigGPU.yaml")
    ap.add_argument("--ckpt", default=None, help="optional trained ckpt dir (else base init)")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(ROOT / args.config)
    tok = build_tokenizer(cfg.base_model, hf_token=os.environ.get(cfg.hf_token_env))
    model = ThinkSparkModel(base_model=cfg.base_model, codebook_size=_codebook_size(cfg),
                            vap_horizon=cfg.vap_horizon,
                            hf_token=os.environ.get(cfg.hf_token_env),
                            gradient_checkpointing=False)
    model.resize_token_embeddings(len(tok))
    if args.ckpt:
        d = Path(args.ckpt if args.ckpt.startswith("/") else ROOT / args.ckpt)
        model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"), strict=False)
    model = model.eval()

    H = model.hidden_size
    print("=" * 62)
    print(f"Input-stream magnitudes   hidden_size={H}  sqrt(H)={H**0.5:.2f}")
    print(f"embed_tokens class: {type(model.embed_tokens).__name__}")
    print("=" * 62)

    # Measure what the BACKBONE ACTUALLY RECEIVES, i.e. the real front-end paths
    # (_text_embeds / _audio_frame_embeds), NOT the raw tables. An earlier version of
    # this script compared model.audio_embed(cb) — the unscaled table — against
    # model.embed_tokens(ids), which Gemma has already scaled, and so kept reporting an
    # imbalance after the scale fix had in fact landed.
    with torch.no_grad():
        ids = torch.randint(0, 1000, (1, 256))
        seg_ids = torch.zeros(1, 256, dtype=torch.long)
        t = model._text_embeds(ids, seg_ids).float()
        cb = torch.randint(0, model.audio_embed.num_embeddings, (1, 256))
        prosody = torch.randn(1, 256, 2)
        state = torch.zeros(1, 256, dtype=torch.long)
        a = model._audio_frame_embeds(cb, prosody, state).float()
        raw_a = model.audio_embed(cb).float()

    def n(x): return float(x.norm(dim=-1).mean())
    nt, na = n(t), n(a)
    print(f"  embed_scale        : {model.embed_scale:10.4f}")
    print(f"  TEXT  stream (_text_embeds)        : {nt:10.4f}")
    print(f"  AUDIO stream (_audio_frame_embeds) : {na:10.4f}")
    print(f"    (raw audio_embed table, unscaled : {n(raw_a):.4f})")
    ratio = nt / max(na, 1e-9)
    print(f"\n  text / audio ratio : {ratio:8.2f}x")
    if ratio > 4:
        print(f"\n  x IMBALANCED. Text enters the residual stream {ratio:.0f}x stronger than")
        print("    audio, so the backbone can largely ignore the audio stream. This is a")
        print("    model-construction bug, not a data or training-length problem.")
    elif ratio > 2:
        print("\n  ! somewhat imbalanced — text is meaningfully stronger than audio.")
    else:
        print("\n  ok — the streams are comparable; look elsewhere for the weak reliance.")
    print("=" * 62)


if __name__ == "__main__":
    main()
