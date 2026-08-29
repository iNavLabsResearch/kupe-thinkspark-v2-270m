#!/usr/bin/env python
"""
Section 11 — drive the live referee over a simulated frame stream.

Loads a trained checkpoint and streams the frames of one encoded utterance (or a frame
record) through StreamingReferee + ReferenceOrchestrator, printing the per-frame
agent-state -> flag (-> spoken) decision log and the decode-latency summary. This is the
loop you wire to your SDK / LiveKit / Pipecat layer in production.

    conda activate llms
    python scripts/09_infer_demo.py --config configs/train_phase2.yaml \
        --ckpt artifacts/thinkspark-v2-350m/phase2/final \
        --encoded data/encoded/<scenario_id>.npz --agent-text "aapka EMI due hai"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import TrainConfig
from thinkspark.dataset import build_tokenizer, DEFAULT_SYSTEM
from thinkspark.model import ThinkSparkModel
from thinkspark.mimi_codec import EncodedAudio
from thinkspark.trainer import _codebook_size
from thinkspark.inference import StreamingReferee, ReferenceOrchestrator, FrameInput
from thinkspark import vocab, metrics as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_phase2.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoded", required=True, help="an .npz shard to stream")
    ap.add_argument("--agent-text", default="")
    ap.add_argument("--start-state", default="TTS_SPEAKING")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(ROOT / args.config)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    import os
    tok = build_tokenizer(cfg.base_model, hf_token=os.environ.get(cfg.hf_token_env))

    model = ThinkSparkModel(base_model=cfg.base_model,
                            codebook_size=_codebook_size(cfg),
                            vap_horizon=cfg.vap_horizon,
                            hf_token=os.environ.get(cfg.hf_token_env),
                            gradient_checkpointing=False)
    model.resize_token_embeddings(len(tok))
    ckpt = Path(args.ckpt if args.ckpt.startswith("/") else ROOT / args.ckpt)
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location=device), strict=False)
    model = model.to(device).eval()

    enc = EncodedAudio.load(ROOT / args.encoded if not args.encoded.startswith("/") else args.encoded)

    referee = StreamingReferee(model, tok, system_prompt=DEFAULT_SYSTEM, device=device)
    referee.set_context(agent_text=args.agent_text, stt_partial="")

    orch = ReferenceOrchestrator(
        referee,
        on_barge_hard=lambda: print("      >> stop_tts + send agent-so-far to LLM"),
        on_barge_soft=lambda: print("      >> duck_tts"),
        on_prefetch=lambda p: print("      >> speculative LLM start"),
        on_commit=lambda: print("      >> play prefetched reply"),
        on_cancel=lambda: print("      >> abort speculative reply"),
        on_turn_end=lambda: print("      >> commit LLM -> TTS"),
        tts_stream=lambda t: print(f"      >> TTS: {t!r}") if t else None,
    )

    # very simple agent-state schedule: play, then go idle after audio ends
    state = args.start_state
    decode = []
    for i in range(enc.num_frames):
        if i > enc.num_frames * 0.6:
            state = "TTS_DONE"
        frame = FrameInput(cb0=int(enc.cb0[i]), energy=float(enc.energy[i]),
                           f0=float(enc.f0[i]), agent_state=state)
        res = orch.handle(frame)
        decode.append(res.decode_ms)

    print("\n--- decision log ---")
    for line in orch.log:
        print(" ", line)
    lat = M.latency_percentiles(decode)
    print(f"\nreferee decode p50/p95 = {lat['p50']:.1f} / {lat['p95']:.1f} ms "
          f"(target p95 <= {vocab.FRAME_MS:.0f} ms frame budget)")


if __name__ == "__main__":
    main()
