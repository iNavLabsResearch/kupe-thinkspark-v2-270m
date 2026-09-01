#!/usr/bin/env python
"""
Continuous LIVE microphone inference (Section 11, real hardware version of
scripts/09_infer_demo.py's simulated-file loop).

Downloads a run's checkpoint from the HF model repo if it isn't already local, opens the
default microphone, encodes real-time audio into Mimi cb0/energy/f0 in 80 ms-frame-sized
chunks, and streams each frame through StreamingReferee + ReferenceOrchestrator exactly
like production would. Every decision is printed live AND appended to a JSONL log file
(one line per frame: timestamp, flag, spoken text if any, decode latency).

Not a KV-cache production stream (see docs/commands/inference.mdx's frame-budget
section) — this re-encodes a short trailing window each chunk, same reference-quality
tradeoff as scripts/09_infer_demo.py, just fed from a real mic instead of a pre-rendered
.npz file.

    conda activate llms
    pip install sounddevice   # only extra dep beyond training requirements

    export HF_TOKEN=hf_...
    python scripts/24_infer_mic.py --run-id 20260901-185848 --tag step1500

    # or point at an already-local checkpoint directly, skips the HF download:
    python scripts/24_infer_mic.py --ckpt artifacts/thinkspark-v2-350m/phase2/runs/20260901-185848/step1500

Stop with Ctrl+C — the log file is flushed after every frame, so nothing is lost.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import time
from pathlib import Path

import numpy as np
import torch

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import TrainConfig, env
from thinkspark.dataset import build_tokenizer, DEFAULT_SYSTEM
from thinkspark.model import ThinkSparkModel
from thinkspark.mimi_codec import MimiEncoder, _MIMI_RATE, _HOP
from thinkspark.trainer import _codebook_size
from thinkspark.inference import StreamingReferee, ReferenceOrchestrator, FrameInput

DEFAULT_MODEL_REPO = "anuj-inavlabs/kupe-thinkspark-audio-270m"
CHUNK_FRAMES = 5   # mic audio is buffered/encoded in groups of 5 frames (400 ms) at a
                   # time — smaller = lower latency but more per-chunk model-load
                   # overhead; 5 is a reasonable default for a demo/logging tool.


def _resolve_ckpt(args) -> Path:
    if args.ckpt:
        ckpt = Path(args.ckpt if args.ckpt.startswith("/") else ROOT / args.ckpt)
        if not (ckpt / "model.pt").exists():
            raise SystemExit(f"no model.pt under {ckpt}")
        return ckpt

    if not args.run_id:
        raise SystemExit("pass --ckpt <local dir> OR --run-id <run> (+ optional --tag)")

    from thinkspark.hf_upload import download_run_checkpoint
    token = env("HF_TOKEN")
    dest = ROOT / "artifacts/thinkspark-v2-350m" / args.phase / "runs" / args.run_id
    print(f"resolving checkpoint: {args.repo} {args.phase}/runs/{args.run_id} "
         f"(tag={args.tag or 'latest'}) ...")
    ckpt = download_run_checkpoint(args.repo, args.phase, args.run_id, dest, token, tag=args.tag)
    if ckpt is None:
        raise SystemExit(f"no checkpoint found for run {args.run_id} on {args.repo} — "
                         f"check --repo/--phase/--run-id/--tag")
    print(f"using checkpoint: {ckpt}")
    return ckpt


def _load_model(cfg: TrainConfig, ckpt_dir: Path, tok, device):
    model = ThinkSparkModel(base_model=cfg.base_model,
                            codebook_size=_codebook_size(cfg),
                            vap_horizon=cfg.vap_horizon,
                            hf_token=os.environ.get(cfg.hf_token_env),
                            gradient_checkpointing=False)
    model.resize_token_embeddings(len(tok))
    state = torch.load(ckpt_dir / "model.pt", map_location=device)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_phase2_bigGPU.yaml")
    ap.add_argument("--ckpt", default=None, help="local checkpoint dir — skips HF download")
    ap.add_argument("--repo", default=DEFAULT_MODEL_REPO)
    ap.add_argument("--phase", default="phase2")
    ap.add_argument("--run-id", default=None, help="e.g. 20260901-185848")
    ap.add_argument("--tag", default=None, help="e.g. step1500 (default: latest for the run)")
    ap.add_argument("--agent-text", default="", help="what the agent is currently saying (context)")
    ap.add_argument("--start-state", default="IDLE",
                    help="IDLE|LLM_GEN|TTS_SPEAKING|TTS_DONE — your real orchestrator "
                        "would drive this from its own state machine; fixed here since "
                        "this is a standalone listening demo")
    ap.add_argument("--mic-device", default=None, help="sounddevice input index/name (default: system default mic)")
    ap.add_argument("--log-out", default="reports/mic_session.jsonl")
    args = ap.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("`pip install sounddevice` first (PortAudio-based mic capture).")

    cfg = TrainConfig.from_yaml(ROOT / args.config)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    ckpt_dir = _resolve_ckpt(args)
    tok = build_tokenizer(cfg.base_model, hf_token=os.environ.get(cfg.hf_token_env))
    model = _load_model(cfg, ckpt_dir, tok, device)
    encoder = MimiEncoder(repo=cfg.mimi_repo, device=device)

    referee = StreamingReferee(model, tok, system_prompt=DEFAULT_SYSTEM, device=device)
    referee.set_context(agent_text=args.agent_text, stt_partial="")

    log_path = ROOT / args.log_out
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("a", encoding="utf-8")
    print(f"logging every frame's decision to {log_path}")

    def _log(flag: str, spoken: str, decode_ms: float):
        rec = {"ts": time.time(), "flag": flag, "spoken": spoken, "decode_ms": round(decode_ms, 2)}
        log_fh.write(json.dumps(rec) + "\n")
        log_fh.flush()

    orch = ReferenceOrchestrator(
        referee,
        on_barge_hard=lambda: print("      >> BARGE_HARD: stop_tts + send agent-so-far to LLM"),
        on_barge_soft=lambda: print("      >> BARGE_SOFT: duck_tts"),
        on_prefetch=lambda p: print(f"      >> PREFETCH_LLM: speculative start ({p})"),
        on_commit=lambda: print("      >> COMMIT_LLM: play prefetched reply"),
        on_cancel=lambda: print("      >> CANCEL_LLM: abort speculative reply"),
        on_turn_end=lambda: print("      >> TURN_END: commit LLM -> TTS"),
        tts_stream=lambda t: print(f"      >> TTS: {t!r}") if t else None,
    )

    hop_samples = _HOP                 # samples per 80 ms frame at 24kHz
    chunk_samples = hop_samples * CHUNK_FRAMES
    audio_q: queue.Queue = queue.Queue()

    def _callback(indata, frames_, time_info, status):
        if status:
            print(f"  ! mic status: {status}")
        audio_q.put(indata[:, 0].copy())   # mono

    print(f"opening microphone (device={args.mic_device or 'default'}, "
         f"{_MIMI_RATE} Hz, {CHUNK_FRAMES} frames / {chunk_samples / _MIMI_RATE * 1000:.0f} ms chunks)")
    print("listening — Ctrl+C to stop\n")

    state = args.start_state
    buf = np.zeros(0, dtype=np.float32)
    n_frames = 0

    try:
        with sd.InputStream(samplerate=_MIMI_RATE, channels=1, dtype="float32",
                            blocksize=chunk_samples, device=args.mic_device,
                            callback=_callback):
            while True:
                chunk = audio_q.get()
                buf = np.concatenate([buf, chunk])
                while len(buf) >= chunk_samples:
                    piece, buf = buf[:chunk_samples], buf[chunk_samples:]
                    enc = encoder.encode_waveform(piece, _MIMI_RATE)
                    for i in range(enc.num_frames):
                        frame = FrameInput(cb0=int(enc.cb0[i]), energy=float(enc.energy[i]),
                                           f0=float(enc.f0[i]), agent_state=state)
                        res = orch.handle(frame)
                        n_frames += 1
                        if res.flag not in ("LISTEN", "HOLD"):   # quieter default log; every
                            print(f"  frame {n_frames:>6} [{state}] -> {res.flag}"
                                 + (f"  spoken={res.spoken!r}" if res.spoken else ""))
                        _log(res.flag, res.spoken, res.decode_ms)
    except KeyboardInterrupt:
        print(f"\nstopped after {n_frames} frames. log: {log_path}")
    finally:
        log_fh.close()


if __name__ == "__main__":
    main()
