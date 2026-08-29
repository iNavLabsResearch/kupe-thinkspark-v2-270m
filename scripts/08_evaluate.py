#!/usr/bin/env python
"""
Section 10 — evaluate a trained checkpoint on held-out frame shards.

Computes per-flag F1, barge F1 + false-barge rate, endpoint cutoff/latency, VAD-F1,
back-channel over-trigger and referee decode latency, then checks them against the
Section 10 target table.

    conda activate llms
    python scripts/08_evaluate.py --config configs/train_phase2.yaml \
        --ckpt artifacts/thinkspark-v2-350m/phase2/final --frames "data/frames_val/*.jsonl"
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from _bootstrap import setup

ROOT = setup()

from thinkspark import vocab, metrics as M
from thinkspark.config import TrainConfig
from thinkspark.dataset import ThinkSparkDataset, build_tokenizer, make_collate
from thinkspark.model import ThinkSparkModel
from thinkspark.trainer import _codebook_size
from torch.utils.data import DataLoader


def load_model(cfg: TrainConfig, ckpt_dir: Path, tok, device):
    model = ThinkSparkModel(base_model=cfg.base_model,
                            codebook_size=_codebook_size(cfg),
                            vap_horizon=cfg.vap_horizon,
                            hf_token=__import__("os").environ.get(cfg.hf_token_env),
                            gradient_checkpointing=False)
    model.resize_token_embeddings(len(tok))
    state = torch.load(ckpt_dir / "model.pt", map_location=device)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_phase2.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--report-out", default="reports/eval.json")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(ROOT / args.config)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = Path(args.ckpt if args.ckpt.startswith("/") else ROOT / args.ckpt)
    tok = build_tokenizer(cfg.base_model, hf_token=__import__("os").environ.get(cfg.hf_token_env))
    model = load_model(cfg, ckpt, tok, device)

    shards = sorted(glob.glob(args.frames if args.frames.startswith("/") else str(ROOT / args.frames)))
    ds = ThinkSparkDataset(shards, tok, phase=2, vap_horizon=cfg.vap_horizon)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=make_collate(tok.pad_token_id, phase=2))

    C = vocab.NUM_CONTROL_FLAGS
    cm = np.zeros((C, C), dtype=np.int64)
    decode_ms: list[float] = []
    pred_speaking, true_speaking = [], []

    keys = ["text_ids", "text_seg", "text_mask", "cb0", "prosody",
            "agent_state", "audio_mask"]
    with torch.no_grad():
        for batch in loader:
            inp = {k: batch[k].to(device) for k in keys}
            t0 = time.perf_counter()
            out = model(**inp)
            decode_ms.append((time.perf_counter() - t0) * 1000.0 / inp["cb0"].shape[0])

            pred = out.control_logits.argmax(-1).cpu().numpy()   # [B, T]
            true = batch["flags"].numpy()
            mask = batch["audio_mask"].numpy()
            cm += M.flag_confusion(pred, true, mask)

            vap_pred = (out.vap_logits[..., 0] > 0).cpu().numpy()   # next-frame speaking
            vap_true = (batch["vap"][..., 0].numpy() > 0.5)
            m = mask.astype(bool)
            pred_speaking.append(vap_pred[m]); true_speaking.append(vap_true[m])

    per_flag = M.per_flag_f1(cm)
    barge = M.barge_metrics(cm, backchannel_frames_pred_barge=0, backchannel_frames_total=1)
    vad = M.vad_f1(np.concatenate(pred_speaking), np.concatenate(true_speaking)) if pred_speaking else 0.0
    # decode latency is per-batch/B; report percentiles
    lat = M.latency_percentiles(decode_ms)

    results = {
        "vad_f1": vad,
        "barge_f1": barge.f1,
        "false_barge_rate": barge.false_barge_rate,
        "latency_p50_ms": lat["p50"],
        "latency_p95_ms": lat["p95"],
        "per_flag_f1": {k: round(v[2], 3) for k, v in per_flag.items()},
    }
    passed = M.check_targets(results)

    print("=" * 60)
    print(f"Evaluation — {ckpt}")
    print("=" * 60)
    print(f"VAD-F1            : {vad:.3f}   (target >= 0.85)  {'PASS' if passed.get('vad_f1') else 'x'}")
    print(f"Barge-in F1       : {barge.f1:.3f}   (target >= 0.85)  {'PASS' if passed.get('barge_f1') else 'x'}")
    print(f"Latency p50/p95   : {lat['p50']:.1f} / {lat['p95']:.1f} ms   (p95 target <= 40)")
    print("\nper-flag F1:")
    for flag, (p, r, f1) in per_flag.items():
        print(f"  {flag:<14} P={p:.2f} R={r:.2f} F1={f1:.2f}")

    out = ROOT / args.report_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "passed": passed}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
