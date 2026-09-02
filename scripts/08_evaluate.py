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
from thinkspark.losses import spoken_ce_loss
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
    ap.add_argument("--max-batches", type=int, default=200,
                    help="cap evaluated batches (0 = all). The full Phase-1 shard set is "
                        "~170k records / ~2.7k batches, which runs for many minutes with "
                        "no output; 200 batches is already a tight estimate.")
    ap.add_argument("--tolerance-frames", type=int, default=None,
                    help="±frames collar for the tolerant control metric (default: the "
                        "config's eval_tolerance_frames). 3 frames = 240 ms.")
    ap.add_argument("--ctrl-event-width", type=float, default=0.0,
                    help="widen 1-frame control events in the EVAL targets too. Default 0 "
                        "= score against the raw point labels (the honest, strict view); "
                        "set it to the training value only to check train/eval agreement.")
    ap.add_argument("--wandb-run-id", default=None,
                    help="log these results into an EXISTING W&B run (e.g. the training "
                        "run this checkpoint came from) instead of a new one — pass the "
                        "run id shown in the training logs, e.g. 20260901-185848")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(ROOT / args.config)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = Path(args.ckpt if args.ckpt.startswith("/") else ROOT / args.ckpt)
    tok = build_tokenizer(cfg.base_model, hf_token=__import__("os").environ.get(cfg.hf_token_env))
    model = load_model(cfg, ckpt, tok, device)

    shards = sorted(glob.glob(args.frames if args.frames.startswith("/") else str(ROOT / args.frames)))
    ds = ThinkSparkDataset(shards, tok, phase=cfg.phase, vap_horizon=cfg.vap_horizon,
                           ctrl_event_width=args.ctrl_event_width)
    tol = (args.tolerance_frames if args.tolerance_frames is not None
           else int(getattr(cfg, "eval_tolerance_frames", 3)))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=getattr(cfg, "num_workers", 2),
                        collate_fn=make_collate(tok.pad_token_id, phase=cfg.phase))
    n_batches = len(loader) if not args.max_batches else min(len(loader), args.max_batches)
    print(f"evaluating {n_batches} batches (batch_size={cfg.batch_size})", flush=True)

    C = vocab.NUM_CONTROL_FLAGS
    cm = np.zeros((C, C), dtype=np.int64)
    tol_acc = None
    # False-barge rate = "the model barged on a look-alike back-channel". It used to be
    # hard-coded as 0/1 below, i.e. it ALWAYS reported 0.00 and passed its target no
    # matter what the model did. Count it for real, over the frames of the two behaviours
    # whose whole point is that barging would be WRONG.
    bc_barge_frames = bc_total_frames = 0
    barge_ids = {vocab.CONTROL_FLAG_TO_ID["BARGE_HARD"], vocab.CONTROL_FLAG_TO_ID["BARGE_SOFT"]}
    lookalike = {"barge_lookalike", "overlap_coop", "backchannel"}
    behaviours = [r.get("behaviour", "") for r in ds.records]
    decode_ms: list[float] = []
    pred_speaking, true_speaking = [], []
    vap_logit_pos: list = []   # raw next-frame VAP logits (for the threshold sweep)
    align_ce: list[float] = [] # Phase-1 only: ASR-style CE on user_text

    # Training runs every forward pass under bf16 autocast (thinkspark/trainer.py), which
    # casts activations on the fly — the backbone itself loads bf16 (thinkspark/model.py)
    # but resize_token_embeddings() above creates the new/expanded embedding rows in
    # float32, so without autocast here too, hidden_states entering the bf16 attention
    # layers mismatch dtype (real error: "expected mat1 and mat2 to have the same dtype,
    # but got: float != c10::BFloat16"). Match training's precision exactly.
    use_bf16 = device == "cuda"
    # MUST match thinkspark.trainer._model_inputs. Omitting spoken_ids/spoken_mask
    # builds a sequence with NO spoken tail, so lm_logits covers only [text+audio] while
    # the labels still mark the tail positions; spoken_ce_loss then shifts logits against
    # the wrong labels (the shapes coincidentally match, so nothing raises) and returns a
    # meaningless CE — observed 17.76 for a checkpoint whose true val align was 3.18.
    keys = ["text_ids", "text_seg", "text_mask", "cb0", "prosody",
            "agent_state", "audio_mask", "spoken_ids", "spoken_mask"]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for bi, batch in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            if bi % 20 == 0:
                print(f"  ... {bi}/{n_batches}", flush=True)
            inp = {k: batch[k].to(device) for k in keys if k in batch}
            t0 = time.perf_counter()
            out = model(**inp)
            decode_ms.append((time.perf_counter() - t0) * 1000.0 / inp["cb0"].shape[0])

            mask = batch["audio_mask"].numpy()
            if cfg.phase == 2:
                pred = out.control_logits.argmax(-1).cpu().numpy()   # [B, T]
                true = batch["flags"].numpy()
                cm += M.flag_confusion(pred, true, mask)
                tol_acc = M.accumulate_tolerant_counts(tol_acc, pred, true, mask,
                                                       tolerance_frames=tol)
                # false-barge accounting over the hard-negative behaviours
                lo = bi * cfg.batch_size
                for r in range(pred.shape[0]):
                    idx = lo + r
                    if idx >= len(behaviours) or behaviours[idx] not in lookalike:
                        continue
                    mrow = mask[r].astype(bool)
                    bc_total_frames += int(mrow.sum())
                    bc_barge_frames += int(np.isin(pred[r][mrow], list(barge_ids)).sum())
            else:
                # Phase 1 trains no control head — its alignment metric is the ASR-style
                # CE on user_text (perplexity = e^CE). The control flags would be pure
                # noise here, so they are neither computed nor reported.
                align_ce.append(float(spoken_ce_loss(
                    out.lm_logits, batch["align_labels"].to(device)).item()))

            vap_logit = out.vap_logits[..., 0].float().cpu().numpy()   # next-frame logit
            vap_pred = (vap_logit > 0)                                 # speaking @ prob>0.5
            vap_true = (batch["vap"][..., 0].numpy() > 0.5)
            m = mask.astype(bool)
            pred_speaking.append(vap_pred[m]); true_speaking.append(vap_true[m])
            vap_logit_pos.append(vap_logit[m])

    per_flag = M.per_flag_f1(cm)
    per_flag_tol = M.tolerant_f1_from_counts(tol_acc) if tol_acc is not None else {}
    barge = M.barge_metrics(cm,
                            backchannel_frames_pred_barge=bc_barge_frames,
                            backchannel_frames_total=max(1, bc_total_frames))
    vad = M.vad_f1(np.concatenate(pred_speaking), np.concatenate(true_speaking)) if pred_speaking else 0.0

    # VAD diagnostics — a VAD-F1 of 0.000 is almost always one of two things, and these
    # numbers tell you which: (a) the VAP head is dead / never fires (pred positive rate
    # ~0) so there are zero true positives, or (b) the fixed >0-logit (prob>0.5) threshold
    # is simply miscalibrated for this head. We report the true/pred base rates plus the
    # best F1 over a logit-threshold sweep, so you can see whether the signal exists at all.
    vad_diag = None
    if pred_speaking:
        tp_all = np.concatenate(true_speaking).astype(bool)
        pp_all = np.concatenate(pred_speaking).astype(bool)   # at the default >0 threshold
        best_f1, best_thr = vad, 0.0
        if vap_logit_pos:
            logits_all = np.concatenate(vap_logit_pos)
            sweep_f1, sweep_thr = M.best_threshold_vad(logits_all, tp_all)
            if sweep_f1 > best_f1:
                best_f1, best_thr = sweep_f1, sweep_thr
            # Persist the calibrated operating point beside the checkpoint so inference
            # uses it instead of the hard-coded prob>0.5 (= logit>0) that this script's
            # headline VAD-F1 assumes. A head trained on an imbalanced target is
            # calibrated to its own base rate; 0 is only correct by luck.
            try:
                (ckpt / "vad_threshold.json").write_text(json.dumps(
                    {"vad_logit_threshold": round(float(best_thr), 4),
                     "f1_at_threshold": round(float(best_f1), 4),
                     "f1_at_zero": round(float(vad), 4)}, indent=2), encoding="utf-8")
            except OSError:
                pass
        vad_diag = {
            "true_speaking_rate": float(tp_all.mean()),
            "pred_speaking_rate": float(pp_all.mean()),
            "best_threshold_f1": round(float(best_f1), 3),
            "best_threshold": round(float(best_thr), 3),
        }
    # decode latency is per-batch/B; report percentiles
    lat = M.latency_percentiles(decode_ms)

    results = {
        "phase": cfg.phase,
        "vad_f1": vad,
        "latency_p50_ms": lat["p50"],
        "latency_p95_ms": lat["p95"],
        "vad_diag": vad_diag,
    }
    if cfg.phase == 2:
        results["barge_f1"] = barge.f1
        results["false_barge_rate"] = barge.false_barge_rate
        results["per_flag_f1"] = {k: round(v[2], 3) for k, v in per_flag.items()}
        results["ctrl_macro_f1"] = round(
            float(np.mean([f1 for _, _, f1 in per_flag.values()])), 3)
        if per_flag_tol:
            results["tolerance_frames"] = tol
            results["per_flag_f1_tol"] = {k: round(v[2], 3) for k, v in per_flag_tol.items()}
            results["ctrl_macro_f1_tol"] = round(
                float(np.mean([f1 for _, _, f1 in per_flag_tol.values()])), 3)
            bt = per_flag_tol["BARGE_HARD"][2], per_flag_tol["BARGE_SOFT"][2]
            results["barge_f1_tol"] = round(float(np.mean(bt)), 3)
    else:
        # Phase-1 alignment quality. Barge / per-flag are deliberately ABSENT: the
        # control head is untrained in Phase 1, so reporting them would be noise that
        # spuriously "fails" the Section 10 targets.
        ce = float(np.mean(align_ce)) if align_ce else float("nan")
        results["align_ce"] = round(ce, 4)
        results["align_perplexity"] = round(float(np.exp(ce)), 3) if align_ce else None
    passed = M.check_targets(results)

    print("=" * 60)
    print(f"Evaluation — {ckpt}")
    print("=" * 60)
    print(f"Phase             : {cfg.phase}")
    print(f"VAD-F1            : {vad:.3f}   (target >= 0.85)  {'PASS' if passed.get('vad_f1') else 'x'}")
    if cfg.phase == 2:
        print(f"Barge-in F1       : {barge.f1:.3f}   (target >= 0.85)  {'PASS' if passed.get('barge_f1') else 'x'}")
        print(f"Ctrl macro-F1     : {results['ctrl_macro_f1']:.3f}   (exact-frame)")
        if per_flag_tol:
            print(f"Ctrl macro-F1 ±{tol}f : {results['ctrl_macro_f1_tol']:.3f}   "
                  f"(±{tol * 80} ms collar — the operational number)")
        print(f"False-barge rate  : {barge.false_barge_rate:.3f}   (target <= 0.05; over "
              f"{bc_total_frames} frames of look-alike/cooperative behaviours)")
    else:
        print(f"Align CE          : {results['align_ce']}   "
              f"(perplexity {results['align_perplexity']})")
    print(f"Latency p50/p95   : {lat['p50']:.1f} / {lat['p95']:.1f} ms   (p95 target <= 40)")
    if vad_diag is not None:
        print(f"VAD diag          : true_speak={vad_diag['true_speaking_rate']:.3f} "
              f"pred_speak={vad_diag['pred_speaking_rate']:.3f} "
              f"| best-threshold F1={vad_diag['best_threshold_f1']:.3f} "
              f"@ logit>{vad_diag['best_threshold']:.2f}")
        if vad_diag["pred_speaking_rate"] < 1e-4 and vad_diag["true_speaking_rate"] > 0.05:
            print("  → VAP head never predicts 'speaking' at the default threshold: the "
                  "head is under-trained (raise lambda_vap_p2) or the >0-logit threshold "
                  "is miscalibrated (use the best-threshold F1 above).")
    if cfg.phase == 2:
        print(f"\nper-flag F1   (exact-frame | ±{tol}-frame collar):")
        for flag, (p, r, f1) in per_flag.items():
            if per_flag_tol:
                pt, rt, ft = per_flag_tol[flag]
                print(f"  {flag:<14} P={p:.2f} R={r:.2f} F1={f1:.2f}   |   "
                      f"P={pt:.2f} R={rt:.2f} F1={ft:.2f}")
            else:
                print(f"  {flag:<14} P={p:.2f} R={r:.2f} F1={f1:.2f}")

    out = ROOT / args.report_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "passed": passed}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if args.wandb_run_id:
        import os as _os
        if _os.environ.get("WANDB_API_KEY") and cfg.wandb_project:
            import wandb
            run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                             id=args.wandb_run_id, resume="allow")
            run.log({f"eval/{k}": v for k, v in results.items() if not isinstance(v, dict)})
            if cfg.phase == 2:
                run.log({f"eval/per_flag_f1/{flag}": f1
                         for flag, (_, _, f1) in per_flag.items()})
            run.finish()
            print(f"logged eval/* metrics into W&B run {args.wandb_run_id}")
        else:
            print("! --wandb-run-id given but WANDB_API_KEY not set or cfg.wandb_project "
                 "is null — skipped W&B logging")


if __name__ == "__main__":
    main()
