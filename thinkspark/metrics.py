"""
Evaluation metrics (Section 10).

Targets from the guide:
    Phase 1        VAD-F1 >= 0.85
    Barge-in       F1 >= 0.85, false-barge <= 5%
    Endpointing    cutoff <= 3%, endpoint latency p95 <= 300 ms
    Back-channel   over-trigger <= 8% (naturalness by LLM-judge >= 4.2)
    Prefetch       useful >= 70%
    Silence-break  trigger precision >= 0.9
    Latency        referee decode p95 <= 40 ms

Definitions (Section 10):
    P = TP/(TP+FP),  R = TP/(TP+FN),  F1 = 2PR/(P+R)
    cutoff rate = (#turns ended while user not done) / #turns
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thinkspark import vocab


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# --------------------------------------------------------------------------- #
def flag_confusion(pred_flags: np.ndarray, true_flags: np.ndarray,
                   mask: np.ndarray | None = None) -> np.ndarray:
    """[C, C] confusion matrix over control flags (rows=true, cols=pred)."""
    C = vocab.NUM_CONTROL_FLAGS
    cm = np.zeros((C, C), dtype=np.int64)
    p = pred_flags.reshape(-1)
    t = true_flags.reshape(-1)
    if mask is not None:
        m = mask.reshape(-1).astype(bool)
        p, t = p[m], t[m]
    for ti, pi in zip(t, p):
        cm[ti, pi] += 1
    return cm


def per_flag_f1(cm: np.ndarray) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for flag, i in vocab.CONTROL_FLAG_TO_ID.items():
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        out[flag] = precision_recall_f1(tp, fp, fn)
    return out


# --------------------------------------------------------------------------- #
@dataclass
class BargeMetrics:
    f1: float
    false_barge_rate: float   # barging on a look-alike back-channel


def barge_metrics(cm: np.ndarray, backchannel_frames_pred_barge: int,
                  backchannel_frames_total: int) -> BargeMetrics:
    """Barge = BARGE_HARD or BARGE_SOFT (merged)."""
    idx = [vocab.CONTROL_FLAG_TO_ID["BARGE_HARD"], vocab.CONTROL_FLAG_TO_ID["BARGE_SOFT"]]
    tp = int(sum(cm[i, j] for i in idx for j in idx))
    fp = int(sum(cm[:, j].sum() for j in idx) - tp)
    fn = int(sum(cm[i, :].sum() for i in idx) - tp)
    _, _, f1 = precision_recall_f1(tp, fp, fn)
    fbr = (backchannel_frames_pred_barge / backchannel_frames_total
           if backchannel_frames_total else 0.0)
    return BargeMetrics(f1=f1, false_barge_rate=fbr)


def cutoff_rate(turns_ended_early: int, total_turns: int) -> float:
    return turns_ended_early / total_turns if total_turns else 0.0


def endpoint_latency_ms(pred_turn_end_frame: list[int],
                        true_turn_end_frame: list[int]) -> dict[str, float]:
    """p50/p95 of |pred - true| endpoint offset, in ms."""
    diffs = [abs(p - t) * vocab.FRAME_MS for p, t in zip(pred_turn_end_frame, true_turn_end_frame)]
    if not diffs:
        return {"p50": 0.0, "p95": 0.0}
    arr = np.array(diffs)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95))}


def vad_f1(pred_speaking: np.ndarray, true_speaking: np.ndarray) -> float:
    p = pred_speaking.reshape(-1).astype(bool)
    t = true_speaking.reshape(-1).astype(bool)
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    return precision_recall_f1(tp, fp, fn)[2]


def backchannel_over_trigger(pred_spoke: int, opportunities: int) -> float:
    """Fraction of silent-opportunity frames where the model wrongly spoke."""
    return pred_spoke / opportunities if opportunities else 0.0


def prefetch_useful_rate(useful: int, total_prefetch: int) -> float:
    return useful / total_prefetch if total_prefetch else 0.0


def latency_percentiles(decode_ms: list[float]) -> dict[str, float]:
    if not decode_ms:
        return {"p50": 0.0, "p95": 0.0}
    arr = np.array(decode_ms)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95))}


# --------------------------------------------------------------------------- #
TARGETS = {
    "vad_f1": (">=", 0.85),
    "barge_f1": (">=", 0.85),
    "false_barge_rate": ("<=", 0.05),
    "cutoff_rate": ("<=", 0.03),
    "endpoint_p95_ms": ("<=", 300.0),
    "backchannel_over_trigger": ("<=", 0.08),
    "prefetch_useful": (">=", 0.70),
    "silence_break_precision": (">=", 0.90),
    "latency_p95_ms": ("<=", 40.0),
    "naturalness_mean": (">=", 4.2),
}


def check_targets(results: dict[str, float]) -> dict[str, bool]:
    passed: dict[str, bool] = {}
    for key, (op, bar) in TARGETS.items():
        if key not in results:
            continue
        v = results[key]
        passed[key] = (v >= bar) if op == ">=" else (v <= bar)
    return passed
