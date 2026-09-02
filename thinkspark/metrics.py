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


def _dilate(x: np.ndarray, tol: int) -> np.ndarray:
    """Binary dilation of a [B, T] mask by `tol` frames on each side (per row)."""
    if tol <= 0:
        return x
    out = x.copy()
    for k in range(1, tol + 1):
        out[:, k:] |= x[:, :-k]
        out[:, :-k] |= x[:, k:]
    return out


def tolerant_per_flag_f1(pred_flags: np.ndarray, true_flags: np.ndarray,
                         mask: np.ndarray | None = None,
                         tolerance_frames: int = 3
                         ) -> dict[str, tuple[float, float, float]]:
    """Per-flag P/R/F1 with a ±`tolerance_frames` collar (the standard VAD/diarization
    scoring convention), computed on [B, T] frame grids.

    `per_flag_f1` above scores exact-frame agreement. For the SUSTAINED flags (LISTEN,
    HOLD, INCOMPLETE, CONTINUE) that is the right metric — they span many frames. For the
    point events (TURN_END, BARGE_*, COMMIT/CANCEL/PREFETCH_LLM, SILENCE_BREAK) it is
    close to meaningless: those labels occupy ONE 80 ms frame, so a referee that fires
    80 ms early is scored as both a false positive and a false negative and earns F1=0
    for a decision that is entirely correct in a live call. That single scoring artifact
    is why run 20260901-214336 reported TURN_END 0.15 / BARGE_SOFT 0.11 / PREFETCH 0.08
    while every sustained flag sat at 0.87+.

    A ±3 frame collar = ±240 ms, comfortably inside the Section 10 endpoint-latency
    budget of 300 ms p95, so a hit inside the collar is a genuine hit, not leniency.
    Report BOTH numbers: exact-frame is the strict view, tolerant is the operational one.
    """
    C = vocab.NUM_CONTROL_FLAGS
    p = np.atleast_2d(pred_flags)
    t = np.atleast_2d(true_flags)
    m = (np.atleast_2d(mask).astype(bool) if mask is not None
         else np.ones_like(p, dtype=bool))
    out: dict[str, tuple[float, float, float]] = {}
    for flag, i in vocab.CONTROL_FLAG_TO_ID.items():
        pi = (p == i) & m
        ti = (t == i) & m
        pi_d = _dilate(pi, tolerance_frames) & m
        ti_d = _dilate(ti, tolerance_frames) & m
        tp_r = int((ti & pi_d).sum())          # true frames matched by a nearby pred
        fn = int(ti.sum()) - tp_r
        tp_p = int((pi & ti_d).sum())          # pred frames justified by a nearby true
        fp = int(pi.sum()) - tp_p
        prec = tp_p / (tp_p + fp) if (tp_p + fp) else 0.0
        rec = tp_r / (tp_r + fn) if (tp_r + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[flag] = (prec, rec, f1)
    return out


def accumulate_tolerant_counts(acc: dict[str, np.ndarray] | None,
                               pred_flags: np.ndarray, true_flags: np.ndarray,
                               mask: np.ndarray | None = None,
                               tolerance_frames: int = 3) -> dict[str, np.ndarray]:
    """Streaming version of `tolerant_per_flag_f1` — accumulate raw counts batch by batch
    (a collar cannot be recovered from a summed confusion matrix, so the counts have to
    be tallied while each batch's frame grid is still in hand). Finish with
    `tolerant_f1_from_counts`."""
    C = vocab.NUM_CONTROL_FLAGS
    if acc is None:
        acc = {k: np.zeros(C, dtype=np.int64) for k in ("tp_r", "fn", "tp_p", "fp")}
    p = np.atleast_2d(pred_flags)
    t = np.atleast_2d(true_flags)
    m = (np.atleast_2d(mask).astype(bool) if mask is not None
         else np.ones_like(p, dtype=bool))
    for i in range(C):
        pi = (p == i) & m
        ti = (t == i) & m
        pi_d = _dilate(pi, tolerance_frames) & m
        ti_d = _dilate(ti, tolerance_frames) & m
        tp_r = int((ti & pi_d).sum())
        tp_p = int((pi & ti_d).sum())
        acc["tp_r"][i] += tp_r
        acc["fn"][i] += int(ti.sum()) - tp_r
        acc["tp_p"][i] += tp_p
        acc["fp"][i] += int(pi.sum()) - tp_p
    return acc


def tolerant_f1_from_counts(acc: dict[str, np.ndarray]
                            ) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for flag, i in vocab.CONTROL_FLAG_TO_ID.items():
        tp_p, fp = int(acc["tp_p"][i]), int(acc["fp"][i])
        tp_r, fn = int(acc["tp_r"][i]), int(acc["fn"][i])
        prec = tp_p / (tp_p + fp) if (tp_p + fp) else 0.0
        rec = tp_r / (tp_r + fn) if (tp_r + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[flag] = (prec, rec, f1)
    return out


def best_threshold_vad(logits: np.ndarray, true_speaking: np.ndarray,
                       n_steps: int = 41) -> tuple[float, float]:
    """(best_f1, best_threshold) over a quantile sweep of the VAP next-frame logit.

    The referee ships with whatever threshold this returns, not a hard-coded 0: a head
    trained on an imbalanced target is calibrated to its own base rate, and forcing
    prob>0.5 throws away a working detector. Save the threshold beside the checkpoint.
    """
    lg = np.asarray(logits).reshape(-1)
    tp = np.asarray(true_speaking).reshape(-1).astype(bool)
    if lg.size == 0:
        return 0.0, 0.0
    best_f1, best_thr = 0.0, 0.0
    for thr in np.unique(np.quantile(lg, np.linspace(0.01, 0.99, n_steps))):
        f1 = vad_f1(lg > thr, tp)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_f1, best_thr


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
