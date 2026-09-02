"""
Energy-based user-speaking (VAD) track — the REAL target for the VAP head.

Why this module exists
----------------------
The VAP auxiliary is supposed to answer "is the USER speaking in each of the next H
80 ms frames". Until now `thinkspark.dataset` built that target out of the frame
record's `speaking_mask`, which is a completely different thing: `speaking_mask` marks
the handful of frames where the *referee* emits back-channel text (see
`thinkspark.frames.build_frames` step 8 — it is the loss mask for the spoken head).

Real, measured consequence of that mix-up (scripts/08_evaluate.py on the Phase-2 run
20260901-214336):

    Phase 1  VAD diag: true_speak=0.989  pred_speak=0.994  -> VAD-F1 0.995
    Phase 2  VAD diag: true_speak=0.003  pred_speak=0.000  -> VAD-F1 0.000

Phase 1 only looked healthy by accident: its frame builder (scripts/P1_02_build_frames.py)
writes `speaking_mask = all 1s`, so "predict speaking everywhere" is the right answer and
the head trivially scores 0.995. Phase 2's `speaking_mask` is ~0.3% positive, so the head
learned "never speaking", its logits collapsed to strongly negative (the eval's threshold
sweep found the optimum at logit > -3.95), and Phase 2 actively DESTROYED the VAP
behaviour Phase 1 had learned. No amount of extra training steps or loss re-weighting
fixes this — the target itself was wrong.

The fix is to derive the speaking track from the audio we already encoded: every clip's
`.npz` carries per-frame log-RMS `energy` on the same 12.5 Hz grid, so a real VAD track
is recoverable for the ENTIRE existing corpus with no regeneration and no re-render.

Threshold
---------
`thinkspark.frames._vap_targets` used `np.percentile(e, 40)`, which by construction
labels exactly 60% of every clip as speech no matter what the audio contains — a
constant, not a detector. Here we use the standard adaptive scheme instead:

    floor = p5(energy)                    (the clip's noise floor)
    peak  = p95(energy)                   (its speech level)
    on    = floor + hi_frac * (peak - floor)
    off   = floor + lo_frac * (peak - floor)      (hysteresis, off < on)

plus hysteresis and minimum-run smoothing so a single dipped frame mid-word does not
chop one utterance into two. Clips with no dynamic range (peak - floor below
`min_dynamic_range` nats, i.e. constant tone / pure silence / a dead render) are treated
as fully-speaking so a broken encode cannot silently poison the target with all-zeros.
"""

from __future__ import annotations

import numpy as np


def speaking_from_energy(
    energy: np.ndarray,
    hi_frac: float = 0.45,
    lo_frac: float = 0.30,
    min_speech_frames: int = 2,    # 160 ms — shorter "speech" runs are noise blips
    min_gap_frames: int = 3,       # 240 ms — shorter gaps are within-word, not pauses
    min_dynamic_range: float = 1.0,
) -> np.ndarray:
    """Per-frame float32 {0.,1.} 'user is speaking', from log-RMS energy [T]."""
    e = np.asarray(energy, dtype=np.float32).reshape(-1)
    T = e.size
    if T == 0:
        return np.zeros(0, dtype=np.float32)

    floor = float(np.percentile(e, 5))
    peak = float(np.percentile(e, 95))
    if peak - floor < min_dynamic_range:
        # No usable contrast — assume the whole clip is the user's utterance rather than
        # emitting an all-zero target the head would happily collapse onto.
        return np.ones(T, dtype=np.float32)

    on = floor + hi_frac * (peak - floor)
    off = floor + lo_frac * (peak - floor)

    speaking = np.zeros(T, dtype=bool)
    active = False
    for t in range(T):
        if active:
            active = e[t] > off
        else:
            active = e[t] > on
        speaking[t] = active

    speaking = _fill_short_runs(speaking, value=False, min_len=min_gap_frames)
    speaking = _fill_short_runs(speaking, value=True, min_len=min_speech_frames)
    return speaking.astype(np.float32)


def speaking_from_npz(path, num_frames: int | None = None) -> np.ndarray | None:
    """Load a Phase-0 `.npz` and return its speaking track, or None if unavailable.

    Used by `thinkspark.dataset` so ALREADY-BUILT frame shards get a correct VAP target
    without being rebuilt — the audio side of the corpus is untouched by this fix.
    """
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return None
    try:
        with np.load(p) as d:
            if "energy" not in d:
                return None
            energy = d["energy"].astype(np.float32)
    except Exception:
        return None
    sp = speaking_from_energy(energy)
    if num_frames is not None:
        sp = sp[:num_frames]
        if sp.size < num_frames:
            sp = np.concatenate([sp, np.zeros(num_frames - sp.size, dtype=np.float32)])
    return sp


def _fill_short_runs(x: np.ndarray, value: bool, min_len: int) -> np.ndarray:
    """Flip runs of `value` shorter than `min_len` to the opposite value."""
    if min_len <= 1 or x.size == 0:
        return x
    out = x.copy()
    start = 0
    for t in range(1, x.size + 1):
        if t == x.size or x[t] != x[start]:
            if bool(x[start]) == value and (t - start) < min_len:
                out[start:t] = not value
            start = t
    return out
