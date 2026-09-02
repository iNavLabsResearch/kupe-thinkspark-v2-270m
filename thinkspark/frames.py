"""
Scenario timeline -> per-frame training labels (Section 5.3, 8.4, 9.1).

Given a validated `Scenario` and (optionally) the Soniox `TTSResult` + Mimi `EncodedAudio`
for its user line, produce the exact per-80 ms-frame supervision the two heads need:

    flags        int64  [T]     control-flag id per frame (control head target)
    agent_state  int64  [T]     agent-state channel (a model INPUT, state machine below)
    vap          float32[T, H]  "is user speaking" in each of the next H frames (VAP aux)
    speaking_mask bool  [T]     frames whose flag carries spoken_text (loss mask for txt)
    vad          float32[T]     "user is speaking" per frame (source of the VAP target)
    spoken_spans list[(frame, text)]  spoken-head targets to tokenize

Timing calibration
------------------
The LLM's `frame_offset`s are relative hints. When a `TTSResult` is available we anchor
the scenario's *primary* event to the real audio via `event_char -> seconds -> frame`,
and shift every other event by the same delta so the whole timeline is frame-accurate.
Without a TTSResult we fall back to the LLM offsets as-is (used by the sample builder).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from thinkspark import vocab
from thinkspark.schema import Scenario
from thinkspark.vad import speaking_from_energy

# Flags that persist until the next event vs. fire on a single frame then revert.
_SUSTAINED = {"LISTEN", "HOLD", "INCOMPLETE", "CONTINUE"}
# The characteristic "primary" flag per behaviour (used as the audio anchor).
_PRIMARY_FLAG = {
    "barge_real": "BARGE_HARD",
    "barge_lookalike": "CONTINUE",
    "backchannel": "LISTEN",
    "overlap_comp": "BARGE_SOFT",
    "overlap_coop": "CONTINUE",
    "endpoint_end": "TURN_END",
    "endpoint_hold": "INCOMPLETE",
    "correction": "TURN_END",
    "incomplete_thinking": "INCOMPLETE",
    "silence_break": "SILENCE_BREAK",
    "prefetch": "PREFETCH_LLM",
    "nonspeech_neg": "LISTEN",
}
_TAIL_FRAMES = 8       # a little room after the last event
_MIN_WINDOW = 16       # never emit a sub-1.3 s window

# --------------------------------------------------------------------------- #
# Event widths — how many frames BEFORE/AFTER a point event still count as that event.
#
# The non-sustained flags below are laid down on exactly ONE 80 ms frame (see step 5 of
# build_frames). In a ~55-frame clip that is 0.2% of the frames, and the per-frame argmax
# metric in scripts/08_evaluate.py is then unforgiving in a way that has nothing to do
# with model quality: a referee that fires TURN_END one frame (80 ms) early scores a
# false positive AND a false negative, i.e. P=R=F1=0 for that clip — even though 80 ms
# early is *better than human* for turn-taking. Measured on run 20260901-214336:
#
#     TURN_END F1=0.15   BARGE_SOFT F1=0.11   BARGE_HARD F1=0.17
#     PREFETCH_LLM F1=0.08   COMMIT_LLM F1=0.11   SILENCE_BREAK F1=0.17
#     ...while LISTEN=0.87 and HOLD=0.88 (the two SUSTAINED, many-frame flags).
#
# That split — every sustained flag passing, every point flag failing — is the signature
# of a labelling/metric problem, not of an under-trained head.
#
# Values are (frames_before, frames_after) at scale 1.0, chosen from what the decision
# actually tolerates in a live call:
#   TURN_END      ±~0.2 s   (Section 10 allows 300 ms endpoint latency p95)
#   BARGE_*       lead-in of the interruption, then the decision holds briefly
#   PREFETCH_LLM  is a *lead-in window*, not an instant — it legitimately spans ~0.5 s
#   COMMIT/CANCEL a short decision window right after the trigger
#   SILENCE_BREAK the re-open decision holds while the agent starts speaking
_EVENT_WIDTH_FRAMES: dict[str, tuple[int, int]] = {
    "TURN_END":      (2, 3),
    "BARGE_SOFT":    (2, 3),
    "BARGE_HARD":    (2, 3),
    "PREFETCH_LLM":  (1, 6),
    "COMMIT_LLM":    (1, 2),
    "CANCEL_LLM":    (1, 2),
    "SILENCE_BREAK": (2, 4),
}
# Lower id = wins when two expanded events collide. Rare, decision-carrying flags must
# survive being overlapped by a wider neighbour, so they are ordered rarest-first.
_EVENT_PRIORITY: list[str] = [
    "BARGE_HARD", "BARGE_SOFT", "TURN_END", "COMMIT_LLM", "CANCEL_LLM",
    "SILENCE_BREAK", "PREFETCH_LLM",
]


def event_width_frames(scale: float = 1.0) -> dict[int, tuple[int, int]]:
    """`_EVENT_WIDTH_FRAMES` as {flag_id: (before, after)}, scaled and floored at 0."""
    out: dict[int, tuple[int, int]] = {}
    for flag, (b, a) in _EVENT_WIDTH_FRAMES.items():
        out[vocab.CONTROL_FLAG_TO_ID[flag]] = (max(0, int(round(b * scale))),
                                               max(0, int(round(a * scale))))
    return out


def expand_event_flags(flags: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Widen single-frame control events into short windows (training labels only).

    Applied as a LABEL TRANSFORM at load time (thinkspark.dataset), not baked into the
    shards: the on-disk record keeps the canonical point events, so the exact-frame
    metric stays available and the transform can be re-tuned without regenerating data.
    Idempotency warning: applying it twice widens twice — apply it in exactly one place.
    """
    if scale <= 0:
        return flags
    widths = event_width_frames(scale)
    T = len(flags)
    out = flags.copy()
    # Paint widest-priority-last so rarer flags end up on top of commoner ones.
    for flag in reversed(_EVENT_PRIORITY):
        fid = vocab.CONTROL_FLAG_TO_ID[flag]
        before, after = widths.get(fid, (0, 0))
        if before == 0 and after == 0:
            continue
        for t in np.flatnonzero(flags == fid):
            out[max(0, t - before):min(T, t + after + 1)] = fid
    return out


@dataclass
class FrameLabels:
    num_frames: int
    flags: np.ndarray                 # int64 [T]
    agent_state: np.ndarray           # int64 [T]
    vap: np.ndarray                   # float32 [T, H]
    speaking_mask: np.ndarray         # bool [T]
    vad: np.ndarray = None            # float32 [T] "user is speaking" (VAP target source)
    spoken_spans: list[tuple[int, str]] = field(default_factory=list)
    audio_frames: int = 0             # how many frames actually have user audio

    def to_record(self, scenario: Scenario, encoded_path: str | None = None) -> dict:
        """A compact, JSON-serialisable training record (one line of a shard)."""
        return {
            "scenario_id": scenario.scenario_id,
            "behaviour": scenario.behaviour,
            "language": scenario.language,
            "domain": scenario.domain,
            "agent_text": scenario.agent_text,
            "user_text": scenario.user_text,
            "num_frames": self.num_frames,
            "audio_frames": self.audio_frames,
            "encoded_path": encoded_path,
            "flags": self.flags.tolist(),
            "agent_state": self.agent_state.tolist(),
            "speaking_mask": self.speaking_mask.astype(int).tolist(),
            # The VAP head's real target source: per-frame "user is speaking", derived
            # from the clip's log-RMS energy (thinkspark.vad). Stored separately from
            # `speaking_mask` because they mean DIFFERENT things — speaking_mask marks
            # the frames that carry back-channel TEXT (the spoken-head loss mask), and
            # feeding it to the VAP head is exactly the bug that produced VAD-F1 0.000.
            "vad": (self.vad.astype(int).tolist() if self.vad is not None
                    else [1] * self.num_frames),
            # VAP is large; store as compact float16 lists only if needed downstream.
            "spoken_spans": [{"frame": f, "text": t} for f, t in self.spoken_spans],
        }


def build_frames(
    scenario: Scenario,
    tts=None,                 # thinkspark.tts_soniox.TTSResult | None
    encoded=None,             # thinkspark.mimi_codec.EncodedAudio | None
    vap_horizon: int = 25,
) -> FrameLabels:
    # 1) audio length in frames -----------------------------------------------
    if encoded is not None:
        audio_frames = int(encoded.num_frames)
    elif tts is not None:
        audio_frames = vocab.seconds_to_frames(tts.duration_s)
    else:
        # no audio: approximate from user_text length (~2.2 words/sec)
        words = max(1, len(scenario.user_text.split()))
        audio_frames = vocab.seconds_to_frames(words / 2.2)

    # 2) calibrate event offsets to real audio --------------------------------
    events = _calibrated_events(scenario, tts, audio_frames)

    # 3) window length --------------------------------------------------------
    last_evt = max((f for f, _, _ in events), default=0)
    T = max(_MIN_WINDOW, audio_frames, last_evt + 1) + _TAIL_FRAMES

    # 4) base flag from the starting agent-state ------------------------------
    base = "HOLD" if scenario.agent_state in ("TTS_SPEAKING", "LLM_GEN") else "LISTEN"
    flags_str = [base] * T

    # 5) lay events onto the frame grid ---------------------------------------
    spoken_spans: list[tuple[int, str]] = []
    events_sorted = sorted(events, key=lambda e: e[0])
    for i, (fr, flag, spoken) in enumerate(events_sorted):
        fr = min(max(0, fr), T - 1)
        if flag in _SUSTAINED:
            nxt = events_sorted[i + 1][0] if i + 1 < len(events_sorted) else T
            for k in range(fr, min(nxt, T)):
                flags_str[k] = flag
        else:
            flags_str[fr] = flag
        if spoken.strip():
            spoken_spans.append((fr, spoken.strip()))

    flags = np.array([vocab.CONTROL_FLAG_TO_ID[f] for f in flags_str], dtype=np.int64)

    # 6) agent-state channel (input) via a small state machine ----------------
    agent_state = _agent_state_track(scenario, flags_str, T)

    # 7) VAP target: is the user speaking in each of the next H frames ---------
    vad = speaking_track(audio_frames, T, encoded)
    vap = _vap_targets(audio_frames, T, vap_horizon, encoded)

    # 8) spoken loss mask -----------------------------------------------------
    speaking_mask = np.zeros(T, dtype=bool)
    for fr, _ in spoken_spans:
        speaking_mask[min(fr, T - 1)] = True

    return FrameLabels(
        num_frames=T,
        flags=flags,
        agent_state=agent_state,
        vap=vap,
        speaking_mask=speaking_mask,
        vad=vad,
        spoken_spans=spoken_spans,
        audio_frames=audio_frames,
    )


# --------------------------------------------------------------------------- #
def _calibrated_events(scenario: Scenario, tts, audio_frames: int):
    """Return [(frame, flag, spoken_text)] shifted so the primary event hits the audio."""
    raw = [(int(t.frame_offset), t.flag, t.spoken_text) for t in scenario.target]
    if not raw:
        return [(0, vocab.DEFAULT_FLAG, "")]

    if tts is None:
        return raw  # trust the LLM offsets (sample / dry-run path)

    primary_flag = _PRIMARY_FLAG.get(scenario.behaviour)
    anchor_frame = tts.frame_at_char(scenario.event_char)

    # find the LLM's primary event to compute the shift
    primary_idx = next((i for i, (_, fl, _) in enumerate(raw) if fl == primary_flag), None)
    if primary_idx is None:
        primary_idx = len(raw) - 1  # fall back to the last event
    shift = anchor_frame - raw[primary_idx][0]

    calibrated = [(max(0, fr + shift), fl, sp) for (fr, fl, sp) in raw]
    return calibrated


def _agent_state_track(scenario: Scenario, flags_str: list[str], T: int) -> np.ndarray:
    """Advance the agent-state channel as the referee's decisions change the floor."""
    state = scenario.agent_state if scenario.agent_state in vocab.AGENT_STATE_TO_ID else "IDLE"
    out = np.empty(T, dtype=np.int64)
    for k in range(T):
        fl = flags_str[k]
        # transitions implied by control decisions
        if fl in ("BARGE_HARD", "CANCEL_LLM"):
            state = "IDLE"
        elif fl in ("TURN_END", "COMMIT_LLM", "PREFETCH_LLM"):
            state = "LLM_GEN"
        elif fl == "SILENCE_BREAK":
            state = "TTS_SPEAKING"
        elif fl == "BARGE_SOFT":
            state = "TTS_SPEAKING"
        out[k] = vocab.AGENT_STATE_TO_ID[state]
    return out


def speaking_track(audio_frames: int, T: int, encoded) -> np.ndarray:
    """Per-frame 'user is speaking' [T], from real energy when the clip is encoded.

    The previous implementation thresholded at `np.percentile(energy, 40)`, which is not
    a detector at all: it labels exactly 60% of EVERY clip as speech regardless of what
    the audio contains. thinkspark.vad uses an adaptive floor/peak threshold with
    hysteresis instead, and falls back to "the audio region is speech" when there is no
    encode to measure.
    """
    speaking = np.zeros(T, dtype=np.float32)
    if encoded is not None and getattr(encoded, "energy", None) is not None \
            and encoded.energy.size:
        sp = speaking_from_energy(encoded.energy[:T])
        speaking[:len(sp)] = sp
    else:
        speaking[:min(audio_frames, T)] = 1.0
    return speaking


def _vap_targets(audio_frames: int, T: int, H: int, encoded) -> np.ndarray:
    """Binary 'user speaking' per frame, expanded to next-H-frame targets."""
    speaking = speaking_track(audio_frames, T, encoded)

    vap = np.zeros((T, H), dtype=np.float32)
    for t in range(T):
        hi = min(t + 1 + H, T)
        chunk = speaking[t + 1:hi]
        vap[t, :len(chunk)] = chunk
    return vap
