"""
Scenario timeline -> per-frame training labels (Section 5.3, 8.4, 9.1).

Given a validated `Scenario` and (optionally) the Soniox `TTSResult` + Mimi `EncodedAudio`
for its user line, produce the exact per-80 ms-frame supervision the two heads need:

    flags        int64  [T]     control-flag id per frame (control head target)
    agent_state  int64  [T]     agent-state channel (a model INPUT, state machine below)
    vap          float32[T, H]  "is user speaking" in each of the next H frames (VAP aux)
    speaking_mask bool  [T]     frames whose flag carries spoken_text (loss mask for txt)
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


@dataclass
class FrameLabels:
    num_frames: int
    flags: np.ndarray                 # int64 [T]
    agent_state: np.ndarray           # int64 [T]
    vap: np.ndarray                   # float32 [T, H]
    speaking_mask: np.ndarray         # bool [T]
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


def _vap_targets(audio_frames: int, T: int, H: int, encoded) -> np.ndarray:
    """Binary 'user speaking' per frame, expanded to next-H-frame targets."""
    speaking = np.zeros(T, dtype=np.float32)
    if encoded is not None and encoded.energy.size:
        e = encoded.energy[:T]
        thr = np.percentile(e, 40) if e.size else -6.0
        speaking[:len(e)] = (e > thr).astype(np.float32)
    else:
        speaking[:min(audio_frames, T)] = 1.0

    vap = np.zeros((T, H), dtype=np.float32)
    for t in range(T):
        hi = min(t + 1 + H, T)
        chunk = speaking[t + 1:hi]
        vap[t, :len(chunk)] = chunk
    return vap
