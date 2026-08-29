"""
The label space for ThinkSpark-v2-350M.

Everything the model can *emit* or *observe* is enumerated here so that the data
generator, the frame builder, the model heads, the losses and the inference loop all
agree on exactly one canonical ordering. Never hard-code these strings elsewhere —
import from this module.

Two output heads (Section 5):
  * control head  -> one CONTROL_FLAG per 80 ms frame (internal, never spoken)
  * spoken head   -> plain multilingual text (sent to TTS), only on some frames

Model inputs the orchestrator feeds every frame (Section 4.3):
  * agent-state flag  (IDLE / LLM_GEN / TTS_SPEAKING / TTS_DONE)
  * agent text (rolling, no timing)  + optional STT partials + system prompt
  * Mimi cb0 audio token + prosody (energy, f0)
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Frame geometry (Section 5, "Frame + parameter math")
# --------------------------------------------------------------------------- #
FRAME_RATE_HZ: float = 12.5          # Mimi cb0 frame rate
FRAME_MS: float = 1000.0 / FRAME_RATE_HZ   # 80 ms
FRAMES_PER_SECOND: float = FRAME_RATE_HZ
FRAMES_PER_HOUR: int = int(FRAME_RATE_HZ * 3600)   # 45_000


def seconds_to_frames(seconds: float) -> int:
    """Round a duration in seconds to a whole number of 80 ms frames."""
    return int(round(seconds * FRAME_RATE_HZ))


def frames_to_seconds(frames: int) -> float:
    return frames / FRAME_RATE_HZ


# --------------------------------------------------------------------------- #
# User-utterance length bands (see DataGenConfig.utterance_length). Each band gives
# the LLM a concrete word-count target (native words; conversational Indic/English is
# ~2-2.5 words/sec, so these map to the stated second ranges) plus a natural-language
# instruction that also tells it to build in realistic mid-utterance structure. Only
# the length differs — behaviour semantics are unchanged across bands.
# --------------------------------------------------------------------------- #
LENGTH_BANDS: dict[str, dict] = {
    "short": {
        "seconds": (1.0, 2.0),
        # min_words 0 -> the field spec prints the ORIGINAL "<= 25 words" phrasing, so
        # any additional "short" generation stays distributionally identical to the
        # existing corpus (which was made before this knob existed).
        "min_words": 0,
        "max_words": 25,
        "instruction": "",   # empty on purpose — no length block for short
    },
    "extended": {
        "seconds": (3.0, 8.0),
        "min_words": 10,
        "max_words": 24,
        "instruction": (
            "LENGTH REQUIREMENT: the user_text must be a REALISTIC, longer spoken turn "
            "of about 3-8 seconds of speech (roughly 10-24 native words) — NOT a terse "
            "one-liner. Write it the way a real person actually talks on a call: two or "
            "three connected clauses, a natural aside or reason, maybe a brief filler or "
            "self-correction mid-way ('umm', 'I mean', 'actually'), so the turn has real "
            "internal rhythm and at least one natural mid-utterance pause BEFORE the true "
            "end. Keep event_char at the genuine end-of-turn boundary (or the true event "
            "point), so any mid-utterance pause is clearly NOT the end. Do not pad with "
            "repetition — it must stay natural and on-domain."
        ),
    },
    "long": {
        # A long, multi-sentence user turn (~12-25 s). This is the efficient way to add
        # many HOURS of training audio: each clip carries far more audio, so you hit an
        # hours target with a fraction of the clips (=> far fewer LLM calls and Soniox
        # stream starts) and it better matches the guide's intended 20-30 s windows.
        # Best for behaviours where a real person genuinely talks for a while (explaining
        # a problem, a detailed request/complaint, thinking out loud); it is NOT natural
        # for terse behaviours (quick back-channel, one-word barge), so mix, don't use it
        # for the whole corpus.
        "seconds": (12.0, 25.0),
        "min_words": 25,
        "max_words": 55,
        "instruction": (
            "LENGTH REQUIREMENT: the user_text must be a LONG, multi-sentence spoken turn "
            "of about 12-25 seconds of speech (roughly 25-55 native words). Write it like "
            "someone actually explaining something at length on a call: several connected "
            "sentences that build on each other (context, then the actual question or "
            "problem, maybe a consequence or example), with natural fillers and at least "
            "two mid-utterance pauses BEFORE the true end. Keep event_char at the genuine "
            "end-of-turn boundary (or the true event point), so the internal pauses are "
            "clearly NOT the end. It must stay natural, coherent, and on-domain — do NOT "
            "pad with repetition or filler just to reach the length."
        ),
    },
}
DEFAULT_LENGTH_BAND = "short"


def length_band(name: str | None) -> dict:
    """Look up a length band, falling back to the default for None/unknown names."""
    return LENGTH_BANDS.get((name or DEFAULT_LENGTH_BAND), LENGTH_BANDS[DEFAULT_LENGTH_BAND])


# --------------------------------------------------------------------------- #
# Control flags — the control head's classes (internal, angle-bracketed in docs,
# never spoken). Order is canonical: index == class id.
# --------------------------------------------------------------------------- #
CONTROL_FLAGS: list[str] = [
    "LISTEN",         # user has the floor -> stay silent
    "HOLD",           # keep current agent audio (also used during LLM_GEN)
    "INCOMPLETE",     # user paused, not finished -> suppress endpoint
    "TURN_END",       # user genuinely finished -> commit LLM -> TTS
    "BARGE_SOFT",     # user started over agent, likely wants floor -> duck TTS
    "BARGE_HARD",     # clear interruption -> stop TTS, send agent-so-far + user
    "CONTINUE",       # overlap is only a back-channel -> ignore, keep speaking
    "PREFETCH_LLM",   # turn-end likely soon -> speculatively start LLM (hide latency)
    "COMMIT_LLM",     # use the prefetched/started reply -> play it
    "CANCEL_LLM",     # user kept talking -> drop the speculative reply
    "SILENCE_BREAK",  # dead air too long -> make the agent re-open the conversation
]
CONTROL_FLAG_TO_ID: dict[str, int] = {f: i for i, f in enumerate(CONTROL_FLAGS)}
ID_TO_CONTROL_FLAG: dict[int, str] = {i: f for i, f in enumerate(CONTROL_FLAGS)}
NUM_CONTROL_FLAGS: int = len(CONTROL_FLAGS)

# The "default / do-nothing" flag while the user holds the floor.
DEFAULT_FLAG: str = "LISTEN"

# Flags that are *rare* and drive the focal-loss / class-weighting (Section 9.1).
RARE_FLAGS: set[str] = {
    "BARGE_HARD", "BARGE_SOFT", "CANCEL_LLM", "COMMIT_LLM",
    "SILENCE_BREAK", "TURN_END", "PREFETCH_LLM",
}

# Flags that are usually accompanied by spoken text (back-channel / thinking / re-open).
SPEAKING_FLAGS: set[str] = {"CONTINUE", "INCOMPLETE", "SILENCE_BREAK", "LISTEN"}


# --------------------------------------------------------------------------- #
# Agent-state channel — a model *input*, fed live by the orchestrator (Section 4.3).
# This replaces TTS char-timestamps and is what kills vendor lock.
# --------------------------------------------------------------------------- #
AGENT_STATES: list[str] = [
    "IDLE",           # nobody on the agent side is talking/thinking
    "LLM_GEN",        # the LLM is producing a reply -> do NOT commit another one
    "TTS_SPEAKING",   # agent audio is playing now (barge decisions matter here)
    "TTS_DONE",       # agent just finished; floor is open
]
AGENT_STATE_TO_ID: dict[str, int] = {s: i for i, s in enumerate(AGENT_STATES)}
NUM_AGENT_STATES: int = len(AGENT_STATES)


# --------------------------------------------------------------------------- #
# Behaviours — the 8 core behaviours expanded into 12 generation buckets
# (Section 5.3 + 8.4 generation prompt enum).
# --------------------------------------------------------------------------- #
BEHAVIOURS: list[str] = [
    "barge_real",          # B1  real interruption -> BARGE_HARD/BARGE_SOFT
    "barge_lookalike",     # B1  look-alike back-channel over agent -> CONTINUE (hard neg)
    "backchannel",         # B2  "haan"/"right" on a clause boundary (context-varied)
    "overlap_comp",        # B3  competitive overlap -> BARGE_SOFT/HARD
    "overlap_coop",        # B3  cooperative overlap -> CONTINUE
    "endpoint_end",        # B4  user genuinely finished -> TURN_END
    "endpoint_hold",       # B4  user paused mid-thought -> INCOMPLETE
    "correction",          # B5  "send to Rahul... no, Rohan" -> HOLD/INCOMPLETE.. TURN_END
    "incomplete_thinking", # B6  trailing off + thinking sound -> INCOMPLETE + spoken
    "silence_break",       # B7  long dead air -> SILENCE_BREAK + spoken re-open
    "prefetch",            # B4  near-end prosody -> PREFETCH_LLM lead-in
    "nonspeech_neg",       # false-trigger robustness (cough/umm/noise) -> LISTEN/HOLD
]
BEHAVIOUR_TO_ID: dict[str, int] = {b: i for i, b in enumerate(BEHAVIOURS)}
NUM_BEHAVIOURS: int = len(BEHAVIOURS)


# --------------------------------------------------------------------------- #
# Languages (Section 8.2). "*_native" = native script with real English words
# inserted (NOT romanized Hinglish/Gujlish).
# --------------------------------------------------------------------------- #
LANGUAGES: list[str] = ["hi", "en", "gu", "hi_en_native", "gu_en_native"]
LANGUAGE_TO_ID: dict[str, int] = {l: i for i, l in enumerate(LANGUAGES)}

# Human-readable script hint used inside the generation prompt.
LANGUAGE_SCRIPT_HINT: dict[str, str] = {
    "hi": "Hindi in Devanagari script",
    "en": "English",
    "gu": "Gujarati in Gujarati script",
    "hi_en_native": "Hindi in Devanagari with real English words inserted (code-mix, native script — e.g. 'aaj weather achha hai' but written in Devanagari)",
    "gu_en_native": "Gujarati in Gujarati script with real English words inserted (e.g. 'aaje meeting chhe')",
}

# Unicode blocks used by the language/script validator (Section 8.5).
DEVANAGARI_RANGE = (0x0900, 0x097F)
GUJARATI_RANGE = (0x0A80, 0x0AFF)


# --------------------------------------------------------------------------- #
# Domains & prosody (generation prompt enum, Section 8.4).
# --------------------------------------------------------------------------- #
DOMAINS: list[str] = ["bfsi_collections", "support", "sales"]
PROSODY: list[str] = ["falling", "rising", "held", "flat", "distressed", "neutral"]
GENDERS: list[str] = ["female", "male"]   # gender-balanced ~50/50 (Section 7.3, 8.3)


# --------------------------------------------------------------------------- #
# Special tokens injected into the model's text stream so a single Gemma sequence
# can carry system prompt + agent text + STT partials + agent-state (Section 4.2).
# These are added to the tokenizer as additional_special_tokens.
# --------------------------------------------------------------------------- #
SPECIAL_TOKENS: dict[str, str] = {
    "sys_bos": "<|sys|>",       # system prompt (persona / domain) start
    "agent_bos": "<|agent|>",   # rolling agent text start
    "stt_bos": "<|stt|>",       # optional user STT partials start
    "state_bos": "<|state|>",   # agent-state token slot
    "audio_bos": "<|audio|>",   # start of the streamed Mimi-token / prosody segment
    "spoken_bos": "<|say|>",    # spoken-head target start (interjection to TTS)
    "spoken_eos": "<|/say|>",   # spoken-head target end
}
ALL_SPECIAL_TOKENS: list[str] = list(SPECIAL_TOKENS.values())


def is_valid_flag(flag: str) -> bool:
    return flag in CONTROL_FLAG_TO_ID


def is_valid_state(state: str) -> bool:
    return state in AGENT_STATE_TO_ID
