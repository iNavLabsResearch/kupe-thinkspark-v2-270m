"""
LLM prompts that make the model write ONE training scenario as strict JSON (Section 8.4).

`build_scenario_prompt(job)` returns (system, user) messages tailored to a single GenJob
so the requested behaviour / language / domain / gender / (silence) is produced. The
generation is *seeded* with per-behaviour guidance and concrete B1-B8 exemplars from
Section 5.3 so the model emits the right flag timeline.
"""

from __future__ import annotations

from thinkspark import vocab
from thinkspark.distribution import GenJob

# --------------------------------------------------------------------------- #
# Static reference blocks injected into every prompt.
# --------------------------------------------------------------------------- #
_FLAG_GLOSSARY = "\n".join(
    f"  {f:<14} {desc}"
    for f, desc in [
        ("LISTEN", "user has the floor; agent stays silent (say nothing)"),
        ("HOLD", "keep current agent audio; also used during LLM_GEN to avoid double-invoke"),
        ("INCOMPLETE", "user paused, not finished; suppress endpoint (often + a thinking sound)"),
        ("TURN_END", "user genuinely finished; commit LLM -> TTS"),
        ("BARGE_SOFT", "user started over agent, likely wants floor; duck TTS"),
        ("BARGE_HARD", "clear interruption; stop TTS, send agent-so-far + user to LLM"),
        ("CONTINUE", "overlap is only a back-channel; ignore, keep speaking"),
        ("PREFETCH_LLM", "turn-end likely soon; speculatively start LLM to hide latency"),
        ("COMMIT_LLM", "use the prefetched/started reply; play it"),
        ("CANCEL_LLM", "user kept talking; drop the speculative reply"),
        ("SILENCE_BREAK", "dead air too long; agent re-opens the conversation"),
    ]
)

# Concrete input->output exemplars (Section 5.3), romanised here for the prompt only.
_BEHAVIOUR_GUIDE: dict[str, str] = {
    "barge_real": (
        "REAL barge-in. agent_state=TTS_SPEAKING, the user clearly takes the floor "
        "(e.g. 'nahi maine pay kar diya'). Timeline: a couple of HOLD frames, then "
        "BARGE_HARD (or BARGE_SOFT if only likely). spoken_text stays empty."
    ),
    "barge_lookalike": (
        "HARD NEGATIVE that looks like a barge but is only a back-channel over the agent "
        "(user says 'haan haan' while agent talks). agent_state=TTS_SPEAKING. Timeline "
        "must resolve to CONTINUE (do NOT interrupt). This is the 50% look-alike half."
    ),
    "backchannel": (
        "User is narrating; on a clause boundary with falling pitch the agent may drop a "
        "short back-channel. agent_state=TTS_DONE or IDLE. If speaking: LISTEN then a "
        "LISTEN/CONTINUE frame with a SHORT spoken_text ('haan' / 'right' / 'bilkul sir'). "
        "Tone must match domain (warm for sales, neutral for collections)."
    ),
    "overlap_comp": (
        "Competitive overlap: both talk and the user wants the floor. agent_state="
        "TTS_SPEAKING, user energy present. Timeline -> BARGE_SOFT or BARGE_HARD."
    ),
    "overlap_coop": (
        "Cooperative overlap: user talks over agent only to affirm. agent_state="
        "TTS_SPEAKING. Timeline -> CONTINUE (keep speaking). spoken_text empty."
    ),
    "endpoint_end": (
        "User genuinely finishes a turn with falling tone. agent_state=IDLE/TTS_DONE. "
        "Timeline: LISTEN... then PREFETCH_LLM as the end nears, then TURN_END, then "
        "COMMIT_LLM. spoken_text empty."
    ),
    "endpoint_hold": (
        "User pauses mid-thought (rising/held pitch, no falling tone). Timeline must be "
        "INCOMPLETE (NOT TURN_END) so the endpoint is suppressed. spoken_text empty."
    ),
    "correction": (
        "Mid-sentence self-correction ('send it to Rahul no to Rohan'). Keep HOLD/"
        "INCOMPLETE across the correction, prefer the LATEST span, then TURN_END at the "
        "true end. event_char marks the correction point. spoken_text empty."
    ),
    "incomplete_thinking": (
        "User trails off with a few silent frames and NO falling tone. Timeline: "
        "INCOMPLETE + a SHORT thinking/continuer spoken_text ('haan haan, aap keh rahe "
        "the?' / 'yeah, go on'). This is the empathy continuer case."
    ),
    "silence_break": (
        "agent_state=IDLE and the user is silent for a long time (>2.5 s / >=32 frames). "
        "Timeline: LISTEN across the dead air, then SILENCE_BREAK with a spoken re-open "
        "('So, shall I share the offer?' / 'Are you still there?')."
    ),
    "prefetch": (
        "Near-end prosody: the user is clearly about to finish. Timeline shows "
        "PREFETCH_LLM firing BEFORE TURN_END (lead-in), then TURN_END, then COMMIT_LLM. "
        "If you add a case where the user keeps talking, end on CANCEL_LLM instead."
    ),
    "nonspeech_neg": (
        "False-trigger robustness: the 'user_text' contains only a cough / 'umm' / filler "
        "/ background noise word, NOT a real turn. Timeline must stay LISTEN or HOLD — the "
        "agent must NOT barge, endpoint, or back-channel. spoken_text empty."
    ),
}

_SILENCE_ADDENDUM = (
    "\nIMPORTANT (negative case): For THIS scenario the natural, correct behaviour is to "
    "stay SILENT. The user's back-channel opportunity does NOT warrant a spoken response "
    "(e.g. the user is mid-sentence, or a spoken back-channel would step on them). Every "
    "target entry must use flag LISTEN (or CONTINUE) with spoken_text = \"\". Emitting a "
    "back-channel word here would be unnatural. event_char still marks the boundary."
)


def _json_field_spec(length_band: str = "short") -> str:
    band = vocab.length_band(length_band)
    lo, hi = band["min_words"], band["max_words"]
    # lo<=0 -> preserve the original upper-bound-only phrasing (the "short" default);
    # a real band with a lower bound prints an explicit range.
    words_spec = f"<= {hi} words" if lo <= 0 else f"about {lo}-{hi} words"
    return (
        "FIELDS (strict JSON, output JSON ONLY, no markdown fence):\n"
        f"  behaviour   : one of {vocab.BEHAVIOURS}\n"
        f"  language    : one of {vocab.LANGUAGES}\n"
        f"  domain      : one of {vocab.DOMAINS}\n"
        "  agent_text  : what the agent is saying (may be \"\")\n"
        f"  agent_state : one of {vocab.AGENT_STATES}\n"
        f"  user_text   : exact line for TTS (native script; {words_spec})\n"
        f"  prosody     : one of {vocab.PROSODY}\n"
        "  event_char  : integer index in user_text where the key event happens\n"
        "  gender      : 'female' or 'male'\n"
        "  persona     : short persona/age hint for voice variety (e.g. 'young woman, polite')\n"
        "  target      : ordered list of {frame_offset:int, flag:str, spoken_text:str}\n"
        "                frame_offset is in 80 ms frames from window start;\n"
        "                flag is from the fixed control vocab; spoken_text is PLAIN words\n"
        "                (NO angle-bracket tags, NO [laugh]/<gasp> — real speakable words).\n"
        "  notes       : one line, why this case is hard\n"
    )


SYSTEM_PROMPT = (
    "You generate ONE training scenario for a full-duplex voice-agent 'floor controller'. "
    "The model you are teaching decides WHEN to listen/interrupt/back-channel — it never "
    "writes the actual answer. Output STRICT JSON only, matching the FIELDS exactly. "
    "Use NATIVE script for Indic languages (Devanagari / Gujarati), never romanized. "
    "spoken_text must be plain speakable words that any TTS can pronounce.\n\n"
    "CONTROL FLAG VOCAB:\n" + _FLAG_GLOSSARY
)


def build_scenario_prompt(job: GenJob) -> tuple[str, str]:
    """Return (system, user) chat messages for one generation job."""
    guide = _BEHAVIOUR_GUIDE.get(job.behaviour, "")
    script_hint = vocab.LANGUAGE_SCRIPT_HINT[job.language]
    silence = _SILENCE_ADDENDUM if job.force_silence else ""
    length_note = vocab.length_band(job.length_band)["instruction"]
    length_block = f"{length_note}\n\n" if length_note else ""

    user = (
        f"Generate ONE scenario.\n"
        f"behaviour  = {job.behaviour}\n"
        f"language   = {job.language}  ({script_hint})\n"
        f"domain     = {job.domain}\n"
        f"gender     = {job.gender}\n\n"
        f"BEHAVIOUR GUIDANCE:\n{guide}\n"
        f"{silence}\n\n"
        f"{length_block}"
        f"{_json_field_spec(job.length_band)}\n"
        "CONSTRAINTS:\n"
        "  - Timeline must be causally ordered by frame_offset and end at the true event.\n"
        "  - For barge_lookalike the target MUST resolve to CONTINUE (never BARGE_*).\n"
        "  - Vary persona/age; keep user_text realistic for the domain.\n"
        "  - Match spoken_text language to 'language' and keep it natural.\n"
        "Return JSON now."
    )
    return SYSTEM_PROMPT, user


# --------------------------------------------------------------------------- #
# Batched generation — N scenarios per LLM call for throughput.
#
# Recommended batch_size <= 15 (config default 12). A single completion asking for many
# structurally-identical JSON objects is where small/fast models (DeepSeek flash-class,
# Gemma-3-27B-class) tend to get "confused": they copy-paste near-duplicate items, let
# language/prosody bleed from one item into the next, or truncate the array if max_tokens
# wasn't scaled up. The mitigations below are baked into the prompt + the caller
# (llm_client / script 02): explicit DISTINCTNESS instruction, max_tokens scaled by n,
# and — critically — every item is validated independently downstream, so one bad item
# in a batch never throws away the good ones.
# --------------------------------------------------------------------------- #
BATCH_SYSTEM_PROMPT = (
    "You generate MULTIPLE training scenarios for a full-duplex voice-agent 'floor "
    "controller' in ONE response. The model you are teaching decides WHEN to "
    "listen/interrupt/back-channel — it never writes the actual answer. Output STRICT "
    "JSON only: a single object {\"scenarios\": [ ... ]} whose array holds EXACTLY the "
    "requested number of scenario objects, each matching the FIELDS exactly. Use NATIVE "
    "script for Indic languages (Devanagari / Gujarati), never romanized. spoken_text "
    "must be plain speakable words that any TTS can pronounce.\n\n"
    "CONTROL FLAG VOCAB:\n" + _FLAG_GLOSSARY
)


def build_batch_scenario_prompt(job: GenJob, n: int) -> tuple[str, str]:
    """Return (system, user) chat messages requesting `n` DISTINCT scenarios at once."""
    guide = _BEHAVIOUR_GUIDE.get(job.behaviour, "")
    script_hint = vocab.LANGUAGE_SCRIPT_HINT[job.language]
    silence = _SILENCE_ADDENDUM if job.force_silence else ""
    band = vocab.length_band(job.length_band)
    length_note = band["instruction"]
    length_block = f"{length_note}\n\n" if length_note else ""
    lo_s, hi_s = band["seconds"]
    length_constraint = (
        f"  - EVERY user_text must meet the LENGTH REQUIREMENT above (a real ~{lo_s:.0f}-{hi_s:.0f} s\n"
        "    turn, not a one-liner) — do not let some items drift back to short lines.\n"
        if length_note else ""
    )

    user = (
        f"Generate {n} DISTINCT scenarios (all with the same facets below, but each with "
        f"a genuinely different user_text / persona / exact timing — NOT paraphrases of "
        f"each other).\n"
        f"behaviour  = {job.behaviour}\n"
        f"language   = {job.language}  ({script_hint})\n"
        f"domain     = {job.domain}\n"
        f"gender     = {job.gender}\n\n"
        f"BEHAVIOUR GUIDANCE:\n{guide}\n"
        f"{silence}\n\n"
        f"{length_block}"
        f"Each object in the \"scenarios\" array uses these FIELDS:\n{_json_field_spec(job.length_band)}\n"
        "CONSTRAINTS:\n"
        "  - Return exactly {n} objects in \"scenarios\", no more, no fewer.\n"
        "  - Every scenario must be DISTINCT: vary persona/age/user_text/exact wording.\n"
        "    Do NOT just swap one word between items — write genuinely different lines.\n"
        "  - Timeline must be causally ordered by frame_offset and end at the true event.\n"
        "  - For barge_lookalike the target MUST resolve to CONTINUE (never BARGE_*).\n"
        "  - Match spoken_text language to 'language' and keep it natural.\n"
        f"{length_constraint}"
        "  - Do NOT let language/prosody/behaviour drift between items — every item shares\n"
        "    the same behaviour/language/domain/gender given above.\n"
        "Return the JSON object now: {\"scenarios\": [ ... ]}"
    ).replace("{n}", str(n))
    return BATCH_SYSTEM_PROMPT, user


# --------------------------------------------------------------------------- #
# LLM-judge prompt for naturalness (Section 8.5, target mean >= 4.2 / 5)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = (
    "You are a strict evaluator of voice-agent back-channel / floor-control data. "
    "Given a scenario, rate the NATURALNESS of the spoken_text interjections and the "
    "appropriateness of the control-flag timeline on a 1-5 scale (5 = a fluent native "
    "speaker in this domain would do exactly this). Output STRICT JSON: "
    '{"score": <int 1-5>, "reason": "<one short line>"}.'
)


def build_judge_prompt(scenario_json: str) -> tuple[str, str]:
    return JUDGE_SYSTEM, f"Scenario:\n{scenario_json}\n\nRate it. JSON only."
