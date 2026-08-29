"""
The Scenario JSON contract (Section 8.4).

The LLM writes ONE training scenario as strict JSON. Each scenario describes a single
20-30 s floor-control window: what the agent is doing, the exact user line to be spoken
by Soniox TTS, a prosody hint, the char-offset where the key event happens, and the
ordered target timeline of {frame_offset, flag, spoken_text}.

`frame_offset` in the raw LLM output is a *relative* hint (event ordering). The real,
frame-accurate offsets are recovered later from Soniox word/char timestamps by
`thinkspark.frames`. We keep the LLM's offsets only as a fallback / sanity anchor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from thinkspark import vocab


@dataclass
class TargetEvent:
    """One entry in the scenario's target timeline."""
    frame_offset: int          # frames from window start (relative hint from the LLM)
    flag: str                  # one of vocab.CONTROL_FLAGS
    spoken_text: str = ""      # PLAIN words for the spoken head ("" = say nothing)

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not vocab.is_valid_flag(self.flag):
            errs.append(f"unknown flag {self.flag!r}")
        if self.frame_offset < 0:
            errs.append(f"negative frame_offset {self.frame_offset}")
        if "<" in self.spoken_text or ">" in self.spoken_text:
            errs.append("spoken_text must be PLAIN words, no angle-bracket tags")
        return errs


@dataclass
class Scenario:
    """A single generated training scenario (matches the Section 8.4 FIELDS block)."""
    behaviour: str             # vocab.BEHAVIOURS
    language: str              # vocab.LANGUAGES
    domain: str                # vocab.DOMAINS
    agent_text: str            # what the agent is saying (may be "")
    agent_state: str           # vocab.AGENT_STATES at window start
    user_text: str             # exact line for Soniox TTS (native script; word budget by length_band)
    prosody: str               # vocab.PROSODY
    event_char: int            # index in user_text where the key event happens
    target: list[TargetEvent]  # ordered control-flag + spoken-text timeline
    notes: str = ""            # one line: why this case is hard
    gender: str = "female"     # requested TTS voice gender (balance ~50/50)
    persona: str = ""          # short persona/age hint for voice variety
    scenario_id: str = ""      # filled by the generator (stable hash)
    length_band: str = "short"  # vocab.LENGTH_BANDS — sets the user_text word budget

    # ------------------------------------------------------------------ #
    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Scenario":
        raw_targets = d.get("target", []) or []
        targets = [
            TargetEvent(
                frame_offset=int(t.get("frame_offset", 0)),
                flag=str(t.get("flag", "")).strip(),
                spoken_text=str(t.get("spoken_text", "") or ""),
            )
            for t in raw_targets
        ]
        return Scenario(
            behaviour=str(d.get("behaviour", "")).strip(),
            language=str(d.get("language", "")).strip(),
            domain=str(d.get("domain", "")).strip(),
            agent_text=str(d.get("agent_text", "") or ""),
            agent_state=str(d.get("agent_state", "IDLE")).strip(),
            user_text=str(d.get("user_text", "") or ""),
            prosody=str(d.get("prosody", "neutral")).strip(),
            event_char=int(d.get("event_char", 0)),
            target=targets,
            notes=str(d.get("notes", "") or ""),
            gender=str(d.get("gender", "female")).strip(),
            persona=str(d.get("persona", "") or ""),
            scenario_id=str(d.get("scenario_id", "") or ""),
            length_band=str(d.get("length_band", "short") or "short").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    # ------------------------------------------------------------------ #
    def validate(self) -> list[str]:
        """Return a list of human-readable schema errors ([] == valid)."""
        errs: list[str] = []
        if self.behaviour not in vocab.BEHAVIOUR_TO_ID:
            errs.append(f"unknown behaviour {self.behaviour!r}")
        if self.language not in vocab.LANGUAGE_TO_ID:
            errs.append(f"unknown language {self.language!r}")
        if self.domain not in vocab.DOMAINS:
            errs.append(f"unknown domain {self.domain!r}")
        if self.agent_state not in vocab.AGENT_STATE_TO_ID:
            errs.append(f"unknown agent_state {self.agent_state!r}")
        if self.prosody not in vocab.PROSODY:
            errs.append(f"unknown prosody {self.prosody!r}")
        if self.gender not in vocab.GENDERS:
            errs.append(f"unknown gender {self.gender!r}")
        if not self.user_text.strip():
            errs.append("empty user_text")
        # word budget follows the scenario's length band (short/extended/long), with 20%
        # headroom so natural LLM overshoot near the band's max doesn't trigger a regen.
        max_words = int(round(vocab.length_band(self.length_band)["max_words"] * 1.2))
        n_words = len(self.user_text.split())
        if n_words > max_words:
            errs.append(f"user_text too long ({n_words} words > {max_words} for "
                        f"'{self.length_band}' band)")
        if not (0 <= self.event_char <= len(self.user_text)):
            errs.append(f"event_char {self.event_char} out of range for user_text")
        if not self.target:
            errs.append("empty target timeline")
        for i, t in enumerate(self.target):
            for e in t.validate():
                errs.append(f"target[{i}]: {e}")
        return errs

    def is_valid(self) -> bool:
        return not self.validate()
