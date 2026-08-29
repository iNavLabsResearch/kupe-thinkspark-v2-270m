#!/usr/bin/env python
"""
Generate small, illustrative sample data WITHOUT any API / GPU (offline, deterministic).

Writes hand-authored representative scenarios and their dry (audio-free) per-frame records
to data/samples/, so you can read the exact on-disk format before spending any budget.
The scenarios here cover the key behaviours — including the deliberate "say nothing"
back-channel negative the model must also learn.

    conda activate llms          # (only needs numpy + the thinkspark package)
    python scripts/make_samples.py

This uses only the pure-python parts of thinkspark (schema, frames, validators) — no
openai, soniox, torch or transformers imports — so it runs anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.schema import Scenario
from thinkspark.frames import build_frames
from thinkspark.validators import validate_scenario


# Hand-authored, representative scenarios (native script where relevant). frame_offset
# values are relative hints; with real Soniox audio they get calibrated (see frames.py).
SAMPLES: list[dict] = [
    {
        "behaviour": "endpoint_end", "language": "en", "domain": "bfsi_collections",
        "agent_text": "", "agent_state": "IDLE", "prosody": "falling", "gender": "female",
        "persona": "young woman, calm",
        "user_text": "my account number is nine eight seven six five four",
        "event_char": 51,
        "target": [
            {"frame_offset": 0, "flag": "LISTEN", "spoken_text": ""},
            {"frame_offset": 18, "flag": "PREFETCH_LLM", "spoken_text": ""},
            {"frame_offset": 24, "flag": "TURN_END", "spoken_text": ""},
            {"frame_offset": 25, "flag": "COMMIT_LLM", "spoken_text": ""},
        ],
        "notes": "clean finish with falling tone; prefetch fires just before turn-end",
        "scenario_id": "sample_endpoint_en",
    },
    {
        "behaviour": "backchannel", "language": "en", "domain": "sales",
        "agent_text": "", "agent_state": "TTS_DONE", "prosody": "falling", "gender": "male",
        "persona": "middle-aged man",
        "user_text": "so basically I want to close my loan account today",
        "event_char": 40,
        "target": [
            {"frame_offset": 0, "flag": "LISTEN", "spoken_text": ""},
            {"frame_offset": 20, "flag": "LISTEN", "spoken_text": "right"},
        ],
        "notes": "warm sales tone; a single short back-channel on the clause boundary",
        "scenario_id": "sample_backchannel_en",
    },
    {
        "behaviour": "backchannel", "language": "hi_en_native", "domain": "support",
        "agent_text": "", "agent_state": "TTS_DONE", "prosody": "rising", "gender": "female",
        "persona": "young woman, hurried",
        # native Devanagari with a real English word inserted ("meeting")
        "user_text": "मेरा meeting अभी चल रही है और मैं बता रही थी कि",
        "event_char": 30,
        "target": [
            {"frame_offset": 0, "flag": "LISTEN", "spoken_text": ""},
            {"frame_offset": 14, "flag": "LISTEN", "spoken_text": ""},
        ],
        "notes": "NEGATIVE / silent case: user is mid-sentence, so saying 'haan' would step on them",
        "scenario_id": "sample_backchannel_silent_hi",
    },
    {
        "behaviour": "barge_real", "language": "hi", "domain": "bfsi_collections",
        "agent_text": "aapka EMI due hai teerah hazaar", "agent_state": "TTS_SPEAKING",
        "prosody": "distressed", "gender": "male", "persona": "angry customer",
        "user_text": "नहीं मैंने पेमेंट कर दिया है सुनिए",
        "event_char": 3,
        "target": [
            {"frame_offset": 0, "flag": "HOLD", "spoken_text": ""},
            {"frame_offset": 2, "flag": "BARGE_HARD", "spoken_text": ""},
        ],
        "notes": "real interruption over live agent audio -> stop TTS immediately",
        "scenario_id": "sample_barge_real_hi",
    },
    {
        "behaviour": "barge_lookalike", "language": "hi", "domain": "sales",
        "agent_text": "toh sir yeh offer aaj tak valid hai", "agent_state": "TTS_SPEAKING",
        "prosody": "neutral", "gender": "female", "persona": "agreeable listener",
        "user_text": "हाँ हाँ ठीक है",
        "event_char": 0,
        "target": [
            {"frame_offset": 0, "flag": "HOLD", "spoken_text": ""},
            {"frame_offset": 3, "flag": "CONTINUE", "spoken_text": ""},
        ],
        "notes": "HARD NEGATIVE: 'haan haan' over the agent is only a back-channel -> do NOT barge",
        "scenario_id": "sample_barge_lookalike_hi",
    },
    {
        "behaviour": "incomplete_thinking", "language": "gu", "domain": "support",
        "agent_text": "", "agent_state": "IDLE", "prosody": "held", "gender": "female",
        "persona": "elderly woman, unsure",
        "user_text": "મારે એક વાત પૂછવી હતી કે",
        "event_char": 24,
        "target": [
            {"frame_offset": 0, "flag": "LISTEN", "spoken_text": ""},
            {"frame_offset": 16, "flag": "INCOMPLETE", "spoken_text": "હા હા, બોલો"},
        ],
        "notes": "user trails off with no falling tone -> suppress endpoint + gentle continuer",
        "scenario_id": "sample_incomplete_gu",
    },
    {
        "behaviour": "silence_break", "language": "en", "domain": "sales",
        "agent_text": "", "agent_state": "IDLE", "prosody": "flat", "gender": "male",
        "persona": "distracted customer",
        "user_text": "umm",
        "event_char": 3,
        "target": [
            {"frame_offset": 0, "flag": "LISTEN", "spoken_text": ""},
            {"frame_offset": 40, "flag": "SILENCE_BREAK", "spoken_text": "Are you still there?"},
        ],
        "notes": "long dead air (> 2.5 s) -> agent re-opens the conversation",
        "scenario_id": "sample_silence_break_en",
    },
    {
        "behaviour": "correction", "language": "hi_en_native", "domain": "bfsi_collections",
        "agent_text": "", "agent_state": "IDLE", "prosody": "falling", "gender": "female",
        "persona": "office worker",
        "user_text": "इसे Rahul को भेज दो नहीं Rohan को भेजो",
        "event_char": 20,
        "target": [
            {"frame_offset": 0, "flag": "LISTEN", "spoken_text": ""},
            {"frame_offset": 12, "flag": "INCOMPLETE", "spoken_text": ""},
            {"frame_offset": 26, "flag": "TURN_END", "spoken_text": ""},
        ],
        "notes": "mid-sentence self-correction; hold through it, prefer the latest span (Rohan)",
        "scenario_id": "sample_correction_hi",
    },
]


def main():
    out_dir = ROOT / "data" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    scen_path = out_dir / "scenarios_sample.jsonl"
    frames_path = out_dir / "frames_sample.jsonl"

    n_ok = 0
    with scen_path.open("w", encoding="utf-8") as fs, frames_path.open("w", encoding="utf-8") as ff:
        for raw in SAMPLES:
            s = Scenario.from_dict(raw)
            res = validate_scenario(s)
            status = "ok" if res.ok else f"INVALID {res.errors}"
            print(f"  {s.scenario_id:<32} {s.behaviour:<20} {status}")
            n_ok += int(res.ok)
            fs.write(s.to_json() + "\n")
            # dry frame build (no TTS, no Mimi) -> uses the LLM offsets directly
            fl = build_frames(s, tts=None, encoded=None, vap_horizon=25)
            ff.write(json.dumps(fl.to_record(s), ensure_ascii=False) + "\n")

    print(f"\nwrote {len(SAMPLES)} scenarios ({n_ok} valid) -> {scen_path}")
    print(f"wrote {len(SAMPLES)} frame records -> {frames_path}")
    print("NOTE: these are DRY (audio-free) records for reading the format. The real")
    print("pipeline calibrates frame offsets to Soniox timings and fills cb0/energy/f0.")


if __name__ == "__main__":
    main()
