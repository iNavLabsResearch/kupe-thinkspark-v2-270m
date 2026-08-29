"""
ThinkSpark-v2-350M — a tiny full-duplex *floor-control* model.

A "conversation referee" built on Gemma-3-270M + Mimi audio tokens. It watches the
caller every 80 ms and emits (a) internal control flags and (b) plain speakable
back-channels, turning any STT -> LLM -> TTS cascade into a full-duplex voice agent
*without* timestamps, agent audio, or vendor lock, while keeping your own Indic LLM.

The package is organised so every stage of the pipeline is an importable module:

    vocab        control flags, agent states, behaviours, languages (the label space)
    schema       the Scenario JSON contract (dataclasses + validation)
    config       typed config loading from YAML + .env
    prompts      LLM prompts that write scenarios as strict JSON
    distribution behaviour/language/gender budget planner (Section 8.1-8.3)
    llm_client   OpenAI-SDK wrapper for scenario + label generation
    tts_soniox   Soniox TTS client (WebSocket, word/char timestamps)
    mimi_codec   Mimi encode -> cb0 tokens + energy + f0
    frames       scenario timeline -> per-frame control-flag + spoken-text labels
    validators   Section 8.5 data-quality gate
    dataset      torch Dataset for Phase-1 / Phase-2
    model        Gemma-3-270M backbone + control head + spoken (LM) head + VAP head
    losses       focal control loss, spoken CE, VAP BCE, alignment loss
    metrics      Section 10 evaluation metrics
    inference    the live referee step() loop (Section 11)

Nothing here trains automatically — the scripts/ folder drives each stage so you can
run them by hand on Kaggle (`conda activate llms`).
"""

__version__ = "2.0.0"
__model_name__ = "thinkspark-v2-350m"

from thinkspark import vocab  # noqa: F401  (convenience re-export)

__all__ = ["vocab", "__version__", "__model_name__"]
