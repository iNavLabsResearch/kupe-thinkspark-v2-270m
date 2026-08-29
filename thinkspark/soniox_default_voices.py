"""
Soniox's named stock ("default") TTS voices — verbatim from the voice list you pasted
(their real `id`/`gender`/description, not fetched live). Used ONLY as a fallback by
scripts/15_create_voice_profiles.py to fill the gap when you submit more reference
clips than the account's cloning cap allows (see `soniox_max_cloned_voices` in
configs/data_gen.yaml) — never used directly by `thinkspark.tts_soniox.resolve_voice()`,
which only ever reads whatever ends up recorded in `voice_profiles.json`.

Order is preserved from your paste; `scripts/15_create_voice_profiles.py` draws from
this list top-to-bottom per gender, only as many as it actually needs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DefaultVoice:
    name: str
    gender: str
    description: str


DEFAULT_VOICE_CATALOG: list[DefaultVoice] = [
    DefaultVoice("Daniel", "male", "A rich, steady male voice with a polished tone, controlled pacing, and a reassuring presence that feels confident and mature."),
    DefaultVoice("Nina", "female", "A bright, expressive female voice with youthful energy, natural rhythm, and a friendly tone that feels warm, engaging, and full of personality."),
    DefaultVoice("Bryce", "male", "A deep, powerful male voice with strong projection and a commanding delivery."),
    DefaultVoice("Kayla", "female", "A natural Gen Z female voice with a casual, low-key, and unpolished style."),
    DefaultVoice("Nora", "female", "A young female voice with a casual, chatty style and an inquisitive tone."),
    DefaultVoice("Emerson", "male", "A warm, elegant male voice with a formal style and a slight American accent."),
    DefaultVoice("Miles", "male", "A deep, smooth British male voice with a relaxed, intimate delivery."),
    DefaultVoice("Imogen", "female", "A clear British female voice with a sincere, down-to-earth, and professional style."),
    DefaultVoice("Alistair", "male", "An older British male voice with a gentle, warm, and grandfatherly sound."),
    DefaultVoice("Bennett", "male", "A deep, articulate male voice with the calm authority of an experienced teacher."),
    DefaultVoice("Harlan", "male", "A polished American male narrator with clear, steady delivery for factual content."),
    DefaultVoice("Emma", "female", "A smooth, natural female voice with a relaxed pace, subtle warmth, and a contemporary tone that feels confident, personable, and easygoing."),
    DefaultVoice("Adrian", "male", "A deep, focused male voice with crisp articulation, measured pacing, and a composed tone that feels authoritative, clear, and professional."),
    DefaultVoice("Grace", "female", "A gentle, soothing female voice with soft clarity, unhurried pacing, and a reassuring tone that feels kind, calm, and comforting."),
    DefaultVoice("Owen", "male", "A grounded male voice with even pacing and a dry, composed tone that feels steady, natural, and quietly confident."),
    DefaultVoice("Mina", "female", "A soft, thoughtful female voice with gentle clarity, steady pacing, and a warm tone that feels composed, sincere, and easy to listen to."),
    DefaultVoice("Kenji", "male", "A calm, precise male voice with smooth clarity, balanced pacing, and a composed tone that feels respectful, modern, and trustworthy."),
    DefaultVoice("Rafael", "male", "A clear, composed male voice with a warm Spanish accent, balanced pacing, and a confident tone that feels approachable and precise."),
    DefaultVoice("Mateo", "male", "A warm, youthful male voice with a soft Spanish accent, clear pacing, and an open tone that feels sincere, friendly, and optimistic."),
    DefaultVoice("Lucia", "female", "A clear, mature female voice with a natural Spanish accent, steady pacing, and a composed tone that feels warm, focused, and approachable."),
    DefaultVoice("Oliver", "male", "A refined male voice with a smooth British accent, gentle pacing, and a calm tone that feels trustworthy, articulate, and reassuring."),
    DefaultVoice("Arthur", "male", "A deep, mature male voice with a rich British accent, measured pacing, and a textured tone that feels composed, assured, and quietly powerful."),
    DefaultVoice("Isla", "female", "A lively female voice with a bright British accent, clear delivery, and expressive energy that feels fresh, friendly, and naturally engaging."),
    DefaultVoice("Victoria", "female", "A poised female voice with a refined British accent, smooth pacing, and a lightly textured tone that feels elegant, confident, and composed."),
    DefaultVoice("Cooper", "male", "A bold male voice with a strong Australian accent, relaxed pacing, and a casual tone that feels confident, rugged, and easygoing."),
    DefaultVoice("Mason", "male", "A relaxed male voice with a natural Australian accent, smooth pacing, and a casual tone that feels friendly, grounded, and effortlessly confident."),
    DefaultVoice("Ruby", "female", "A confident female voice with a natural Australian accent, lively pacing, and a warm tone that feels personable, sharp, and engaging."),
    DefaultVoice("Arjun", "male", "A deep male voice with a natural Indian accent, warm resonance, and an easygoing tone that feels friendly, grounded, and confident."),
    DefaultVoice("Rohan", "male", "A lively male voice with a natural Indian accent, expressive rhythm, and confident energy that feels charismatic, upbeat, and full of personality."),
    DefaultVoice("Priya", "female", "A clear female voice with a natural Indian accent, warm pacing, and a composed tone that feels helpful, attentive, and easy to trust."),
    DefaultVoice("Trevor", "male", "A casual male voice with a slightly gritty tone and a confident, unrehearsed style."),
    DefaultVoice("Evan", "male", "A friendly, clear male voice that makes detailed information easy to follow."),
    DefaultVoice("Nathan", "male", "A deep, gentle male voice with a soft, calming delivery."),
    DefaultVoice("Karan", "male", "A deep, friendly male voice with an Indian accent and a light, upbeat touch."),
    DefaultVoice("Wesley", "male", "A rich, deep male voice with a smooth and authoritative sound."),
    DefaultVoice("Curtis", "male", "A deep, resonant male voice with a relaxed and comforting tone."),
    DefaultVoice("Preston", "male", "A confident, friendly male voice with lively emphasis and a persuasive tone."),
    DefaultVoice("Walter", "male", "A deep, mature male voice with an earnest, kind, and sincere delivery."),
    DefaultVoice("Russell", "male", "A deep, gravelly male voice with a warm and intimate delivery."),
    DefaultVoice("Tunde", "male", "A young nigerian male voice with a deep, melodic sound."),
    DefaultVoice("Nigel", "male", "A warm, neutral British male voice with a clean and balanced delivery."),
    DefaultVoice("Shane", "male", "A pleasant male voice with clear speech and steady, easy pacing."),
    DefaultVoice("Silas", "male", "An older, deep male voice with a rugged Southern drawl and a wise, unhurried style."),
    DefaultVoice("Reid", "male", "A calm, precise male voice with a steady and trustworthy delivery."),
    DefaultVoice("Emilio", "male", "A lively male voice with a Mexican accent and a warm, joyful energy."),
    DefaultVoice("Dominic", "male", "A deep male voice with a wide emotional range and a bold, dramatic delivery."),
    DefaultVoice("Elliot", "male", "A soft-spoken British male voice with warm, calm, and even delivery."),
    DefaultVoice("Hugo", "male", "A deep British male voice with a dramatic, polished storytelling style."),
    DefaultVoice("Sebastian", "male", "A deep, gravelly British male voice with an intimate and understated sarcastic tone."),
    DefaultVoice("Haruto", "male", "A young male voice with a calm, smooth, and measured delivery."),
    DefaultVoice("Freddie", "male", "A youthful British male voice with a warm, friendly, and enthusiastic style."),
    DefaultVoice("Logan", "male", "A warm, conversational male voice with a natural and quietly confident delivery."),
    DefaultVoice("Poppy", "female", "A youthful British female voice with bright energy and an easy, casual style."),
    DefaultVoice("Harper", "female", "A relaxed female voice with natural pauses and a casual, unscripted delivery."),
    DefaultVoice("Cordelia", "female", "A calm, polished female character voice with an elegant but menacing edge."),
    DefaultVoice("Reese", "female", "A young female voice with confident energy and a bold, determined delivery."),
    DefaultVoice("Hazel", "female", "A warm female voice with a clear, natural, and understated delivery."),
    DefaultVoice("Juliet", "female", "A warm, soothing female voice with slow, poetic phrasing."),
    DefaultVoice("Piper", "female", "A cheerful female voice with upbeat energy and a bright, lively delivery."),
    DefaultVoice("Reyna", "female", "A mature female voice with a powerful theatrical range and a warm, commanding presence."),
    DefaultVoice("Sloane", "female", "A crisp, articulate female narrator with enough range to keep long stories engaging."),
    DefaultVoice("Iris", "female", "A clear, patient female voice with a gentle and reassuring tone."),
    DefaultVoice("Freya", "female", "A clear, bright British female voice with a neutral and easy-to-follow delivery."),
    DefaultVoice("Brooke", "female", "A bright American female voice with a casual, friendly conversational style."),
    DefaultVoice("Bonnie", "female", "A soft Australian female voice with a gentle, calming delivery."),
    DefaultVoice("Arabella", "female", "A rich, silky British female voice with a soft and sophisticated delivery."),
    DefaultVoice("Colleen", "female", "A warm, clear female voice that moves naturally between storytelling and instruction."),
    DefaultVoice("Margo", "female", "A dry female voice with flat delivery and sharp, understated sarcasm."),
    DefaultVoice("Bianca", "female", "A serious female voice with a Brazilian accent and crisp, deliberate delivery."),
    DefaultVoice("Sari", "female", "A calm female voice with a gentle, deep tone and clear delivery."),
    DefaultVoice("Maya", "female", "A steady, clear voice with a natural presence and measured delivery that feels confident, warm, and easy to listen to."),
    DefaultVoice("Noah", "male", "A lively, youthful male voice with crisp clarity, quick natural pacing, and an upbeat tone that feels friendly, expressive, and modern."),
    DefaultVoice("Jack", "male", "A friendly, confident male voice with clear articulation, steady energy, and a natural tone that feels approachable, upbeat, and sincere."),
    DefaultVoice("Claire", "female", "A polished, articulate female voice with a bright tone, smooth pacing, and a confident presence that feels refined, clear, and approachable."),
    DefaultVoice("Sofia", "female", "A bright, friendly female voice with a natural Spanish accent, clear articulation, and an inviting tone that feels warm, confident, and easy to follow."),
    DefaultVoice("Elise", "female", "A warm female voice with a natural Australian accent, clear pronunciation, and a confident tone that feels supportive, polished, and easy to follow."),
    DefaultVoice("Meera", "female", "A polished female voice with a natural Indian accent, crisp articulation, and a steady tone that feels professional, reassuring, and dependable."),
]


def default_voices_by_gender(gender: str) -> list[DefaultVoice]:
    g = gender.lower()
    return [v for v in DEFAULT_VOICE_CATALOG if v.gender.lower() == g]
