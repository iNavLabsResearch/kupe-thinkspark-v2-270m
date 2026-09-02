"""
Smoke test: ARTPARK-IISc/DhVaani-0.5 zero-shot TTS with Hindi text.

Loads HF_TOKEN from the project .env (never hardcode the token here), clones from
tests/fixtures/hindi_prompt.wav, and writes a 24 kHz WAV.

Usage (from kupe-thinkspark-v2-350m root):

    conda activate llms   # or whatever env has torch + transformers + soundfile
    python tests/test_dhvaani_hindi.py

Optional knobs:

    python tests/test_dhvaani_hindi.py \\
        --text "नमस्ते, आप कैसे हैं?" \\
        --prompt-wav tests/fixtures/hindi_prompt.wav \\
        --prompt-text "सामाजिक जिम्मेदारी निभाना हर नागरिक का कर्तव्य है।" \\
        --out tests/out_dhvaani_hindi.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thinkspark.config import env, load_env  # noqa: E402

DEFAULT_TEXT = "नमस्ते, आप कैसे हैं? आज का दिन कैसा चल रहा है?"
DEFAULT_PROMPT_WAV = ROOT / "tests" / "fixtures" / "hindi_prompt.wav"
# Transcript of tests/fixtures/hindi_prompt.wav (Soniox Hindi sample).
DEFAULT_PROMPT_TEXT = "सामाजिक जिम्मेदारी निभाना हर नागरिक का कर्तव्य है।"
DEFAULT_OUT = ROOT / "tests" / "out_dhvaani_hindi.wav"
MODEL_ID = "ARTPARK-IISc/DhVaani-0.5"


def main() -> int:
    load_env()  # reads ROOT/.env into os.environ (does not overwrite existing)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Hindi text to synthesize")
    parser.add_argument(
        "--prompt-wav",
        type=Path,
        default=DEFAULT_PROMPT_WAV,
        help="Reference voice WAV (a few seconds)",
    )
    parser.add_argument(
        "--prompt-text",
        default=DEFAULT_PROMPT_TEXT,
        help="Transcript of --prompt-wav",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output WAV path")
    parser.add_argument("--num-step", type=int, default=16, help="Flow steps (16=quality, 8=faster)")
    args = parser.parse_args()

    hf_token = env("HF_TOKEN", required=True)
    if not args.prompt_wav.is_file():
        raise SystemExit(f"prompt wav not found: {args.prompt_wav}")

    import torch
    import soundfile as sf
    from transformers import AutoModel

    # huggingface_hub / transformers pick up HF_TOKEN from the environment; also pass
    # token= explicitly so gated downloads work even if env wiring differs.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_ID} on {device} …")
    model = (
        AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            token=hf_token,
        )
        .to(device)
        .eval()
    )

    print(f"synthesize Hindi text ({len(args.text)} chars) …")
    synth_kwargs = dict(
        text=args.text,
        prompt_wav=str(args.prompt_wav),
        prompt_text=args.prompt_text,
    )
    # num_step is optional on some wrappers; pass only if supported.
    try:
        audio = model.synthesize(**synth_kwargs, num_step=args.num_step)
    except TypeError:
        audio = model.synthesize(**synth_kwargs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sr = int(getattr(model, "sampling_rate", 24000))
    sf.write(str(args.out), audio, sr)
    dur = float(getattr(audio, "shape", [0])[0]) / sr if hasattr(audio, "shape") else 0.0
    print(f"wrote {args.out}  ({dur:.2f}s @ {sr} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
