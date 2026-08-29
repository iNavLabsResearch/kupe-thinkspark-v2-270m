"""
Typed configuration loading from YAML + environment (.env).

Every script takes a `--config path.yaml`. Secrets (API keys, HF token) live only in
the environment / .env and are never written into a config file. `load_env()` reads a
`.env` next to the project root without adding a hard python-dotenv dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# .env loading (tiny, dependency-free)
# --------------------------------------------------------------------------- #
def load_env(path: str | os.PathLike | None = None) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (does not overwrite)."""
    env_path = Path(path) if path else PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.environ.get(name, default)
    if required and not val:
        raise RuntimeError(
            f"Missing required env var {name!r}. Copy .env.example -> .env and set it."
        )
    return val


# --------------------------------------------------------------------------- #
# Data-generation config (Section 8)
# --------------------------------------------------------------------------- #
@dataclass
class DataGenConfig:
    total_hours: float = 55.0            # target user-audio hours (Section 8)
    avg_window_seconds: float = 22.0     # mean scenario length (20-30 s windows)
    # Target spoken length of the USER line, stamped onto every job in the plan so the
    # LLM writes user_text of the right length (longer text -> longer Soniox audio ->
    # longer frame windows; the render/frame/encode pipeline is length-agnostic, so this
    # single knob is all that's needed). One of thinkspark.vocab.LENGTH_BANDS.
    #   "short"    ~1-2 s  — terse one-liners (the original corpus; keep for coverage)
    #   "extended" ~3-8 s  — multi-clause turns with natural mid-utterance pauses
    # Default "short" keeps the existing plan/keys byte-identical; the extended corpus is
    # generated additively via configs/data_gen_extended.yaml (separate output files).
    utterance_length: str = "short"
    # OpenAI-SDK LLM (scenario/label writer). base_url lets you point at any
    # OpenAI-compatible endpoint (OpenAI, Sarvam v2, vLLM, Together, a DeepSeek/Gemma3
    # host, ...). e.g. deepseek-v3-flash, gemma-3-27b-it via your chosen provider.
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_temperature: float = 0.9
    llm_max_tokens: int = 1200          # PER SCENARIO; scaled up x batch_size at call time
    llm_concurrency: int = 30           # concurrent worker threads hitting the LLM
    # Batched generation: ask for `batch_size` DISTINCT scenarios per LLM call instead of
    # one-at-a-time, for throughput. Kept modest (<=15 recommended) because bigger batches
    # risk near-duplicate / cross-contaminated items from small-fast models (Section 8.4
    # batching note) — every item is still independently schema-validated and only the
    # bad ones are re-requested, never the whole batch.
    batch_size: int = 12
    llm_max_retries: int = 5
    # LLM-judge for naturalness (Section 8.5)
    judge_model: str = "gpt-4o-mini"
    # Soniox TTS — verified against a real working client (kupe-tts), not guessed.
    # NOTE: this is the TTS endpoint, distinct from Soniox's STT endpoint
    # (stt-rt.soniox.com) — an earlier version of this config pointed at STT by
    # mistake, which silently produced empty audio (see thinkspark.tts_soniox docstring).
    soniox_ws_url: str = "wss://tts-rt.soniox.com/tts-websocket"
    soniox_model: str = "tts-rt-v2"
    soniox_sample_rate: int = 24000
    soniox_max_retries: int = 5     # exponential backoff; rate-limit hits back off harder
    # THE render-speed ceiling. Soniox's default account limit is 3 CONCURRENT STREAMS,
    # counted account-wide across ALL your WebSocket connections (soniox.com/docs/tts/rt/
    # limits-and-quotas). So AT MOST 3 clips can ever be generating at once, no matter how
    # many sockets you open. This value = the render script's worker-thread count, one
    # active stream per thread, so 3 already saturates that cap. (The separate "5 streams
    # per connection" figure is a per-connection sub-limit; it does NOT raise the 3
    # account-wide cap — trying to run 3x5=15 concurrent just trips the cap and gets you
    # rate-limited, i.e. SLOWER.) The ONLY way to go faster than 3-at-once is to request a
    # higher "Concurrent streams" limit in the Soniox Console (the docs say this one CAN
    # be raised), then set this to that number — the code scales cleanly (N threads, N
    # connections, N concurrent streams).
    soniox_concurrency: int = 3
    # Soniox's other documented limit is 100 NEW STREAM STARTS/MINUTE, account-wide (same
    # docs page). thinkspark.tts_soniox.SonioxTTS proactively paces starts under this (see
    # `_throttle_stream_start`) instead of only reacting to 429s after the fact. 90 leaves
    # a 10% safety margin under 100; push toward ~99 for a bit more if you never see 429s.
    # For very short clips this (not soniox_concurrency) is the binding limit; for the
    # ~3-8 s extended corpus it never binds, so concurrency is the limit there. If you get
    # the Console to raise BOTH limits, raise this alongside soniox_concurrency.
    soniox_max_stream_starts_per_min: int = 90
    # Voice profiles (scripts/15_create_voice_profiles.py) — resolve_voice() uses ONLY
    # what's in this file, never a Soniox catalog fetch. Mix of "cloned" (your own ref
    # clips) and "default" (named Soniox stock voices, used to fill the gap when you
    # submit more ref clips than the account can actually clone).
    voice_profiles_path: str = "data/voice_refs/voice_profiles.json"
    # Account-wide cap on TOTAL cloned voices, reported by you against your real Soniox
    # account (not documented publicly by Soniox as of this writing — override if yours
    # differs). scripts/15_create_voice_profiles.py clones up to this many across BOTH
    # genders combined, proportional to how many ref clips you submitted per gender, and
    # fills any shortfall with named default voices instead of cloning past the cap.
    soniox_max_cloned_voices: int = 20
    # Mimi codec
    mimi_repo: str = "kyutai/mimi"
    mimi_sample_rate: int = 24000
    # distribution overrides (empty -> use vocab defaults)
    behaviour_shares: dict[str, float] = field(default_factory=dict)
    language_shares: dict[str, float] = field(default_factory=dict)
    domain_shares: dict[str, float] = field(default_factory=dict)
    # generation sharding
    num_parts: int = 8                    # break generation into N resumable parts
    seed: int = 1234

    # ---- cost tracking (Section 13 budget) --------------------------------
    # Fill these in for whichever LLM provider/model you actually point llm_model at
    # (DeepSeek V3/V4 flash, Gemma-3-27B, gpt-4o-mini, ...) — pricing varies by provider
    # and changes over time, so it is NOT hard-coded; 0.0 just means "cost unknown".
    llm_price_in_per_1m_usd: float = 0.0   # USD per 1M input/prompt tokens
    llm_price_out_per_1m_usd: float = 0.0  # USD per 1M output/completion tokens
    soniox_price_per_hour_usd: float = 0.70  # ~verified against soniox.com/pricing (Section 13)
    inr_per_usd: float = 83.0              # only used for the printed INR budget summary
    budget_inr_target: float = 5000.0      # Section 13 target; scripts/11_monitor.py tracks % used
    # SQLite run/cost tracking DB (Section 8, audit log — see thinkspark.db)
    db_path: str = "data/thinkspark_runs.db"

    @staticmethod
    def from_yaml(path: str | os.PathLike) -> "DataGenConfig":
        raw = _read_yaml(path)
        return _fill(DataGenConfig, raw)


# --------------------------------------------------------------------------- #
# Training config (Section 9)
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    phase: int = 2                       # 1 = modality alignment, 2 = referee
    base_model: str = "google/gemma-3-270m"
    hf_token_env: str = "HF_TOKEN"
    # data
    frames_dir: str = "data/frames"
    encoded_dir: str = "data/encoded"
    seq_len: int = 1024
    # optimisation (Section 9.4)
    optimizer: str = "adamw"
    weight_decay: float = 0.1
    lr: float = 1e-4                     # Phase-2 default; Phase-1 uses 2e-4
    warmup_ratio: float = 0.03
    batch_size: int = 8
    grad_accum: int = 4                  # effective batch 32-64
    epochs: int = 4                      # P1: 1-2, P2: 3-5
    precision: str = "bf16"
    grad_checkpointing: bool = True
    frame_drop_aug: float = 0.05         # 5% frame drop-augmentation robustness
    # LoRA / partial unfreeze (Phase 1 uses LoRA)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # loss weights (Section 9.1)
    lambda_vap: float = 0.3              # Phase-1 alignment aux
    lambda_ctrl: float = 1.0            # Phase-2 control head
    lambda_txt: float = 0.5             # Phase-2 spoken head
    lambda_vap_p2: float = 0.2          # Phase-2 VAP aux
    focal_gamma: float = 2.0
    vap_horizon: int = 25               # future 80 ms bins for VAP aux (~2 s)
    # io
    out_dir: str = "artifacts/thinkspark-v2-350m"
    log_every: int = 20
    save_every: int = 500
    seed: int = 1234
    # ddp (2x T4)
    ddp: bool = False

    @staticmethod
    def from_yaml(path: str | os.PathLike) -> "TrainConfig":
        raw = _read_yaml(path)
        return _fill(TrainConfig, raw)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read_yaml(path: str | os.PathLike) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _fill(cls, raw: dict[str, Any]):
    """Construct a dataclass, ignoring unknown keys, keeping declared defaults."""
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in raw.items() if k in known}
    return cls(**kwargs)
