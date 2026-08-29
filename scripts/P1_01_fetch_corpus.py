#!/usr/bin/env python
"""
Phase 1 data acquisition (Section 7) — download the free open corpora that teach the
model to *hear* (Mimi tokens -> language + prosody), before Phase 2 teaches it to
*referee*. This was previously a "bring your own" gap in the pipeline; this script
closes it with real, verified sources (see configs/phase1_corpus.yaml for citations).

Sources per language (see configs/phase1_corpus.yaml, each entry has a `weight` and a
`note` explaining why it's in the mix):
    en   LibriSpeech (60%) + FLEURS (40%)
    hi   Kathbath (45%) + Shrutilipi (40%) + FLEURS (15%)
    gu   Kathbath (45%) + Shrutilipi (35%) + IndicTTS-Gujarati (20%)

All free (CC0 or a click-through research license), and — as of this config — none
gated. (Common Voice was the one gated source here; it was removed entirely on
2026-08-29 after Mozilla pulled it off Hugging Face — "Effective October 2025, Mozilla
Common Voice datasets are now exclusively available through Mozilla Data Collective."
Not a token/gating issue, the HF repo itself is empty. See configs/phase1_corpus.yaml's
header comment for the full story and what replaced/absorbed its weight per language.)

Built for Kaggle's constraints: every source is streamed (`datasets` library,
`streaming=True`) so nothing downloads more than what's needed — we stop per-source the
moment its share of the language's target_hours is reached, so disk/bandwidth stay
bounded regardless of how large the upstream dataset actually is (Shrutilipi alone is
6400h+; we only ever pull the ~40-50h this config asks for it).

Resumable: a JSONL manifest (`data/phase1_raw/manifest.jsonl`) is the source of truth —
on restart, already-written clips per source are skipped (re-streamed past, not
re-saved) and the run continues toward the same per-source target.

    conda activate llms
    pip install datasets soundfile
    # no HF_TOKEN needed — every default source is public/ungated

    # 1. see the plan without downloading anything:
    python scripts/P1_01_fetch_corpus.py --config configs/phase1_corpus.yaml --dry-run

    # 2. fetch (resumable — safe to stop and re-run):
    python scripts/P1_01_fetch_corpus.py --config configs/phase1_corpus.yaml

    # 3. then encode + build Phase-1 frames (see scripts/P1_02_build_frames.py):
    python scripts/00_encode_audio.py --wav-dir data/phase1_raw/hi --out-dir data/encoded
    python scripts/P1_02_build_frames.py --lang hi
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import env

# Column-name candidates tried, in order, when a source doesn't set an explicit
# text_col/gender_col override — different HF datasets name these differently and we'd
# rather auto-detect than hard-crash on a guess that turns out wrong for one source.
_TEXT_COL_CANDIDATES = ["sentence", "text", "transcription", "transcript", "normalized_text"]
_GENDER_COL_CANDIDATES = ["gender", "sex", "speaker_gender"]


# --------------------------------------------------------------------------- #
@dataclass
class SourceSpec:
    id: str
    hf_dataset: str
    hf_config: str | None
    split: str
    gated: bool = False
    weight: float = 1.0
    note: str = ""
    audio_col: str | None = None    # override; else auto-detect the Audio-typed column
    text_col: str | None = None     # override; else try _TEXT_COL_CANDIDATES
    gender_col: str | None = None   # override; else try _GENDER_COL_CANDIDATES

    @staticmethod
    def from_dict(d: dict) -> "SourceSpec":
        known = {f for f in SourceSpec.__dataclass_fields__}
        return SourceSpec(**{k: v for k, v in d.items() if k in known})


@dataclass
class Phase1CorpusConfig:
    target_hours: dict[str, float]
    sources: dict[str, list[SourceSpec]]
    gender_balance: bool = True
    sample_rate: int = 24000
    min_clip_seconds: float = 1.5
    max_clip_seconds: float = 20.0

    @staticmethod
    def from_yaml(path: str) -> "Phase1CorpusConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        sources = {
            lang: [SourceSpec.from_dict(s) for s in specs]
            for lang, specs in (raw.get("sources") or {}).items()
        }
        return Phase1CorpusConfig(
            target_hours=raw.get("target_hours", {}),
            sources=sources,
            gender_balance=bool(raw.get("gender_balance", True)),
            sample_rate=int(raw.get("sample_rate", 24000)),
            min_clip_seconds=float(raw.get("min_clip_seconds", 1.5)),
            max_clip_seconds=float(raw.get("max_clip_seconds", 20.0)),
        )


# --------------------------------------------------------------------------- #
def _manifest_path(out_dir: Path) -> Path:
    return out_dir / "manifest.jsonl"


def _existing_written(manifest_path: Path) -> dict[tuple[str, str], dict]:
    """
    {(lang, source_id): {"count": N, "hours": H, "female": F, "male": M}} from what's
    already on disk — the resume checkpoint. File-based, same pattern as script 02.
    """
    stats: dict[tuple[str, str], dict] = {}
    if not manifest_path.exists():
        return stats
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (rec.get("lang"), rec.get("source"))
        s = stats.setdefault(key, {"count": 0, "hours": 0.0, "female": 0, "male": 0})
        s["count"] += 1
        s["hours"] += rec.get("duration_s", 0.0) / 3600.0
        g = rec.get("gender")
        if g in ("female", "male"):
            s[g] += 1
    return stats


def _clip_id(lang: str, source_id: str, row_index: int) -> str:
    raw = f"{lang}|{source_id}|{row_index}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _pick_column(candidates: list[str], feature_names: list[str]) -> str | None:
    for c in candidates:
        if c in feature_names:
            return c
    return None


def _relative_or_absolute(path: Path) -> str:
    """Store paths relative to ROOT when possible (portable manifest); fall back to the
    absolute path if --out-dir was ever pointed outside the project (e.g. in tests)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _detect_audio_column(row: dict) -> str | None:
    for k, v in row.items():
        if isinstance(v, dict) and "array" in v and "sampling_rate" in v:
            return k
    return None


# --------------------------------------------------------------------------- #
def fetch_source(
    cfg: Phase1CorpusConfig,
    lang: str,
    spec: SourceSpec,
    out_dir: Path,
    manifest_fh,
    already: dict,
    hf_token: str | None,
    dry_run: bool,
) -> dict:
    """Stream one source, writing wavs + manifest rows until this source's share of
    target_hours is reached (or the stream runs out). Returns a small result summary."""
    target_h = cfg.target_hours.get(lang, 0.0) * spec.weight
    have_h = already.get((lang, spec.id), {}).get("hours", 0.0)
    have_n = already.get((lang, spec.id), {}).get("count", 0)

    if have_h >= target_h:
        return {"lang": lang, "source": spec.id, "status": "already_done",
               "have_hours": have_h, "target_hours": target_h}

    if dry_run:
        return {"lang": lang, "source": spec.id, "status": "dry_run",
               "have_hours": have_h, "target_hours": target_h,
               "remaining_hours": max(0.0, target_h - have_h),
               "hf_dataset": spec.hf_dataset, "hf_config": spec.hf_config,
               "gated": spec.gated}

    try:
        from datasets import Audio, load_dataset
    except ImportError:
        raise SystemExit("`datasets` not installed. `pip install datasets soundfile`.")

    try:
        import soundfile as sf
    except ImportError:
        raise SystemExit("`soundfile` not installed. `pip install soundfile`.")

    try:
        ds = load_dataset(
            spec.hf_dataset, spec.hf_config, split=spec.split,
            streaming=True, token=hf_token if spec.gated else None,
        )
    except Exception as e:
        msg = str(e)
        if spec.gated and ("gated" in msg.lower() or "401" in msg or "403" in msg):
            raise SystemExit(
                f"'{spec.hf_dataset}' is gated. Visit "
                f"https://huggingface.co/datasets/{spec.hf_dataset}, log in, and click "
                f"'Agree and access repository' — then re-run with HF_TOKEN set."
            )
        raise SystemExit(f"failed to open {spec.hf_dataset}/{spec.hf_config}: {e}")

    ds = ds.cast_column(spec.audio_col or "audio", Audio(sampling_rate=cfg.sample_rate))

    # FLAT per-language dir (not lang/source/) — scripts/00_encode_audio.py globs
    # "*.wav" non-recursively, so this must match exactly what it expects. The source
    # id is folded into the filename instead, so provenance is still visible.
    lang_dir = out_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_for_resume = have_n
    audio_col = spec.audio_col
    text_col = spec.text_col
    gender_col = spec.gender_col
    row_index = -1

    for row in ds:
        row_index += 1
        if audio_col is None:
            audio_col = spec.audio_col or _detect_audio_column(row)
            if audio_col is None:
                raise SystemExit(
                    f"couldn't find an audio column in {spec.hf_dataset}/{spec.hf_config} "
                    f"— set `audio_col:` explicitly in configs/phase1_corpus.yaml"
                )
        if text_col is None:
            text_col = spec.text_col or _pick_column(_TEXT_COL_CANDIDATES, list(row.keys()))
            if text_col is None:
                raise SystemExit(
                    f"couldn't find a transcript column in {spec.hf_dataset}/{spec.hf_config} "
                    f"(tried {_TEXT_COL_CANDIDATES}) — set `text_col:` explicitly in "
                    f"configs/phase1_corpus.yaml"
                )
        if gender_col is None and cfg.gender_balance:
            gender_col = spec.gender_col or _pick_column(_GENDER_COL_CANDIDATES, list(row.keys()))
            # may legitimately stay None (Kathbath/Shrutilipi don't carry gender) — fine

        if skipped_for_resume > 0:
            skipped_for_resume -= 1
            continue  # already saved from a previous run; skip the write, keep streaming

        audio = row.get(audio_col)
        if not audio or "array" not in audio:
            continue
        arr, sr = audio["array"], audio["sampling_rate"]
        duration_s = len(arr) / float(sr)
        if duration_s < cfg.min_clip_seconds or duration_s > cfg.max_clip_seconds:
            continue

        transcript = str(row.get(text_col, "") or "").strip()
        if not transcript:
            continue

        gender = None
        if gender_col:
            raw_g = str(row.get(gender_col, "") or "").lower()
            if raw_g.startswith("f"):
                gender = "female"
            elif raw_g.startswith("m"):
                gender = "male"

        clip_id = _clip_id(lang, spec.id, row_index)
        wav_path = lang_dir / f"{spec.id}_{clip_id}.wav"
        sf.write(str(wav_path), arr, sr)

        manifest_fh.write(json.dumps({
            "id": clip_id, "lang": lang, "source": spec.id,
            "wav_path": _relative_or_absolute(wav_path),
            "transcript": transcript, "gender": gender,
            "duration_s": round(duration_s, 3),
        }, ensure_ascii=False) + "\n")
        manifest_fh.flush()

        written += 1
        have_h += duration_s / 3600.0
        if have_h >= target_h:
            break

    return {"lang": lang, "source": spec.id, "status": "ok",
           "written": written, "have_hours": round(have_h, 3), "target_hours": round(target_h, 3)}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_corpus.yaml")
    ap.add_argument("--out-dir", default="data/phase1_raw")
    ap.add_argument("--lang", default=None, help="only fetch this language (en/hi/gu)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (targets, gating, remaining hours) without downloading")
    args = ap.parse_args()

    cfg = Phase1CorpusConfig.from_yaml(ROOT / args.config)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(out_dir)
    already = _existing_written(manifest_path)
    hf_token = env("HF_TOKEN")

    langs = [args.lang] if args.lang else list(cfg.sources.keys())

    print("=" * 68)
    print(f"ThinkSpark-v2-350M — Phase-1 corpus {'plan' if args.dry_run else 'fetch'}")
    print("=" * 68)

    manifest_fh = None if args.dry_run else manifest_path.open("a", encoding="utf-8")
    results = []
    try:
        for lang in langs:
            target_h = cfg.target_hours.get(lang, 0.0)
            print(f"\n[{lang}]  target = {target_h:.0f}h")
            for spec in cfg.sources.get(lang, []):
                gate_note = "  (GATED — needs HF_TOKEN + one-time browser accept)" if spec.gated else ""
                print(f"  - {spec.id:<14} weight={spec.weight:.2f}  "
                     f"{spec.hf_dataset}/{spec.hf_config or spec.split}{gate_note}")
                r = fetch_source(cfg, lang, spec, out_dir, manifest_fh, already, hf_token, args.dry_run)
                results.append(r)
                if r["status"] == "dry_run":
                    print(f"      have={r['have_hours']:.2f}h  target={r['target_hours']:.2f}h  "
                         f"remaining={r['remaining_hours']:.2f}h")
                elif r["status"] == "already_done":
                    print(f"      already done: {r['have_hours']:.2f}h >= {r['target_hours']:.2f}h target")
                else:
                    print(f"      wrote {r['written']} clips -> {r['have_hours']:.2f}h "
                         f"(target {r['target_hours']:.2f}h)")
    finally:
        if manifest_fh:
            manifest_fh.close()

    total_h = sum(r.get("have_hours", 0.0) for r in results if r["lang"] in langs) if not args.dry_run else None
    print("\n" + "=" * 68)
    if args.dry_run:
        remaining = sum(r.get("remaining_hours", 0.0) for r in results)
        print(f"plan only — nothing downloaded. ~{remaining:.1f}h remaining across all sources shown.")
        print("re-run without --dry-run to actually fetch.")
    else:
        print(f"done. manifest -> {manifest_path}")
        print("next: python scripts/00_encode_audio.py --wav-dir data/phase1_raw/<lang> --out-dir data/encoded")
        print("      python scripts/P1_02_build_frames.py --lang <lang>")


if __name__ == "__main__":
    main()
