#!/usr/bin/env python
"""
Phase 1 data acquisition (Section 7) — download the free open corpora that teach the
model to *hear* (Mimi tokens -> language + prosody), before Phase 2 teaches it to
*referee*. This was previously a "bring your own" gap in the pipeline; this script
closes it with real, verified sources (see configs/phase1_corpus.yaml for citations).

This is the single-source-at-a-time CLI. For a much faster local run — concurrent
downloads across sources, encoding + frame-building starting the moment each language's
downloads finish (not waiting for everything), plus continuous background upload to
Hugging Face — see scripts/P1_00_pipeline.py instead. Both share the exact same fetch
logic (thinkspark.phase1_corpus.fetch_source), so results are identical either way.

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
6400h+; we only ever pull the ~40-50h this config asks for it). Transient network
failures (a real observed one: '[Errno 9] Bad file descriptor' from a Kaggle network
blip) are retried automatically rather than crashing the whole fetch — see
thinkspark.phase1_corpus's module docstring for the retry design.

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

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import env
from thinkspark.phase1_corpus import (
    Phase1CorpusConfig, existing_written, fetch_source, manifest_path,
)


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
    manifest_file = manifest_path(out_dir)
    already = existing_written(manifest_file)
    hf_token = env("HF_TOKEN")

    langs = [args.lang] if args.lang else list(cfg.sources.keys())

    print("=" * 68)
    print(f"ThinkSpark-v2-350M — Phase-1 corpus {'plan' if args.dry_run else 'fetch'}")
    print("=" * 68)

    manifest_fh = None if args.dry_run else manifest_file.open("a", encoding="utf-8")
    results = []
    try:
        for lang in langs:
            target_h = cfg.target_hours.get(lang, 0.0)
            print(f"\n[{lang}]  target = {target_h:.0f}h")
            for spec in cfg.sources.get(lang, []):
                gate_note = "  (GATED — needs HF_TOKEN + one-time browser accept)" if spec.gated else ""
                print(f"  - {spec.id:<14} weight={spec.weight:.2f}  "
                     f"{spec.hf_dataset}/{spec.hf_config or spec.split}{gate_note}")
                r = fetch_source(cfg, lang, spec, out_dir, ROOT, manifest_fh, already,
                                 hf_token, args.dry_run,
                                 log_fn=lambda m: print(f"      {m}"))
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

    print("\n" + "=" * 68)
    if args.dry_run:
        remaining = sum(r.get("remaining_hours", 0.0) for r in results)
        print(f"plan only — nothing downloaded. ~{remaining:.1f}h remaining across all sources shown.")
        print("re-run without --dry-run to actually fetch.")
    else:
        print(f"done. manifest -> {manifest_file}")
        print("next: python scripts/00_encode_audio.py --wav-dir data/phase1_raw/<lang> --out-dir data/encoded")
        print("      python scripts/P1_02_build_frames.py --lang <lang>")


if __name__ == "__main__":
    main()
