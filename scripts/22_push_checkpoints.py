#!/usr/bin/env python
"""
Push trained model checkpoints to a Hugging Face MODEL repo (default
anuj-inavlabs/kupe-thinkspark-audio-270m).

The trainer (thinkspark.trainer.Trainer.save) writes each checkpoint as a directory:
    <out_dir>/<tag>/model.pt        state dict
    <out_dir>/<tag>/config.json     the exact TrainConfig it was trained with
    <out_dir>/<tag>/tokenizer files
where <tag> is `step<N>` (periodic) or `final`. Phase-1 out_dir is
artifacts/thinkspark-v2-350m/phase1, Phase-2 is .../phase2 (see configs/train_*.yaml).

This uploads them under `<phase>/<tag>/...` in the model repo, so Phase-1 and Phase-2
checkpoints live side by side without colliding:
    phase1/final/model.pt, phase1/step500/model.pt, ...
    phase2/final/model.pt, ...

    conda activate llms
    export HF_TOKEN=hf_...   # WRITE access

    # push every checkpoint under a phase's out_dir (default: only the `final/` one):
    python scripts/22_push_checkpoints.py --ckpt-dir artifacts/thinkspark-v2-350m/phase1 --phase phase1
    python scripts/22_push_checkpoints.py --ckpt-dir artifacts/thinkspark-v2-350m/phase1 --phase phase1 --all
    # or one specific checkpoint dir:
    python scripts/22_push_checkpoints.py --ckpt-dir artifacts/thinkspark-v2-350m/phase1/final --phase phase1 --single
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import env
from thinkspark.hf_upload import create_commit_with_backoff, ensure_repo

DEFAULT_MODEL_REPO = "anuj-inavlabs/kupe-thinkspark-audio-270m"

MODEL_CARD = """\
---
license: gemma
base_model: google/gemma-3-270m
library_name: pytorch
tags:
  - thinkspark
  - full-duplex
  - turn-taking
  - voice-agent
  - mimi
pipeline_tag: audio-classification
---

# ThinkSpark-audio-270M

A 270M-parameter [Gemma-3-270M](https://huggingface.co/google/gemma-3-270m)-based
full-duplex floor-controller for voice agents — it reads a stream of
[Mimi](https://huggingface.co/kyutai/mimi) audio tokens and decides turn-taking
(when to listen, speak, back-channel, or yield the floor), rather than transcribing
speech.

Trained in two phases (see the
[kupe-thinkspark-v2-270m](https://github.com/iNavLabsResearch/kupe-thinkspark-v2-270m)
repo for the full recipe):

- **Phase 1 — modality alignment**: on ~400-450h of free open audio (LibriSpeech,
  AI4Bharat Kathbath/Shrutilipi, IndicTTS, FLEURS across en/hi/gu), teaching the model
  that Mimi tokens carry language + prosody.
- **Phase 2 — referee fine-tune**: on a synthetic turn-taking corpus, teaching the
  actual floor-control decision (per-frame control flags + back-channel timing).

## Checkpoints

```
phase1/final/model.pt     Phase-1 (alignment) final weights
phase1/step<N>/model.pt   Phase-1 periodic checkpoints (if pushed with --all)
phase2/final/model.pt     Phase-2 (referee) final weights — the deployable model
phase2/step<N>/model.pt   Phase-2 periodic checkpoints (if pushed with --all)
```

Each checkpoint directory also carries the exact `config.json` it was trained with and
the tokenizer. Load with `thinkspark.model` + `thinkspark.inference` from the source
repo.

## License

Inherits Gemma's license (base model is `google/gemma-3-270m`).
"""


def _checkpoint_dirs(ckpt_dir: Path, args) -> list[Path]:
    """Which checkpoint dir(s) to push. --single: `ckpt_dir` itself is one checkpoint.
    Otherwise `ckpt_dir` is a phase out_dir holding <tag>/ subdirs: --all pushes every
    tag, default pushes only `final/` (the one you almost always actually want)."""
    if args.single:
        if not (ckpt_dir / "model.pt").exists():
            raise SystemExit(f"--single given but {ckpt_dir}/model.pt doesn't exist")
        return [ckpt_dir]
    tags = sorted(p for p in ckpt_dir.iterdir() if p.is_dir() and (p / "model.pt").exists())
    if not tags:
        raise SystemExit(f"no <tag>/model.pt checkpoints found under {ckpt_dir} — "
                         f"has training saved anything yet?")
    if args.all:
        return tags
    final = ckpt_dir / "final"
    if (final / "model.pt").exists():
        return [final]
    # no final/ yet (training still running / interrupted) — push the newest step*/ so
    # something useful lands rather than erroring.
    newest = max(tags, key=lambda p: p.stat().st_mtime)
    print(f"no final/ checkpoint yet — pushing newest instead: {newest.name}")
    return [newest]


def push_checkpoints(ckpt_dir: Path, phase: str, repo: str, private: bool, args) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    token = env("HF_TOKEN", required=True)
    api = HfApi(token=token)
    ensure_repo(api, repo, private, repo_type="model")   # clear 401/403 messages on failure

    dirs = _checkpoint_dirs(ckpt_dir, args)
    print(f"pushing {len(dirs)} checkpoint dir(s) to model repo {repo} under {phase}/ ...")

    for d in dirs:
        tag = d.name
        files = [f for f in sorted(d.rglob("*")) if f.is_file()]
        if not files:
            print(f"  {tag}: empty, skipping")
            continue
        ops = [CommitOperationAdd(path_in_repo=f"{phase}/{tag}/{f.relative_to(d)}",
                                  path_or_fileobj=str(f)) for f in files]
        print(f"  {tag}: {len(ops)} file(s) "
             f"({sum(f.stat().st_size for f in files) / 1e6:.1f}MB)")
        create_commit_with_backoff(
            api, repo=repo, repo_type="model", operations=ops,
            commit_message=f"{phase}: checkpoint {tag}",
            max_retries=args.max_retries, base_backoff=args.backoff,
            max_backoff=args.max_backoff, log_fn=print,
        )

    # model card (README.md) — small, always safe to overwrite with the current version
    card_path = ROOT / "artifacts" / "_model_card.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(MODEL_CARD, encoding="utf-8")
    create_commit_with_backoff(
        api, repo=repo, repo_type="model",
        operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card_path))],
        commit_message="update model card",
        max_retries=args.max_retries, base_backoff=args.backoff, max_backoff=args.max_backoff,
        log_fn=print,
    )
    print(f"done -> https://huggingface.co/{repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True,
                    help="a phase out_dir (holds <tag>/ subdirs) or, with --single, one "
                        "checkpoint dir directly")
    ap.add_argument("--phase", required=True, choices=["phase1", "phase2"],
                    help="which phase these checkpoints are from — sets the repo subfolder")
    ap.add_argument("--repo", default=DEFAULT_MODEL_REPO,
                    help=f"HF MODEL repo to push to (default {DEFAULT_MODEL_REPO})")
    ap.add_argument("--private", action="store_true", help="create the model repo private")
    ap.add_argument("--all", action="store_true",
                    help="push EVERY checkpoint (all step*/ + final/), not just final/")
    ap.add_argument("--single", action="store_true",
                    help="--ckpt-dir is one checkpoint dir directly, not a phase out_dir")
    ap.add_argument("--backoff", type=float, default=20.0)
    ap.add_argument("--max-backoff", type=float, default=300.0)
    ap.add_argument("--max-retries", type=int, default=10)
    args = ap.parse_args()

    ckpt_dir = ROOT / args.ckpt_dir if not Path(args.ckpt_dir).is_absolute() else Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise SystemExit(f"{ckpt_dir} doesn't exist")

    print("=" * 68)
    print(f"ThinkSpark-audio-270M — push {args.phase} checkpoints -> {args.repo}")
    print("=" * 68)
    push_checkpoints(ckpt_dir, args.phase, args.repo, args.private, args)


if __name__ == "__main__":
    main()
