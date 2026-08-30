"""
Run-scoped checkpointing + live HF upload + resume, shared by scripts/06_train_phase1.py
and scripts/07_train_phase2.py so the wiring lives in one tested place.

Every training invocation is a RUN with its own id (a timestamp by default, or your own
via --run-id). Checkpoints for a run go to `<out_dir>/runs/<run_id>/<tag>/` locally and,
if a --push-repo is set, are uploaded live DURING training to
`<phase>/runs/<run_id>/<tag>/` on the model repo (plus a `latest.json` pointer). So an
interrupted run can be resumed — from local disk if it's still there, otherwise pulled
back from HF — and continues from the last saved step (model + optimizer + position all
restored). `--fresh` forces starting from scratch, ignoring any existing checkpoints for
the resolved run id.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from thinkspark.hf_upload import (
    download_run_checkpoint, log, make_checkpoint_uploader,
)


def add_run_args(ap, *, default_repo: str) -> None:
    """Register the run/checkpoint/resume CLI flags on an argparse parser."""
    ap.add_argument("--run-id", default=None,
                    help="run id for this training run (checkpoints go to "
                        "<out_dir>/runs/<run-id>/ and, if pushing, <phase>/runs/<run-id>/ "
                        "on the model repo). Default: a fresh UTC timestamp. Pass an "
                        "EXISTING run id (with --resume, or without --fresh) to continue "
                        "that run.")
    ap.add_argument("--resume", action="store_true",
                    help="resume the run named by --run-id from its latest checkpoint "
                        "(local if present, else pulled from --push-repo). Requires "
                        "--run-id. Mutually exclusive with --fresh.")
    ap.add_argument("--fresh", action="store_true",
                    help="start from scratch (step 0), ignoring any existing checkpoints "
                        "for the resolved run id. Mutually exclusive with --resume.")
    ap.add_argument("--push-repo", default=default_repo,
                    help=f"HF MODEL repo to upload each checkpoint to live during "
                        f"training (default {default_repo}). Pass --no-push to train "
                        f"without uploading.")
    ap.add_argument("--no-push", action="store_true",
                    help="disable live checkpoint upload to HF (train + save locally only)")
    ap.add_argument("--push-private", action="store_true",
                    help="create the model repo private on the first upload")


def _newest_local_checkpoint(run_dir: Path) -> Path | None:
    """Newest usable checkpoint dir under a run dir: prefer final/, else the most-recent
    step*/ that actually has a model.pt."""
    if not run_dir.exists():
        return None
    final = run_dir / "final"
    if (final / "model.pt").exists():
        return final
    steps = [p for p in run_dir.iterdir() if p.is_dir() and (p / "model.pt").exists()]
    if not steps:
        return None
    return max(steps, key=lambda p: p.stat().st_mtime)


def wire_run(trainer, cfg, args, *, phase: str, root: Path) -> str:
    """Resolve the run id, point the trainer's checkpoints at the run dir, optionally
    resume, and attach the live-upload hook. Returns the resolved run id.

    Only the main process (rank 0) resumes-from-HF / uploads — every other DDP rank just
    shares the run id and the same local run dir."""
    if args.resume and args.fresh:
        raise SystemExit("--resume and --fresh are mutually exclusive")
    if args.resume and not args.run_id:
        raise SystemExit("--resume needs --run-id (which run to resume)")

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime())

    base_out = Path(cfg.out_dir)
    run_dir = base_out / "runs" / run_id
    trainer.out_dir = run_dir
    if getattr(trainer, "_is_main", True):
        run_dir.mkdir(parents=True, exist_ok=True)

    push_enabled = bool(args.push_repo) and not args.no_push
    token = os.environ.get(cfg.hf_token_env)

    is_main = getattr(trainer, "_is_main", True)

    # ---- resume (main process only) --------------------------------------------------
    if is_main and not args.fresh:
        ckpt = _newest_local_checkpoint(run_dir)
        if ckpt is None and push_enabled:
            # not on this box — try pulling the run back from HF
            log(f"no local checkpoint for run {run_id}; checking {args.push_repo} ...")
            ckpt = download_run_checkpoint(args.push_repo, phase, run_id, run_dir, token)
        if ckpt is not None:
            print(f"resuming run {run_id} from {ckpt}")
            trainer.load_checkpoint(ckpt)
        elif args.resume:
            raise SystemExit(f"--resume given but no checkpoint found for run {run_id} "
                             f"(neither locally at {run_dir} nor on {args.push_repo})")
        else:
            print(f"starting fresh run {run_id} (no existing checkpoint found)")
    elif is_main:
        print(f"starting FRESH run {run_id} (--fresh: ignoring any existing checkpoints)")

    # ---- live upload hook ------------------------------------------------------------
    if push_enabled and is_main:
        trainer.on_checkpoint = make_checkpoint_uploader(
            args.push_repo, phase, run_id, args.push_private, token)
        print(f"live checkpoint upload ON -> {args.push_repo}:{phase}/runs/{run_id}/")
    elif is_main:
        print("live checkpoint upload OFF (--no-push) — checkpoints saved locally only")

    return run_id
