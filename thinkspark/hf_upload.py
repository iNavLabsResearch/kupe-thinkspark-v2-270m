"""
Shared Hugging Face upload helpers — logging, backoff/retry, repo-creation with clear
401-vs-403 error messages, and commit-packing. Used by scripts/13_upload_hf.py (Phase-2
audio+text+timestamps upload) and scripts/P1_00_pipeline.py (Phase-1 continuous
background upload), so both share the exact same tested retry/error-handling behavior.
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc_now()}] {msg}", flush=True)


def is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(n in msg for n in ("429", "rate limit", "ratelimit", "too many requests", "quota", "throttl"))


def is_transient_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(t in msg for t in ("timeout", "temporarily", "connection", "reset", "502", "503", "504"))


def sleep_with_jitter(seconds: float) -> None:
    if seconds <= 0:
        return
    jitter = seconds * 0.15 * (random.random() * 2 - 1)
    time.sleep(max(0.5, seconds + jitter))


def npz_repo_path(lang: str, filename: str) -> str:
    """Where to upload an encoded `.npz` under, bucketed into subdirectories to stay
    under HF/git's real, hard limit of ~10,000 files per directory. Real observed
    failure this fixes: a single language's flat `encoded/<lang>/` directory hit that
    cap mid-upload (25,704 clips for `en` alone) and HF rejected every further commit
    with "too many files per directory", even though the commit itself was fine — the
    directory as a whole was just too big.

    Buckets by the clip_id's own hash (the last 16 hex chars of the filename stem,
    before the extension — see `thinkspark.phase1_corpus.clip_id`, which produces
    exactly that) into up to 256 near-evenly-distributed subdirectories, so even a
    multi-hundred-thousand-clip language stays well under the per-directory limit.
    Filenames are `<source_id>_<clip_id>.npz` — slicing the LAST 16 chars of the stem
    (not splitting on "_") is deliberate: some source ids (kathbath, shrutilipi,
    indictts) don't contain underscores but this is robust even if one ever did.

    Existing files already uploaded flatly (before this fix existed) are left exactly
    where they are — nothing here touches or migrates them. `scripts/19_fetch_training_data.py`
    globs recursively (`encoded/<lang>/**/*.npz`) so it picks up both the old flat files
    and new bucketed ones from the same repo without needing any cleanup step."""
    stem = filename.rsplit(".", 1)[0]
    cid = stem[-16:]
    return f"encoded/{lang}/{cid[:2]}/{filename}"


def create_commit_with_backoff(api, *, repo: str, operations: list, commit_message: str,
                               max_retries: int = 10, base_backoff: float = 20.0,
                               max_backoff: float = 300.0, log_fn=log,
                               repo_type: str = "dataset") -> None:
    """Retries a repo commit on rate-limit (429) or transient network errors, with
    exponential backoff + jitter. Any other error is raised immediately. `repo_type`
    is "dataset" by default (Phase-1/2 data uploads); pass "model" to push model
    checkpoints to a model repo."""
    attempt = 0
    backoff = base_backoff
    while True:
        try:
            api.create_commit(repo_id=repo, repo_type=repo_type,
                              operations=operations, commit_message=commit_message)
            return
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            if is_rate_limit_error(exc):
                wait = min(max_backoff, backoff)
                log_fn(f"rate-limited ({type(exc).__name__}): sleeping {wait:.1f}s "
                      f"(attempt {attempt}/{max_retries})")
                sleep_with_jitter(wait)
                backoff = min(max_backoff, backoff * 2.0)
                continue
            if is_transient_error(exc):
                wait = min(max_backoff, backoff)
                log_fn(f"transient error ({type(exc).__name__}): sleeping {wait:.1f}s "
                      f"(attempt {attempt}/{max_retries}) — {exc}")
                sleep_with_jitter(wait)
                backoff = min(max_backoff, backoff * 1.8)
                continue
            raise


def ensure_repo(api, repo: str, private: bool, repo_type: str = "dataset") -> None:
    """Create the repo if it doesn't exist yet, with error messages that distinguish a
    genuinely bad/expired token (401) from a valid token that simply has no create-rights
    in `repo`'s namespace (403 — e.g. an org you're not a member of). `repo_type` is
    "dataset" by default; pass "model" for a model-checkpoint repo."""
    try:
        api.create_repo(repo_id=repo, repo_type=repo_type, private=private, exist_ok=True)
    except Exception as exc:
        msg = str(exc).lower()
        if "401" in msg or "unauthorized" in msg or "invalid user token" in msg:
            raise RuntimeError(
                "HF token rejected (401) — the token itself is invalid/expired/read-only. "
                "Set HF_TOKEN=hf_... with WRITE access (https://huggingface.co/settings/tokens)."
            ) from exc
        if "403" in msg and ("rights" in msg or "namespace" in msg or "forbidden" in msg):
            namespace = repo.split("/", 1)[0] if "/" in repo else repo
            raise RuntimeError(
                f"HF rejected repo creation (403) — your token IS valid, but your "
                f"account has no repo-create rights under the '{namespace}' namespace. "
                f"If '{namespace}' is an org, you need to actually be a member with "
                f"write/create rights there. Easiest fix: use your own username instead, "
                f"e.g. --repo <your-hf-username>/{repo.split('/', 1)[-1] if '/' in repo else repo}."
            ) from exc
        if "already" not in msg and "409" not in msg and "exist" not in msg:
            raise


def make_checkpoint_uploader(repo: str, phase: str, run_id: str, private: bool,
                             token: str | None):
    """Returns an `on_checkpoint(tag, ckpt_dir)` callable that uploads a checkpoint dir
    live during training, into `<phase>/runs/<run_id>/<tag>/...` on a MODEL repo. Used by
    the training scripts so an interrupted run can be resumed from HF. Also uploads a
    small `<phase>/runs/<run_id>/latest.json` pointer so a resumer knows the newest tag
    without listing the whole repo. Repo is created (idempotently) on the first call."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    ensured = {"done": False}

    def _on_checkpoint(tag: str, ckpt_dir) -> None:
        from pathlib import Path
        ckpt_dir = Path(ckpt_dir)
        if not ensured["done"]:
            ensure_repo(api, repo, private, repo_type="model")
            ensured["done"] = True

        base = f"{phase}/runs/{run_id}"
        files = [f for f in sorted(ckpt_dir.rglob("*")) if f.is_file()]
        ops = [CommitOperationAdd(path_in_repo=f"{base}/{tag}/{f.relative_to(ckpt_dir)}",
                                  path_or_fileobj=str(f)) for f in files]
        # a tiny pointer to the newest checkpoint tag for easy resume
        import json as _json
        import tempfile
        ptr = Path(tempfile.gettempdir()) / f"_latest_{run_id}.json"
        ptr.write_text(_json.dumps({"latest_tag": tag}), encoding="utf-8")
        ops.append(CommitOperationAdd(path_in_repo=f"{base}/latest.json", path_or_fileobj=str(ptr)))

        log(f"uploading checkpoint {tag} -> {repo}:{base}/{tag}/ ({len(files)} files)")
        create_commit_with_backoff(
            api, repo=repo, repo_type="model", operations=ops,
            commit_message=f"{phase} run {run_id}: checkpoint {tag}",
        )

    return _on_checkpoint


def download_run_checkpoint(repo: str, phase: str, run_id: str, dest_dir,
                            token: str | None, tag: str | None = None):
    """Download a run's checkpoint from a MODEL repo into `dest_dir/<tag>/` so training
    can resume from it. If `tag` is None, reads `<phase>/runs/<run_id>/latest.json` to
    find the newest tag. Returns the local Path to the checkpoint dir, or None if the run
    doesn't exist on the repo yet (fresh start)."""
    import json as _json
    from pathlib import Path

    from huggingface_hub import hf_hub_download, snapshot_download

    base = f"{phase}/runs/{run_id}"
    if tag is None:
        try:
            latest_path = hf_hub_download(repo_id=repo, repo_type="model", token=token,
                                          filename=f"{base}/latest.json")
            tag = _json.loads(Path(latest_path).read_text())["latest_tag"]
        except Exception:
            return None   # no such run / no latest pointer yet -> fresh start

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        snap = snapshot_download(repo_id=repo, repo_type="model", token=token,
                                 allow_patterns=[f"{base}/{tag}/*"], local_dir=str(dest_dir / ".hf_resume_tmp"))
    except Exception:
        return None
    src = Path(snap) / base / tag
    if not (src / "model.pt").exists():
        return None
    final = dest_dir / tag
    final.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in src.iterdir():
        shutil.move(str(f), str(final / f.name))
    shutil.rmtree(dest_dir / ".hf_resume_tmp", ignore_errors=True)
    return final


def pack_by_file_count(items: list, files_per_item: int, files_per_commit: int) -> list[list]:
    """Greedy-pack a flat list of items (each contributing `files_per_item` files to a
    commit, e.g. 2 for a wav+timestamps pair) into commits of up to `files_per_commit`
    files. Generic version of scripts/13_upload_hf.py's original pack_commits."""
    commits: list[list] = []
    cur: list = []
    cur_files = 0
    for item in items:
        if cur and cur_files + files_per_item > files_per_commit:
            commits.append(cur)
            cur, cur_files = [], 0
        cur.append(item)
        cur_files += files_per_item
    if cur:
        commits.append(cur)
    return commits
