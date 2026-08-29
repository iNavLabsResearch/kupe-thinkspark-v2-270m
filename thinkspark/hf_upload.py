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


def create_commit_with_backoff(api, *, repo: str, operations: list, commit_message: str,
                               max_retries: int = 10, base_backoff: float = 20.0,
                               max_backoff: float = 300.0, log_fn=log) -> None:
    """Retries a dataset-repo commit on rate-limit (429) or transient network errors,
    with exponential backoff + jitter. Any other error is raised immediately."""
    attempt = 0
    backoff = base_backoff
    while True:
        try:
            api.create_commit(repo_id=repo, repo_type="dataset",
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


def ensure_repo(api, repo: str, private: bool) -> None:
    """Create the dataset repo if it doesn't exist yet, with error messages that
    distinguish a genuinely bad/expired token (401) from a valid token that simply has
    no create-rights in `repo`'s namespace (403 — e.g. an org you're not a member of)."""
    try:
        api.create_repo(repo_id=repo, repo_type="dataset", private=private, exist_ok=True)
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
