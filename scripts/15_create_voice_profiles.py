#!/usr/bin/env python
"""
Build voice_profiles.json — a mix of YOUR OWN cloned voices (from data/voice_refs/) and
Soniox's named stock voices, used to fill the gap when your account's cloning cap is
smaller than the number of reference clips you collected (Section 8.3's "gender-
balanced voice profiles"). This project uses ONLY what ends up in this file — Soniox's
full live catalog is never fetched or used; `thinkspark/soniox_default_voices.py` is a
small, explicitly-chosen list of named stock voices you supplied, not a live fetch.

Drop reference clips into data/voice_refs/ (see data/voice_refs/README.md for the
required `<gender>_<name>.<ext>` naming convention), then:

    conda activate llms
    python scripts/15_create_voice_profiles.py --config configs/data_gen.yaml --dry-run
    python scripts/15_create_voice_profiles.py --config configs/data_gen.yaml

Writes data/voice_refs/voice_profiles.json. thinkspark/tts_soniox.py::resolve_voice()
rotates deterministically across whatever voices are in that file for each gender —
03_render_user_audio.py needs no changes to pick them up, but it WILL refuse to start
(before any spend) if a gender it needs has zero voices (cloned or default) yet.

Cap-aware allocation
---------------------
Soniox caps how many voices ONE account can have cloned at once — `soniox_max_cloned_
voices` in configs/data_gen.yaml (default 20; reported against a real account, override
if yours differs). If you submit more reference clips than that cap allows:
  1. The clone budget (cap minus voices already cloned by this project) is split across
     genders proportional to how many NOT-yet-cloned clips each gender still has
     (largest-remainder rounding, never over-allocating past what a gender actually has).
  2. Clips that don't fit the budget are NOT cloned — instead, that many named Soniox
     stock voices of the matching gender (thinkspark/soniox_default_voices.py) are added
     to fill the shortfall, so the final per-gender voice COUNT still matches what you
     submitted, just sourced from stock voices instead of your own clips for the overflow.
Every run recomputes this from scratch against what's already in voice_profiles.json, so
re-running after adding more clips (or raising the cap) just tops up the difference.

Cleanup
-------
`--cleanup` deletes every "cloned" entry from your Soniox account too (best-effort
`DELETE /v1/voices/{id}`, same as kupe-backend's delete_cloned_voice — a failure there
doesn't block removing it locally) and resets voice_profiles.json to empty, with an
"are you sure?" confirmation first. Use this before rebuilding from scratch, e.g. after
changing which clips you're using or raising/lowering the cap.

Protocol (verified against kupe-backend/app/services/providers_service.py::clone_voice
+ _soniox_recompute_and_wait + delete_cloned_voice — NOT guessed):

  1. POST https://api.soniox.com/v1/voices  (multipart/form-data)
       data:  name=<display name>
       files: file=(filename, raw audio bytes, content-type)
     -> {"id": "...", "filename": "...", "models": [...]}

  2. A freshly cloned voice is NOT immediately usable with every TTS model — Soniox
     prepares ("recomputes") each voice for a specific model asynchronously. Using it
     with tts-rt-v2 before that finishes fails with error_type "voice_not_prepared" (or
     error_code 409). So for every new voice:
       POST https://api.soniox.com/v1/voices/{voice_id}/recompute   json={"model": model}
     then poll (kupe-backend: 8 attempts, 1.5s apart):
       GET  https://api.soniox.com/v1/voices/{voice_id}
     until models[] has an entry for `model` with status "ready" or "computed".

  3. DELETE https://api.soniox.com/v1/voices/{voice_id}  — best-effort, used by --cleanup.

Resumable: a filename already recorded in voice_profiles.json (by content hash) is
skipped for (re-)cloning; a default-voice fill is only topped up by however many are
newly missing on this run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import time
import uuid
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import DataGenConfig, env  # noqa: E402
from thinkspark.tts_soniox import SONIOX_API_BASE  # noqa: E402
from thinkspark.soniox_default_voices import default_voices_by_gender  # noqa: E402

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
_NAME_RE = re.compile(r"^(female|male|f|m)_([A-Za-z0-9_-]+)\.\w+$", re.IGNORECASE)
_GENDER_ALIAS = {"f": "female", "m": "male", "female": "female", "male": "male"}
GENDERS = ("female", "male")

RECOMPUTE_POLL_ATTEMPTS = 8      # same cadence as kupe-backend's verified implementation
RECOMPUTE_POLL_INTERVAL_S = 1.5


# --------------------------------------------------------------------------- #
def parse_ref_filename(path: Path) -> tuple[str, str] | None:
    """`<gender>_<name>.<ext>` -> (gender, display_name), or None if it doesn't match."""
    m = _NAME_RE.match(path.name)
    if not m:
        return None
    gender = _GENDER_ALIAS[m.group(1).lower()]
    name = m.group(2)
    return gender, name


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def scan_ref_folder(folder: Path) -> tuple[list[dict], list[Path]]:
    """Returns (matched [{path, gender, name}], skipped [Path, ...])."""
    matched, skipped = [], []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        parsed = parse_ref_filename(path)
        if parsed is None:
            skipped.append(path)
            continue
        gender, name = parsed
        matched.append({"path": path, "gender": gender, "name": name})
    return matched, skipped


def allocate_by_largest_remainder(weights: dict[str, int], total: int) -> dict[str, int]:
    """
    Split an integer `total` across keys proportional to `weights`, largest-remainder
    rounding so the parts always sum to exactly `total` — and never exceed a key's own
    weight (excess capacity is redistributed to whichever key still has room, in
    weight order, since a quota above what a gender actually has is meaningless here).
    """
    keys = list(weights)
    sum_w = sum(weights.values())
    if sum_w <= 0 or total <= 0:
        return {k: 0 for k in keys}
    total = min(total, sum_w)   # never allocate more than exists in total across all keys

    raw = {k: (weights[k] * total / sum_w) for k in keys}
    alloc = {k: min(weights[k], int(raw[k])) for k in keys}
    remainder = total - sum(alloc.values())

    # hand out leftover units by largest fractional remainder first, skipping any key
    # already at its own weight cap
    order = sorted(keys, key=lambda k: raw[k] - int(raw[k]), reverse=True)
    i = 0
    while remainder > 0 and any(alloc[k] < weights[k] for k in keys):
        k = order[i % len(order)]
        if alloc[k] < weights[k]:
            alloc[k] += 1
            remainder -= 1
        i += 1
    return alloc


# --------------------------------------------------------------------------- #
# HTTP — plain urllib (no new dependency), multipart built by hand for the clone
# upload, matching kupe-backend's httpx calls field-for-field.
# --------------------------------------------------------------------------- #
def _multipart_body(fields: dict[str, str], file_field: str, filename: str,
                    file_bytes: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = f"----thinkspark-{uuid.uuid4().hex}"
    parts = []
    for key, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n"
            .encode("utf-8")
        )
    parts.append(
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
         f"filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n").encode("utf-8")
        + file_bytes + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _request(method: str, url: str, api_key: str, *, json_body: dict | None = None,
            body: bytes | None = None, content_type: str | None = None) -> dict:
    import urllib.error
    import urllib.request

    headers = {"Authorization": f"Bearer {api_key}"}
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif body is not None:
        data = body
        headers["Content-Type"] = content_type

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            body_json = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            body_json = {"error_message": raw.decode("utf-8", "replace")[:300]}
        body_json["_http_status"] = e.code
        return body_json


def clone_voice(api_key: str, name: str, path: Path) -> dict:
    """POST /v1/voices — real Soniox voice cloning, multipart upload."""
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body, ctype = _multipart_body({"name": name}, "file", path.name, path.read_bytes(), content_type)
    return _request("POST", f"{SONIOX_API_BASE}/v1/voices", api_key, body=body, content_type=ctype)


def delete_voice(api_key: str, voice_id: str) -> None:
    """DELETE /v1/voices/{id} — best-effort, matches kupe-backend's delete_cloned_voice
    (a failure here shouldn't block removing the voice from your local records)."""
    try:
        _request("DELETE", f"{SONIOX_API_BASE}/v1/voices/{voice_id}", api_key)
    except Exception:
        pass


def recompute_and_wait(api_key: str, voice_id: str, model: str, log) -> None:
    """POST /v1/voices/{id}/recompute then poll GET /v1/voices/{id} until `model` is
    ready — mirrors kupe-backend's _soniox_recompute_and_wait exactly (same endpoints,
    same poll cadence): recompute returns before the model is actually usable, so a bare
    retry right after it would just hit voice_not_prepared again."""
    resp = _request("POST", f"{SONIOX_API_BASE}/v1/voices/{voice_id}/recompute", api_key,
                    json_body={"model": model})
    if resp.get("_http_status", 0) >= 400:
        raise RuntimeError(f"recompute failed: {resp}")

    for attempt in range(RECOMPUTE_POLL_ATTEMPTS):
        time.sleep(RECOMPUTE_POLL_INTERVAL_S)
        status = _request("GET", f"{SONIOX_API_BASE}/v1/voices/{voice_id}", api_key)
        if status.get("_http_status", 0) >= 400:
            continue
        models = status.get("models", [])
        entry = next((m for m in models if m.get("model") == model), None)
        if entry and entry.get("status") in ("ready", "computed"):
            log(f"    model '{model}' ready (poll {attempt + 1}/{RECOMPUTE_POLL_ATTEMPTS})")
            return
        if entry and entry.get("status") == "error":
            raise RuntimeError(f"voice preparation failed: {entry.get('error_message', 'unknown error')}")
    raise RuntimeError(
        f"voice preparation for '{model}' still not ready after "
        f"{RECOMPUTE_POLL_ATTEMPTS * RECOMPUTE_POLL_INTERVAL_S:.0f}s — try re-running this "
        f"script shortly, it will pick up right where it left off."
    )


# --------------------------------------------------------------------------- #
def load_profiles(profiles_path: Path) -> dict:
    if profiles_path.exists():
        try:
            data = json.loads(profiles_path.read_text(encoding="utf-8"))
            # backfill "source" for entries written before this field existed —
            # everything from the old version of this script was a real clone
            for v in data.get("voices", []):
                v.setdefault("source", "cloned")
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"voices": []}


def save_profiles(profiles_path: Path, data: dict) -> None:
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    profiles_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
def _run_cleanup(args, cfg: DataGenConfig, profiles_path: Path) -> None:
    profiles = load_profiles(profiles_path)
    cloned = [v for v in profiles["voices"] if v.get("source") == "cloned"]
    default = [v for v in profiles["voices"] if v.get("source") == "default"]

    print("=" * 68)
    print("ThinkSpark-v2-350M — voice profile cleanup")
    print("=" * 68)
    print(f"{profiles_path}")
    print(f"  {len(cloned)} cloned voice(s) — will be deleted from your Soniox account too "
         f"(best-effort) and from this file")
    print(f"  {len(default)} default (stock) voice reference(s) — will be removed from this "
         f"file only (nothing to delete on Soniox, they aren't yours)")

    if not profiles["voices"]:
        print("\nnothing to clean up — voice_profiles.json is already empty.")
        return

    try:
        answer = input(f"\nDelete all {len(profiles['voices'])} voice profile record(s) and "
                       f"reset {profiles_path.name}? Type 'yes' to confirm: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\ncancelled.")
        return
    if answer.lower() != "yes":
        print("cancelled — nothing deleted.")
        return

    if cloned:
        api_key = env("SONIOX_API_KEY", required=True)
        for v in cloned:
            print(f"  deleting '{v.get('name')}' ({v.get('gender')}) -> "
                 f"voice_id={v.get('voice_id')} ...")
            delete_voice(api_key, v["voice_id"])

    save_profiles(profiles_path, {"voices": []})
    print(f"\ndone — {profiles_path} reset to empty. Re-run this script (without --cleanup) "
         "to rebuild it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_gen.yaml")
    ap.add_argument("--ref-dir", default="data/voice_refs")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the full plan (clone / default-fill), hit no API, write nothing")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete every cloned voice (Soniox account + local record) and "
                         "reset voice_profiles.json; asks for confirmation first")
    args = ap.parse_args()

    cfg = DataGenConfig.from_yaml(ROOT / args.config)
    ref_dir = ROOT / args.ref_dir
    profiles_path = ROOT / cfg.voice_profiles_path
    ref_dir.mkdir(parents=True, exist_ok=True)

    if args.cleanup:
        _run_cleanup(args, cfg, profiles_path)
        return

    matched, skipped = scan_ref_folder(ref_dir)

    print("=" * 68)
    print("ThinkSpark-v2-350M — voice profiles (your clones + default fill-in)")
    print("=" * 68)
    print(f"scanning {ref_dir}")
    print(f"found {len(matched)} usable clip(s), {len(skipped)} skipped (bad filename)")

    if skipped:
        print("\nskipped (rename to `<gender>_<name>.<ext>` — see data/voice_refs/README.md):")
        for p in skipped[:10]:
            print(f"  {p.name}")
        if len(skipped) > 10:
            print(f"  … +{len(skipped) - 10} more")

    submitted_by_gender = {g: [m for m in matched if m["gender"] == g] for g in GENDERS}
    if not matched:
        print("\nnothing to do. Drop reference clips into "
             f"{ref_dir} named like `female_anita.wav` / `male_rahul.wav`, then re-run.")
        return

    profiles = load_profiles(profiles_path)
    cloned_entries = [v for v in profiles["voices"] if v.get("source") == "cloned"]
    default_entries = [v for v in profiles["voices"] if v.get("source") == "default"]
    by_hash = {v["content_hash"]: v for v in cloned_entries if v.get("content_hash")}

    # split each gender's submitted clips into already-cloned vs. still-needing-a-decision
    already_done = {g: [] for g in GENDERS}
    candidates = {g: [] for g in GENDERS}
    for g in GENDERS:
        for item in submitted_by_gender[g]:
            h = file_hash(item["path"])
            if h in by_hash:
                already_done[g].append(by_hash[h])
            else:
                item["content_hash"] = h
                candidates[g].append(item)

    already_cloned_total = len(cloned_entries)
    cap = cfg.soniox_max_cloned_voices
    remaining_budget = max(0, cap - already_cloned_total)

    quota = allocate_by_largest_remainder({g: len(candidates[g]) for g in GENDERS}, remaining_budget)
    clone_this_run = {g: sorted(candidates[g], key=lambda it: it["path"].name)[:quota[g]] for g in GENDERS}
    skip_this_run = {g: sorted(candidates[g], key=lambda it: it["path"].name)[quota[g]:] for g in GENDERS}

    already_default_names = {g: {v["name"] for v in default_entries if v["gender"] == g} for g in GENDERS}
    default_pick: dict[str, list] = {g: [] for g in GENDERS}
    for g in GENDERS:
        needed = len(skip_this_run[g])
        available = [dv for dv in default_voices_by_gender(g) if dv.name not in already_default_names[g]]
        default_pick[g] = available[:needed]
        if len(default_pick[g]) < needed:
            print(f"\nWARNING: ran out of unused default {g} voices "
                 f"({needed} needed, only {len(default_pick[g])} available) — "
                 f"add more clips within the cap instead, or raise soniox_max_cloned_voices.")

    print(f"\naccount cap: {cap} total cloned voices  "
         f"(already cloned by this project: {already_cloned_total}, "
         f"remaining budget: {remaining_budget})")
    for g in GENDERS:
        print(f"  {g:<6}: {len(submitted_by_gender[g])} submitted  "
             f"= {len(already_done[g])} already cloned "
             f"+ {len(clone_this_run[g])} to clone now "
             f"+ {len(skip_this_run[g])} over cap -> filled with default voices "
             f"({[dv.name for dv in default_pick[g]]})")

    if args.dry_run:
        print("\nDRY RUN — no API calls made, voice_profiles.json not written.")
        for g in GENDERS:
            if clone_this_run[g]:
                print(f"\nwould clone ({g}):")
                for item in clone_this_run[g]:
                    print(f"  {item['name']:<20} <- {item['path'].name}")
            if skip_this_run[g]:
                print(f"\nwould NOT clone, over cap ({g}) — audio ignored, using default instead:")
                for item in skip_this_run[g]:
                    print(f"  {item['name']:<20} <- {item['path'].name}")
        return

    # default-voice entries cost nothing and can't fail — write them first
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    added_defaults = 0
    for g in GENDERS:
        for dv in default_pick[g]:
            profiles["voices"].append({
                "voice_id": dv.name,     # Soniox stock voices ARE their own voice id
                "name": dv.name,
                "gender": dv.gender,
                "description": dv.description,
                "source": "default",
                "added_at": now,
            })
            added_defaults += 1
    if added_defaults:
        save_profiles(profiles_path, profiles)
        print(f"\nadded {added_defaults} default (stock) voice(s) to {profiles_path}")

    to_clone_flat = [(g, item) for g in GENDERS for item in clone_this_run[g]]
    if not to_clone_flat:
        print("\nno new clips to clone this run.")
    else:
        api_key = env("SONIOX_API_KEY", required=True)
        model = cfg.soniox_model
        print()
        n_ok, n_fail = 0, 0
        for i, (g, item) in enumerate(to_clone_flat, 1):
            print(f"[{i}/{len(to_clone_flat)}] cloning '{item['name']}' ({g}) "
                 f"from {item['path'].name} ...")
            try:
                resp = clone_voice(api_key, item["name"], item["path"])
                if resp.get("_http_status", 0) >= 400 or "id" not in resp:
                    raise RuntimeError(f"clone failed: {resp}")
                voice_id = resp["id"]
                print(f"    cloned -> voice_id={voice_id}")

                print(f"    preparing voice for model '{model}' (recompute + poll)...")
                recompute_and_wait(api_key, voice_id, model, log=print)

                profiles["voices"].append({
                    "voice_id": voice_id,
                    "name": item["name"],
                    "gender": g,
                    "source": "cloned",
                    "source_file": item["path"].name,
                    "content_hash": item["content_hash"],
                    "model_prepared": model,
                    "cloned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                save_profiles(profiles_path, profiles)   # save after every success — resumable
                n_ok += 1
            except Exception as e:
                print(f"    FAILED: {e}")
                n_fail += 1

        print("\n" + "=" * 68)
        print(f"done: {n_ok} cloned + prepared, {n_fail} failed")
        if n_fail:
            print("re-run this script to retry the failed clip(s) — already-cloned ones are "
                 "skipped, so nothing is re-uploaded.")

    final_cloned = sum(1 for v in profiles["voices"] if v["source"] == "cloned")
    final_default = sum(1 for v in profiles["voices"] if v["source"] == "default")
    print(f"\nwrote {profiles_path}")
    for g in GENDERS:
        n_g_cloned = sum(1 for v in profiles["voices"] if v["gender"] == g and v["source"] == "cloned")
        n_g_default = sum(1 for v in profiles["voices"] if v["gender"] == g and v["source"] == "default")
        print(f"  {g}: {n_g_cloned} cloned + {n_g_default} default = {n_g_cloned + n_g_default} total")
    print(f"totals: {final_cloned} cloned + {final_default} default "
         f"(cap {cap}, {final_cloned}/{cap} used)")
    print("\nscripts/03_render_user_audio.py reads this file automatically — nothing else "
         "to configure.")


if __name__ == "__main__":
    main()
