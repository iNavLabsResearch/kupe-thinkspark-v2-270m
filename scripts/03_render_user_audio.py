#!/usr/bin/env python
"""
Section 8.4 step 2 — render the USER line of each scenario with Soniox TTS, with the
same live colored dashboard treatment as scenario generation (script 02): real-time
progress, audio hours done/target, cost spent/projected/gap-vs-budget, throughput+ETA,
and rate-limit tracking with automatic exponential backoff (thinkspark.tts_soniox).

Only user audio is synthesised (the agent side stays text + a state flag). We request
character-level timestamps from Soniox's TTS response itself (return_timestamps=true —
no separate STT pass needed) so the frame builder can place events frame-accurately.
These timestamps are used ONLY to build training data (Section 8.4 → 04_build_frames.py);
inference never needs them (Section 4.3's agent-state channel replaces timing entirely).
Each scenario's audio goes to data/audio/<scenario_id>.wav and its word timings (grouped
from Soniox's character timestamps) to a sidecar JSON. Resumable: scenarios whose wav
already exists are skipped. Empty/zero-duration audio is treated as a failure, never a
silent "success" — see the `duration_s <= 0.0` check in `render_one`.

Concurrency renders several clips in parallel (config `soniox_concurrency`, default **3**
— Soniox's real default account-wide concurrent-stream limit; raise only if your account
has a higher one). A failure classified as a rate limit (429 / "rate limit" / "quota" in
the error) is retried with exponential backoff inside the client itself (see
thinkspark.tts_soniox); if it still fails after retries, it's logged and shown distinctly
(⏳ rate-limited) so you can see whether you need to lower --concurrency.

    conda activate llms
    export SONIOX_API_KEY=...
    python scripts/03_render_user_audio.py --config configs/data_gen.yaml \
        --in data/scenarios/scenarios_all.jsonl
    python scripts/03_render_user_audio.py --config configs/data_gen.yaml \
        --in data/scenarios/scenarios_all.jsonl -j 3     # lower concurrency if rate-limited
    python scripts/03_render_user_audio.py --config configs/data_gen.yaml \
        --in data/scenarios/scenarios_all.jsonl --quiet  # plain logs, no live dashboard

    # delete ALL rendered audio for this --audio-dir + its tts_calls/hf_sync SQLite rows
    # (asks "are you sure?" first, same as script 02's --cleanup). Scoped: never touches
    # scenario generation data (llm_calls/unit_evals/scenario_registry) — see script 02
    # for cleaning that up instead.
    python scripts/03_render_user_audio.py --config configs/data_gen.yaml --cleanup
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import DataGenConfig
from thinkspark.schema import Scenario
from thinkspark.tts_soniox import (
    SonioxTTS, SonioxRateLimitError, soniox_cost_usd, _load_custom_voice_profiles,
)
from thinkspark.tts_stream import TTSStream
from thinkspark.db import RunDB


def render_one(tts: SonioxTTS, s: Scenario, audio_dir: Path, cfg: DataGenConfig,
               db: RunDB, run_id: str, stream: TTSStream) -> str:
    """Render one scenario's user audio. Returns 'ok' | 'skipped' | 'failed'."""
    sid = s.scenario_id or "unknown"
    wav_path = audio_dir / f"{sid}.wav"
    meta_path = audio_dir / f"{sid}.words.json"
    if wav_path.exists() and meta_path.exists():
        return "skipped"

    t0 = time.perf_counter()
    try:
        result = tts.synthesize(
            text=s.user_text, language=s.language, gender=s.gender,
            wav_path=wav_path, keep_audio=False,
        )
    except SonioxRateLimitError as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        db.log_tts_call(run_id, sid, len(s.user_text), 0.0, 0.0, status="rate_limited",
                        latency_ms=latency_ms, error=str(e)[:500])
        # emit() always updates stream.stats (cheap bookkeeping); the quiet-mode print
        # is an ADDITION for terminal visibility, not a replacement — stream.stats
        # must stay accurate in both modes since main()'s quiet-mode summary reads it.
        stream.emit("rate_limited", scenario_id=sid)
        if not stream.enabled:
            print(f"  ! rate-limited: {sid}: {e}")
        return "failed"
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        db.log_tts_call(run_id, sid, len(s.user_text), 0.0, 0.0, status="error",
                        latency_ms=latency_ms, error=str(e)[:500])
        stream.emit("clip_failed", scenario_id=sid, error=str(e)[:120])
        if not stream.enabled:
            print(f"  ! TTS failed for {sid}: {e}")
        return "failed"

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Defensive: empty/zero-duration audio must NEVER be recorded as a success — this
    # is exactly the failure mode a wrong endpoint/protocol produces (connects fine,
    # "succeeds", but the wav is empty). Treat it as a real failure so it's retried on
    # the next run instead of silently poisoning the corpus. tts.synthesize() already
    # wrote wav_path unconditionally before we got here — delete that empty file so
    # nothing lingers on disk looking like a valid (but corrupt) render.
    if result.duration_s <= 0.0:
        wav_path.unlink(missing_ok=True)
        err = "empty audio returned (0.0s duration) — check Soniox response format"
        db.log_tts_call(run_id, sid, len(s.user_text), 0.0, 0.0, status="error",
                        latency_ms=latency_ms, error=err)
        stream.emit("clip_failed", scenario_id=sid, error=err)
        if not stream.enabled:
            print(f"  ! empty audio for {sid} — not writing a corrupt file")
        return "failed"

    meta = {
        "scenario_id": sid,
        "duration_s": result.duration_s,
        "sample_rate": result.sample_rate,
        "chars_synthesized": result.chars_synthesized,
        "words": [asdict(w) for w in result.words],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    # Real per-request Soniox pricing (token-based — see thinkspark.tts_soniox), not a
    # flat $/hour estimate: accounts for this specific request's text length + audio
    # length, so cost accumulates accurately and progressively as clips render.
    cost = soniox_cost_usd(chars=len(s.user_text), duration_s=result.duration_s)
    db.log_tts_call(run_id, sid, len(s.user_text), result.duration_s, cost,
                    status="ok", latency_ms=latency_ms)
    stream.emit("clip_ok", scenario_id=sid, duration_s=result.duration_s,
               cost_usd=cost, latency_ms=latency_ms)
    if not stream.enabled:
        print(f"  ✓ {sid}  {result.duration_s:.1f}s  ${cost:.5f}  {latency_ms:.0f}ms")
    return "ok"


def _run_live_loop(futures: set, stream: TTSStream) -> None:
    pending = set(futures)
    try:
        from rich.live import Live

        with Live(stream.render(), refresh_per_second=8, transient=True, screen=False) as live:
            while pending:
                done, pending = wait(pending, timeout=0.15, return_when=FIRST_COMPLETED)
                live.update(stream.render())
                for fut in done:
                    fut.result()
    except ImportError:
        last_plain = ""
        while pending:
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            plain = stream.render()
            if plain != last_plain:
                print("\033[2J\033[H" + plain, flush=True)
                last_plain = plain
            for fut in done:
                fut.result()


def _count_audio_files(audio_dir: Path) -> tuple[int, int]:
    if not audio_dir.exists():
        return 0, 0
    return len(list(audio_dir.glob("*.wav"))), len(list(audio_dir.glob("*.words.json")))


def _run_cleanup(args, cfg: DataGenConfig, audio_dir: Path) -> None:
    """
    Delete every rendered wav + words.json in `audio_dir`, plus their tts_calls/hf_sync
    SQLite rows, after interactive confirmation — same "are you sure?" pattern as
    scripts/02_generate_scripts.py --cleanup. Deliberately SCOPED: never touches
    llm_calls/unit_evals/scenario_registry (scenario generation, a separate concern with
    its own --cleanup in script 02) — see thinkspark.db.RunDB.wipe_tts_data.
    """
    db_path = ROOT / cfg.db_path
    n_wav, n_meta = _count_audio_files(audio_dir)
    db_counts = RunDB.counts_if_exists(db_path)

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        tbl = Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column(style="bold")
        tbl.add_column()
        tbl.add_row("audio dir", str(audio_dir.relative_to(ROOT)) if audio_dir.is_relative_to(ROOT) else str(audio_dir))
        tbl.add_row("wav files", str(n_wav))
        tbl.add_row("words.json sidecars", str(n_meta))
        tbl.add_row("sqlite db", str(db_path.relative_to(ROOT)) if db_path.is_relative_to(ROOT) else str(db_path))
        tbl.add_row("tts call rows", str(db_counts["tts_calls"]))
        tbl.add_row("hf_sync rows", str(db_counts["hf_sync"]))
        tbl.add_row("(untouched)", "llm_calls, unit_evals, scenario_registry, runs")
        Console().print(
            Panel(tbl, title="[bold red]Cleanup — the following will be deleted[/]", border_style="red")
        )
    except Exception:
        print("Cleanup will delete:")
        print(f"  audio dir: {audio_dir} ({n_wav} wav, {n_meta} words.json)")
        print(f"  db: {db_counts['tts_calls']} tts_calls rows, {db_counts['hf_sync']} hf_sync rows "
             f"(llm_calls/unit_evals/scenario_registry/runs left untouched)")

    try:
        answer = input("\nDelete all rendered audio + tts_calls/hf_sync rows? "
                       "Type 'yes' to confirm: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\ncancelled.")
        raise SystemExit(0)

    if answer.lower() != "yes":
        print("cancelled — nothing deleted.")
        raise SystemExit(0)

    removed_wav = removed_meta = 0
    if audio_dir.exists():
        for p in audio_dir.glob("*.wav"):
            p.unlink()
            removed_wav += 1
        for p in audio_dir.glob("*.words.json"):
            p.unlink()
            removed_meta += 1

    db_removed = {"tts_calls": 0, "hf_sync": 0}
    if db_path.exists():
        db = RunDB(db_path)
        db_removed = db.wipe_tts_data()
        db.close()

    summary = (f"removed {removed_wav} wav, {removed_meta} words.json, "
              f"{db_removed['tts_calls']} tts_calls rows, {db_removed['hf_sync']} hf_sync rows")
    try:
        from rich.console import Console
        Console().print(f"[green]cleaned up:[/] {summary}")
    except Exception:
        print(f"cleaned up: {summary}")
    print("re-run scripts/03_render_user_audio.py to regenerate — nothing else was touched.")


def main():
    ap = argparse.ArgumentParser(
        description="Render Soniox user audio with a live colored progress dashboard.",
    )
    ap.add_argument("--config", default="configs/data_gen.yaml")
    ap.add_argument("--in", dest="in_path", default=None,
                    help="scenarios JSONL (required unless --cleanup)")
    ap.add_argument("--audio-dir", default="data/audio")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", "-j", type=int, default=None,
                    help="override config soniox_concurrency")
    ap.add_argument("--quiet", action="store_true",
                    help="disable the live colored dashboard (minimal plain logs)")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete ALL rendered audio in --audio-dir + tts_calls/hf_sync "
                         "SQLite rows; asks for confirmation. Scoped — never touches "
                         "scenario generation data (see script 02's own --cleanup for that)")
    args = ap.parse_args()

    cfg = DataGenConfig.from_yaml(ROOT / args.config)
    audio_dir = ROOT / args.audio_dir

    if args.cleanup:
        _run_cleanup(args, cfg, audio_dir)
        return

    if not args.in_path:
        raise SystemExit("--in is required (unless --cleanup)")

    # resolve relative to the repo root, same as every other cfg path (db_path, audio_dir, ...) —
    # so `voice_profiles.json` is found regardless of the caller's cwd
    cfg.voice_profiles_path = str(ROOT / cfg.voice_profiles_path)
    tts = SonioxTTS.from_config(cfg)
    concurrency = max(1, args.concurrency or cfg.soniox_concurrency)

    audio_dir.mkdir(parents=True, exist_ok=True)

    lines = [l for l in Path(ROOT / args.in_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        lines = lines[:args.limit]
    scenarios = [Scenario.from_dict(json.loads(l)) for l in lines]

    # Split into already-rendered (resumable skip) vs. remaining work — one pass, one
    # source of truth, so the dashboard's target always matches what actually runs.
    remaining: list[Scenario] = []
    already_skipped = 0
    # Corpus-cumulative meters: sum the audio hours + cost already on disk from previous
    # runs so the "audio hours / target" meter doesn't reset to zero on every re-run.
    # The .words.json sidecar is authoritative for what's actually rendered; cost is
    # recomputed from that clip's real duration + its user_text length (same formula the
    # live path uses), so no separate cost store is needed.
    prior_audio_hours = 0.0
    prior_cost_usd = 0.0
    for s in scenarios:
        wav_path = audio_dir / f"{s.scenario_id}.wav"
        meta_path = audio_dir / f"{s.scenario_id}.words.json"
        if wav_path.exists() and meta_path.exists():
            already_skipped += 1
            try:
                dur = float(json.loads(meta_path.read_text(encoding="utf-8")).get("duration_s", 0.0))
            except (OSError, ValueError, json.JSONDecodeError):
                dur = 0.0
            if dur > 0.0:
                prior_audio_hours += dur / 3600.0
                prior_cost_usd += soniox_cost_usd(chars=len(s.user_text), duration_s=dur)
        else:
            remaining.append(s)

    # Fail fast, before spending a single Soniox call: this project uses ONLY your own
    # cloned voices (no catalog fallback — thinkspark.tts_soniox.resolve_voice), so
    # every gender actually needed by the remaining work must already have at least one
    # cloned profile in voice_profiles.json.
    custom_profiles = _load_custom_voice_profiles(cfg.voice_profiles_path)
    needed_genders = sorted({s.gender for s in remaining})
    missing_genders = [g for g in needed_genders if not custom_profiles.get(g.lower())]
    if missing_genders:
        raise SystemExit(
            f"No cloned voice profiles for gender(s) {missing_genders} in "
            f"{cfg.voice_profiles_path} — this project uses ONLY your own cloned "
            f"voices, never Soniox's built-in catalog. Add reference clips to "
            f"data/voice_refs/ (see its README) named '<gender>_<name>.wav', then run:\n"
            f"  python scripts/15_create_voice_profiles.py --config {args.config}\n"
            f"before rendering."
        )
    n_custom = sum(len(v) for v in custom_profiles.values())
    print(f"voice profiles: {n_custom} of your own cloned voice(s) loaded from "
         f"{cfg.voice_profiles_path} ({', '.join(f'{g}={len(v)}' for g, v in sorted(custom_profiles.items()))})")

    db = RunDB(ROOT / cfg.db_path)
    run_id = db.start_run("render_tts", cfg.__dict__, vars(args))

    stream = TTSStream(
        target=len(remaining), already_skipped=already_skipped,
        target_hours=cfg.total_hours, price_per_hour=cfg.soniox_price_per_hour_usd,
        budget_inr_target=cfg.budget_inr_target, inr_per_usd=cfg.inr_per_usd,
        concurrency=concurrency, enabled=not args.quiet,
        base_audio_hours=prior_audio_hours, base_cost_usd=prior_cost_usd,
    )

    # Explicit and unambiguous: the dashboard's "target"/progress bar shows only what
    # THIS run still needs to do (remaining), not the file's total scenario count —
    # print the arithmetic once, plainly, so "target=8936" is never mistaken for "only
    # 8936 of my 9008 scenarios were found" (72 skipped + 8936 remaining = 9008 total).
    print(f"input file: {len(scenarios)} scenario(s) total "
         f"({already_skipped} already rendered + {len(remaining)} remaining this run)")
    if args.quiet:
        print(f"run_id={run_id}  concurrency={concurrency}")
        print(f"rendering {len(remaining)} scenarios ({already_skipped} already done) -> {audio_dir}")

    lock = threading.Lock()
    counts = {"ok": 0, "skipped": already_skipped, "failed": 0}

    def _worker(s: Scenario):
        status = render_one(tts, s, audio_dir, cfg, db, run_id, stream)
        with lock:
            counts[status] += 1

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(_worker, s) for s in remaining}
            if args.quiet:
                for fut in futures:
                    fut.result()
            else:
                _run_live_loop(futures, stream)
    finally:
        summary = db.run_summary(run_id)
        db.close()

    if not args.quiet:
        stream.print_final()
    else:
        s = stream.stats
        print(f"done: rendered={counts['ok']} skipped={counts['skipped']} "
             f"failed={counts['failed']} (of which rate_limited={s.rate_limited}) -> {audio_dir}")
        print(f"audio hours: {s.audio_hours:.3f}h / {s.target_hours:.1f}h target")
        print(f"cost this run: ${summary['tts_cost_usd']:.4f}  "
             f"(@ ${cfg.soniox_price_per_hour_usd}/h)")
        print(f"projected total: ${s.projected_cost_usd:.4f}   "
             f"gap (left to spend): ${s.gap_usd:.4f}   "
             f"{s.budget_pct:.1f}% of INR {cfg.budget_inr_target:.0f} budget (projected)")


if __name__ == "__main__":
    main()
