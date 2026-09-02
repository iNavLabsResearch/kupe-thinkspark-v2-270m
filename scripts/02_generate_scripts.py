#!/usr/bin/env python
"""
Section 8.4 — generate scenarios as strict JSON with the OpenAI-SDK LLM, BATCHED.

Each LLM call asks for `batch_size` (config, default 12) DISTINCT scenarios at once
instead of one-at-a-time — much faster for a given rate limit / concurrency budget.
Concurrency (config `llm_concurrency`, default 30) is how many worker threads issue
batched calls in parallel; each thread works through one job's remaining count in
batch-sized chunks.

Batching risk & mitigation (read this before raising batch_size): asking a small/fast
model (DeepSeek-flash-class, Gemma-3-27B-class) for many structurally-identical JSON
objects in one completion can make it repeat near-duplicates or let language/behaviour
bleed between items. We mitigate this three ways:
  1. The prompt (thinkspark.prompts.build_batch_scenario_prompt) explicitly demands
     DISTINCT items and forbids paraphrase-only variation.
  2. max_tokens is scaled by batch_size so the JSON array isn't truncated mid-response.
  3. Every item is schema-validated INDEPENDENTLY after parsing; only the missing count
     is re-requested on the next pass — one bad item never discards a whole good batch.
If you still see repeats/drift in practice, drop `batch_size` in configs/data_gen.yaml
(1 = fully back to single-scenario calls, most reliable, slowest).

Resumability: the JSONL shard is the source of truth — on restart we count how many
scenarios already exist per job (by `_job_key`) and only top up the remainder, so a
killed/corrupted run always continues from exactly where it left off. Every LLM call
(cost, tokens, batch outcome) is additionally logged to a SQLite DB (thinkspark.db) for
a full audit trail, independent of the shard file.

Live output: a colored SSE-style event stream (batch start → each scenario → batch done)
refreshes in the terminal while workers run. When generation finishes, Section 8.5 data
quality evaluation runs automatically unless you pass `--no-eval`.

Simple by default (same idea as kupe-thinkspark / kupe-tts): just run it. The plan is
auto-built on first run (identical to running 01_plan_distribution.py yourself — same
function, same numbers), and with no --part/--all flag it targets the FULL corpus and
resumes automatically on every re-run, exactly like a `--fresh`-less restart elsewhere in
kupe. Nothing here runs a second time by accident — a finished target is a no-op.

    conda activate llms
    export OPENAI_API_KEY=...                 # (or put it in .env)

    python scripts/02_generate_scripts.py                     # plan auto-built, full corpus, resumes
    python scripts/02_generate_scripts.py -j 10                # override concurrency
    python scripts/02_generate_scripts.py --limit 20            # smoke test a few scenarios first

    # watch live cost/latency/pass-fail charts in a second terminal:
    python scripts/11_monitor.py
    # or build a self-contained HTML chart report any time:
    python scripts/12_build_report.py

    # advanced: split across Kaggle's 9h/session budget by hand instead of one long run
    python scripts/02_generate_scripts.py --part 0
    python scripts/02_generate_scripts.py --part 1
    ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import DataGenConfig
from thinkspark.db import RunDB
from thinkspark.distribution import DEFAULT_LANGUAGE_SHARES, GenJob
from thinkspark import distribution as dist
from thinkspark.gen_stream import GenerationStream, run_data_quality_eval
from thinkspark.llm_client import LLMClient
from thinkspark.prompts import build_batch_scenario_prompt
from thinkspark.schema import Scenario
from thinkspark.validators import validate_scenario

# Per-scenario token budget baked into cfg.llm_max_tokens; scale by batch (+ headroom for
# JSON structure/keys) so a full batch response never gets cut off mid-array.
_BATCH_TOKEN_OVERHEAD = 200


def _load_jobs(plan_path: Path) -> list[GenJob]:
    jobs = []
    for line in plan_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            jobs.append(GenJob(**json.loads(line)))
    return jobs


def _existing_counts(shard_path: Path) -> dict[str, int]:
    """Count already-generated scenarios per job key for resumability (file = truth)."""
    counts: dict[str, int] = {}
    if not shard_path.exists():
        return counts
    for line in shard_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a truncated last line from a hard crash
        key = d.get("_job_key")
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _scenario_id(job: GenJob, idx: int, corpus_tag: str = "") -> str:
    """Deterministic id for (job, index) within a corpus namespace.

    `corpus_tag` namespaces the hash so an ADDITIVE corpus (a deficit top-up) cannot
    reproduce the ids of an existing one. Without it, a second run over the same
    behaviours/languages/length band emits the identical id sequence, and because the id
    is the .wav / .npz filename in the shared data/audio + data/encoded directories, the
    top-up would overwrite the original clips instead of adding to them. Empty tag = the
    original hash input, so ids already generated are unchanged.
    """
    raw = f"{job.key}|{idx}" if not corpus_tag else f"{corpus_tag}|{job.key}|{idx}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _parse_range(range_str: str, total: int) -> tuple[int, int]:
    """Parse '--range START:END' into validated (start, end) bounds, 0 <= start < end <= total."""
    try:
        start_s, end_s = range_str.split(":")
        start, end = int(start_s), int(end_s)
    except ValueError:
        raise SystemExit(
            f"--range must be START:END (e.g. --range 0:3000), got {range_str!r}"
        )
    if start < 0 or end <= start:
        raise SystemExit(f"--range START:END needs 0 <= START < END, got {range_str!r}")
    if end > total:
        raise SystemExit(f"--range end {end} is past the plan total ({total} scenarios)")
    return start, end


def _job_targets(jobs: list[GenJob], range_bounds: tuple[int, int] | None):
    """
    Walk `jobs` in their fixed plan order (same order every run — see
    thinkspark.distribution.write_plan) and assign each job a stable global-index span
    [job_start, job_start + job.count). Returns (job, target_count, global_offset)
    triples:
      - no range: target_count = job.count (full job), global_offset = job_start
        (every job included, exactly today's behaviour before --range existed)
      - with a range: target_count = size of the overlap between the job's span and
        [range_start, range_end); jobs with zero overlap are skipped entirely.
    This is the ONLY place a --range is turned into per-job work — the balanced
    distribution itself (thinkspark.distribution) is never touched.

    Correctness note: the file-based per-job resume (`already = min(target_count,
    done.get(job.key))` in `_run_job`) guarantees no duplicate or missing scenarios
    even if ranges are re-run out of order or overlap — a job's total production is
    always clipped to its own count regardless of which range(s) touched it. What is
    NOT guaranteed for out-of-order ranges is that a scenario's recorded
    `global_index` in `scenario_registry` exactly matches its conceptual position
    within THIS range slice (it's derived from the job's natural fill order, which is
    precise for the common sequential workflow — 0:3000, then 3000:6000, ... — but is
    a best-effort label, not a strict guarantee, otherwise). Run ranges in increasing
    order for exact bookkeeping; run them in any order for correct, complete coverage.
    """
    out = []
    cursor = 0
    for job in jobs:
        job_start, job_end = cursor, cursor + job.count
        cursor = job_end
        if range_bounds is None:
            out.append((job, job.count, job_start))
            continue
        lo = max(job_start, range_bounds[0])
        hi = min(job_end, range_bounds[1])
        if hi > lo:
            out.append((job, hi - lo, job_start))
    return out


def _load_scenarios(shard_path: Path) -> list[Scenario]:
    if not shard_path.exists():
        return []
    out: list[Scenario] = []
    for line in shard_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(Scenario.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
    return out


def _flag_chain(scenario: Scenario) -> str:
    flags = [t.flag for t in scenario.target[:4]]
    if len(scenario.target) > 4:
        flags.append("…")
    return "→".join(flags) if flags else "—"


def generate_batch(
    client: LLMClient,
    job: GenJob,
    start_idx: int,
    n: int,
    db: RunDB,
    run_id: str,
    stream: GenerationStream,
    global_offset: int = 0,
    corpus_tag: str = "",
) -> list[Scenario]:
    """
    One LLM call requesting `n` scenarios; returns only the ones that pass unit-level
    evaluation.

    Unit-level eval (Section 8.5, per scenario): every parsed item is run through
    `validate_scenario()` — schema, control-vocab, and language/script checks — the
    moment it's produced, not just at the end on the whole corpus. A pass is logged to
    the DB (`unit_evals`, status='pass') AND recorded in `scenario_registry` — the
    SQLite-side "what was actually generated" record, keyed by scenario_id and tagged
    with its `global_index` (= global_offset + its position within this job), which is
    what `--range START:END` reads back to know what's already done in a given slice.
    A fail is logged as status='fail' with an incrementing, job-scoped `fail_flag`
    ('fail1' the first time this job fails, 'fail2' the second, ...) so repeated
    failures are traceable across the whole run — and the caller's retry loop
    (`_run_job`) regenerates that slot in the next batch. The corpus-wide Section 8.5
    report (`run_data_quality_eval`) still runs once more on the complete shard at the
    end — this is the fast, per-item gate that runs continuously.
    """
    if stream.enabled:
        stream.emit("batch_start", job_key=job.key, n=n, behaviour=job.behaviour, language=job.language)

    system, user = build_batch_scenario_prompt(job, n)
    max_tokens = client.max_tokens * n + _BATCH_TOKEN_OVERHEAD
    t0 = time.perf_counter()
    try:
        data, usage = client.chat_json_with_usage(system, user, max_tokens=max_tokens)
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        db.log_llm_call(
            run_id, job.key, client.model, n, 0, 0, 0, 0, 0.0, latency_ms, "api_error", str(e)[:500]
        )
        if stream.enabled:
            stream.emit("api_error", job_key=job.key, error=str(e)[:500])
        else:
            print(f"  ! LLM error ({job.behaviour}/{job.language}, n={n}): {e}")
        return []

    latency_ms = (time.perf_counter() - t0) * 1000.0
    raw_items = data.get("scenarios", [])
    if not isinstance(raw_items, list):
        raw_items = [data]  # tolerate a model that ignores the wrapper for n=1

    scenarios: list[Scenario] = []
    for offset, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        item.update({
            "behaviour": job.behaviour,
            "language": job.language,
            "domain": job.domain,
            "gender": job.gender,
            "length_band": job.length_band,   # so validate() uses the right word budget
        })
        s = Scenario.from_dict(item)
        s.scenario_id = _scenario_id(job, start_idx + offset, corpus_tag)

        # --- unit-level evaluation (Section 8.5, this ONE scenario) --------------
        res = validate_scenario(s)
        if res.ok:
            scenarios.append(s)
            db.log_unit_eval(run_id, job.key, s.scenario_id, "pass")
            db.log_scenario(run_id, job.key, s.scenario_id,
                            global_index=global_offset + start_idx + offset)
            if stream.enabled:
                stream.emit(
                    "scenario_ok",
                    scenario_id=s.scenario_id,
                    behaviour=s.behaviour,
                    language=s.language,
                    user_text=s.user_text,
                    flags=_flag_chain(s),
                )
        else:
            fail_flag = stream.next_fail_flag(job.key)
            db.log_unit_eval(
                run_id, job.key, s.scenario_id, "fail", fail_flag,
                json.dumps(res.errors[:5], ensure_ascii=False),
            )
            if stream.enabled:
                stream.emit("scenario_invalid", job_key=job.key, errors=res.errors[:3], fail_flag=fail_flag)
            else:
                print(f"  x unit-eval {fail_flag.upper()} ({job.behaviour}/{job.language}): "
                      f"{res.errors[:2]} -> will regenerate")

    cost = usage.cost_usd(client.price_in_per_1m, client.price_out_per_1m)
    status = "ok" if scenarios else ("parse_error" if not raw_items else "all_invalid")
    db.log_llm_call(
        run_id, job.key, client.model, n, len(raw_items), len(scenarios),
        usage.prompt_tokens, usage.completion_tokens, cost, latency_ms, status,
    )
    if stream.enabled:
        stream.emit(
            "batch_done",
            job_key=job.key,
            requested=n,
            valid=len(scenarios),
            latency_ms=latency_ms,
            cost_usd=cost,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
    return scenarios


def _run_live_loop(
    futures: set,
    stream: GenerationStream,
    limit: int | None,
    produced_ref: list[int],
) -> None:
    """Drain futures while refreshing the colored live dashboard."""
    pending = set(futures)
    try:
        from rich.live import Live

        with Live(stream.render(), refresh_per_second=8, transient=True, screen=False) as live:
            while pending:
                done, pending = wait(pending, timeout=0.15, return_when=FIRST_COMPLETED)
                live.update(stream.render())
                for fut in done:
                    fut.result()
                if limit is not None and produced_ref[0] >= limit:
                    for f in pending:
                        f.cancel()
                    pending.clear()
    except ImportError:
        last_plain = ""
        while pending:
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            plain = stream.render()
            if plain != last_plain:
                print("\033[2J\033[H" + plain, flush=True)
                last_plain = plain
            for fut in done:
                if limit is not None and produced_ref[0] >= limit:
                    continue
                fut.result()


def _shard_path(args, out_dir: Path) -> Path:
    # Default (no --part given) targets the FULL plan — the simple, one-command path.
    # --part N is still there for splitting a big run across Kaggle sessions by hand.
    if args.part is not None:
        return out_dir / f"scenarios_part{args.part:02d}.jsonl"
    return out_dir / "scenarios_all.jsonl"


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _run_cleanup(args, cfg: DataGenConfig, shard_path: Path) -> None:
    """Delete scenario shard + SQLite audit DB after interactive confirmation."""
    db_path = ROOT / cfg.db_path
    report_path = ROOT / args.report_out
    n_scenarios = _count_jsonl_lines(shard_path)
    db_counts = RunDB.counts_if_exists(db_path)

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        tbl = Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column(style="bold")
        tbl.add_column()
        tbl.add_row("scenario shard", str(shard_path.relative_to(ROOT)))
        tbl.add_row("scenarios in shard", str(n_scenarios))
        tbl.add_row("sqlite db", str(db_path.relative_to(ROOT)))
        tbl.add_row("llm call rows", str(db_counts["llm_calls"]))
        tbl.add_row("unit-eval rows", str(db_counts["unit_evals"]))
        tbl.add_row("scenario registry rows", str(db_counts["scenario_registry"]))
        tbl.add_row("tts call rows", str(db_counts["tts_calls"]))
        tbl.add_row("run rows", str(db_counts["runs"]))
        if report_path.exists():
            tbl.add_row("quality report", str(report_path.relative_to(ROOT)))
        Console().print(
            Panel(tbl, title="[bold red]Cleanup — the following will be deleted[/]", border_style="red")
        )
    except Exception:
        print("Cleanup will delete:")
        print(f"  shard: {shard_path} ({n_scenarios} scenarios)")
        print(f"  db:    {db_path} ({db_counts['llm_calls']} llm calls, "
              f"{db_counts['unit_evals']} unit-evals, {db_counts['scenario_registry']} "
              f"registered scenarios, {db_counts['runs']} runs)")
        if report_path.exists():
            print(f"  report: {report_path}")

    try:
        answer = input("\nDelete all records? Type 'yes' to confirm: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\ncancelled.")
        raise SystemExit(0)

    if answer.lower() != "yes":
        print("cancelled — nothing deleted.")
        raise SystemExit(0)

    removed = []
    if shard_path.exists():
        shard_path.unlink()
        removed.append(str(shard_path.name))
    for sidecar in (f"{db_path}-wal", f"{db_path}-shm"):
        p = Path(sidecar)
        if p.exists():
            p.unlink()
    if db_path.exists():
        db = RunDB(db_path)
        db.wipe_all()
        db.close()
        db_path.unlink()
        removed.append(db_path.name)
    if report_path.exists():
        report_path.unlink()
        removed.append(report_path.name)

    try:
        from rich.console import Console
        Console().print(f"[green]cleaned up:[/] {', '.join(removed) or '(nothing was on disk)'}")
    except Exception:
        print(f"cleaned up: {', '.join(removed) or '(nothing was on disk)'}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate ThinkSpark scenarios with live colored streaming output.",
    )
    ap.add_argument("--config", default="configs/data_gen.yaml")
    ap.add_argument("--plan-dir", default="data/plan")
    ap.add_argument("--out-dir", default="data/scenarios")
    ap.add_argument("--part", type=int, default=None,
                    help="advanced: run only this plan-part shard, for splitting a big "
                         "run across sessions by hand. Default (omitted) targets the "
                         "FULL plan — the simple one-command path.")
    ap.add_argument("--all", action="store_true",
                    help="explicit alias for the default (full plan); kept for scripts "
                         "that already call it, has no effect beyond the default")
    ap.add_argument("--range", dest="range_str", default=None,
                    help="advanced: generate only global scenario indices [START,END) "
                         "out of the full plan's ~9k target, e.g. --range 0:3000 then "
                         "--range 3000:6000 then --range 6000:9009 — a sequential "
                         "alternative to --part's hash-based shards. All writes land in "
                         "the same data/scenarios/scenarios_all.jsonl; mutually "
                         "exclusive with --part.")
    ap.add_argument("--limit", type=int, default=None, help="cap scenarios (smoke test)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override config batch_size (1 = single-scenario calls)")
    ap.add_argument("--concurrency", "-j", type=int, default=None,
                    help="override config llm_concurrency (parallel LLM worker threads)")
    ap.add_argument("--quiet", action="store_true",
                    help="disable the live colored stream (minimal plain logs)")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip automatic Section 8.5 data-quality evaluation at the end")
    ap.add_argument("--judge", action="store_true",
                    help="include LLM-judge naturalness pass in post-run evaluation")
    ap.add_argument("--judge-n", type=int, default=200,
                    help="how many scenarios to sample for the LLM judge (default 200)")
    ap.add_argument("--report-out", default="reports/data_quality.txt",
                    help="where to write the evaluation text report")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete scenario shard + cost DB (+ report); asks for confirmation")
    args = ap.parse_args()

    cfg = DataGenConfig.from_yaml(ROOT / args.config)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = _shard_path(args, out_dir)

    if args.cleanup:
        _run_cleanup(args, cfg, shard_path)
        return

    if args.range_str is not None and args.part is not None:
        raise SystemExit("--range and --part are mutually exclusive (--range always "
                         "operates on the full plan's global index space).")

    batch_size = max(1, args.batch_size or cfg.batch_size)
    concurrency = max(1, args.concurrency or cfg.llm_concurrency)

    plan_dir = ROOT / args.plan_dir

    # Auto-build the plan on first run — identical numbers to running
    # 01_plan_distribution.py yourself (same dist.write_plan() call, so the balanced
    # distribution can never drift between the two entry points). No-op if it already
    # exists. This is what makes `python scripts/02_generate_scripts.py` work with zero
    # setup, the same way the sibling kupe-thinkspark/kupe-tts generators do.
    plan_summary = dist.load_or_write_plan(cfg, plan_dir)
    if not args.quiet:
        try:
            from rich.console import Console
            Console().print(
                f"[dim]plan: {plan_summary['total_scenarios']} scenarios "
                f"({plan_summary['silent_backchannel']} deliberate silent-backchannel)[/]"
            )
        except Exception:
            pass

    if args.part is not None:
        plan_path = plan_dir / f"plan_part{args.part:02d}.jsonl"
    else:
        plan_path = plan_dir / "plan.jsonl"

    jobs = _load_jobs(plan_path)

    range_bounds = None
    if args.range_str is not None:
        range_bounds = _parse_range(args.range_str, plan_summary["total_scenarios"])
        if not args.quiet:
            try:
                from rich.console import Console
                Console().print(
                    f"[dim]range: [{range_bounds[0]},{range_bounds[1]}) of "
                    f"{plan_summary['total_scenarios']} total scenarios[/]"
                )
            except Exception:
                pass
        else:
            print(f"range: [{range_bounds[0]},{range_bounds[1]}) of "
                  f"{plan_summary['total_scenarios']} total scenarios")

    job_targets = _job_targets(jobs, range_bounds)  # [(job, target_count, global_offset), ...]

    done = _existing_counts(shard_path)
    already_done = sum(min(tc, done.get(j.key, 0)) for j, tc, _ in job_targets)
    plan_total = sum(tc for _, tc, _ in job_targets)
    target = args.limit if args.limit is not None else plan_total
    client = LLMClient.for_generation(cfg)

    db = RunDB(ROOT / cfg.db_path)
    run_id = db.start_run("generate", cfg.__dict__, vars(args))

    stream = GenerationStream(
        target=target,
        already_done=already_done if args.limit is None else 0,
        model=cfg.llm_model,
        concurrency=concurrency,
        batch_size=batch_size,
        shard_name=shard_path.name,
        enabled=not args.quiet,
    )

    if args.quiet:
        print(f"run_id={run_id}  batch_size={batch_size}  concurrency={concurrency}  "
              f"model={cfg.llm_model}")
        print(f"generating from {plan_path.name} -> {shard_path.name} "
              f"({len(job_targets)} jobs in scope)")

    lock = threading.Lock()
    produced_ref = [0]
    session_scenarios: list[Scenario] = []
    fout = shard_path.open("a", encoding="utf-8")

    def _run_job(job: GenJob, target_count: int, global_offset: int):
        already = min(target_count, done.get(job.key, 0))
        remaining = target_count - already
        idx = already
        made = 0
        consecutive_empty = 0
        while remaining > 0:
            with lock:
                if args.limit is not None and produced_ref[0] >= args.limit:
                    break
                if args.limit is not None:
                    remaining = min(remaining, max(0, args.limit - produced_ref[0]))
            if remaining <= 0:
                break
            n = min(batch_size, remaining)
            scenarios = generate_batch(client, job, idx, n, db, run_id, stream=stream,
                                       global_offset=global_offset,
                                       corpus_tag=getattr(cfg, "corpus_tag", "") or "")
            written = 0
            with lock:
                for s in scenarios:
                    if args.limit is not None and produced_ref[0] >= args.limit:
                        break
                    rec = s.to_dict()
                    rec["_job_key"] = job.key
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    session_scenarios.append(s)
                    produced_ref[0] += 1
                    made += 1
                    written += 1
                fout.flush()
            idx += n
            remaining -= written

            # unit-eval failures inside this batch mean some slots weren't filled —
            # make the regeneration explicit (terminal stream + monitor can both see it
            # via the unit_evals fail rows already logged in generate_batch above).
            if written < n:
                needed = n - written
                if stream.enabled:
                    stream.emit("regenerate", job_key=job.key, needed=needed)
                else:
                    print(f"  ↻ regenerating {needed} scenario(s) for job {job.key} "
                          f"(unit-eval fail — retrying next batch)")

            if len(scenarios) == 0:
                consecutive_empty += 1
                if consecutive_empty >= 5:
                    if stream.enabled:
                        stream.emit("job_give_up", job_key=job.key, remaining=remaining)
                    else:
                        print(f"  ! giving up on job {job.key} after 5 empty batches "
                              f"({remaining} scenarios never produced)")
                    break
            else:
                consecutive_empty = 0
        return made

    summary = {"llm_cost_usd": 0.0, "llm_calls": 0, "prompt_tokens": 0,
               "completion_tokens": 0, "scenarios_valid": 0, "scenarios_requested": 0}
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(_run_job, j, tc, off) for j, tc, off in job_targets}
            if args.quiet:
                for fut in as_completed(futures):
                    if args.limit is not None and produced_ref[0] >= args.limit:
                        for f in futures:
                            f.cancel()
                        break
                    fut.result()
            else:
                _run_live_loop(futures, stream, args.limit, produced_ref)
    finally:
        fout.close()
        summary = db.run_summary(run_id)
        unit_summary = db.unit_eval_summary(run_id)
        db.close()

    produced = produced_ref[0]
    if not args.quiet:
        stream.print_final(unit_summary=unit_summary)
    else:
        print(f"done: {produced} new scenarios -> {shard_path}")
        print(f"cost this run: ${summary['llm_cost_usd']:.4f}  "
              f"({summary['llm_calls']} calls, {summary['prompt_tokens']}+"
              f"{summary['completion_tokens']} tokens, "
              f"{summary['scenarios_valid']}/{summary['scenarios_requested']} valid)")
        ue_rate = (unit_summary["passed"] / unit_summary["total"] * 100.0) if unit_summary["total"] else 0.0
        print(f"unit-eval (§8.5, per scenario): {unit_summary['passed']}/{unit_summary['total']} "
              f"passed ({ue_rate:.1f}%), {unit_summary['failed']} failed-then-regenerated")

    if cfg.llm_price_in_per_1m_usd == 0 and cfg.llm_price_out_per_1m_usd == 0 and args.quiet:
        print("  (pricing not set in config -> cost_usd is 0; fill llm_price_*_per_1m_usd "
              "for your chosen model/provider to see real cost)")

    if args.no_eval:
        return

    eval_scenarios = session_scenarios if args.limit is not None else _load_scenarios(shard_path)
    judge = LLMClient.for_judge(cfg) if args.judge else None
    _, exit_code = run_data_quality_eval(
        eval_scenarios,
        target_shares=DEFAULT_LANGUAGE_SHARES,
        judge_client=judge,
        judge_n=args.judge_n if args.judge else 0,
        report_out=str(ROOT / args.report_out),
        skip_balance=args.limit is not None,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
