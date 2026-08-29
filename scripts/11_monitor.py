#!/usr/bin/env python
"""
Live monitor for data generation — run this in a SEPARATE terminal while
scripts/02_generate_scripts.py (and/or 03_render_user_audio.py) are running.

Reads ONLY from the SQLite DB (thinkspark.db.ReadOnlyDB, WAL-mode-safe concurrent
reader — never writes, never touches the JSONL shards) and refreshes every
`--interval` seconds with everything you asked to see:

    - cost spent so far            (LLM + TTS, USD and INR)
    - expected / projected total cost   (extrapolated from the observed cost-per-item
                                          rate, against the plan's target scenario count)
    - how much budget is left      (projected total vs configs/data_gen.yaml budget_inr_target)
    - how much got done so far     (scenarios valid / target, audio hours / target)
    - latency                      (mean / p50 / p95 per LLM batch call and per TTS call,
                                     plus an estimated seconds-per-scenario)
    - failures vs passes           (status breakdown + last few error messages)
    - throughput + ETA             (scenarios/min over a recent window, minutes remaining)
    - unit-level eval (§8.5)       (per-scenario pass/fail, checked the instant it's
                                     generated, with the job-scoped 'failN' retry tag —
                                     see thinkspark.db unit_evals + scripts/02's
                                     generate_batch())

Uses `rich` for a live-refreshing dashboard if installed, else falls back to a plain
ANSI-clear text loop (no hard rich dependency).

    conda activate llms
    python scripts/11_monitor.py --config configs/data_gen.yaml
    python scripts/11_monitor.py --config configs/data_gen.yaml --once   # single snapshot, exit
    python scripts/11_monitor.py --interval 10                            # slower refresh
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import DataGenConfig
from thinkspark.db import ReadOnlyDB
from thinkspark import distribution as dist

RECENT_WINDOW_S = 600.0    # look at the last 10 min for throughput / "current speed"
RECENT_ROWS_CAP = 5000     # cap per-row fetches (latency/error samples) for perf


# --------------------------------------------------------------------------- #
def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _target_scenarios(cfg: DataGenConfig) -> int:
    summary_path = ROOT / "data" / "plan" / "plan_summary.json"
    if summary_path.exists():
        try:
            return int(json.loads(summary_path.read_text())["total_scenarios"])
        except Exception:
            pass
    return dist.total_scenarios(cfg)


# --------------------------------------------------------------------------- #
def gather_stats(db: ReadOnlyDB, cfg: DataGenConfig, target_scenarios: int) -> dict:
    now = time.time()

    # ---- LLM aggregate totals (all runs, all time) ---------------------------
    n_calls, req_sum, valid_sum, ptok, ctok, cost_sum = db.fetchone(
        "SELECT COUNT(*), COALESCE(SUM(requested_n),0), COALESCE(SUM(valid_n),0), "
        "COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), "
        "COALESCE(SUM(cost_usd),0) FROM llm_calls"
    )
    status_rows = db.fetchall(
        "SELECT status, COUNT(*) FROM llm_calls GROUP BY status"
    )
    status_counts = {row[0] or "unknown": row[1] for row in status_rows}
    n_ok = status_counts.get("ok", 0)
    n_fail = n_calls - n_ok

    # recent rows for latency distribution + throughput + error tail
    recent = db.fetchall(
        "SELECT requested_n, valid_n, latency_ms, status, job_key, error, created_at "
        "FROM llm_calls ORDER BY id DESC LIMIT ?", (RECENT_ROWS_CAP,)
    )
    all_latency = [r[2] for r in recent if r[2] is not None]
    per_item_latency = [r[2] / r[0] for r in recent if r[0] and r[2] is not None]
    # Require a minimum time span before trusting the "recent window" rate — with high
    # concurrency, many calls can land within the same second or two right after a burst,
    # which would otherwise spike throughput/ETA to nonsense (division by ~0 seconds).
    MIN_SPAN_S = 30.0
    recent_window = [r for r in recent if r[6] and (now - r[6]) <= RECENT_WINDOW_S]
    recent_valid = sum(r[1] or 0 for r in recent_window)
    recent_span_s = (now - min(r[6] for r in recent_window)) if recent_window else 0.0
    throughput_per_min = (
        recent_valid / (recent_span_s / 60.0) if recent_span_s >= MIN_SPAN_S else 0.0
    )

    first_ts = db.fetchone("SELECT MIN(created_at) FROM llm_calls")[0]
    if not throughput_per_min and first_ts and valid_sum:
        elapsed_s = max(MIN_SPAN_S, now - first_ts)
        throughput_per_min = valid_sum / (elapsed_s / 60.0)

    remaining_scn = max(0, target_scenarios - valid_sum)
    eta_min = (remaining_scn / throughput_per_min) if throughput_per_min > 0 else None

    projected_llm_cost = (cost_sum / valid_sum * target_scenarios) if valid_sum else None

    last_errors = [
        {"job_key": r[4], "status": r[3], "error": (r[5] or "")[:120]}
        for r in recent if r[3] != "ok"
    ][:5]

    llm = {
        "calls": n_calls, "ok": n_ok, "failed": n_fail,
        "status_counts": status_counts,
        "requested": req_sum, "valid": valid_sum,
        "pass_rate": (valid_sum / req_sum) if req_sum else 0.0,
        "prompt_tokens": ptok, "completion_tokens": ctok,
        "cost_usd": cost_sum,
        "latency_mean_ms": (sum(all_latency) / len(all_latency)) if all_latency else 0.0,
        "latency_p50_ms": _percentile(all_latency, 50),
        "latency_p95_ms": _percentile(all_latency, 95),
        "sec_per_scenario": (sum(per_item_latency) / len(per_item_latency) / 1000.0)
                            if per_item_latency else 0.0,
        "throughput_per_min": throughput_per_min,
        "eta_min": eta_min,
        "projected_cost_usd": projected_llm_cost,
        "last_errors": last_errors,
        "target": target_scenarios,
        "remaining": remaining_scn,
    }

    # ---- TTS aggregate totals -------------------------------------------------
    n_tts, tts_ok, tts_secs, tts_cost = db.fetchone(
        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END),0), "
        "COALESCE(SUM(duration_s),0), COALESCE(SUM(cost_usd),0) FROM tts_calls"
    )
    tts_fail = n_tts - tts_ok
    tts_rate_limited = db.fetchone(
        "SELECT COUNT(*) FROM tts_calls WHERE status='rate_limited'"
    )[0]
    tts_recent = db.fetchall(
        "SELECT latency_ms, status, scenario_id, error FROM tts_calls "
        "ORDER BY id DESC LIMIT ?", (RECENT_ROWS_CAP,)
    )
    tts_latency = [r[0] for r in tts_recent if r[0] is not None]
    tts_last_errors = [
        {"scenario_id": r[2], "status": r[1], "error": (r[3] or "")[:120]}
        for r in tts_recent if r[1] != "ok"
    ][:5]

    target_hours = cfg.total_hours
    avg_hours_per_scn = (tts_secs / 3600.0 / tts_ok) if tts_ok else None
    projected_tts_cost = (
        avg_hours_per_scn * target_scenarios * cfg.soniox_price_per_hour_usd
        if avg_hours_per_scn else target_hours * cfg.soniox_price_per_hour_usd
    )

    tts = {
        "calls": n_tts, "ok": tts_ok, "failed": tts_fail, "rate_limited": tts_rate_limited,
        "audio_hours": tts_secs / 3600.0, "target_hours": target_hours,
        "cost_usd": tts_cost,
        "latency_mean_ms": (sum(tts_latency) / len(tts_latency)) if tts_latency else 0.0,
        "latency_p50_ms": _percentile(tts_latency, 50),
        "latency_p95_ms": _percentile(tts_latency, 95),
        "projected_cost_usd": projected_tts_cost,
        "last_errors": tts_last_errors,
    }

    # ---- budget roll-up ---------------------------------------------------
    spent_usd = llm["cost_usd"] + tts["cost_usd"]
    proj_llm = llm["projected_cost_usd"] if llm["projected_cost_usd"] is not None else llm["cost_usd"]
    projected_total_usd = proj_llm + tts["projected_cost_usd"]
    budget_inr = cfg.budget_inr_target
    budget = {
        "spent_usd": spent_usd, "spent_inr": spent_usd * cfg.inr_per_usd,
        "projected_total_usd": projected_total_usd,
        "projected_total_inr": projected_total_usd * cfg.inr_per_usd,
        "remaining_projected_usd": max(0.0, projected_total_usd - spent_usd),
        "remaining_projected_inr": max(0.0, projected_total_usd - spent_usd) * cfg.inr_per_usd,
        "budget_inr_target": budget_inr,
        "pct_spent": (spent_usd * cfg.inr_per_usd / budget_inr * 100.0) if budget_inr else 0.0,
        "pct_projected": (projected_total_usd * cfg.inr_per_usd / budget_inr * 100.0)
                         if budget_inr else 0.0,
    }

    # ---- Unit-level eval (Section 8.5, per scenario) -------------------------
    # Every generated scenario is validated the moment it's produced (schema/vocab/
    # script — see thinkspark.validators.validate_scenario). A fail gets a job-scoped
    # 'failN' tag and the caller regenerates that slot. This is the fast, continuous
    # gate — separate from the one-shot corpus-wide report (scripts/05_validate_data.py).
    ue_total, ue_pass, ue_fail = db.fetchone(
        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN status='pass' THEN 1 ELSE 0 END),0), "
        "COALESCE(SUM(CASE WHEN status='fail' THEN 1 ELSE 0 END),0) FROM unit_evals"
    ) if _table_exists(db, "unit_evals") else (0, 0, 0)

    ue_recent_fails: list[dict] = []
    max_retry = 0
    if ue_fail:
        rows = db.fetchall(
            "SELECT job_key, fail_flag, errors FROM unit_evals WHERE status='fail' "
            "ORDER BY id DESC LIMIT ?", (RECENT_ROWS_CAP,)
        )
        ue_recent_fails = [
            {"job_key": r[0], "fail_flag": r[1], "errors": _short_errors(r[2])}
            for r in rows[:5]
        ]
        for r in rows:
            m = re.match(r"fail(\d+)", r[1] or "")
            if m:
                max_retry = max(max_retry, int(m.group(1)))

    unit_eval = {
        "total": ue_total, "passed": ue_pass, "failed": ue_fail,
        "pass_rate": (ue_pass / ue_total) if ue_total else 0.0,
        "max_retry": max_retry,
        "recent_fails": ue_recent_fails,
    }

    return {"llm": llm, "tts": tts, "budget": budget, "unit_eval": unit_eval, "updated_at": now}


def _table_exists(db: ReadOnlyDB, name: str) -> bool:
    row = db.fetchone(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return row is not None


def _short_errors(raw_json: str | None) -> str:
    if not raw_json:
        return ""
    try:
        errs = json.loads(raw_json)
        return "; ".join(errs[:2]) if isinstance(errs, list) else str(errs)[:100]
    except Exception:
        return str(raw_json)[:100]


# --------------------------------------------------------------------------- #
def render_plain(stats: dict, cfg: DataGenConfig):
    import sys
    l, t, b, u = stats["llm"], stats["tts"], stats["budget"], stats["unit_eval"]
    sys.stdout.write("\x1b[2J\x1b[H")  # clear screen, cursor home

    print("=" * 72)
    print(f" ThinkSpark-v2-350M — data generation monitor   "
         f"(refreshed {time.strftime('%H:%M:%S')})")
    print("=" * 72)

    print("\n[UNIT-LEVEL EVAL]  (§8.5 check on each generated scenario, immediately)")
    print(f"  checked        : {u['total']}   passed={u['passed']}  failed={u['failed']}   "
         f"pass_rate={u['pass_rate']*100:.1f}%")
    print(f"  worst retries  : {'fail' + str(u['max_retry']) if u['max_retry'] else 'none'} "
         f"(highest fail-count reached by any single job before it succeeded)")
    if u["recent_fails"]:
        print("  recent fails   :")
        for e in u["recent_fails"]:
            print(f"    - [{e['fail_flag']}] {e['job_key']}: {e['errors']}")

    print(f"\n[LLM GENERATION]  model={cfg.llm_model}  batch_size={cfg.batch_size}  "
         f"concurrency={cfg.llm_concurrency}")
    print(f"  scenarios      : {l['valid']} / {l['target']} valid "
         f"({100*l['valid']/max(1,l['target']):.1f}%)   remaining={l['remaining']}")
    print(f"  calls          : {l['calls']} total   ok={l['ok']}  failed={l['failed']}   "
         f"pass_rate={l['pass_rate']*100:.1f}%")
    if l["status_counts"]:
        print(f"  by status      : {l['status_counts']}")
    print(f"  tokens         : {l['prompt_tokens']} in / {l['completion_tokens']} out")
    print(f"  latency (ms)   : mean={l['latency_mean_ms']:.0f}  p50={l['latency_p50_ms']:.0f}  "
         f"p95={l['latency_p95_ms']:.0f}   ~{l['sec_per_scenario']:.2f}s/scenario")
    eta = f"{l['eta_min']:.1f} min" if l["eta_min"] is not None else "n/a"
    print(f"  throughput     : {l['throughput_per_min']:.1f} scenarios/min   ETA={eta}")
    print(f"  cost so far    : ${l['cost_usd']:.4f}")
    proj = f"${l['projected_cost_usd']:.4f}" if l["projected_cost_usd"] is not None else "n/a (no data yet)"
    print(f"  projected cost : {proj}")
    if l["last_errors"]:
        print("  recent errors  :")
        for e in l["last_errors"]:
            print(f"    - [{e['status']}] {e['job_key']}: {e['error']}")

    print(f"\n[TTS RENDERING]  target={t['target_hours']:.1f}h")
    print(f"  audio hours    : {t['audio_hours']:.3f} / {t['target_hours']:.1f} h "
         f"({100*t['audio_hours']/max(1e-6,t['target_hours']):.1f}%)")
    print(f"  calls          : {t['calls']} total   ok={t['ok']}  failed={t['failed']}  "
         f"(rate_limited={t['rate_limited']})")
    print(f"  latency (ms)   : mean={t['latency_mean_ms']:.0f}  p50={t['latency_p50_ms']:.0f}  "
         f"p95={t['latency_p95_ms']:.0f}")
    print(f"  cost so far    : ${t['cost_usd']:.4f}")
    print(f"  projected cost : ${t['projected_cost_usd']:.4f}")
    if t["last_errors"]:
        print("  recent errors  :")
        for e in t["last_errors"]:
            print(f"    - [{e['status']}] {e['scenario_id']}: {e['error']}")

    print(f"\n[BUDGET]  target = INR {b['budget_inr_target']:.0f}")
    print(f"  spent so far      : ${b['spent_usd']:.4f}  (~INR {b['spent_inr']:.0f})   "
         f"{b['pct_spent']:.1f}% of budget")
    print(f"  projected total   : ${b['projected_total_usd']:.4f}  "
         f"(~INR {b['projected_total_inr']:.0f})   {b['pct_projected']:.1f}% of budget")
    print(f"  projected remaining: ${b['remaining_projected_usd']:.4f}  "
         f"(~INR {b['remaining_projected_inr']:.0f})")
    print("\n(Ctrl+C to stop monitoring — the generator keeps running independently)")


def render_rich(stats: dict, cfg: DataGenConfig):
    from rich.table import Table
    from rich.panel import Panel
    from rich.console import Group
    from rich.markup import escape

    l, t, b, u = stats["llm"], stats["tts"], stats["budget"], stats["unit_eval"]

    unit_tbl = Table.grid(padding=(0, 2))
    unit_tbl.add_column(justify="left", style="bold")
    unit_tbl.add_column(justify="left")
    unit_tbl.add_row("Checked", f"{u['total']}  passed={u['passed']}  failed={u['failed']}  "
                     f"pass_rate={u['pass_rate']*100:.1f}%")
    unit_tbl.add_row("Worst retries", ("fail" + str(u["max_retry"])) if u["max_retry"] else "none")
    if u["recent_fails"]:
        # escape() guards against '[fail1]' etc. being parsed as Rich console markup —
        # unescaped square brackets silently swallow their contents instead of erroring.
        unit_tbl.add_row("Recent fails", "\n".join(
            escape(f"[{e['fail_flag']}] {e['job_key']}: {e['errors']}") for e in u["recent_fails"]))

    llm_tbl = Table.grid(padding=(0, 2))
    llm_tbl.add_column(justify="left", style="bold")
    llm_tbl.add_column(justify="left")
    llm_tbl.add_row("Scenarios", f"{l['valid']} / {l['target']} "
                    f"({100*l['valid']/max(1,l['target']):.1f}%)  remaining={l['remaining']}")
    llm_tbl.add_row("Calls", f"{l['calls']} total  ok={l['ok']}  failed={l['failed']}  "
                    f"pass_rate={l['pass_rate']*100:.1f}%  {l['status_counts']}")
    llm_tbl.add_row("Tokens", f"{l['prompt_tokens']} in / {l['completion_tokens']} out")
    llm_tbl.add_row("Latency", f"mean={l['latency_mean_ms']:.0f}ms  p50={l['latency_p50_ms']:.0f}ms  "
                    f"p95={l['latency_p95_ms']:.0f}ms  ~{l['sec_per_scenario']:.2f}s/scenario")
    eta = f"{l['eta_min']:.1f} min" if l["eta_min"] is not None else "n/a"
    llm_tbl.add_row("Throughput", f"{l['throughput_per_min']:.1f} scenarios/min   ETA={eta}")
    llm_tbl.add_row("Cost", f"${l['cost_usd']:.4f} so far   -> projected "
                    f"{'$%.4f' % l['projected_cost_usd'] if l['projected_cost_usd'] is not None else 'n/a'}")
    if l["last_errors"]:
        llm_tbl.add_row("Recent errors", "\n".join(
            escape(f"[{e['status']}] {e['job_key']}: {e['error']}") for e in l["last_errors"]))

    tts_tbl = Table.grid(padding=(0, 2))
    tts_tbl.add_column(justify="left", style="bold")
    tts_tbl.add_column(justify="left")
    tts_tbl.add_row("Audio", f"{t['audio_hours']:.3f} / {t['target_hours']:.1f} h "
                    f"({100*t['audio_hours']/max(1e-6,t['target_hours']):.1f}%)")
    tts_tbl.add_row("Calls", f"{t['calls']} total  ok={t['ok']}  failed={t['failed']}  "
                    f"(rate_limited={t['rate_limited']})")
    tts_tbl.add_row("Latency", f"mean={t['latency_mean_ms']:.0f}ms  p50={t['latency_p50_ms']:.0f}ms  "
                    f"p95={t['latency_p95_ms']:.0f}ms")
    tts_tbl.add_row("Cost", f"${t['cost_usd']:.4f} so far   -> projected ${t['projected_cost_usd']:.4f}")
    if t["last_errors"]:
        tts_tbl.add_row("Recent errors", "\n".join(
            escape(f"[{e['status']}] {e['scenario_id']}: {e['error']}") for e in t["last_errors"]))

    budget_tbl = Table.grid(padding=(0, 2))
    budget_tbl.add_column(justify="left", style="bold")
    budget_tbl.add_column(justify="left")
    budget_tbl.add_row("Spent so far", f"${b['spent_usd']:.4f}  (~INR {b['spent_inr']:.0f})  "
                       f"{b['pct_spent']:.1f}% of budget")
    budget_tbl.add_row("Projected total", f"${b['projected_total_usd']:.4f}  "
                       f"(~INR {b['projected_total_inr']:.0f})  {b['pct_projected']:.1f}% of budget")
    budget_tbl.add_row("Projected remaining", f"${b['remaining_projected_usd']:.4f}  "
                       f"(~INR {b['remaining_projected_inr']:.0f})")
    budget_tbl.add_row("Budget target", f"INR {b['budget_inr_target']:.0f}  (Section 13)")

    return Group(
        Panel(unit_tbl, title="Unit-level eval — §8.5 check per scenario, immediately", border_style="yellow"),
        Panel(llm_tbl, title=f"LLM generation — {cfg.llm_model} "
             f"(batch={cfg.batch_size}, concurrency={cfg.llm_concurrency})", border_style="cyan"),
        Panel(tts_tbl, title="Soniox TTS rendering", border_style="magenta"),
        Panel(budget_tbl, title="Budget", border_style="green"),
    )


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_gen.yaml")
    ap.add_argument("--interval", type=float, default=5.0, help="refresh seconds")
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--target-scenarios", type=int, default=None,
                    help="override the target scenario count (else read data/plan/plan_summary.json)")
    args = ap.parse_args()

    cfg = DataGenConfig.from_yaml(ROOT / args.config)
    db_path = ROOT / cfg.db_path
    target_scenarios = args.target_scenarios or _target_scenarios(cfg)

    if not db_path.exists():
        raise SystemExit(
            f"no DB found at {db_path} yet — start scripts/02_generate_scripts.py "
            f"(or 03_render_user_audio.py) first, then run this monitor alongside it."
        )

    db = ReadOnlyDB(db_path)
    try:
        use_rich = False
        try:
            from rich.live import Live
            use_rich = True
        except Exception:
            use_rich = False

        if args.once:
            stats = gather_stats(db, cfg, target_scenarios)
            if use_rich:
                from rich.console import Console
                Console().print(render_rich(stats, cfg))
            else:
                render_plain(stats, cfg)
            return

        if use_rich:
            with Live(refresh_per_second=1) as live:
                while True:
                    stats = gather_stats(db, cfg, target_scenarios)
                    live.update(render_rich(stats, cfg))
                    time.sleep(args.interval)
        else:
            print("(tip: `pip install rich` for a nicer live dashboard — falling back to plain text)")
            while True:
                stats = gather_stats(db, cfg, target_scenarios)
                render_plain(stats, cfg)
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nmonitor stopped (generation, if running elsewhere, is unaffected).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
