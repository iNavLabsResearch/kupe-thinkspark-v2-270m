#!/usr/bin/env python
"""
Self-contained HTML chart report — dark theme + Chart.js, same convention as
kupe-tts/text_scripts/build_cost_eda_html.py so it feels like the rest of the kupe family.

Reads ONLY the SQLite DB (thinkspark.db.ReadOnlyDB, safe alongside a running generator)
plus data/plan/plan_summary.json for the TARGET side of every chart — it never
recomputes or touches the distribution/validation logic itself, so the report can never
disagree with what 01/02/05 actually produced or planned.

Shows, at a glance:
    - stat cards: scenarios done/target, unit-eval pass rate, cost spent, projected
      total, TTS audio hours/target, budget % used
    - ACTUAL vs TARGET bars by behaviour and by language — the fastest way to see the
      corpus is still exactly the balanced distribution planned in Section 8.1-8.2
    - cumulative LLM spend over time
    - unit-level eval pass/fail (Section 8.5, per scenario, from the fail-flag chain)

    conda activate llms
    python scripts/12_build_report.py --config configs/data_gen.yaml
    python scripts/12_build_report.py --out reports/generation_report.html
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import DataGenConfig
from thinkspark.db import ReadOnlyDB
from thinkspark import distribution as dist


def _split_job_key(job_key: str) -> tuple[str, str, str, str]:
    """job_key = 'behaviour|language|domain|gender|force_silence' — see GenJob.key."""
    parts = (job_key or "").split("|")
    parts += [""] * (5 - len(parts))
    return parts[0], parts[1], parts[2], parts[3]


def build(cfg: DataGenConfig, db: ReadOnlyDB, plan_summary: dict) -> dict:
    rows = db.fetchall(
        "SELECT job_key, requested_n, valid_n, cost_usd, prompt_tokens, "
        "completion_tokens, created_at FROM llm_calls ORDER BY id"
    )

    by_behaviour: dict[str, dict] = defaultdict(lambda: {"actual": 0, "cost": 0.0})
    by_language: dict[str, dict] = defaultdict(lambda: {"actual": 0, "cost": 0.0})
    by_gender: dict[str, int] = defaultdict(int)

    cum = 0.0
    cum_points: list[tuple[int, float]] = []
    total_cost = total_requested = total_valid = 0

    for i, (job_key, requested_n, valid_n, cost_usd, ptok, ctok, created_at) in enumerate(rows):
        behaviour, language, domain, gender = _split_job_key(job_key)
        valid_n = valid_n or 0
        cost_usd = cost_usd or 0.0

        by_behaviour[behaviour]["actual"] += valid_n
        by_behaviour[behaviour]["cost"] += cost_usd
        by_language[language]["actual"] += valid_n
        by_language[language]["cost"] += cost_usd
        by_gender[gender] += valid_n

        cum += cost_usd
        total_cost += cost_usd
        total_requested += requested_n or 0
        total_valid += valid_n
        cum_points.append((i + 1, cum))

    # downsample the cumulative line like kupe-tts does, for a light HTML file
    step = max(1, len(cum_points) // 200 or 1)
    cum_sampled = [p for j, p in enumerate(cum_points) if j % step == 0 or j == len(cum_points) - 1]

    n_calls, n_ok, ptok_sum, ctok_sum = db.fetchone(
        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END),0), "
        "COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0) FROM llm_calls"
    )

    # unit-level eval (Section 8.5, per scenario)
    ue_total, ue_pass, ue_fail = db.fetchone(
        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN status='pass' THEN 1 ELSE 0 END),0), "
        "COALESCE(SUM(CASE WHEN status='fail' THEN 1 ELSE 0 END),0) FROM unit_evals"
    )
    fail_rows = db.fetchall("SELECT job_key, fail_flag FROM unit_evals WHERE status='fail'")
    max_retry = 0
    fails_by_behaviour: dict[str, int] = defaultdict(int)
    for job_key, fail_flag in fail_rows:
        behaviour, *_ = _split_job_key(job_key)
        fails_by_behaviour[behaviour] += 1
        m = re.match(r"fail(\d+)", fail_flag or "")
        if m:
            max_retry = max(max_retry, int(m.group(1)))

    # TTS
    n_tts, tts_ok, tts_secs, tts_cost = db.fetchone(
        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END),0), "
        "COALESCE(SUM(duration_s),0), COALESCE(SUM(cost_usd),0) FROM tts_calls"
    )

    # ---- assemble behaviour/language tables against the PLAN's targets --------
    target_beh = plan_summary.get("by_behaviour", {})
    target_lang = plan_summary.get("by_language", {})
    target_total = plan_summary.get("total_scenarios", 0)

    behaviour_rows = []
    all_behaviours = sorted(set(target_beh) | set(by_behaviour))
    for name in all_behaviours:
        target = target_beh.get(name, 0)
        actual = by_behaviour.get(name, {"actual": 0, "cost": 0.0})["actual"]
        cost = by_behaviour.get(name, {"actual": 0, "cost": 0.0})["cost"]
        behaviour_rows.append({
            "name": name, "target": target, "actual": actual,
            "pct": round(100.0 * actual / target, 1) if target else 0.0,
            "cost": round(cost, 4), "fails": fails_by_behaviour.get(name, 0),
        })

    language_rows = []
    all_languages = sorted(set(target_lang) | set(by_language))
    for name in all_languages:
        target = target_lang.get(name, 0)
        actual = by_language.get(name, {"actual": 0, "cost": 0.0})["actual"]
        language_rows.append({
            "name": name, "target": target, "actual": actual,
            "pct": round(100.0 * actual / target, 1) if target else 0.0,
        })

    projected_llm_cost = (total_cost / total_valid * target_total) if total_valid else 0.0
    avg_hours_per_scn = (tts_secs / 3600.0 / tts_ok) if tts_ok else None
    projected_tts_cost = (
        avg_hours_per_scn * target_total * cfg.soniox_price_per_hour_usd
        if avg_hours_per_scn else cfg.total_hours * cfg.soniox_price_per_hour_usd
    )
    spent_usd = total_cost + (tts_cost or 0.0)
    projected_total_usd = projected_llm_cost + projected_tts_cost

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": cfg.llm_model,
        "batch_size": cfg.batch_size,
        "concurrency": cfg.llm_concurrency,
        "target_total": target_total,
        "actual_total": total_valid,
        "pct_done": round(100.0 * total_valid / target_total, 1) if target_total else 0.0,
        "llm_calls": n_calls, "llm_calls_ok": n_ok,
        "prompt_tokens": ptok_sum, "completion_tokens": ctok_sum,
        "llm_cost_usd": round(total_cost, 4),
        "projected_llm_cost_usd": round(projected_llm_cost, 4),
        "unit_eval": {
            "total": ue_total, "pass": ue_pass, "fail": ue_fail,
            "pass_rate": round(100.0 * ue_pass / ue_total, 1) if ue_total else 0.0,
            "max_retry": max_retry,
        },
        "tts": {
            "calls": n_tts, "ok": tts_ok, "audio_hours": round((tts_secs or 0) / 3600.0, 3),
            "target_hours": cfg.total_hours, "cost_usd": round(tts_cost or 0.0, 4),
            "projected_cost_usd": round(projected_tts_cost, 4),
        },
        "budget": {
            "spent_usd": round(spent_usd, 4), "spent_inr": round(spent_usd * cfg.inr_per_usd, 2),
            "projected_total_usd": round(projected_total_usd, 4),
            "projected_total_inr": round(projected_total_usd * cfg.inr_per_usd, 2),
            "target_inr": cfg.budget_inr_target,
            "pct_of_budget": round(100.0 * projected_total_usd * cfg.inr_per_usd / cfg.budget_inr_target, 1)
                            if cfg.budget_inr_target else 0.0,
        },
        "by_gender": dict(by_gender),
        "behaviour_rows": behaviour_rows,
        "language_rows": language_rows,
        "cum_cost": [{"x": x, "y": round(y, 4)} for x, y in cum_sampled],
    }


# --------------------------------------------------------------------------- #
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ThinkSpark-v2-350M — generation report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1419; --panel: #1a222c; --line: #2a3542; --text: #e7eef6;
    --muted: #8b9aab; --accent: #3dd6c6; --warn: #f0b429; --danger: #ff6b6b; --ok: #6bcb77;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
    background: radial-gradient(1200px 600px at 10% -10%, #1b2a33 0%, var(--bg) 55%);
    color: var(--text); line-height: 1.45;
  }
  header {
    padding: 24px 32px 14px; border-bottom: 1px solid var(--line);
    display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap;
  }
  header h1 { margin: 0 0 6px; font-size: 1.55rem; letter-spacing: -0.02em; }
  header p { margin: 0; color: var(--muted); font-size: 0.9rem; }
  main { padding: 22px 32px 48px; max-width: 1400px; margin: 0 auto; }
  h2 { font-size: 1.05rem; margin: 26px 0 12px; }
  .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
  .card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px;
  }
  .card .label { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
  .card .value { font-size: 1.35rem; font-weight: 650; margin-top: 4px; }
  .card .sub { color: var(--muted); font-size: 0.8rem; margin-top: 2px; }
  .accent { color: var(--accent); } .warn { color: var(--warn); }
  .danger { color: var(--danger); } .ok { color: var(--ok); }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
  .charts { display: grid; gap: 16px; grid-template-columns: 1fr 1fr; }
  @media (max-width: 900px) { .charts { grid-template-columns: 1fr; } }
  table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
  th, td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
  th { color: var(--muted); font-weight: 600; position: sticky; top: 0; background: #1e2833; }
  .table-wrap { max-height: 420px; overflow: auto; border: 1px solid var(--line); border-radius: 10px; }
  .bar { height: 8px; background: #243040; border-radius: 99px; overflow: hidden; width: 90px; display: inline-block; vertical-align: middle; }
  .bar > span { display: block; height: 100%; background: linear-gradient(90deg, #2bbbad, #6bcb77); }
  .bar.over > span { background: linear-gradient(90deg, #f0b429, #ff6b6b); }
</style>
</head>
<body>
<header>
  <div>
    <h1 id="title">ThinkSpark-v2-350M — generation report</h1>
    <p id="meta">Loading…</p>
  </div>
</header>
<main>
  <h2>Overview</h2>
  <div class="grid" id="overviewCards"></div>

  <h2>Unit-level eval (§8.5, checked per scenario, immediately)</h2>
  <div class="grid" id="unitCards"></div>

  <h2>Budget</h2>
  <div class="grid" id="budgetCards"></div>

  <h2>Scenarios by behaviour — actual vs. planned target</h2>
  <div class="panel"><canvas id="behChart" height="90"></canvas></div>
  <div class="table-wrap" style="margin-top:12px">
    <table id="behTable">
      <thead><tr><th>Behaviour</th><th>Actual</th><th>Target</th><th>Progress</th><th>Cost</th><th>Unit-eval fails</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <h2>Scenarios by language — actual vs. planned target</h2>
  <div class="panel"><canvas id="langChart" height="90"></canvas></div>
  <div class="table-wrap" style="margin-top:12px">
    <table id="langTable">
      <thead><tr><th>Language</th><th>Actual</th><th>Target</th><th>Progress</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <h2>Cumulative LLM spend</h2>
  <div class="panel"><canvas id="cumChart" height="90"></canvas></div>
</main>
<script>
const DATA = __DATA__;

function fmtInt(n) { return Number(n||0).toLocaleString(); }
function fmt(n,d=2) { return Number(n||0).toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d}); }
function money(usd) { return "$" + fmt(usd, 4); }

document.getElementById("meta").textContent =
  `${DATA.model} · batch=${DATA.batch_size} concurrency=${DATA.concurrency} · generated ${DATA.generated_at}`;

document.getElementById("overviewCards").innerHTML = [
  ["Scenarios", `${fmtInt(DATA.actual_total)} / ${fmtInt(DATA.target_total)}`, "accent", `${DATA.pct_done}% done`],
  ["LLM calls", `${fmtInt(DATA.llm_calls_ok)} / ${fmtInt(DATA.llm_calls)}`, "", "ok / total"],
  ["Tokens", `${fmtInt(DATA.prompt_tokens)} in / ${fmtInt(DATA.completion_tokens)} out`, "", ""],
  ["LLM cost so far", money(DATA.llm_cost_usd), "warn", `-> projected ${money(DATA.projected_llm_cost_usd)}`],
  ["TTS audio", `${fmt(DATA.tts.audio_hours,2)}h / ${fmt(DATA.tts.target_hours,0)}h`, "accent", `${DATA.tts.ok} rendered ok`],
  ["TTS cost so far", money(DATA.tts.cost_usd), "warn", `-> projected ${money(DATA.tts.projected_cost_usd)}`],
].map(([l,v,c,s]) => `<div class="card"><div class="label">${l}</div><div class="value ${c}">${v}</div><div class="sub">${s}</div></div>`).join("");

const u = DATA.unit_eval;
document.getElementById("unitCards").innerHTML = [
  ["Checked", fmtInt(u.total), "", "scenarios validated the instant they're produced"],
  ["Passed", fmtInt(u.pass), "ok", `${u.pass_rate}% pass rate`],
  ["Failed (then regenerated)", fmtInt(u.fail), u.fail ? "danger" : "", ""],
  ["Worst retries", u.max_retry ? `fail${u.max_retry}` : "none", u.max_retry >= 3 ? "danger" : "", "highest fail-count for any one job"],
].map(([l,v,c,s]) => `<div class="card"><div class="label">${l}</div><div class="value ${c}">${v}</div><div class="sub">${s}</div></div>`).join("");

const b = DATA.budget;
document.getElementById("budgetCards").innerHTML = [
  ["Spent so far", `${money(b.spent_usd)}`, "warn", `~INR ${fmtInt(b.spent_inr)}`],
  ["Projected total", `${money(b.projected_total_usd)}`, "danger", `~INR ${fmtInt(b.projected_total_inr)}`],
  ["Budget target", `INR ${fmtInt(b.target_inr)}`, "", "Section 13"],
  ["% of budget (projected)", `${b.pct_of_budget}%`, b.pct_of_budget > 100 ? "danger" : "ok", ""],
].map(([l,v,c,s]) => `<div class="card"><div class="label">${l}</div><div class="value ${c}">${v}</div><div class="sub">${s}</div></div>`).join("");

document.querySelector("#behTable tbody").innerHTML = DATA.behaviour_rows.map(r => `
  <tr>
    <td><strong>${r.name}</strong></td>
    <td>${fmtInt(r.actual)}</td>
    <td>${fmtInt(r.target)}</td>
    <td><div class="bar ${r.pct>110?'over':''}"><span style="width:${Math.min(r.pct,100)}%"></span></div> ${r.pct}%</td>
    <td>${money(r.cost)}</td>
    <td>${r.fails ? r.fails : "—"}</td>
  </tr>`).join("");

document.querySelector("#langTable tbody").innerHTML = DATA.language_rows.map(r => `
  <tr>
    <td><strong>${r.name}</strong></td>
    <td>${fmtInt(r.actual)}</td>
    <td>${fmtInt(r.target)}</td>
    <td><div class="bar ${r.pct>110?'over':''}"><span style="width:${Math.min(r.pct,100)}%"></span></div> ${r.pct}%</td>
  </tr>`).join("");

new Chart(document.getElementById("behChart"), {
  type: "bar",
  data: {
    labels: DATA.behaviour_rows.map(r => r.name),
    datasets: [
      { label: "Actual", data: DATA.behaviour_rows.map(r => r.actual),
        backgroundColor: "#3dd6c6aa", borderColor: "#3dd6c6", borderWidth: 1 },
      { label: "Target", data: DATA.behaviour_rows.map(r => r.target),
        backgroundColor: "#8b9aab55", borderColor: "#8b9aab", borderWidth: 1 },
    ]
  },
  options: {
    plugins: { legend: { labels: { color: "#e7eef6" } } },
    scales: {
      x: { ticks: { color: "#8b9aab" }, grid: { color: "#2a3542" } },
      y: { ticks: { color: "#8b9aab" }, grid: { color: "#2a3542" } }
    }
  }
});

new Chart(document.getElementById("langChart"), {
  type: "bar",
  data: {
    labels: DATA.language_rows.map(r => r.name),
    datasets: [
      { label: "Actual", data: DATA.language_rows.map(r => r.actual),
        backgroundColor: "#f0b429aa", borderColor: "#f0b429", borderWidth: 1 },
      { label: "Target", data: DATA.language_rows.map(r => r.target),
        backgroundColor: "#8b9aab55", borderColor: "#8b9aab", borderWidth: 1 },
    ]
  },
  options: {
    plugins: { legend: { labels: { color: "#e7eef6" } } },
    scales: {
      x: { ticks: { color: "#8b9aab" }, grid: { color: "#2a3542" } },
      y: { ticks: { color: "#8b9aab" }, grid: { color: "#2a3542" } }
    }
  }
});

new Chart(document.getElementById("cumChart"), {
  type: "line",
  data: {
    labels: DATA.cum_cost.map(p => String(p.x)),
    datasets: [{ label: "Cumulative $", data: DATA.cum_cost.map(p => p.y),
      borderColor: "#6bcb77", backgroundColor: "#6bcb7733", fill: true, tension: 0.25, pointRadius: 0 }]
  },
  options: {
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: "#8b9aab", maxTicksLimit: 12 }, grid: { color: "#2a3542" } },
      y: { ticks: { color: "#8b9aab" }, grid: { color: "#2a3542" } }
    }
  }
});
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_gen.yaml")
    ap.add_argument("--plan-dir", default="data/plan")
    ap.add_argument("--out", default="reports/generation_report.html")
    args = ap.parse_args()

    cfg = DataGenConfig.from_yaml(ROOT / args.config)
    db_path = ROOT / cfg.db_path
    if not db_path.exists():
        raise SystemExit(f"no DB found at {db_path} — run scripts/02_generate_scripts.py first.")

    plan_summary = dist.load_or_write_plan(cfg, ROOT / args.plan_dir)

    db = ReadOnlyDB(db_path)
    try:
        data = build(cfg, db, plan_summary)
    finally:
        db.close()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"  {data['actual_total']}/{data['target_total']} scenarios ({data['pct_done']}%)  "
          f"unit-eval {data['unit_eval']['pass_rate']}% pass  "
          f"${data['llm_cost_usd']:.4f} spent -> ${data['projected_llm_cost_usd']:.4f} projected")


if __name__ == "__main__":
    main()
