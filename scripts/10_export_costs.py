#!/usr/bin/env python
"""
Export every logged LLM + TTS call to a flat CSV cost report, and print a summary
against the Section 13 INR 5000 budget.

Reads from the SQLite DB (`data/thinkspark_runs.db` by default — see thinkspark.db) that
scripts 02 (generate) and 03 (render_user_audio) log to on every call. Safe to run at any
time, including mid-run — it's a read-only export.

    conda activate llms
    python scripts/10_export_costs.py
    python scripts/10_export_costs.py --config configs/data_gen.yaml --out reports/cost_report.csv
"""

from __future__ import annotations

import argparse
import sqlite3

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import DataGenConfig
from thinkspark.db import RunDB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_gen.yaml")
    ap.add_argument("--out", default="reports/cost_report.csv")
    args = ap.parse_args()

    cfg = DataGenConfig.from_yaml(ROOT / args.config)
    db_path = ROOT / cfg.db_path
    if not db_path.exists():
        raise SystemExit(f"no DB found at {db_path} — run scripts 02/03 first.")

    db = RunDB(db_path)
    out_path = db.export_costs_csv(ROOT / args.out)

    # aggregate totals directly (read-only queries)
    conn = sqlite3.connect(str(db_path))
    llm_cost, llm_calls, ptok, ctok = conn.execute(
        "SELECT COALESCE(SUM(cost_usd),0), COUNT(*), COALESCE(SUM(prompt_tokens),0), "
        "COALESCE(SUM(completion_tokens),0) FROM llm_calls").fetchone()
    valid, requested = conn.execute(
        "SELECT COALESCE(SUM(valid_n),0), COALESCE(SUM(requested_n),0) FROM llm_calls"
    ).fetchone()
    tts_cost, tts_calls, tts_secs = conn.execute(
        "SELECT COALESCE(SUM(cost_usd),0), COUNT(*), COALESCE(SUM(duration_s),0) "
        "FROM tts_calls").fetchone()
    n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    conn.close()
    db.close()

    total_usd = llm_cost + tts_cost
    total_inr = total_usd * cfg.inr_per_usd

    print("=" * 60)
    print("ThinkSpark-v2-350M — cost report (all runs, all time)")
    print("=" * 60)
    print(f"runs logged        : {n_runs}")
    print(f"LLM calls          : {llm_calls}   ${llm_cost:.4f}   "
         f"({ptok}+{ctok} tokens, {valid}/{requested} scenarios valid)")
    print(f"TTS calls          : {tts_calls}   ${tts_cost:.4f}   "
         f"({tts_secs / 3600.0:.3f} h audio)")
    print(f"TOTAL              : ${total_usd:.4f}  (~₹{total_inr:.0f} @ "
         f"₹{cfg.inr_per_usd}/$)   [Section 13 budget target: ₹5000]")
    if cfg.llm_price_in_per_1m_usd == 0 and cfg.llm_price_out_per_1m_usd == 0:
        print("\n  NOTE: llm_price_*_per_1m_usd is 0 in config -> LLM cost above is $0.")
        print("  Set it in configs/data_gen.yaml for your chosen model/provider")
        print("  (DeepSeek V3/V4-flash, Gemma-3-27B, ...) to see the real number.")
    print(f"\nper-call CSV -> {out_path}")


if __name__ == "__main__":
    main()
