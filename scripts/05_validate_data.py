#!/usr/bin/env python
"""
Section 8.5 — evaluate the generated data before training on junk.

Runs the five checks (schema / vocab / script / balance / naturalness) over a scenarios
shard and prints a report with pass bars. The LLM-judge naturalness pass is optional
(`--judge`, samples `--judge-n` scenarios) so validation is free by default.

    conda activate llms
    python scripts/05_validate_data.py --in data/scenarios/scenarios_all.jsonl
    python scripts/05_validate_data.py --in data/scenarios/scenarios_all.jsonl --judge --judge-n 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import DataGenConfig
from thinkspark.schema import Scenario
from thinkspark.distribution import DEFAULT_LANGUAGE_SHARES
from thinkspark.llm_client import LLMClient
from thinkspark.gen_stream import run_data_quality_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_gen.yaml")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--judge", action="store_true", help="run LLM-judge naturalness")
    ap.add_argument("--judge-n", type=int, default=200)
    ap.add_argument("--report-out", default="reports/data_quality.txt")
    args = ap.parse_args()

    cfg = DataGenConfig.from_yaml(ROOT / args.config)
    lines = [l for l in Path(ROOT / args.in_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    scenarios = [Scenario.from_dict(json.loads(l)) for l in lines]

    judge = LLMClient.for_judge(cfg) if args.judge and scenarios else None
    _, exit_code = run_data_quality_eval(
        scenarios,
        target_shares=DEFAULT_LANGUAGE_SHARES,
        judge_client=judge,
        judge_n=args.judge_n if args.judge else 0,
        report_out=str(ROOT / args.report_out),
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
