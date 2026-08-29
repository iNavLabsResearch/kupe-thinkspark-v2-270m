#!/usr/bin/env python
"""
Section 8.1-8.3 — build the balanced generation plan and write it to disk.

Produces `data/plan/plan.jsonl` (one GenJob per line) split into `num_parts` shards.
Prints a summary so you can eyeball the behaviour/language/gender balance before
spending any budget. Optional — `scripts/02_generate_scripts.py` auto-builds the same
plan on first run if you skip this, but running it explicitly first is the fastest way
to sanity-check the distribution.

    conda activate llms
    python scripts/01_plan_distribution.py --config configs/data_gen.yaml
"""

from __future__ import annotations

import argparse

from _bootstrap import setup

ROOT = setup()

from thinkspark.config import DataGenConfig
from thinkspark import distribution as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_gen.yaml")
    ap.add_argument("--out", default="data/plan")
    args = ap.parse_args()

    cfg = DataGenConfig.from_yaml(ROOT / args.config)
    jobs, parts, summary = dist.write_plan(cfg, ROOT / args.out)

    print("=" * 60)
    print(f"ThinkSpark-v2-350M generation plan  ({cfg.total_hours} h target)")
    print("=" * 60)
    print(f"total scenarios : {summary['total_scenarios']}")
    print(f"parts           : {cfg.num_parts}  (jobs per part ~ {len(jobs)//cfg.num_parts})")
    print(f"silent backchnl : {summary['silent_backchannel']} "
          f"(deliberate 'say nothing' cases)")
    print("\nby behaviour:")
    for b, n in sorted(summary["by_behaviour"].items()):
        print(f"  {b:<22} {n:>6}")
    print("\nby language:")
    for l, n in sorted(summary["by_language"].items()):
        print(f"  {l:<14} {n:>6}")
    print("\nby gender:")
    for g, n in sorted(summary["by_gender"].items()):
        print(f"  {g:<8} {n:>6}")
    print(f"\nwrote plan to {ROOT / args.out}")
    print("\nnext:  python scripts/02_generate_scripts.py   (resumes automatically)")


if __name__ == "__main__":
    main()
