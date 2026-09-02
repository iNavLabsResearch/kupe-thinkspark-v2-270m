#!/usr/bin/env python
"""
Phase-2 corpus audit — answer "do I actually need to generate more data?" with numbers.

Reads the EXISTING frame shards (plus their .npz encodes, if present) and reports:

  * control-flag frame counts + share, and the number of distinct CLIPS carrying each
    flag at all. The second number is the one that matters for the rare decisions: a
    flag that appears on 1 frame in each of 800 clips has 800 learnable examples, and a
    flag present in 30 clips does not, no matter how many total frames the corpus has.
  * the same broken down by behaviour and language, so a shortfall can be traced to the
    generation bucket that should have produced it.
  * the real user-speaking rate from the audio (thinkspark.vad) — the VAP head's actual
    target base rate, which the eval's `true_speak` line should agree with.
  * spoken-span coverage (how many clips carry back-channel text at all).

Then it compares each flag's clip count against `--min-clips-per-flag` and, if anything
falls short, writes a TOP-UP generation config whose `behaviour_shares` are proportional
to the deficit — so the next generation run spends its whole budget on exactly the
behaviours that are missing, and nothing is regenerated. If nothing is short, it says so
and writes nothing: generation is not free and this is the gate.

    conda activate llms
    python scripts/28_data_audit.py --frames "data/frames/*.jsonl"
    python scripts/28_data_audit.py --frames "data/frames/*.jsonl" \
        --min-clips-per-flag 3000 --topup-out configs/data_gen_topup.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from _bootstrap import setup

ROOT = setup()

from thinkspark import vocab
from thinkspark.vad import speaking_from_npz

# Which generation behaviour is the natural producer of each control flag. Used to turn
# a per-flag deficit into a per-behaviour generation budget.
FLAG_TO_BEHAVIOURS: dict[str, list[str]] = {
    "TURN_END":      ["endpoint_end", "correction"],
    "INCOMPLETE":    ["endpoint_hold", "incomplete_thinking"],
    "BARGE_HARD":    ["barge_real", "overlap_comp"],
    "BARGE_SOFT":    ["overlap_comp", "barge_real"],
    "CONTINUE":      ["barge_lookalike", "overlap_coop"],
    "PREFETCH_LLM":  ["prefetch"],
    "COMMIT_LLM":    ["endpoint_end", "prefetch"],
    "CANCEL_LLM":    ["prefetch", "endpoint_hold"],
    "SILENCE_BREAK": ["silence_break"],
    "LISTEN":        ["backchannel", "nonspeech_neg"],
    "HOLD":          ["nonspeech_neg", "backchannel"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames/*.jsonl")
    ap.add_argument("--min-clips-per-flag", type=int, default=2000,
                    help="how many DISTINCT clips must carry a flag before it is "
                        "considered adequately covered")
    ap.add_argument("--vad-sample", type=int, default=400,
                    help="how many clips to open for the speaking-rate estimate (0 = skip)")
    ap.add_argument("--topup-out", default="configs/data_gen_topup.yaml",
                    help="where to write the deficit-driven top-up config (only written "
                        "when there IS a deficit)")
    ap.add_argument("--topup-template", default="configs/data_gen_extended.yaml",
                    help="config whose LLM/TTS/pricing settings the top-up inherits")
    ap.add_argument("--report-out", default="reports/data_audit.json")
    args = ap.parse_args()

    shards = sorted(glob.glob(args.frames if args.frames.startswith("/")
                              else str(ROOT / args.frames)))
    if not shards:
        raise SystemExit(f"no frame shards matched {args.frames}")

    frame_counts: Counter = Counter()
    clip_counts: Counter = Counter()
    by_behaviour: Counter = Counter()
    by_language: Counter = Counter()
    flag_by_behaviour: dict[str, Counter] = defaultdict(Counter)
    total_frames = 0
    n_clips = 0
    n_with_spoken = 0
    enc_paths: list[str] = []

    for shard in shards:
        for line in Path(shard).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_clips += 1
            flags = rec.get("flags") or []
            total_frames += len(flags)
            present = set()
            for fid in flags:
                name = vocab.ID_TO_CONTROL_FLAG.get(int(fid))
                if name is None:
                    continue
                frame_counts[name] += 1
                present.add(name)
            beh = rec.get("behaviour", "?")
            by_behaviour[beh] += 1
            by_language[rec.get("language", "?")] += 1
            for name in present:
                clip_counts[name] += 1
                flag_by_behaviour[name][beh] += 1
            if rec.get("spoken_spans"):
                n_with_spoken += 1
            if rec.get("encoded_path"):
                enc_paths.append(rec["encoded_path"])

    print("=" * 72)
    print(f"Phase-2 corpus audit — {len(shards)} shard(s), {n_clips} clips, "
          f"{total_frames} frames ({total_frames / vocab.FRAMES_PER_HOUR:.1f} h)")
    print("=" * 72)
    print(f"{'flag':<15}{'frames':>10}{'share':>9}{'clips':>9}{'clip%':>8}   top behaviours")
    deficits: dict[str, int] = {}
    for flag in vocab.CONTROL_FLAGS:
        fr = frame_counts.get(flag, 0)
        cl = clip_counts.get(flag, 0)
        top = ", ".join(f"{b}({n})" for b, n in flag_by_behaviour[flag].most_common(2))
        short = "  SHORT" if cl < args.min_clips_per_flag else ""
        if cl < args.min_clips_per_flag:
            deficits[flag] = args.min_clips_per_flag - cl
        print(f"{flag:<15}{fr:>10}{100.0 * fr / max(1, total_frames):>8.3f}%"
              f"{cl:>9}{100.0 * cl / max(1, n_clips):>7.1f}%   {top}{short}")

    print(f"\nclips with back-channel text : {n_with_spoken} "
          f"({100.0 * n_with_spoken / max(1, n_clips):.1f}%)")
    print("\nby behaviour:")
    for b, n in by_behaviour.most_common():
        print(f"  {b:<22}{n:>8}")
    print("\nby language:")
    for l, n in by_language.most_common():
        print(f"  {l:<22}{n:>8}")

    # ---- real speaking rate (the VAP head's target base rate) -----------------
    speak_rate = None
    if args.vad_sample and enc_paths:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(enc_paths), size=min(args.vad_sample, len(enc_paths)),
                          replace=False)
        rates = []
        for i in pick:
            p = enc_paths[int(i)]
            sp = speaking_from_npz(p if Path(p).is_absolute() else ROOT / p)
            if sp is not None and sp.size:
                rates.append(float(sp.mean()))
        if rates:
            speak_rate = float(np.mean(rates))
            print(f"\nuser-speaking rate (VAP target, {len(rates)} clips sampled) : "
                  f"{speak_rate:.3f}")
            print("  scripts/08_evaluate.py's `true_speak` should now match this. If it "
                  "still reports ~0.003, the run is reading a stale shard/checkout.")

    report = {
        "shards": len(shards), "clips": n_clips, "frames": total_frames,
        "hours": round(total_frames / vocab.FRAMES_PER_HOUR, 2),
        "frame_counts": dict(frame_counts), "clip_counts": dict(clip_counts),
        "by_behaviour": dict(by_behaviour), "by_language": dict(by_language),
        "clips_with_spoken_text": n_with_spoken,
        "speaking_rate": speak_rate,
        "min_clips_per_flag": args.min_clips_per_flag,
        "deficits": deficits,
    }
    out = ROOT / args.report_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    # ---- verdict + top-up plan ------------------------------------------------
    print("\n" + "=" * 72)
    if not deficits:
        print(f"VERDICT: no top-up needed — every control flag appears in at least "
              f"{args.min_clips_per_flag} clips.")
        print("Fix the labels/loss and retrain on what you already have; generating more "
              "of a corpus that is already covered buys nothing.")
        print("=" * 72)
        return

    print("VERDICT: top-up needed. Short flags: " +
          ", ".join(f"{f} (+{n} clips)" for f, n in sorted(deficits.items())))
    shares = _behaviour_shares_from_deficits(deficits)
    _write_topup_config(ROOT / args.topup_out, ROOT / args.topup_template, shares,
                        deficits, args.min_clips_per_flag)
    cfg_rel = args.topup_out
    print(f"\nwrote {ROOT / args.topup_out} — generate ONLY the deficit with:")
    print(f"  python scripts/02_generate_scripts.py --config {cfg_rel} \\")
    print( "      --plan-dir data/plan_topup --out-dir data/scenarios_topup")
    print(f"  python scripts/03_render_user_audio.py --config {cfg_rel} \\")
    print( "      --in data/scenarios_topup/scenarios_all.jsonl --audio-dir data/audio_topup")
    print(f"  python scripts/00_encode_audio.py --config {cfg_rel} \\")
    print( "      --audio-dir data/audio_topup --out-dir data/encoded")
    print(f"  python scripts/04_build_frames.py --config {cfg_rel} \\")
    print( "      --in data/scenarios_topup/scenarios_all.jsonl \\")
    print( "      --audio-dir data/audio_topup --frames-out data/frames/frames_topup.jsonl")
    print("\nEvery one of those writes to its OWN directory and appends a NEW shard next to "
          "the existing ones — nothing already generated is touched or regenerated. To fold "
          "the new clips into the HF Parquet corpus, run scripts/23_repack_phase2_to_parquet.py "
          "over data/audio_topup — it appends new shard files, it does not rewrite old ones.")
    print("=" * 72)


def _behaviour_shares_from_deficits(deficits: dict[str, int]) -> dict[str, float]:
    """Spread each flag's deficit over the behaviours that produce it, then normalise."""
    weight: Counter = Counter()
    for flag, need in deficits.items():
        behaviours = FLAG_TO_BEHAVIOURS.get(flag, [])
        if not behaviours:
            continue
        # the first-listed behaviour is the primary producer; give it twice the weight
        w = [2.0] + [1.0] * (len(behaviours) - 1)
        tot = sum(w)
        for b, wi in zip(behaviours, w):
            weight[b] += need * wi / tot
    total = sum(weight.values()) or 1.0
    return {b: round(v / total, 4) for b, v in sorted(weight.items(), key=lambda kv: -kv[1])}


def _write_topup_config(out_path: Path, template: Path, shares: dict[str, float],
                        deficits: dict[str, int], min_clips: int) -> None:
    import yaml

    base = yaml.safe_load(template.read_text(encoding="utf-8")) if template.exists() else {}
    # Sizing: the largest single deficit sets the scenario count, since behaviour_shares
    # then splits that budget in deficit proportion.
    need = max(deficits.values())
    avg_seconds = float(base.get("avg_window_seconds", 5.5))
    base.update({
        # behaviour_shares splits this budget in deficit proportion, so the corpus has to
        # be big enough that the LARGEST deficit is covered by its own share. Floored at
        # 0.1 h so a tiny deficit still produces a runnable plan rather than 0 scenarios.
        "total_hours": max(0.1, round(need * avg_seconds / 3600.0, 2)),
        "avg_window_seconds": avg_seconds,
        "behaviour_shares": shares,
        "db_path": "data/thinkspark_runs_topup.db",
        # MUST be distinct from every corpus already generated — scenario_ids are
        # md5(corpus_tag | job.key | idx) and name the shared data/audio + data/encoded
        # files, so an empty tag here would overwrite existing clips.
        "corpus_tag": "topup",
    })
    header = (
        "# AUTO-GENERATED by scripts/28_data_audit.py — a DEFICIT-ONLY top-up corpus.\n"
        "#\n"
        f"# Flags below the {min_clips}-clip coverage bar, and how many clips each is short:\n"
        + "".join(f"#     {f:<15} +{n}\n" for f, n in sorted(deficits.items())) +
        "#\n"
        "# `behaviour_shares` is proportional to those deficits, so the whole budget goes\n"
        "# to the behaviours that actually produce the missing decisions.\n"
        "#\n"
        "# Nothing already generated is re-generated or overwritten:\n"
        "#   * scripts/02 resumes against the shard in its --out-dir, and this corpus uses a\n"
        "#     FRESH --out-dir (data/scenarios_topup), so it starts from an empty count and\n"
        "#     never re-requests an existing scenario;\n"
        "#   * `corpus_tag` below namespaces the generated scenario_ids, so the new clips\n"
        "#     cannot collide with existing .wav/.npz filenames in the shared data/audio and\n"
        "#     data/encoded directories;\n"
        "#   * the frames land in their own shard (data/frames/frames_topup.jsonl) alongside\n"
        "#     the existing ones, which the training --frames glob already picks up.\n"
        "#\n"
        f"# Inherits LLM/TTS/pricing settings from {template.name}.\n\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + yaml.safe_dump(base, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")


if __name__ == "__main__":
    main()
