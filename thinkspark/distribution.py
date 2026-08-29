"""
Behaviour / language / gender budget planner (Sections 8.1-8.3).

Turns a target of N total-hours into an exact, integer-balanced list of "generation
jobs" — one job = (behaviour, language, domain, gender, prosody) plus how many
scenarios to make for it. The generator (`scripts/02_generate_scripts.py`) then walks
this plan, so the corpus is balanced *by construction* instead of hoping the LLM does it.

Key nuance the user asked for (back-channel realism): the `backchannel` bucket is split
so that a fraction of its scenarios are the *negative* case — the natural, correct move
is to stay silent (`<LISTEN>` with empty spoken_text). Always emitting "haan"/"right"
would be unnatural, so we deliberately keep a silence share. See BACKCHANNEL_SILENCE_SHARE.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from thinkspark import vocab
from thinkspark.config import DataGenConfig

# --------------------------------------------------------------------------- #
# Canonical distribution (Section 8.1). Shares sum to 1.0.
# --------------------------------------------------------------------------- #
BEHAVIOUR_BUCKETS: dict[str, dict] = {
    # bucket_name: {share, behaviours: [(behaviour, weight_within_bucket)]}
    "clean_turn_taking": {"share": 0.24, "behaviours": [("endpoint_end", 0.6), ("endpoint_hold", 0.4)]},
    "barge":             {"share": 0.16, "behaviours": [("barge_real", 0.5), ("barge_lookalike", 0.5)]},  # 50/50 hard negatives (obj. 9)
    "backchannel":       {"share": 0.12, "behaviours": [("backchannel", 1.0)]},
    "overlap":           {"share": 0.12, "behaviours": [("overlap_comp", 0.5), ("overlap_coop", 0.5)]},
    "incomplete_think":  {"share": 0.10, "behaviours": [("incomplete_thinking", 1.0)]},
    "correction":        {"share": 0.08, "behaviours": [("correction", 1.0)]},
    "silence_break":     {"share": 0.06, "behaviours": [("silence_break", 1.0)]},
    "prefetch":          {"share": 0.06, "behaviours": [("prefetch", 1.0)]},
    "nonspeech_neg":     {"share": 0.06, "behaviours": [("nonspeech_neg", 1.0)]},
}

# Default language shares (Section 8.2).
DEFAULT_LANGUAGE_SHARES: dict[str, float] = {
    "hi": 0.30, "en": 0.26, "gu": 0.24, "hi_en_native": 0.10, "gu_en_native": 0.10,
}

# Domains roughly even, sales-heavy because back-channel tone varies most there.
DEFAULT_DOMAIN_SHARES: dict[str, float] = {
    "sales": 0.40, "bfsi_collections": 0.35, "support": 0.25,
}

# Fraction of the `backchannel` bucket whose *correct* label is silence (<LISTEN>, no
# spoken text). This is the "don't always say haan" realism the model must learn.
BACKCHANNEL_SILENCE_SHARE: float = 0.35

# The `barge` bucket already carries hard negatives (barge_lookalike -> CONTINUE), so
# no extra silence split is needed there.


@dataclass
class GenJob:
    behaviour: str
    language: str
    domain: str
    gender: str
    count: int
    # a hint passed to the prompt so the LLM knows to make the "silent/negative" case
    force_silence: bool = False
    # target spoken length of the user line (thinkspark.vocab.LENGTH_BANDS); "short" is
    # the original behaviour and is kept OUT of `key` for backward compatibility.
    length_band: str = "short"

    @property
    def key(self) -> str:
        base = f"{self.behaviour}|{self.language}|{self.domain}|{self.gender}|{int(self.force_silence)}"
        # Only extend the key for non-default bands, so existing corpora (whose _job_key
        # was written before length bands existed) resume with byte-identical keys.
        return base if self.length_band == "short" else f"{base}|{self.length_band}"


def _normalise(shares: dict[str, float]) -> dict[str, float]:
    total = sum(shares.values()) or 1.0
    return {k: v / total for k, v in shares.items()}


def total_scenarios(cfg: DataGenConfig) -> int:
    """How many scenarios ~= total_hours of audio at the mean window length."""
    seconds = cfg.total_hours * 3600.0
    return max(1, int(round(seconds / cfg.avg_window_seconds)))


def build_plan(cfg: DataGenConfig) -> list[GenJob]:
    """
    Produce a fully-balanced, integer list of GenJobs summing to ~total_scenarios(cfg).

    Balancing order (largest split first) so rounding error is absorbed at the leaves:
        behaviour bucket  ->  behaviour  ->  language  ->  domain  ->  gender
    Gender is forced to an exact 50/50 within every leaf (Section 7.3).
    """
    n = total_scenarios(cfg)

    lang_shares = _normalise(cfg.language_shares or DEFAULT_LANGUAGE_SHARES)
    dom_shares = _normalise(cfg.domain_shares or DEFAULT_DOMAIN_SHARES)
    beh_shares = cfg.behaviour_shares or None  # optional per-behaviour override

    jobs: list[GenJob] = []

    for bucket in BEHAVIOUR_BUCKETS.values():
        bucket_n = int(round(bucket["share"] * n))
        for behaviour, w in bucket["behaviours"]:
            beh_n = int(round(bucket_n * w))
            if beh_shares and behaviour in beh_shares:  # explicit override wins
                beh_n = int(round(beh_shares[behaviour] * n))
            for language, lw in lang_shares.items():
                lang_n = int(round(beh_n * lw))
                for domain, dw in dom_shares.items():
                    dom_n = int(round(lang_n * dw))
                    if dom_n <= 0:
                        continue
                    # gender-balanced 50/50
                    fem = dom_n // 2
                    male = dom_n - fem
                    for gender, g_n in (("female", fem), ("male", male)):
                        if g_n <= 0:
                            continue
                        _emit_gender_job(jobs, behaviour, language, domain, gender, g_n)

    # stamp the plan-wide utterance length band onto every job (drives prompt length;
    # "short" is the default and leaves job.key backward-compatible — see GenJob.key)
    band = getattr(cfg, "utterance_length", "short")
    for j in jobs:
        j.length_band = band
    return jobs


def _emit_gender_job(jobs, behaviour, language, domain, gender, count):
    """Split the backchannel bucket into speak/silence sub-jobs; else emit one job."""
    if behaviour == "backchannel" and count >= 2:
        silent = int(round(count * BACKCHANNEL_SILENCE_SHARE))
        speak = count - silent
        if speak > 0:
            jobs.append(GenJob(behaviour, language, domain, gender, speak, force_silence=False))
        if silent > 0:
            jobs.append(GenJob(behaviour, language, domain, gender, silent, force_silence=True))
    else:
        jobs.append(GenJob(behaviour, language, domain, gender, count, force_silence=False))


def partition_plan(jobs: list[GenJob], num_parts: int) -> list[list[GenJob]]:
    """
    Deterministically split the plan into `num_parts` roughly-equal-work shards so the
    generation can be run over several Kaggle sessions (Section 9.3). We bucket by a
    stable hash of the job key so re-running with the same num_parts is reproducible.
    """
    parts: list[list[GenJob]] = [[] for _ in range(num_parts)]
    for job in jobs:
        h = int(hashlib.md5(job.key.encode("utf-8")).hexdigest(), 16)
        parts[h % num_parts].append(job)
    return parts


def plan_summary(jobs: list[GenJob]) -> dict:
    """Aggregate counts for logging / the distribution report."""
    by_beh: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    by_gender: dict[str, int] = {}
    silent_backchannel = 0
    total = 0
    for j in jobs:
        by_beh[j.behaviour] = by_beh.get(j.behaviour, 0) + j.count
        by_lang[j.language] = by_lang.get(j.language, 0) + j.count
        by_gender[j.gender] = by_gender.get(j.gender, 0) + j.count
        total += j.count
        if j.behaviour == "backchannel" and j.force_silence:
            silent_backchannel += j.count
    return {
        "total_scenarios": total,
        "by_behaviour": by_beh,
        "by_language": by_lang,
        "by_gender": by_gender,
        "silent_backchannel": silent_backchannel,
    }


# --------------------------------------------------------------------------- #
def write_plan(cfg: DataGenConfig, plan_dir) -> tuple[list[GenJob], list[list[GenJob]], dict]:
    """
    Build the balanced plan and persist it to `plan_dir` — this is the ONE place the
    plan is ever built or written. Both `scripts/01_plan_distribution.py` (explicit,
    inspectable) and `scripts/02_generate_scripts.py` (auto-build on first run, for a
    zero-setup `python scripts/02_generate_scripts.py`) call this exact function, so
    the corpus's balance/distribution can never drift between the two entry points.

    Returns (jobs, parts, summary) and writes:
        plan_dir/plan.jsonl              full job list
        plan_dir/plan_part{NN}.jsonl      cfg.num_parts resumable shards
        plan_dir/plan_summary.json        aggregate counts (also used by the monitor
                                           and the HTML report as the "target" side of
                                           every actual-vs-target comparison)
    """
    import json
    from pathlib import Path

    jobs = build_plan(cfg)
    parts = partition_plan(jobs, cfg.num_parts)
    summary = plan_summary(jobs)

    out_dir = Path(plan_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "plan.jsonl").open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j.__dict__, ensure_ascii=False) + "\n")

    for i, part in enumerate(parts):
        with (out_dir / f"plan_part{i:02d}.jsonl").open("w", encoding="utf-8") as f:
            for j in part:
                f.write(json.dumps(j.__dict__, ensure_ascii=False) + "\n")

    (out_dir / "plan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return jobs, parts, summary


def load_or_write_plan(cfg: DataGenConfig, plan_dir) -> dict:
    """
    Ensure `plan_dir/plan.jsonl` (+ parts + summary) exists, building it via
    `write_plan()` if this is the first run — the thing that lets
    `scripts/02_generate_scripts.py` be run with zero setup, like the sibling
    kupe-thinkspark/kupe-tts projects. Returns the plan summary dict either way.
    """
    import json
    from pathlib import Path

    out_dir = Path(plan_dir)
    plan_path = out_dir / "plan.jsonl"
    summary_path = out_dir / "plan_summary.json"
    if plan_path.exists() and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    _, _, summary = write_plan(cfg, plan_dir)
    return summary
