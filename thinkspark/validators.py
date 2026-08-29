"""
Section 8.5 data-quality gate — "before you train on junk".

Five checks with explicit pass bars:

    Schema valid       JSON parse + field/enum check           >= 99%
    Vocab compliance   flags in fixed set; no brackets in text  100%
    Language/script    Unicode-block check per language slice   >= 98%
    Balance            count per behaviour/lang/gender           within +-2% of target
    Naturalness        LLM-judge 1-5 on spoken_text             mean >= 4.2

`validate_scenario()` runs the cheap per-item checks. `corpus_report()` aggregates a
whole file and evaluates the balance + naturalness bars. The LLM-judge is optional and
driven by the caller (script 05) so validation stays offline by default.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from thinkspark import vocab
from thinkspark.schema import Scenario


@dataclass
class ItemResult:
    ok: bool
    schema_ok: bool
    vocab_ok: bool
    script_ok: bool
    errors: list[str] = field(default_factory=list)


def _in_block(ch: str, lo: int, hi: int) -> bool:
    return lo <= ord(ch) <= hi


def script_compliance(text: str, language: str) -> bool:
    """Fraction of alphabetic chars in the expected block must clear a threshold."""
    if not text.strip():
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True  # numeric/punctuation-only lines (rare) pass trivially
    base = language.split("_")[0]
    is_native_mix = language.endswith("_native")

    if base == "en":
        # English (or English-with-code-mix): mostly Latin
        latin = sum(1 for c in letters if ord(c) < 0x0250)
        return latin / len(letters) >= 0.7

    lo, hi = (vocab.DEVANAGARI_RANGE if base == "hi" else vocab.GUJARATI_RANGE)
    indic = sum(1 for c in letters if _in_block(c, lo, hi))
    # native-mix slices legitimately contain Latin English words -> lower bar
    bar = 0.45 if is_native_mix else 0.85
    return indic / len(letters) >= bar


def validate_scenario(s: Scenario) -> ItemResult:
    errors = s.validate()
    schema_ok = not errors

    # vocab compliance: flags in set + no bracket tags in spoken text
    vocab_ok = True
    for t in s.target:
        if not vocab.is_valid_flag(t.flag):
            vocab_ok = False
        if any(ch in t.spoken_text for ch in "<>[]"):
            vocab_ok = False
            errors.append("bracket tag in spoken_text")

    # barge_lookalike must never resolve to a real barge
    if s.behaviour == "barge_lookalike":
        if any(t.flag in ("BARGE_HARD", "BARGE_SOFT") for t in s.target):
            vocab_ok = False
            errors.append("barge_lookalike resolved to BARGE_* (should be CONTINUE)")

    script_ok = script_compliance(s.user_text, s.language)
    if not script_ok:
        errors.append(f"script mismatch for language {s.language}")

    ok = schema_ok and vocab_ok and script_ok
    return ItemResult(ok=ok, schema_ok=schema_ok, vocab_ok=vocab_ok,
                      script_ok=script_ok, errors=errors)


# --------------------------------------------------------------------------- #
@dataclass
class CorpusReport:
    n: int
    schema_pass: float
    vocab_pass: float
    script_pass: float
    by_behaviour: dict[str, int]
    by_language: dict[str, int]
    by_gender: dict[str, int]
    balance_ok: bool
    naturalness_mean: float | None = None
    words_mean: float = 0.0
    words_median: int = 0
    words_p10: int = 0
    words_p90: int = 0

    def summary(self) -> str:
        lines = [
            f"scenarios          : {self.n}",
            f"schema pass        : {self.schema_pass:.3f}   (bar >= 0.99)",
            f"vocab pass         : {self.vocab_pass:.3f}   (bar  = 1.00)",
            f"script pass        : {self.script_pass:.3f}   (bar >= 0.98)",
            f"balance within 2%  : {self.balance_ok}",
            # user_text length (proxy for spoken duration at ~2.2 words/sec) — lets you
            # confirm an 'extended' corpus is actually longer, not still one-liners
            f"user_text words    : mean {self.words_mean:.1f}, median {self.words_median}, "
            f"p10 {self.words_p10}, p90 {self.words_p90}  "
            f"(~{self.words_median / 2.2:.1f}s median @ 2.2 wps)",
            f"by_behaviour       : {dict(sorted(self.by_behaviour.items()))}",
            f"by_language        : {dict(sorted(self.by_language.items()))}",
            f"by_gender          : {dict(sorted(self.by_gender.items()))}",
        ]
        if self.naturalness_mean is not None:
            lines.append(f"naturalness mean   : {self.naturalness_mean:.2f} (bar >= 4.2)")
        return "\n".join(lines)


def corpus_report(scenarios: list[Scenario],
                  target_shares: dict[str, float] | None = None,
                  naturalness_scores: list[int] | None = None) -> CorpusReport:
    n = len(scenarios)
    if n == 0:
        return CorpusReport(0, 0, 0, 0, {}, {}, {}, False)

    schema = vocab_c = script = 0
    by_beh: Counter = Counter()
    by_lang: Counter = Counter()
    by_gender: Counter = Counter()
    for s in scenarios:
        r = validate_scenario(s)
        schema += int(r.schema_ok)
        vocab_c += int(r.vocab_ok)
        script += int(r.script_ok)
        by_beh[s.behaviour] += 1
        by_lang[s.language] += 1
        by_gender[s.gender] += 1

    # balance check: language slice within +-2% of target (Section 8.2)
    balance_ok = True
    if target_shares:
        for lang, share in target_shares.items():
            got = by_lang.get(lang, 0) / n
            if abs(got - share) > 0.02:
                balance_ok = False
    # gender must be ~50/50
    if abs(by_gender.get("female", 0) - by_gender.get("male", 0)) / n > 0.04:
        balance_ok = False

    nat = None
    if naturalness_scores:
        nat = sum(naturalness_scores) / len(naturalness_scores)

    # user_text length distribution (words) — a cheap proxy for spoken duration
    counts = sorted(len((s.user_text or "").split()) for s in scenarios)

    def _pct(p: float) -> int:
        return counts[min(len(counts) - 1, int(p * len(counts)))]

    return CorpusReport(
        n=n,
        schema_pass=schema / n,
        vocab_pass=vocab_c / n,
        script_pass=script / n,
        by_behaviour=dict(by_beh),
        by_language=dict(by_lang),
        by_gender=dict(by_gender),
        balance_ok=balance_ok,
        naturalness_mean=nat,
        words_mean=sum(counts) / n,
        words_median=_pct(0.5),
        words_p10=_pct(0.10),
        words_p90=_pct(0.90),
    )
