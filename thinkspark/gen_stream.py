"""
Colored live terminal stream for scenario generation (SSE-style event feed).

Used by scripts/02_generate_scripts.py to show batches and validated scenarios as they
land, plus an instant post-run data-quality evaluation panel.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from thinkspark.schema import Scenario
from thinkspark.validators import corpus_report, validate_scenario


# --------------------------------------------------------------------------- #
@dataclass
class StreamStats:
    target: int = 0
    produced: int = 0
    already_done: int = 0
    invalid: int = 0
    api_errors: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def total_done(self) -> int:
        return self.already_done + self.produced

    @property
    def elapsed_s(self) -> float:
        return max(0.001, time.time() - self.start_time)

    @property
    def rate_per_min(self) -> float:
        return self.produced / (self.elapsed_s / 60.0)

    @property
    def eta_min(self) -> float | None:
        remaining = max(0, self.target - self.total_done)
        if self.rate_per_min <= 0 or remaining == 0:
            return None
        return remaining / self.rate_per_min


@dataclass
class _Event:
    ts: float
    kind: str
    data: dict[str, Any]


def _clip(text: str, n: int = 52) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _flag_chain(scenario: Scenario) -> str:
    flags = [t.flag for t in scenario.target[:4]]
    if len(scenario.target) > 4:
        flags.append("…")
    return "→".join(flags) if flags else "—"


class GenerationStream:
    """Thread-safe SSE-style event log + stats for the live Rich dashboard."""

    def __init__(
        self,
        *,
        target: int,
        already_done: int,
        model: str,
        concurrency: int,
        batch_size: int,
        shard_name: str,
        enabled: bool = True,
        max_events: int = 48,
    ):
        self.enabled = enabled
        self.model = model
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.shard_name = shard_name
        self.stats = StreamStats(target=target, already_done=already_done)
        self._lock = threading.Lock()
        self._events: deque[_Event] = deque(maxlen=max_events)
        self._fail_counts: dict[str, int] = {}  # job_key -> cumulative unit-eval fails

    # ------------------------------------------------------------------ #
    def next_fail_flag(self, job_key: str) -> str:
        """
        Section 8.5 unit-level check failed for one scenario in this job — return the
        next job-scoped fail tag ('fail1' the first time, 'fail2' the second, ...).
        Thread-safe; call exactly once per failed scenario.
        """
        with self._lock:
            n = self._fail_counts.get(job_key, 0) + 1
            self._fail_counts[job_key] = n
            return f"fail{n}"

    def max_fail_number(self) -> int:
        """Highest fail-count reached by any single job — a proxy for 'stuck' jobs."""
        with self._lock:
            return max(self._fail_counts.values(), default=0)

    # ------------------------------------------------------------------ #
    def emit(self, kind: str, **data: Any) -> None:
        with self._lock:
            self._events.append(_Event(time.time(), kind, data))
            if kind == "scenario_ok":
                self.stats.produced += 1
            elif kind == "scenario_invalid":
                self.stats.invalid += 1
            elif kind == "batch_done":
                self.stats.llm_calls += 1
                self.stats.cost_usd += float(data.get("cost_usd", 0.0))
                self.stats.prompt_tokens += int(data.get("prompt_tokens", 0))
                self.stats.completion_tokens += int(data.get("completion_tokens", 0))
            elif kind == "api_error":
                self.stats.api_errors += 1
                self.stats.llm_calls += 1

    def snapshot(self) -> tuple[StreamStats, list[_Event]]:
        with self._lock:
            return StreamStats(**vars(self.stats)), list(self._events)

    # ------------------------------------------------------------------ #
    def render(self):
        try:
            return self._render_rich()
        except Exception:
            return self._render_plain()

    def _render_rich(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        stats, events = self.snapshot()
        pct = (stats.total_done / stats.target * 100.0) if stats.target else 0.0
        bar_w = 40
        filled = int(bar_w * min(100.0, pct) / 100.0)
        bar = "[green]" + "█" * filled + "[/][dim]" + "░" * (bar_w - filled) + "[/]"
        eta = f"{stats.eta_min:.0f}m" if stats.eta_min is not None else "—"
        progress_line = (
            f"{bar}  {pct:5.1f}%  {stats.total_done}/{stats.target}  "
            f"•  {stats.rate_per_min:.1f}/min  •  ETA {eta}"
        )

        header = Table.grid(padding=(0, 2))
        header.add_column(style="bold cyan")
        header.add_column()
        header.add_row("model", self.model)
        header.add_row("shard", self.shard_name)
        header.add_row(
            "workers",
            f"concurrency={self.concurrency}  batch_size={self.batch_size}",
        )
        header.add_row("progress", progress_line)

        meta = Table.grid(padding=(0, 2))
        meta.add_column(style="dim")
        meta.add_column()
        meta.add_row("cost", f"${stats.cost_usd:.4f}")
        meta.add_row("llm calls", str(stats.llm_calls))
        meta.add_row("invalid", str(stats.invalid))
        meta.add_row("api errors", str(stats.api_errors))
        meta.add_row("tokens", f"{stats.prompt_tokens}+{stats.completion_tokens}")

        feed = Text()
        for ev in events[-20:]:
            ts = time.strftime("%H:%M:%S", time.localtime(ev.ts))
            feed.append(f"{ts} ", style="dim")
            feed.append(_format_event_rich(ev))
            feed.append("\n")

        if not events:
            feed.append("waiting for first batch…", style="dim italic")

        body = Group(
            header,
            meta,
            Panel(feed, title="[bold]live stream[/]", border_style="blue"),
        )
        return Panel(body, title="[bold magenta]ThinkSpark scenario generation[/]", border_style="magenta")

    def _render_plain(self) -> str:
        stats, events = self.snapshot()
        lines = [
            f"ThinkSpark generation — {self.shard_name}",
            f"  {stats.total_done}/{stats.target} scenarios  "
            f"({stats.rate_per_min:.1f}/min  ${stats.cost_usd:.4f})",
            f"  concurrency={self.concurrency}  batch={self.batch_size}  model={self.model}",
            "  ---",
        ]
        for ev in events[-12:]:
            ts = time.strftime("%H:%M:%S", time.localtime(ev.ts))
            lines.append(f"  {ts} {_format_event_plain(ev)}")
        return "\n".join(lines)

    def print_final(self, unit_summary: dict | None = None) -> None:
        stats, _ = self.snapshot()
        max_retry = self.max_fail_number()
        ue_line = ""
        if unit_summary:
            ue_total = unit_summary.get("total", 0)
            ue_pass = unit_summary.get("passed", 0)
            ue_rate = (ue_pass / ue_total * 100.0) if ue_total else 0.0
            ue_line = f"{ue_pass}/{ue_total} passed ({ue_rate:.1f}%)"
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table

            tbl = Table(show_header=False, box=None, padding=(0, 1))
            tbl.add_column(style="bold")
            tbl.add_column()
            tbl.add_row("new scenarios", str(stats.produced))
            tbl.add_row("total in shard", str(stats.total_done))
            tbl.add_row("llm calls", str(stats.llm_calls))
            if unit_summary:
                tbl.add_row("unit-eval (§8.5)", ue_line)
            tbl.add_row("invalid items", str(stats.invalid))
            tbl.add_row("max retries (1 slot)", f"fail{max_retry}" if max_retry else "—")
            tbl.add_row("api errors", str(stats.api_errors))
            tbl.add_row("cost", f"${stats.cost_usd:.4f}")
            tbl.add_row("elapsed", f"{stats.elapsed_s / 60.0:.1f} min")
            tbl.add_row("throughput", f"{stats.rate_per_min:.1f} scenarios/min")
            Console().print(
                Panel(tbl, title="[bold green]Generation complete[/]", border_style="green")
            )
        except Exception:
            print(
                f"\n=== done: +{stats.produced} scenarios "
                f"({stats.total_done} total)  ${stats.cost_usd:.4f}  "
                f"{stats.rate_per_min:.1f}/min "
                + (f" unit-eval: {ue_line}" if unit_summary else "")
                + (f"  max retries: fail{max_retry}" if max_retry else "")
                + " ==="
            )


def _format_event_rich(ev: _Event):
    from rich.text import Text

    d = ev.data
    if ev.kind == "batch_start":
        t = Text()
        t.append("▶ batch ", style="bold yellow")
        t.append(f"{d.get('job_key', '?')} ", style="cyan")
        t.append(f"n={d.get('n', '?')}", style="dim")
        return t
    if ev.kind == "scenario_ok":
        t = Text()
        t.append("✓ ", style="bold green")
        t.append(f"{d.get('scenario_id', '?')} ", style="green")
        t.append(f"{d.get('behaviour', '?')}/{d.get('language', '?')} ", style="cyan")
        t.append(f'"{_clip(d.get("user_text", ""))}" ', style="white")
        t.append(d.get("flags", ""), style="dim")
        return t
    if ev.kind == "scenario_invalid":
        t = Text()
        flag = d.get("fail_flag", "")
        t.append(f"✗ unit-eval {flag.upper()} ", style="bold red")
        t.append(f"{d.get('job_key', '?')} ", style="red")
        t.append(_clip("; ".join(d.get("errors", [])), 55), style="dim red")
        return t
    if ev.kind == "regenerate":
        t = Text()
        t.append("↻ regenerating ", style="bold yellow")
        t.append(f"{d.get('needed', '?')} slot(s) ", style="yellow")
        t.append(f"for {d.get('job_key', '?')}", style="dim yellow")
        return t
    if ev.kind == "batch_done":
        t = Text()
        t.append("◀ batch ", style="bold blue")
        t.append(
            f"{d.get('valid', 0)}/{d.get('requested', 0)} valid  "
            f"{d.get('latency_ms', 0):.0f}ms  ${d.get('cost_usd', 0):.4f}",
            style="blue",
        )
        return t
    if ev.kind == "api_error":
        t = Text()
        t.append("! api error ", style="bold red")
        t.append(f"{d.get('job_key', '?')} ", style="red")
        t.append(_clip(str(d.get("error", "")), 70), style="dim red")
        return t
    if ev.kind == "job_give_up":
        t = Text()
        t.append("! giving up ", style="bold red")
        t.append(str(d.get("job_key", "?")), style="red")
        return t
    return Text(f"{ev.kind} {d}", style="dim")


def _format_event_plain(ev: _Event) -> str:
    d = ev.data
    if ev.kind == "batch_start":
        return f"▶ batch {d.get('job_key')} n={d.get('n')}"
    if ev.kind == "scenario_ok":
        return (
            f"✓ {d.get('scenario_id')} {d.get('behaviour')}/{d.get('language')} "
            f"\"{_clip(d.get('user_text', ''))}\" {d.get('flags', '')}"
        )
    if ev.kind == "scenario_invalid":
        flag = d.get("fail_flag", "")
        return f"✗ unit-eval {flag.upper()} {d.get('job_key')} {'; '.join(d.get('errors', []))}"
    if ev.kind == "regenerate":
        return f"↻ regenerating {d.get('needed')} slot(s) for {d.get('job_key')}"
    if ev.kind == "batch_done":
        return (
            f"◀ batch {d.get('valid')}/{d.get('requested')} valid "
            f"{d.get('latency_ms', 0):.0f}ms"
        )
    if ev.kind == "api_error":
        return f"! api error {d.get('job_key')} {d.get('error')}"
    if ev.kind == "job_give_up":
        return f"! giving up {d.get('job_key')}"
    return f"{ev.kind} {d}"


# --------------------------------------------------------------------------- #
def run_data_quality_eval(
    scenarios: list[Scenario],
    *,
    target_shares: dict[str, float] | None,
    judge_client=None,
    judge_n: int = 0,
    report_out: str | None = None,
    skip_balance: bool = False,
) -> tuple[Any, int]:
    """
    Run Section 8.5 checks and print a colored summary. Returns (CorpusReport, exit_code).

    When `skip_balance` is True (e.g. `--limit` smoke tests), the balance bar is marked
    SKIPPED — only schema / vocab / script gates affect the exit code.
    """
    nat_scores: list[int] | None = None
    if judge_client and scenarios and judge_n > 0:
        from thinkspark.prompts import build_judge_prompt

        sample = random.sample(scenarios, min(judge_n, len(scenarios)))
        nat_scores = []
        try:
            from rich.console import Console
            from rich.progress import Progress

            console = Console()
            with Progress(transient=True) as prog:
                task = prog.add_task("[cyan]LLM judge naturalness…", total=len(sample))
                for s in sample:
                    sys_p, usr_p = build_judge_prompt(s.to_json())
                    try:
                        res = judge_client.chat_json(sys_p, usr_p)
                        nat_scores.append(int(res.get("score", 0)))
                    except Exception as e:
                        console.print(f"[red]judge error:[/] {e}")
                    prog.advance(task)
        except Exception:
            for s in sample:
                sys_p, usr_p = build_judge_prompt(s.to_json())
                try:
                    res = judge_client.chat_json(sys_p, usr_p)
                    nat_scores.append(int(res.get("score", 0)))
                except Exception as e:
                    print(f"  ! judge error: {e}")

    report = corpus_report(scenarios, target_shares=target_shares, naturalness_scores=nat_scores)
    text = report.summary()
    exit_code = 1 if (report.schema_pass < 0.99 or report.vocab_pass < 1.0 or report.script_pass < 0.98) else 0
    if not skip_balance and not report.balance_ok:
        exit_code = 1

    try:
        _print_eval_rich(report, exit_code, skip_balance=skip_balance)
    except Exception:
        print("=" * 60)
        print(text)
        if skip_balance:
            print("(balance check skipped — use a full run without --limit)")
        print("=" * 60)
        print("PASS" if exit_code == 0 else "FAIL")

    if report_out:
        from pathlib import Path

        out = Path(report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        try:
            from rich.console import Console

            Console().print(f"[dim]wrote report →[/] {out}")
        except Exception:
            print(f"wrote report -> {out}")

    return report, exit_code


def _print_eval_rich(report, exit_code: int, *, skip_balance: bool = False) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    def _bar(value: float, threshold: float, higher_is_better: bool = True) -> str:
        ok = value >= threshold if higher_is_better else value <= threshold
        return "[green]PASS[/]" if ok else "[red]FAIL[/]"

    tbl = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    tbl.add_column("check")
    tbl.add_column("value", justify="right")
    tbl.add_column("bar")
    tbl.add_column("status", justify="center")

    tbl.add_row("scenarios", str(report.n), "", "")
    tbl.add_row(
        "schema pass", f"{report.schema_pass:.3f}", ">= 0.99", _bar(report.schema_pass, 0.99),
    )
    tbl.add_row(
        "vocab pass", f"{report.vocab_pass:.3f}", "= 1.00", _bar(report.vocab_pass, 1.0),
    )
    tbl.add_row(
        "script pass", f"{report.script_pass:.3f}", ">= 0.98", _bar(report.script_pass, 0.98),
    )
    if skip_balance:
        tbl.add_row("balance ±2%", "—", "smoke test", "[dim yellow]SKIP[/]")
    else:
        tbl.add_row(
            "balance ±2%",
            "yes" if report.balance_ok else "no",
            "",
            "[green]PASS[/]" if report.balance_ok else "[red]FAIL[/]",
        )
    if report.naturalness_mean is not None:
        tbl.add_row(
            "naturalness mean",
            f"{report.naturalness_mean:.2f}",
            ">= 4.2",
            _bar(report.naturalness_mean, 4.2),
        )

    detail = Table.grid(padding=(0, 2))
    detail.add_column(style="dim")
    detail.add_column()
    detail.add_row("by_behaviour", str(dict(sorted(report.by_behaviour.items()))))
    detail.add_row("by_language", str(dict(sorted(report.by_language.items()))))
    detail.add_row("by_gender", str(dict(sorted(report.by_gender.items()))))

    if skip_balance:
        title = (
            "[bold green]Data quality — PASS (smoke test)[/]"
            if exit_code == 0
            else "[bold red]Data quality — FAIL (smoke test)[/]"
        )
        border = "green" if exit_code == 0 else "red"
    else:
        title = (
            "[bold green]Data quality evaluation — PASS[/]"
            if exit_code == 0
            else "[bold red]Data quality evaluation — FAIL[/]"
        )
        border = "green" if exit_code == 0 else "red"

    console = Console()
    console.print(Panel(tbl, title=title, border_style=border))
    console.print(detail)
    if skip_balance:
        console.print(
            "[dim]Balance skipped: --limit only fills the first plan jobs "
            "(one behaviour/language), not the full corpus mix.[/]"
        )


__all__ = ["GenerationStream", "run_data_quality_eval"]
