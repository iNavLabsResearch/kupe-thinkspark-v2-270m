"""
Colored live terminal stream for TTS rendering (Section 8.4 step 2) — the same treatment
`thinkspark.gen_stream.GenerationStream` gives scenario generation, applied to
`scripts/03_render_user_audio.py`.

Shows, live, in the terminal: rendered/skipped/failed/rate-limited counts, audio hours
done vs. the `total_hours` target, cost spent vs. projected vs. the configured budget gap,
throughput + ETA, and the last few events (including which scenario hit a rate limit).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
@dataclass
class TTSStats:
    target: int = 0                 # scenarios to render THIS run (after skip) — NOT
                                    # the input file's total; see total_in_file below
    already_skipped: int = 0        # already had a wav on disk
    total_in_file: int = 0          # already_skipped + target — the file's real total,
                                    # shown explicitly so "target" is never mistaken
                                    # for "only N scenarios found in my file"
    rendered: int = 0
    failed: int = 0
    rate_limited: int = 0           # subset of `failed` that was rate-limit-classified
    # audio_hours / cost_usd are CORPUS-CUMULATIVE: seeded with everything already on
    # disk from previous runs (base_*), then this run's clips add on top — so re-running
    # never resets the "audio hours / 55h target" meter back to zero.
    audio_hours: float = 0.0
    cost_usd: float = 0.0
    base_audio_hours: float = 0.0   # hours already rendered in prior runs (informational)
    rendered_prior: int = 0         # clips already on disk before this run (informational)
    target_hours: float = 0.0       # cfg.total_hours — the project's overall TTS target
    price_per_hour: float = 0.70
    budget_inr_target: float = 5000.0
    inr_per_usd: float = 83.0
    start_time: float = field(default_factory=time.time)

    @property
    def done(self) -> int:
        return self.rendered + self.failed

    @property
    def elapsed_s(self) -> float:
        return max(0.001, time.time() - self.start_time)

    # Same guard as thinkspark's other live monitors: don't trust a rate computed over
    # a near-zero elapsed window — early on (or with a burst of near-instant mocked/
    # cached results) this can otherwise divide by ~0 seconds and spike to a nonsense
    # rate/ETA. Floor the divisor at 5s of real elapsed time.
    _MIN_ELAPSED_S = 5.0

    @property
    def rate_per_min(self) -> float:
        elapsed = max(self.elapsed_s, self._MIN_ELAPSED_S)
        return self.rendered / (elapsed / 60.0)

    @property
    def eta_min(self) -> float | None:
        remaining = max(0, self.target - self.done)
        if self.rate_per_min <= 0 or remaining == 0:
            return None
        return remaining / self.rate_per_min

    @property
    def projected_cost_usd(self) -> float:
        """
        Progressive: once real clips have rendered, extrapolate from the OBSERVED
        $/hour rate for THIS corpus (real text lengths + audio durations, via
        thinkspark.tts_soniox.soniox_cost_usd) — not the generic flat-rate estimate.
        Only falls back to price_per_hour * target_hours before any data exists.
        """
        # audio_hours includes prior runs' audio, so a real observed rate exists even
        # before this run's first clip renders — don't gate on `rendered` here.
        if self.audio_hours <= 0:
            return self.target_hours * self.price_per_hour
        observed_rate_per_hour = self.cost_usd / self.audio_hours
        remaining_hours = max(0.0, self.target_hours - self.audio_hours)
        return self.cost_usd + remaining_hours * observed_rate_per_hour

    @property
    def gap_usd(self) -> float:
        """projected total - spent so far -> what's still left to spend."""
        return max(0.0, self.projected_cost_usd - self.cost_usd)

    @property
    def budget_pct(self) -> float:
        proj_inr = self.projected_cost_usd * self.inr_per_usd
        return (proj_inr / self.budget_inr_target * 100.0) if self.budget_inr_target else 0.0


@dataclass
class _Event:
    ts: float
    kind: str
    data: dict[str, Any]


def _clip(text: str, n: int = 46) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


# --------------------------------------------------------------------------- #
class TTSStream:
    """Thread-safe event log + stats for the live Rich TTS-rendering dashboard."""

    def __init__(self, *, target: int, already_skipped: int, target_hours: float,
                price_per_hour: float, budget_inr_target: float, inr_per_usd: float,
                concurrency: int, enabled: bool = True, max_events: int = 40,
                base_audio_hours: float = 0.0, base_cost_usd: float = 0.0):
        self.enabled = enabled
        self.concurrency = concurrency
        self.stats = TTSStats(
            target=target, already_skipped=already_skipped,
            total_in_file=target + already_skipped, target_hours=target_hours,
            price_per_hour=price_per_hour, budget_inr_target=budget_inr_target,
            inr_per_usd=inr_per_usd,
            # seed cumulative meters with what's already rendered on disk
            audio_hours=base_audio_hours, cost_usd=base_cost_usd,
            base_audio_hours=base_audio_hours, rendered_prior=already_skipped,
        )
        self._lock = threading.Lock()
        self._events: deque[_Event] = deque(maxlen=max_events)

    def emit(self, kind: str, **data: Any) -> None:
        with self._lock:
            self._events.append(_Event(time.time(), kind, data))
            if kind == "clip_ok":
                self.stats.rendered += 1
                self.stats.audio_hours += float(data.get("duration_s", 0.0)) / 3600.0
                self.stats.cost_usd += float(data.get("cost_usd", 0.0))
            elif kind == "clip_failed":
                self.stats.failed += 1
            elif kind == "rate_limited":
                # rate_limited IS a failure (render_one returns "failed" for it too) —
                # count it in both so `failed` is always the true total and
                # `rate_limited` is the breakdown of how many were rate-limit-related.
                self.stats.failed += 1
                self.stats.rate_limited += 1

    def snapshot(self) -> tuple[TTSStats, list[_Event]]:
        with self._lock:
            return TTSStats(**vars(self.stats)), list(self._events)

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

        s, events = self.snapshot()
        pct = (s.done / s.target * 100.0) if s.target else 0.0
        bar_w = 40
        filled = int(bar_w * min(100.0, pct) / 100.0)
        bar = "[green]" + "█" * filled + "[/][dim]" + "░" * (bar_w - filled) + "[/]"
        eta = f"{s.eta_min:.0f}m" if s.eta_min is not None else "—"
        progress_line = (
            f"{bar}  {pct:5.1f}%  {s.done}/{s.target}  "
            f"•  {s.rate_per_min:.1f}/min  •  ETA {eta}"
        )

        header = Table.grid(padding=(0, 2))
        header.add_column(style="bold cyan")
        header.add_column()
        header.add_row("workers", f"concurrency={self.concurrency}")
        header.add_row("input file", f"{s.total_in_file} scenarios total "
                       f"({s.already_skipped} already rendered + {s.target} this run)")
        header.add_row("progress", progress_line)

        audio = Table.grid(padding=(0, 2))
        audio.add_column(style="dim")
        audio.add_column()
        prior_note = (f"  [dim](incl. {s.base_audio_hours:.3f}h from {s.rendered_prior} "
                      f"already on disk)[/]" if s.base_audio_hours > 0 else "")
        audio.add_row("audio hours", f"{s.audio_hours:.3f}h / {s.target_hours:.1f}h target "
                      f"({100*s.audio_hours/max(1e-6,s.target_hours):.1f}%){prior_note}")
        audio.add_row("rendered/failed", f"{s.rendered} ok  /  {s.failed} failed "
                      f"({s.rate_limited} rate-limited)")

        cost = Table.grid(padding=(0, 2))
        cost.add_column(style="dim")
        cost.add_column()
        cost.add_row("spent", f"${s.cost_usd:.4f}")
        cost.add_row("projected total", f"${s.projected_cost_usd:.4f}")
        cost.add_row("gap (left to spend)", f"${s.gap_usd:.4f}")
        cost.add_row("budget", f"{s.budget_pct:.1f}% of ₹{s.budget_inr_target:.0f} target "
                     f"(projected)")

        feed = Text()
        for ev in events[-16:]:
            ts = time.strftime("%H:%M:%S", time.localtime(ev.ts))
            feed.append(f"{ts} ", style="dim")
            feed.append(_format_event_rich(ev))
            feed.append("\n")
        if not events:
            feed.append("waiting for first clip…", style="dim italic")

        body = Group(
            header, audio, cost,
            Panel(feed, title="[bold]live stream[/]", border_style="blue"),
        )
        return Panel(body, title="[bold magenta]ThinkSpark Soniox TTS rendering[/]", border_style="magenta")

    def _render_plain(self) -> str:
        s, events = self.snapshot()
        lines = [
            "ThinkSpark Soniox TTS rendering",
            f"  input file: {s.total_in_file} total ({s.already_skipped} already rendered "
            f"+ {s.target} this run)",
            f"  {s.done}/{s.target} ({s.rate_per_min:.1f}/min)  "
            f"audio={s.audio_hours:.3f}/{s.target_hours:.1f}h  "
            f"spent=${s.cost_usd:.4f} projected=${s.projected_cost_usd:.4f} gap=${s.gap_usd:.4f}",
            f"  ok={s.rendered} failed={s.failed} rate_limited={s.rate_limited} "
            f"skipped={s.already_skipped}  concurrency={self.concurrency}",
            "  ---",
        ]
        for ev in events[-10:]:
            ts = time.strftime("%H:%M:%S", time.localtime(ev.ts))
            lines.append(f"  {ts} {_format_event_plain(ev)}")
        return "\n".join(lines)

    def print_final(self) -> None:
        s, _ = self.snapshot()
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table

            tbl = Table(show_header=False, box=None, padding=(0, 1))
            tbl.add_column(style="bold")
            tbl.add_column()
            tbl.add_row("input file total", str(s.total_in_file))
            tbl.add_row("rendered", str(s.rendered))
            tbl.add_row("failed", str(s.failed))
            tbl.add_row("rate-limited", str(s.rate_limited))
            tbl.add_row("skipped (already done)", str(s.already_skipped))
            tbl.add_row("audio hours", f"{s.audio_hours:.3f}h / {s.target_hours:.1f}h target")
            tbl.add_row("spent", f"${s.cost_usd:.4f}")
            tbl.add_row("projected total", f"${s.projected_cost_usd:.4f}")
            tbl.add_row("gap (left to spend)", f"${s.gap_usd:.4f}")
            tbl.add_row("budget", f"{s.budget_pct:.1f}% of ₹{s.budget_inr_target:.0f}")
            tbl.add_row("elapsed", f"{s.elapsed_s / 60.0:.1f} min")
            tbl.add_row("throughput", f"{s.rate_per_min:.1f} clips/min")
            Console().print(
                Panel(tbl, title="[bold green]TTS rendering complete[/]", border_style="green")
            )
        except Exception:
            print(
                f"\n=== done: rendered={s.rendered} failed={s.failed} "
                f"rate_limited={s.rate_limited} audio={s.audio_hours:.3f}h/{s.target_hours:.1f}h "
                f"spent=${s.cost_usd:.4f} projected=${s.projected_cost_usd:.4f} "
                f"gap=${s.gap_usd:.4f} ({s.budget_pct:.1f}% of budget) ==="
            )


def _format_event_rich(ev: _Event):
    from rich.text import Text

    d = ev.data
    if ev.kind == "clip_ok":
        t = Text()
        t.append("✓ ", style="bold green")
        t.append(f"{d.get('scenario_id', '?')} ", style="green")
        t.append(f"{d.get('duration_s', 0):.1f}s ", style="cyan")
        t.append(f"${d.get('cost_usd', 0):.5f} ", style="dim")
        t.append(f"{d.get('latency_ms', 0):.0f}ms", style="dim")
        return t
    if ev.kind == "clip_failed":
        t = Text()
        t.append("✗ failed ", style="bold red")
        t.append(f"{d.get('scenario_id', '?')} ", style="red")
        t.append(_clip(str(d.get("error", "")), 60), style="dim red")
        return t
    if ev.kind == "rate_limited":
        t = Text()
        t.append("⏳ rate-limited ", style="bold yellow")
        t.append(f"{d.get('scenario_id', '?')} ", style="yellow")
        t.append("— backed off and retried", style="dim yellow")
        return t
    return Text(f"{ev.kind} {d}", style="dim")


def _format_event_plain(ev: _Event) -> str:
    d = ev.data
    if ev.kind == "clip_ok":
        return (f"✓ {d.get('scenario_id')} {d.get('duration_s', 0):.1f}s "
               f"${d.get('cost_usd', 0):.5f} {d.get('latency_ms', 0):.0f}ms")
    if ev.kind == "clip_failed":
        return f"✗ failed {d.get('scenario_id')} {d.get('error')}"
    if ev.kind == "rate_limited":
        return f"⏳ rate-limited {d.get('scenario_id')} — backed off and retried"
    return f"{ev.kind} {d}"


__all__ = ["TTSStream", "TTSStats"]
