"""
SQLite tracking for every data-generation run (Section 8, user requirement).

Design choice — **the JSONL shard stays the source of truth for content and resume**
(script 02 already resumes by scanning `_job_key` counts in the shard; that keeps working
unmodified even if this DB file is deleted or corrupted). This DB is an **audit + cost
log layered on top**: one row per LLM call and per TTS call, with token counts / duration
and USD cost, plus a `runs` row per script invocation. That way a crash mid-run loses at
most the in-flight batch (the shard file is fsync'd after every successful write), and you
always have a full cost trail even across many resumed sessions.

Tables
------
runs         one row per script invocation (run_id, stage, config snapshot)
llm_calls    one row per LLM request (batch or single), tokens + cost + status
tts_calls    one row per Soniox TTS request, chars/duration + cost + status
unit_evals   one row per GENERATED SCENARIO — the Section 8.5 unit-level check
             (schema/vocab/script via validate_scenario()) run immediately after
             parsing, pass or fail. A fail gets an incrementing `fail_flag`
             ('fail1', 'fail2', ...) scoped to its job, so repeated failures on the
             same job are traceable; the caller then regenerates that slot. This is
             the per-item gate BEFORE the corpus-wide Section 8.5 report
             (thinkspark.gen_stream.run_data_quality_eval) runs at the end.
scenario_registry   one row per scenario that actually made it into the shard (passed
             unit-eval and was written). This is the SQLite-side record of "what was
             generated" — scenario_id, its job, and its `global_index` (position in the
             full plan's linear scenario order, stable across every run). The JSONL
             shard is still the content source of truth; this table is what lets you
             query progress by --range (e.g. "how many of [3000,6000) are done?")
             straight from the DB instead of re-scanning the shard file.

Thread-safety: a single sqlite3 connection in WAL mode, guarded by one lock. Fine for
this workload (occasional short writes from up to ~30 concurrent worker threads).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    stage           TEXT NOT NULL,          -- 'generate' | 'render_tts' | 'other'
    started_at      REAL NOT NULL,
    config_snapshot TEXT,
    args_snapshot   TEXT,
    host            TEXT
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,
    job_key           TEXT,
    model             TEXT,
    requested_n       INTEGER,              -- scenarios asked for in this batch
    returned_n        INTEGER,              -- items parsed out of the response
    valid_n           INTEGER,              -- items that passed schema validation
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    cost_usd          REAL,
    latency_ms        REAL,
    status            TEXT,                 -- ok | parse_error | api_error
    error             TEXT,
    created_at        REAL
);

CREATE TABLE IF NOT EXISTS tts_calls (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    scenario_id  TEXT,
    chars        INTEGER,
    duration_s   REAL,                      -- length of the SYNTHESISED audio
    cost_usd     REAL,
    latency_ms   REAL,                      -- WALL-CLOCK time the TTS call took
    status       TEXT,                      -- ok | error
    error        TEXT,
    created_at   REAL
);

CREATE TABLE IF NOT EXISTS unit_evals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    job_key      TEXT,
    scenario_id  TEXT,
    status       TEXT,                      -- pass | fail
    fail_flag    TEXT,                      -- '' | 'fail1' | 'fail2' | ... (job-scoped)
    errors       TEXT,                      -- JSON list of validator error strings
    created_at   REAL
);

CREATE TABLE IF NOT EXISTS scenario_registry (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    job_key       TEXT,
    scenario_id   TEXT NOT NULL,
    global_index  INTEGER,                  -- position in the full plan's linear order
    created_at    REAL
);

CREATE TABLE IF NOT EXISTS hf_sync (
    scenario_id  TEXT NOT NULL,
    repo_id      TEXT NOT NULL,
    synced_at    REAL,
    PRIMARY KEY (scenario_id, repo_id)
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_run   ON llm_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_job   ON llm_calls(job_key);
CREATE INDEX IF NOT EXISTS idx_tts_calls_run   ON tts_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_unit_evals_run  ON unit_evals(run_id);
CREATE INDEX IF NOT EXISTS idx_unit_evals_job  ON unit_evals(job_key);
CREATE UNIQUE INDEX IF NOT EXISTS idx_scenario_registry_sid ON scenario_registry(scenario_id);
CREATE INDEX IF NOT EXISTS idx_scenario_registry_run   ON scenario_registry(run_id);
CREATE INDEX IF NOT EXISTS idx_scenario_registry_job   ON scenario_registry(job_key);
CREATE INDEX IF NOT EXISTS idx_scenario_registry_gidx  ON scenario_registry(global_index);
"""


class RunDB:
    """Open (or create) the tracking DB and expose small logging helpers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._ensure_column("tts_calls", "latency_ms", "REAL")  # migration for older DB files
        self._lock = threading.Lock()

    def _ensure_column(self, table: str, column: str, coltype: str) -> None:
        """Best-effort ALTER TABLE ADD COLUMN, tolerant of it already existing."""
        cols = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            self._conn.commit()

    # ------------------------------------------------------------------ #
    def start_run(self, stage: str, config_snapshot: dict, args_snapshot: dict) -> str:
        import socket
        run_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, stage, started_at, config_snapshot, "
                "args_snapshot, host) VALUES (?,?,?,?,?,?)",
                (run_id, stage, time.time(), json.dumps(config_snapshot, default=str),
                 json.dumps(args_snapshot, default=str), socket.gethostname()),
            )
            self._conn.commit()
        return run_id

    def log_llm_call(self, run_id: str, job_key: str, model: str, requested_n: int,
                     returned_n: int, valid_n: int, prompt_tokens: int,
                     completion_tokens: int, cost_usd: float, latency_ms: float,
                     status: str, error: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO llm_calls (run_id, job_key, model, requested_n, returned_n, "
                "valid_n, prompt_tokens, completion_tokens, cost_usd, latency_ms, status, "
                "error, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, job_key, model, requested_n, returned_n, valid_n, prompt_tokens,
                 completion_tokens, cost_usd, latency_ms, status, error, time.time()),
            )
            self._conn.commit()

    def log_tts_call(self, run_id: str, scenario_id: str, chars: int, duration_s: float,
                     cost_usd: float, status: str, latency_ms: float = 0.0,
                     error: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO tts_calls (run_id, scenario_id, chars, duration_s, cost_usd, "
                "latency_ms, status, error, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, scenario_id, chars, duration_s, cost_usd, latency_ms, status,
                 error, time.time()),
            )
            self._conn.commit()

    def mark_tts_call_corrected(self, scenario_id: str, new_status: str, error: str = "") -> int:
        """
        Retroactively correct EVERY historical tts_calls row for `scenario_id` that was
        logged as 'ok' but is now known to have been corrupt (e.g. a run made against
        the old wrong Soniox endpoint — see scripts/14_cleanup_corrupt_audio.py). Keeps
        cost/audio-hours aggregates (thinkspark.gather_stats, the HTML report) honest
        instead of counting a false success forever. Returns rows updated.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE tts_calls SET status=?, error=? "
                "WHERE scenario_id=? AND status='ok'",
                (new_status, error, scenario_id),
            )
            self._conn.commit()
            return cur.rowcount

    def log_unit_eval(self, run_id: str, job_key: str, scenario_id: str, status: str,
                      fail_flag: str = "", errors: str = "") -> None:
        """
        Record the Section 8.5 unit-level check for ONE generated scenario. `status` is
        'pass' or 'fail'; on a fail, `fail_flag` is the caller's job-scoped incrementing
        tag ('fail1', 'fail2', ...) and `errors` is a JSON-encoded list of validator
        messages. This is what makes a failed-then-regenerated scenario traceable in the
        terminal stream and in scripts/11_monitor.py.
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO unit_evals (run_id, job_key, scenario_id, status, fail_flag, "
                "errors, created_at) VALUES (?,?,?,?,?,?,?)",
                (run_id, job_key, scenario_id, status, fail_flag, errors, time.time()),
            )
            self._conn.commit()

    def log_scenario(self, run_id: str, job_key: str, scenario_id: str,
                     global_index: int | None = None) -> None:
        """
        Record a scenario that actually made it into the shard (passed unit-eval).
        `INSERT OR IGNORE` because scenario_id is unique — a re-run that re-derives the
        same id (same job + slot) just no-ops instead of erroring.
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO scenario_registry (run_id, job_key, scenario_id, "
                "global_index, created_at) VALUES (?,?,?,?,?)",
                (run_id, job_key, scenario_id, global_index, time.time()),
            )
            self._conn.commit()

    def registered_count(self, *, job_key: str | None = None,
                         global_range: tuple[int, int] | None = None) -> int:
        """
        How many scenarios are recorded in scenario_registry — optionally scoped to one
        job, or to a [start, end) global-index range (what a --range run just covers).
        """
        with self._lock:
            if global_range is not None:
                lo, hi = global_range
                return self._conn.execute(
                    "SELECT COUNT(*) FROM scenario_registry "
                    "WHERE global_index >= ? AND global_index < ?", (lo, hi)
                ).fetchone()[0]
            if job_key:
                return self._conn.execute(
                    "SELECT COUNT(*) FROM scenario_registry WHERE job_key=?", (job_key,)
                ).fetchone()[0]
            return self._conn.execute("SELECT COUNT(*) FROM scenario_registry").fetchone()[0]

    def mark_hf_synced(self, scenario_id: str, repo_id: str) -> None:
        """Record that scenario_id has been uploaded to repo_id — resumability for
        scripts/13_upload_hf.py, same pattern as kupe-tts's hf_synced_at column."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO hf_sync (scenario_id, repo_id, synced_at) "
                "VALUES (?,?,?)",
                (scenario_id, repo_id, time.time()),
            )
            self._conn.commit()

    def hf_synced_ids(self, repo_id: str) -> set[str]:
        """Every scenario_id already uploaded to repo_id — the resume checkpoint."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT scenario_id FROM hf_sync WHERE repo_id=?", (repo_id,)
            ).fetchall()
        return {r[0] for r in rows}

    def clear_hf_sync(self, repo_id: str) -> int:
        """Delete every hf_sync row for repo_id — pairs with actually deleting the
        content from that repo remotely (scripts/P1_00_pipeline.py --cleanup), so a
        subsequent real run doesn't think everything is already uploaded and skip it.
        Returns the number of rows removed."""
        with self._lock:
            n = self._conn.execute(
                "SELECT COUNT(*) FROM hf_sync WHERE repo_id=?", (repo_id,)
            ).fetchone()[0]
            self._conn.execute("DELETE FROM hf_sync WHERE repo_id=?", (repo_id,))
            self._conn.commit()
        return n

    def unit_eval_summary(self, run_id: str | None = None) -> dict:
        """Pass/fail counts for one run, or across all runs if run_id is None."""
        with self._lock:
            if run_id:
                cur = self._conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(CASE WHEN status='pass' THEN 1 ELSE 0 END),0), "
                    "COALESCE(SUM(CASE WHEN status='fail' THEN 1 ELSE 0 END),0) "
                    "FROM unit_evals WHERE run_id=?", (run_id,))
            else:
                cur = self._conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(CASE WHEN status='pass' THEN 1 ELSE 0 END),0), "
                    "COALESCE(SUM(CASE WHEN status='fail' THEN 1 ELSE 0 END),0) FROM unit_evals")
            total, passed, failed = cur.fetchone()
        return {"total": total, "passed": passed, "failed": failed}

    # ------------------------------------------------------------------ #
    def run_summary(self, run_id: str) -> dict:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(cost_usd),0), COALESCE(SUM(prompt_tokens),0), "
                "COALESCE(SUM(completion_tokens),0), COALESCE(SUM(valid_n),0), "
                "COALESCE(SUM(requested_n),0) FROM llm_calls WHERE run_id=?", (run_id,))
            n_calls, llm_cost, ptok, ctok, valid, requested = cur.fetchone()
            cur = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(cost_usd),0), COALESCE(SUM(duration_s),0) "
                "FROM tts_calls WHERE run_id=?", (run_id,))
            n_tts, tts_cost, tts_secs = cur.fetchone()
        return {
            "run_id": run_id,
            "llm_calls": n_calls, "llm_cost_usd": round(llm_cost, 4),
            "prompt_tokens": ptok, "completion_tokens": ctok,
            "scenarios_valid": valid, "scenarios_requested": requested,
            "tts_calls": n_tts, "tts_cost_usd": round(tts_cost, 4),
            "tts_audio_hours": round(tts_secs / 3600.0, 3),
            "total_cost_usd": round(llm_cost + tts_cost, 4),
        }

    def export_costs_csv(self, out_path: str | Path) -> Path:
        """Dump a flat, per-call CSV covering every llm_call + tts_call ever logged."""
        import csv
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["kind", "run_id", "stage", "job_key_or_scenario", "model",
                       "requested_or_chars", "valid_or_duration_s", "prompt_tokens",
                       "completion_tokens", "cost_usd", "latency_ms", "status",
                       "error", "created_at"])
            rows = self._conn.execute(
                "SELECT l.run_id, r.stage, l.job_key, l.model, l.requested_n, l.valid_n, "
                "l.prompt_tokens, l.completion_tokens, l.cost_usd, l.latency_ms, l.status, "
                "l.error, l.created_at FROM llm_calls l JOIN runs r ON l.run_id=r.run_id"
            ).fetchall()
            for row in rows:
                w.writerow(["llm", *row[:3], row[3], row[4], row[5], row[6], row[7],
                           row[8], row[9], row[10], row[11], row[12]])
            rows = self._conn.execute(
                "SELECT t.run_id, r.stage, t.scenario_id, t.chars, t.duration_s, "
                "t.cost_usd, t.latency_ms, t.status, t.error, t.created_at "
                "FROM tts_calls t JOIN runs r ON t.run_id=r.run_id"
            ).fetchall()
            for row in rows:
                w.writerow(["tts", row[0], row[1], row[2], "soniox", row[3], row[4],
                           "", "", row[5], row[6], row[7], row[8], row[9]])
        return out_path

    def close(self):
        with self._lock:
            self._conn.close()

    def wipe_all(self) -> dict[str, int]:
        """Delete every run / LLM / TTS / unit-eval / registry / hf_sync row. Returns
        counts removed. hf_sync is included so a --cleanup + regenerate cycle doesn't
        leave stale 'already uploaded' markers for scenario_ids that get re-derived
        with the same id (deterministic md5 of job+index) but different content."""
        with self._lock:
            n_llm = self._conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
            n_tts = self._conn.execute("SELECT COUNT(*) FROM tts_calls").fetchone()[0]
            n_runs = self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            n_unit = self._conn.execute("SELECT COUNT(*) FROM unit_evals").fetchone()[0]
            n_reg = self._conn.execute("SELECT COUNT(*) FROM scenario_registry").fetchone()[0]
            n_hf = self._conn.execute("SELECT COUNT(*) FROM hf_sync").fetchone()[0]
            self._conn.execute("DELETE FROM llm_calls")
            self._conn.execute("DELETE FROM tts_calls")
            self._conn.execute("DELETE FROM unit_evals")
            self._conn.execute("DELETE FROM scenario_registry")
            self._conn.execute("DELETE FROM hf_sync")
            self._conn.execute("DELETE FROM runs")
            self._conn.commit()
        return {"llm_calls": n_llm, "tts_calls": n_tts, "runs": n_runs, "unit_evals": n_unit,
               "scenario_registry": n_reg, "hf_sync": n_hf}

    def wipe_tts_data(self) -> dict[str, int]:
        """
        Scoped cleanup for scripts/03_render_user_audio.py --cleanup: clears ONLY
        `tts_calls` + `hf_sync` (both TTS/audio-specific) — deliberately leaves
        `llm_calls`, `unit_evals`, `scenario_registry`, and `runs` untouched, since
        those track scenario GENERATION (a separate stage with its own --cleanup in
        script 02) and have nothing to do with rendered audio. `hf_sync` is cleared
        too: after deleting local audio, any 'already uploaded' marker would otherwise
        make scripts/13_upload_hf.py wrongly skip re-uploading the corrected audio for
        the same scenario_id. Returns counts removed.
        """
        with self._lock:
            n_tts = self._conn.execute("SELECT COUNT(*) FROM tts_calls").fetchone()[0]
            n_hf = self._conn.execute("SELECT COUNT(*) FROM hf_sync").fetchone()[0]
            self._conn.execute("DELETE FROM tts_calls")
            self._conn.execute("DELETE FROM hf_sync")
            self._conn.commit()
        return {"tts_calls": n_tts, "hf_sync": n_hf}

    @staticmethod
    def counts_if_exists(path: str | Path) -> dict[str, int]:
        empty = {"llm_calls": 0, "tts_calls": 0, "runs": 0, "unit_evals": 0,
                "scenario_registry": 0, "hf_sync": 0}
        p = Path(path)
        if not p.exists():
            return empty
        conn = sqlite3.connect(str(p), timeout=10)
        try:
            out = dict(empty)
            out["llm_calls"] = conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
            out["tts_calls"] = conn.execute("SELECT COUNT(*) FROM tts_calls").fetchone()[0]
            out["runs"] = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            for table in ("unit_evals", "scenario_registry", "hf_sync"):
                try:
                    out[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                except sqlite3.OperationalError:
                    pass  # older DB file predating this table
            return out
        finally:
            conn.close()


@contextmanager
def open_db(path: str | Path):
    db = RunDB(path)
    try:
        yield db
    finally:
        db.close()


class ReadOnlyDB:
    """
    A separate, read-only connection meant for `scripts/11_monitor.py` — safe to open
    from a second process WHILE the generator/renderer is writing (WAL mode allows
    concurrent readers), and `query_only` guards against this class ever writing.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._conn = sqlite3.connect(str(self.path), timeout=10)
        self._conn.execute("PRAGMA query_only=1;")

    def exists(self) -> bool:
        return self.path.exists()

    def fetchone(self, sql: str, params: tuple = ()) -> tuple:
        return self._conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list[tuple]:
        return self._conn.execute(sql, params).fetchall()

    def close(self):
        self._conn.close()
