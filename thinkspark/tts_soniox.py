"""
Soniox TTS client for rendering the USER side of every scenario (Section 8.4, step 2).

Only the user's line is ever synthesised — the agent side stays text + a state flag, so
zero agent audio is generated and the model never consumes agent audio (Section 8).

We request character-level timestamps (`return_timestamps=true`) so the frame builder can
place each control-flag event on a frame-accurate offset (this is the ONLY place
timestamps are used; inference is timestamp-free) — see `TTSResult.frame_at_char`.

Protocol (verified against a real, working client — kupe-tts/tts_scripts/soniox/client.py
— NOT guessed): Soniox realtime TTS is a WebSocket at `wss://tts-rt.soniox.com/tts-websocket`
(distinct from the STT endpoint). One call = TWO sends on one connection:
  1. a config message: api_key, model, language, voice, audio_format, sample_rate,
     stream_id, return_timestamps
  2. a text message: {"text": ..., "text_end": true, "stream_id": ...}
The server streams back JSON frames keyed by `stream_id` carrying:
  - "audio": base64 WAV bytes (streamed with placeholder RIFF/data sizes — concatenate
    then patch, see `_patch_streaming_wav`)
  - "timestamps": {"characters": [...], "character_start_times_seconds": [...],
    "character_end_times_seconds": [...]} — CHARACTER-level, not word-level; we group
    on whitespace into WordSpans ourselves (`_group_words`)
  - "audio_end": true when audio is fully sent
  - "terminated": true when the stream is fully done (the actual completion signal —
    NOT "finished" or a type=="final" field, which don't exist in the real protocol)
  - "error_code" (+ "error_type", "error_message"): a real error, not a truthy "error" field

Voices: Soniox voice IDs are character NAMES (e.g. "Priya", "Arjun", "Maya", "Daniel"),
never a "language-gender" string — every voice works with all 60+ supported languages
(soniox.com/docs/tts/concepts/voices: "pick a voice once and keep the same speaker across
your whole product"), so `language` alone controls what's spoken; `voice` only controls
timbre. By explicit instruction, this project uses ONLY your own cloned voices
(scripts/15_create_voice_profiles.py -> data/voice_refs/voice_profiles.json) — never
Soniox's built-in catalog. `resolve_voice()` rotates deterministically across YOUR
profiles per scenario (hash of the text) and raises a clear, actionable error if none
exist yet for the requested gender — it never silently substitutes a catalog voice.

Rate limits (https://soniox.com/docs/tts/rt/limits-and-quotas): 100 stream starts/minute,
3 concurrent streams account-wide by default (raise `soniox_concurrency` in config only
if your account has a higher limit), 5 active streams per WebSocket connection, 2 minutes
of audio per stream. Both limits are respected proactively, not just reacted to:
concurrency is still bounded by the render script's thread pool (`soniox_concurrency`),
and `_throttle_stream_start` paces new streams under `soniox_max_stream_starts_per_min`
(default 90, a safety margin under the real 100) — a small, evenly-spread wait here is
cheaper than bursting past the limit and paying `_synthesize_ws`'s exponential 429
backoff instead. Each worker thread also REUSES one persistent WS connection across all
the scenarios it renders (`_get_ws`, thread-local) instead of reconnecting per scenario
— well within the documented 5-streams-per-connection allowance since each thread only
ever runs one stream on its connection at a time — cutting the TCP+TLS handshake cost
out of every request but the first per thread. Together these make a long render
noticeably faster without changing how many streams run at once.

Pricing: token-based, not a flat $/hour — $4.00 / 1M input text tokens (~0.3 tokens/char)
+ $21.50 / 1M output audio tokens (~30,000 tokens/hour of audio). See `soniox_cost_usd()`;
these constants are the same ones kupe-tts's tts_pricing.py verified against Soniox's
pricing page. `cfg.soniox_price_per_hour_usd` (~$0.70/h) stays as a coarse fallback only
for projecting cost *before* any real calls have happened.

Result contract
---------------
`TTSResult`:
    audio          : float32 mono numpy array (or None if you only saved to disk)
    sample_rate    : int
    duration_s     : float
    words          : list[WordSpan]  (text, start_s, end_s, start_char, end_char)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np

from thinkspark import vocab
from thinkspark.config import DataGenConfig, env

# ---- verified Soniox constants (kupe-tts/tts_scripts/soniox/paths.py) --------------
SONIOX_TTS_WS_DEFAULT = "wss://tts-rt.soniox.com/tts-websocket"
SONIOX_MODEL_DEFAULT = "tts-rt-v2"
SONIOX_API_BASE = "https://api.soniox.com"
MAX_STREAM_AUDIO_SEC = 120.0   # fixed 2-minute-per-stream cap

# ---- your own cloned voices ONLY (scripts/15_create_voice_profiles.py) -------------
# By explicit instruction: no Soniox built-in catalog voices, ever — only voice IDs
# you cloned yourself from your own reference clips.
_custom_voices_lock = threading.Lock()
_custom_voices_cache: dict[str, dict[str, list[str]]] = {}  # profiles_path -> {gender: [voice_id,...]}


def _load_custom_voice_profiles(profiles_path: str | os.PathLike) -> dict[str, list[str]]:
    """
    Load YOUR OWN cloned voice IDs from scripts/15_create_voice_profiles.py's output
    (`voice_profiles.json`) — {gender: [voice_id, ...]}. Cached per path; missing/empty
    file just returns {} (no profiles cloned yet — resolve_voice() then raises clearly
    rather than silently reaching for a catalog voice).
    """
    key = str(profiles_path)
    with _custom_voices_lock:
        if key in _custom_voices_cache:
            return _custom_voices_cache[key]
        by_gender: dict[str, list[str]] = {}
        try:
            path = Path(profiles_path)
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                for v in data.get("voices", []):
                    gender = str(v.get("gender", "")).lower()
                    vid = v.get("voice_id")
                    if gender and vid:
                        by_gender.setdefault(gender, []).append(vid)
        except (OSError, json.JSONDecodeError):
            pass
        _custom_voices_cache[key] = by_gender
        return by_gender


def clear_custom_voice_cache() -> None:
    """Test/CLI hook — forces the next resolve_voice() call to re-read the JSON file
    instead of serving a stale in-process cache."""
    with _custom_voices_lock:
        _custom_voices_cache.clear()


def resolve_voice(gender: str, profiles_path: str | os.PathLike, seed_text: str = "") -> str:
    """
    Pick one of YOUR OWN cloned voice IDs for `gender`, rotating deterministically
    across them by a hash of `seed_text` (same text -> same voice, reproducible;
    different scenarios -> different voices, real speaker diversity across the corpus —
    Section 8.3's "gender-balanced voice profiles", but sourced entirely from clips you
    collected yourself). Deliberately NEVER falls back to Soniox's built-in catalog —
    raises a clear, actionable error instead if no clones exist yet for this gender.
    """
    names = _load_custom_voice_profiles(profiles_path).get(gender.lower(), [])
    if not names:
        raise RuntimeError(
            f"No cloned voice profiles found for gender='{gender}' in {profiles_path}. "
            f"This project uses ONLY your own cloned voices (no Soniox catalog "
            f"fallback) — add reference clips to data/voice_refs/ named "
            f"'{gender}_<name>.wav' (see data/voice_refs/README.md), then run "
            f"`python scripts/15_create_voice_profiles.py --config configs/data_gen.yaml` "
            f"before rendering."
        )
    if len(names) == 1 or not seed_text:
        return names[0]
    h = int(hashlib.md5(seed_text.encode("utf-8")).hexdigest(), 16)
    return names[h % len(names)]

# ---- verified per-request pricing (kupe-tts/text_scripts/tts_pricing.py) -----------
SONIOX_INPUT_USD_PER_M = 4.00
SONIOX_OUTPUT_USD_PER_M = 21.50
SONIOX_CHARS_TO_INPUT_TOKENS = 0.3      # 1 char ~= 0.3 input text tokens
SONIOX_OUTPUT_TOKENS_PER_HOUR = 30_000  # 1 hour of audio ~= 30,000 output tokens


def soniox_cost_usd(chars: int, duration_s: float) -> float:
    """Real per-request Soniox TTS cost — token-based, not a flat $/hour estimate."""
    input_tokens = chars * SONIOX_CHARS_TO_INPUT_TOKENS
    output_tokens = (duration_s / 3600.0) * SONIOX_OUTPUT_TOKENS_PER_HOUR
    return (input_tokens / 1_000_000.0) * SONIOX_INPUT_USD_PER_M + \
           (output_tokens / 1_000_000.0) * SONIOX_OUTPUT_USD_PER_M


class SonioxRateLimitError(RuntimeError):
    """Raised when a Soniox call exhausted its retries and the failure looked like a
    rate limit (429 / 'rate limit' / 'quota' in the error) — callers can catch this
    separately from a generic failure to track/display rate-limit hits distinctly."""


_RATE_LIMIT_MARKERS = ("rate limit", "ratelimit", "429", "too many requests", "quota")


def _looks_rate_limited(msg: str) -> bool:
    low = msg.lower()
    return any(marker in low for marker in _RATE_LIMIT_MARKERS)


# --------------------------------------------------------------------------- #
@dataclass
class WordSpan:
    text: str
    start_s: float
    end_s: float
    start_char: int
    end_char: int


@dataclass
class TTSResult:
    sample_rate: int
    duration_s: float
    words: list[WordSpan] = field(default_factory=list)
    audio: np.ndarray | None = None      # float32 mono in [-1, 1]
    wav_path: str | None = None
    chars_synthesized: int = 0           # len(input text) — used for real per-request cost

    def char_to_seconds(self, char_index: int) -> float:
        """Map a char offset in user_text to a wall-clock time using word spans."""
        if not self.words:
            # linear fallback across the whole utterance (no timestamps captured —
            # should only happen if a scenario had zero words, not the normal path)
            total_chars = max(1, self.chars_synthesized or char_index + 1)
            return self.duration_s * min(1.0, char_index / total_chars)
        for w in self.words:
            if w.start_char <= char_index <= w.end_char:
                span = max(1, w.end_char - w.start_char)
                frac = (char_index - w.start_char) / span
                return w.start_s + frac * (w.end_s - w.start_s)
        if char_index <= self.words[0].start_char:
            return self.words[0].start_s
        return self.words[-1].end_s

    def frame_at_char(self, char_index: int) -> int:
        return vocab.seconds_to_frames(self.char_to_seconds(char_index))


@dataclass
class SonioxTTS:
    ws_url: str = SONIOX_TTS_WS_DEFAULT
    model: str = SONIOX_MODEL_DEFAULT
    sample_rate: int = 24000
    api_key: str | None = None
    max_retries: int = 5          # exponential backoff; rate-limit hits back off harder
    connect_timeout_s: float = 60.0
    recv_timeout_s: float = 45.0
    voice_profiles_path: str | None = None   # scripts/15_create_voice_profiles.py output
    # Proactive pacing of NEW stream starts, sized under Soniox's documented 100/minute
    # account-wide limit (soniox.com/docs/tts/rt/limits-and-quotas) as a safety margin —
    # see `_throttle_stream_start`. Shared across every thread using this SAME instance.
    max_stream_starts_per_min: int = 90

    def __post_init__(self):
        self.api_key = self.api_key or env("SONIOX_API_KEY", required=True)
        # one persistent WS connection PER WORKER THREAD, reused across every scenario
        # that thread renders (see `_get_ws`) — avoids paying a fresh TCP+TLS handshake
        # per scenario, which is pure overhead Soniox's protocol doesn't require (docs:
        # a single connection supports up to 5 active streams, so sequential reuse by
        # one thread is well within spec). Self-healing: any send/recv failure drops
        # the thread's connection so the next attempt just reconnects, same as before.
        self._local = threading.local()
        self._start_lock = threading.Lock()
        self._start_times: deque[float] = deque()   # sliding 60s window, shared by all threads

    @classmethod
    def from_config(cls, cfg: DataGenConfig) -> "SonioxTTS":
        return cls(ws_url=cfg.soniox_ws_url, model=cfg.soniox_model,
                  sample_rate=cfg.soniox_sample_rate, max_retries=cfg.soniox_max_retries,
                  voice_profiles_path=cfg.voice_profiles_path,
                  max_stream_starts_per_min=cfg.soniox_max_stream_starts_per_min)

    # ------------------------------------------------------------------ #
    def synthesize(
        self,
        text: str,
        language: str,
        gender: str = "female",
        speaker: str | None = None,
        wav_path: str | os.PathLike | None = None,
        keep_audio: bool = False,
    ) -> TTSResult:
        """
        Synthesize `text` to speech with character-level timestamps, grouped into
        WordSpans. If `wav_path` is given, the (already-valid) WAV bytes returned by
        Soniox are written straight to disk; kept in memory as a float32 array only
        when `keep_audio=True` (to bound RAM during a 55h render).
        """
        wav_bytes, char_timeline = self._synthesize_ws(text, language, gender, speaker)
        meta = _parse_wav_meta(wav_bytes)
        duration = meta["duration_sec"] if meta else 0.0

        if wav_path is not None:
            Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
            Path(wav_path).write_bytes(wav_bytes)

        audio = None
        if keep_audio and meta:
            with wave.open(BytesIO(wav_bytes), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        words = _group_words(char_timeline)

        return TTSResult(
            sample_rate=meta["sample_rate"] if meta else self.sample_rate,
            duration_s=duration,
            words=words,
            audio=audio,
            wav_path=str(wav_path) if wav_path is not None else None,
            chars_synthesized=len(text),
        )

    # ------------------------------------------------------------------ #
    def _synthesize_ws(self, text, language, gender, speaker):
        """
        Retry wrapper with exponential backoff. A failure classified as a rate limit
        (429 / 'rate limit' / 'quota' — see `_looks_rate_limited`) backs off harder
        (base 3s vs. 1s) and, once retries are exhausted, raises `SonioxRateLimitError`
        specifically so callers (the render script's live dashboard) can count/display
        rate-limit hits separately from other failures.
        """
        last_err: Exception | None = None
        was_rate_limited = False
        for attempt in range(self.max_retries):
            try:
                return self._synthesize_ws_once(text, language, gender, speaker)
            except Exception as e:
                last_err = e
                rate_limited = _looks_rate_limited(str(e))
                was_rate_limited = was_rate_limited or rate_limited
                if attempt == self.max_retries - 1:
                    break
                base = 3.0 if rate_limited else 1.0
                sleep = min(base * (2 ** attempt), 60.0)
                time.sleep(sleep)

        if was_rate_limited:
            raise SonioxRateLimitError(
                f"Soniox TTS rate-limited after {self.max_retries} retries: {last_err}"
            ) from last_err
        raise RuntimeError(f"Soniox TTS failed after {self.max_retries} retries: {last_err}") from last_err

    def _throttle_stream_start(self) -> None:
        """
        Block (briefly) until starting a new stream stays under
        `max_stream_starts_per_min` — a sliding 60s window shared by every worker
        thread. This makes the OVERALL render faster, not slower: a proactive, evenly-
        paced wait here is far cheaper than bursting past Soniox's real limit, getting
        a 429, and paying `_synthesize_ws`'s exponential backoff (which starts at
        3s and doubles) — this waits at most a fraction of a second, and only when
        actually needed.
        """
        while True:
            with self._start_lock:
                now = time.monotonic()
                while self._start_times and now - self._start_times[0] > 60.0:
                    self._start_times.popleft()
                if len(self._start_times) < self.max_stream_starts_per_min:
                    self._start_times.append(now)
                    return
                wait_s = 60.0 - (now - self._start_times[0]) + 0.05
            time.sleep(max(wait_s, 0.01))

    def _get_ws(self):
        """
        Return this THREAD's persistent connection, reconnecting only if it doesn't
        exist yet or was dropped (by a prior error, or the socket actually closing).
        Never shared across threads (`threading.local`), so this never risks exceeding
        the account's concurrent-stream cap — it's strictly connection reuse for
        sequential requests on the same worker, not added parallelism.
        """
        try:
            from websocket import WebSocketBadStatusException, create_connection
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "websocket-client not installed. `pip install websocket-client`."
            ) from e

        ws = getattr(self._local, "ws", None)
        if ws is not None and ws.connected:
            return ws
        try:
            ws = create_connection(self.ws_url, timeout=self.connect_timeout_s)
        except WebSocketBadStatusException as e:
            status = getattr(e, "status_code", None)
            if status == 429:
                raise RuntimeError(f"Soniox connect failed: HTTP 429 rate limited ({e})") from e
            raise RuntimeError(f"Soniox connect failed: HTTP {status} ({e})") from e
        ws.settimeout(self.recv_timeout_s)
        self._local.ws = ws
        return ws

    def _synthesize_ws_once(self, text, language, gender, speaker):
        """
        ONE attempt at the real two-message Soniox protocol, over this thread's reused
        connection (see `_get_ws`). Returns (wav_bytes, char_timeline) where
        char_timeline is [(character, start_sec, end_sec), ...] in text order.
        """
        self._throttle_stream_start()

        stream_id = f"tts-{uuid.uuid4().hex[:12]}"
        config = {
            "api_key": self.api_key,
            "model": self.model,
            "language": language.split("_")[0],   # 'hi_en_native' -> 'hi'
            "voice": speaker or resolve_voice(gender, self.voice_profiles_path, seed_text=text),
            "audio_format": "wav",
            "sample_rate": self.sample_rate,
            "stream_id": stream_id,
            "return_timestamps": True,
        }

        ws = self._get_ws()
        try:
            ws.send(json.dumps(config))
            ws.send(json.dumps({"text": text, "text_end": True, "stream_id": stream_id}))

            audio_chunks: list[bytes] = []
            raw_frames: list[dict] = []
            terminated = False
            t0 = time.perf_counter()
            while not terminated:
                if time.perf_counter() - t0 > self.recv_timeout_s:
                    raise RuntimeError(f"Soniox recv timed out after {self.recv_timeout_s}s "
                                       f"waiting for 'terminated'")
                msg = ws.recv()
                if not msg:
                    raise RuntimeError("Soniox connection closed before 'terminated'")
                data = json.loads(msg) if isinstance(msg, str) else None
                if data is None:
                    continue
                if data.get("error_code") is not None:
                    raise RuntimeError(
                        f"Soniox error {data.get('error_code')} "
                        f"({data.get('error_type')}): {data.get('error_message')}"
                    )
                audio_b64 = data.get("audio")
                if audio_b64:
                    audio_chunks.append(base64.b64decode(audio_b64))
                ts = data.get("timestamps")
                if isinstance(ts, dict):
                    raw_frames.append(ts)
                if data.get("terminated"):
                    terminated = True
        except Exception:
            # the connection's state is unknown after any error mid-stream (could be
            # desynced, half-closed, etc.) — drop it so the NEXT attempt (this one's
            # retry, or this thread's next scenario) opens a fresh one instead of
            # reusing something possibly broken. Success deliberately does NOT close
            # the connection — that's the whole point, see `_get_ws`.
            try:
                ws.close()
            except Exception:
                pass
            self._local.ws = None
            raise

        wav_bytes = _patch_streaming_wav(b"".join(audio_chunks))
        char_timeline = _flatten_char_timeline(raw_frames)
        return wav_bytes, char_timeline


# --------------------------------------------------------------------------- #
# WAV + character-timestamp helpers, ported from kupe-tts/tts_scripts/soniox/audio.py
# (verified against the real Soniox response format — do not "simplify" these back to
# guessed field names; that's the exact bug this file replaced).
# --------------------------------------------------------------------------- #
def _patch_streaming_wav(data: bytes) -> bytes:
    """Soniox streams WAV with RIFF/data sizes set to 0xFFFFFFFF. Patch after concat."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data
    buf = bytearray(data)
    struct.pack_into("<I", buf, 4, len(buf) - 8)
    pos = 12
    while pos + 8 <= len(buf):
        chunk_id = bytes(buf[pos:pos + 4])
        chunk_size = struct.unpack_from("<I", buf, pos + 4)[0]
        if chunk_id == b"data":
            pcm_len = len(buf) - (pos + 8)
            struct.pack_into("<I", buf, pos + 4, pcm_len)
            break
        if chunk_size == 0xFFFFFFFF:
            break
        pos += 8 + chunk_size + (chunk_size % 2)
    return bytes(buf)


def _parse_wav_meta(data: bytes) -> dict | None:
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos = 12
    rate = channels = bits = None
    pcm_len = None
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        if chunk_id == b"fmt " and chunk_size >= 16:
            _fmt, channels, rate, _br, _ba, bits = struct.unpack_from("<HHIIHH", data, pos + 8)
        if chunk_id == b"data":
            pcm_len = chunk_size if chunk_size != 0xFFFFFFFF else len(data) - (pos + 8)
            break
        if chunk_size == 0xFFFFFFFF:
            break
        pos += 8 + chunk_size + (chunk_size % 2)
    if not rate or not channels or not bits or pcm_len is None:
        return None
    frame_bytes = channels * (bits // 8)
    nframes = pcm_len // frame_bytes if frame_bytes else 0
    return {
        "sample_rate": rate, "channels": channels, "sample_width": bits // 8,
        "nframes": nframes, "duration_sec": nframes / float(rate) if rate else 0.0,
    }


def _flatten_char_timeline(raw_frames: list[dict]) -> list[tuple[str, float, float]]:
    """Concatenate every incremental {"characters": [...], start/end times} frame into
    one ordered (character, start_sec, end_sec) list, matching text order."""
    out: list[tuple[str, float, float]] = []
    for frame in raw_frames:
        chars = frame.get("characters") or []
        starts = frame.get("character_start_times_seconds") or []
        ends = frame.get("character_end_times_seconds") or []
        n = min(len(chars), len(starts), len(ends))
        for i in range(n):
            out.append((str(chars[i]), float(starts[i]), float(ends[i])))
    return out


def _group_words(char_timeline: list[tuple[str, float, float]]) -> list[WordSpan]:
    """Group Soniox's character-level timestamps into words, split on whitespace,
    tracking the character INDEX into the original text (needed by
    TTSResult.char_to_seconds / frame_at_char for Section 8.4 event calibration)."""
    words: list[WordSpan] = []
    buf: list[tuple[str, float, float]] = []
    buf_start_idx: int | None = None

    def flush(end_idx: int):
        nonlocal buf, buf_start_idx
        if buf:
            words.append(WordSpan(
                text="".join(c for c, _, _ in buf),
                start_s=buf[0][1], end_s=buf[-1][2],
                start_char=buf_start_idx, end_char=end_idx,
            ))
        buf = []
        buf_start_idx = None

    for idx, (ch, s, e) in enumerate(char_timeline):
        if ch.isspace():
            flush(idx - 1)
        else:
            if buf_start_idx is None:
                buf_start_idx = idx
            buf.append((ch, s, e))
    flush(len(char_timeline) - 1)
    return words
