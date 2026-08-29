"""
The live referee running frame-by-frame with prefetch (Section 11).

This wraps the trained model in the exact orchestration loop from the guide:

    every 80 ms frame:
        tok, e, f0 = mimi_stream_encode(mic_frame)
        st = agent_state()          # IDLE|LLM_GEN|TTS_SPEAKING|TTS_DONE (NO timestamps)
        flag, spoken = model.step(prompt, st, agent_text, stt_partial?, tok, e, f0)
        match flag:
            BARGE_HARD    -> stop_tts();  send_to_llm(agent_text_so_far, user_turn)
            BARGE_SOFT    -> duck_tts()
            PREFETCH_LLM  -> if st==LLM_GEN: start_llm(async, user_partial)  # hide latency
            COMMIT_LLM    -> reply = await_llm();  tts_stream(reply)
            CANCEL_LLM    -> abort_llm()
            TURN_END      -> if no prefetch: start_llm(); else COMMIT
            SILENCE_BREAK -> tts_stream(spoken or llm_reopen())
            INCOMPLETE|HOLD|LISTEN|CONTINUE -> pass
        if spoken: tts_stream(spoken)   # back-channel / thinking / forceful-interrupt

The model keeps a rolling text context and a streaming KV cache; only the newest audio
frame is embedded each step, so per-frame decode is a single incremental transformer
step (Section 5, "Decode cost per frame").

`ReferenceOrchestrator` is a pure-python reference implementation with pluggable
callbacks (`on_barge`, `on_prefetch`, `tts_stream`, ...). Wire these to your SDK /
LiveKit / Pipecat layer in production; here they default to logging so the loop can be
driven from a simulated frame stream in scripts/09_infer_demo.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from thinkspark import vocab


# --------------------------------------------------------------------------- #
@dataclass
class FrameInput:
    cb0: int            # Mimi cb0 token for this 80 ms frame
    energy: float       # per-frame log-RMS energy
    f0: float           # per-frame f0 (Hz; 0 = unvoiced)
    agent_state: str    # IDLE|LLM_GEN|TTS_SPEAKING|TTS_DONE


@dataclass
class StepResult:
    flag: str
    spoken: str = ""
    decode_ms: float = 0.0


# --------------------------------------------------------------------------- #
class StreamingReferee:
    """
    Thin adapter around a trained ThinkSparkModel that keeps a rolling context and
    exposes a per-frame `step()`. This reference version re-runs the model over a small
    trailing window each frame for clarity; swap in a KV-cache stream for production
    (the model already runs KV-cache-friendly — see model.forward use_cache).
    """

    def __init__(self, model, tokenizer, system_prompt: str,
                 device: str = "cpu", window_frames: int = 96,
                 spoken_max_new: int = 12):
        self.model = model.eval()
        self.tok = tokenizer
        self.device = device
        self.system_prompt = system_prompt
        self.window = window_frames
        self.spoken_max_new = spoken_max_new
        # rolling buffers
        self._cb0: list[int] = []
        self._energy: list[float] = []
        self._f0: list[float] = []
        self._state: list[int] = []
        self.agent_text = ""
        self.stt_partial = ""

    # ------------------------------------------------------------------ #
    def reset(self):
        self._cb0.clear(); self._energy.clear(); self._f0.clear(); self._state.clear()

    def set_context(self, agent_text: str = "", stt_partial: str = ""):
        self.agent_text = agent_text
        self.stt_partial = stt_partial

    # ------------------------------------------------------------------ #
    def step(self, frame: FrameInput) -> StepResult:
        import time
        import torch

        self._cb0.append(frame.cb0)
        self._energy.append(frame.energy)
        self._f0.append(frame.f0)
        self._state.append(vocab.AGENT_STATE_TO_ID.get(frame.agent_state, 0))

        # keep only the trailing window
        if len(self._cb0) > self.window:
            for buf in (self._cb0, self._energy, self._f0, self._state):
                del buf[0]

        batch = self._make_batch(torch)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(**batch)
        decode_ms = (time.perf_counter() - t0) * 1000.0

        # newest frame's control flag
        ctrl = out.control_logits[0, -1]                      # [num_flags]
        flag_id = int(ctrl.argmax().item())
        flag = vocab.ID_TO_CONTROL_FLAG[flag_id]

        spoken = ""
        if flag in vocab.SPEAKING_FLAGS or flag == "SILENCE_BREAK":
            spoken = self._maybe_generate_spoken(out, batch, torch)

        return StepResult(flag=flag, spoken=spoken, decode_ms=decode_ms)

    # ------------------------------------------------------------------ #
    def _make_batch(self, torch):
        from thinkspark.model import SEG_SYS, SEG_AGENT, SEG_STT
        sp = vocab.SPECIAL_TOKENS

        ids: list[int] = []; seg: list[int] = []
        for marker, text, sid in (
            (sp["sys_bos"], self.system_prompt, SEG_SYS),
            (sp["agent_bos"], self.agent_text, SEG_AGENT),
            (sp["stt_bos"], self.stt_partial, SEG_STT),
        ):
            piece = self.tok(marker + " " + text, add_special_tokens=False)["input_ids"]
            ids.extend(piece); seg.extend([sid] * len(piece))

        T = len(self._cb0)
        f0n = np.array(self._f0, dtype=np.float32)
        f0n = np.where(f0n > 1.0, np.log(np.maximum(f0n, 1.0)) - np.log(150.0), 0.0)
        prosody = np.stack([np.array(self._energy, dtype=np.float32), f0n], axis=-1)

        dev = self.device
        L = lambda a, dt: torch.tensor(a, dtype=dt, device=dev).unsqueeze(0)
        return {
            "text_ids": L(ids, torch.long),
            "text_seg": L(seg, torch.long),
            "text_mask": L([1] * len(ids), torch.long),
            "cb0": L(self._cb0, torch.long),
            "prosody": torch.tensor(prosody, dtype=torch.float32, device=dev).unsqueeze(0),
            "agent_state": L(self._state, torch.long),
            "audio_mask": L([1] * T, torch.long),
        }

    def _maybe_generate_spoken(self, out, batch, torch) -> str:
        """
        Greedy-decode a short spoken interjection from the spoken head, seeded with the
        <|say|> token. Kept intentionally short (few tokens) — back-channels are 1-3 words.
        Returns "" if the model emits <|/say|> immediately (the silent case).
        """
        sp = vocab.SPECIAL_TOKENS
        say_bos = self.tok(sp["spoken_bos"], add_special_tokens=False)["input_ids"]
        say_eos = self.tok(sp["spoken_eos"], add_special_tokens=False)["input_ids"][0]

        cur = list(say_bos)
        for _ in range(self.spoken_max_new):
            b = dict(batch)
            b["spoken_ids"] = torch.tensor(cur, dtype=torch.long, device=self.device).unsqueeze(0)
            b["spoken_mask"] = torch.ones_like(b["spoken_ids"])
            with torch.no_grad():
                o = self.model(**b)
            nxt = int(o.lm_logits[0, -1].argmax().item())
            if nxt == say_eos:
                break
            cur.append(nxt)
        text = self.tok.decode(cur[len(say_bos):], skip_special_tokens=True).strip()
        return text


# --------------------------------------------------------------------------- #
@dataclass
class ReferenceOrchestrator:
    """
    Pure-python floor controller that turns StepResults into orchestration actions
    (Section 11). Callbacks default to logging so it runs headless.
    """
    referee: StreamingReferee
    on_barge_hard: Callable[[], None] = lambda: None
    on_barge_soft: Callable[[], None] = lambda: None
    on_prefetch: Callable[[str], None] = lambda partial: None
    on_commit: Callable[[], None] = lambda: None
    on_cancel: Callable[[], None] = lambda: None
    on_turn_end: Callable[[], None] = lambda: None
    tts_stream: Callable[[str], None] = lambda text: None
    log: list[str] = field(default_factory=list)

    prefetch_active: bool = False

    def handle(self, frame: FrameInput) -> StepResult:
        res = self.referee.step(frame)
        flag, spoken = res.flag, res.spoken

        if flag == "BARGE_HARD":
            self.on_barge_hard()
        elif flag == "BARGE_SOFT":
            self.on_barge_soft()
        elif flag == "PREFETCH_LLM":
            if frame.agent_state != "LLM_GEN":
                self.on_prefetch(self.referee.stt_partial)
                self.prefetch_active = True
        elif flag == "COMMIT_LLM":
            self.on_commit()
            self.prefetch_active = False
        elif flag == "CANCEL_LLM":
            self.on_cancel()
            self.prefetch_active = False
        elif flag == "TURN_END":
            if not self.prefetch_active:
                self.on_prefetch(self.referee.stt_partial)
            self.on_turn_end()
        elif flag == "SILENCE_BREAK":
            self.tts_stream(spoken or "")
        # INCOMPLETE|HOLD|LISTEN|CONTINUE -> pass

        if spoken and flag != "SILENCE_BREAK":
            self.tts_stream(spoken)   # back-channel / thinking / forceful interrupt

        self.log.append(f"{frame.agent_state:<12} -> {flag:<13} {('say=' + spoken) if spoken else ''}")
        return res
