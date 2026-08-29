"""
torch Dataset + collate for Phase-1 (alignment) and Phase-2 (referee) training.

A training example is one frame-record (produced by scripts/04_build_frames.py) plus its
Mimi-encoded audio shard (.npz from Phase 0). The dataset assembles the three sequence
segments the model consumes:

    text segment   : <|sys|> system/persona  <|agent|> agent_text  <|stt|> stt_partials
    audio segment  : cb0 tokens + prosody (energy,f0) + agent_state, per 80 ms frame
    spoken tail     : the target text the spoken head must produce
                      Phase 1 -> the user transcript (ASR-style alignment)
                      Phase 2 -> the concatenated back-channel / thinking interjections

The control head is supervised at the audio-segment positions (per-frame flags); the
spoken head is supervised at the tail (masked LM). See thinkspark.losses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from thinkspark import vocab, frames as frame_lib
from thinkspark.model import SEG_SYS, SEG_AGENT, SEG_STT

DEFAULT_SYSTEM = "You are a polite Indic voice agent. Decide when to listen, hold, interrupt, or back-channel."


# --------------------------------------------------------------------------- #
def build_tokenizer(base_model: str, hf_token: str | None = None):
    """Load the Gemma tokenizer and register ThinkSpark special tokens."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    tok.add_special_tokens({"additional_special_tokens": vocab.ALL_SPECIAL_TOKENS})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# --------------------------------------------------------------------------- #
class ThinkSparkDataset(Dataset):
    def __init__(self, shard_paths: list[str], tokenizer, phase: int = 2,
                 seq_len: int = 1024, vap_horizon: int = 25,
                 system_prompt: str = DEFAULT_SYSTEM):
        self.records: list[dict] = []
        for p in shard_paths:
            for line in Path(p).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.tok = tokenizer
        self.phase = phase
        self.seq_len = seq_len
        self.vap_horizon = vap_horizon
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.records)

    # ------------------------------------------------------------------ #
    def _encode_text(self, rec) -> tuple[list[int], list[int]]:
        """Build token ids + segment ids for the text context segment."""
        sp = vocab.SPECIAL_TOKENS
        ids: list[int] = []
        seg: list[int] = []

        def add(marker: str, text: str, seg_id: int):
            piece = self.tok(marker + " " + text, add_special_tokens=False)["input_ids"]
            ids.extend(piece)
            seg.extend([seg_id] * len(piece))

        add(sp["sys_bos"], self.system_prompt, SEG_SYS)
        add(sp["agent_bos"], rec.get("agent_text", "") or "", SEG_AGENT)
        # STT partials optional; use the user_text as a rolling partial proxy in training
        add(sp["stt_bos"], rec.get("user_text", "") or "", SEG_STT)
        return ids, seg

    def _spoken_tail(self, rec) -> list[int]:
        sp = vocab.SPECIAL_TOKENS
        if self.phase == 1:
            target = rec.get("user_text", "") or ""
        else:
            spans = rec.get("spoken_spans", [])
            target = " ".join(s["text"] for s in spans if s.get("text"))
        if not target.strip():
            return []  # nothing to say (the silent back-channel case)
        text = f"{sp['spoken_bos']} {target} {sp['spoken_eos']}"
        return self.tok(text, add_special_tokens=False)["input_ids"]

    # ------------------------------------------------------------------ #
    def __getitem__(self, i):
        rec = self.records[i]
        enc_path = rec.get("encoded_path")
        T = int(rec["num_frames"])

        # audio streams (fall back to zeros if the shard was built dry, e.g. samples)
        if enc_path and Path(enc_path).exists():
            d = np.load(enc_path)
            cb0 = d["cb0"].astype(np.int64)[:T]
            energy = d["energy"].astype(np.float32)[:T]
            f0 = d["f0"].astype(np.float32)[:T]
        else:
            cb0 = np.zeros(T, dtype=np.int64)
            energy = np.zeros(T, dtype=np.float32)
            f0 = np.zeros(T, dtype=np.float32)
        cb0 = _pad_or_trim(cb0, T, 0)
        energy = _pad_or_trim(energy, T, 0.0)
        f0 = _pad_or_trim(f0, T, 0.0)
        prosody = np.stack([energy, _norm_f0(f0)], axis=-1)     # [T, 2]

        flags = np.array(rec["flags"], dtype=np.int64)[:T]
        flags = _pad_or_trim(flags, T, vocab.CONTROL_FLAG_TO_ID[vocab.DEFAULT_FLAG])
        agent_state = np.array(rec["agent_state"], dtype=np.int64)[:T]
        agent_state = _pad_or_trim(agent_state, T, vocab.AGENT_STATE_TO_ID["IDLE"])

        # VAP targets recomputed from stored speaking info
        speaking = np.array(rec.get("speaking_mask", [1] * T), dtype=np.float32)[:T]
        speaking = _pad_or_trim(speaking, T, 0.0)
        vap = _vap_from_speaking(speaking, self.vap_horizon)

        text_ids, text_seg = self._encode_text(rec)
        spoken_ids = self._spoken_tail(rec)

        return {
            "text_ids": text_ids,
            "text_seg": text_seg,
            "cb0": cb0,
            "prosody": prosody,
            "agent_state": agent_state,
            "flags": flags,
            "vap": vap,
            "spoken_ids": spoken_ids,
        }


# --------------------------------------------------------------------------- #
def collate(batch, pad_id: int, phase: int = 2):
    """Pad the three segments to batch maxima and assemble tensors + labels."""
    Lt = max(len(b["text_ids"]) for b in batch)
    T = max(len(b["cb0"]) for b in batch)
    S = max((len(b["spoken_ids"]) for b in batch), default=0)
    S = max(S, 1)  # keep a tail slot even if all-empty
    B = len(batch)

    text_ids = np.full((B, Lt), pad_id, dtype=np.int64)
    text_seg = np.zeros((B, Lt), dtype=np.int64)
    text_mask = np.zeros((B, Lt), dtype=np.int64)

    cb0 = np.zeros((B, T), dtype=np.int64)
    prosody = np.zeros((B, T, 2), dtype=np.float32)
    agent_state = np.zeros((B, T), dtype=np.int64)
    audio_mask = np.zeros((B, T), dtype=np.int64)
    flags = np.full((B, T), vocab.CONTROL_FLAG_TO_ID[vocab.DEFAULT_FLAG], dtype=np.int64)
    Hv = batch[0]["vap"].shape[-1]
    vap = np.zeros((B, T, Hv), dtype=np.float32)

    spoken_ids = np.full((B, S), pad_id, dtype=np.int64)
    spoken_mask = np.zeros((B, S), dtype=np.int64)

    for i, b in enumerate(batch):
        lt = len(b["text_ids"]); t = len(b["cb0"]); s = len(b["spoken_ids"])
        text_ids[i, :lt] = b["text_ids"]
        text_seg[i, :lt] = b["text_seg"]
        text_mask[i, :lt] = 1
        cb0[i, :t] = b["cb0"]
        prosody[i, :t] = b["prosody"]
        agent_state[i, :t] = b["agent_state"]
        audio_mask[i, :t] = 1
        flags[i, :t] = b["flags"]
        vap[i, :t] = b["vap"]
        if s:
            spoken_ids[i, :s] = b["spoken_ids"]
            spoken_mask[i, :s] = 1

    # labels over the FULL sequence: -100 everywhere except the spoken tail
    L_total = Lt + T + S
    labels = np.full((B, L_total), -100, dtype=np.int64)
    tail_off = Lt + T
    for i, b in enumerate(batch):
        s = len(b["spoken_ids"])
        if s:
            labels[i, tail_off:tail_off + s] = b["spoken_ids"]

    out = {
        "text_ids": torch.from_numpy(text_ids),
        "text_seg": torch.from_numpy(text_seg),
        "text_mask": torch.from_numpy(text_mask),
        "cb0": torch.from_numpy(cb0),
        "prosody": torch.from_numpy(prosody),
        "agent_state": torch.from_numpy(agent_state),
        "audio_mask": torch.from_numpy(audio_mask),
        "flags": torch.from_numpy(flags),
        "vap": torch.from_numpy(vap),
        "spoken_ids": torch.from_numpy(spoken_ids),
        "spoken_mask": torch.from_numpy(spoken_mask),
    }
    # Both phases reuse the same masked-LM tail; expose it under both names so the
    # phase-specific loss (Phase1Loss reads align_labels, Phase2Loss reads
    # spoken_labels) always finds its key.
    labels_t = torch.from_numpy(labels)
    out["align_labels"] = labels_t
    out["spoken_labels"] = labels_t
    return out


def make_collate(pad_id: int, phase: int = 2):
    def _fn(batch):
        return collate(batch, pad_id=pad_id, phase=phase)
    return _fn


# --------------------------------------------------------------------------- #
def _pad_or_trim(x: np.ndarray, n: int, fill):
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    pad = np.full((n - len(x),) + x.shape[1:], fill, dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


def _norm_f0(f0: np.ndarray) -> np.ndarray:
    """Log-normalise voiced f0 to a stable range; unvoiced (0) -> 0."""
    out = np.zeros_like(f0)
    voiced = f0 > 1.0
    out[voiced] = (np.log(f0[voiced]) - np.log(150.0))  # centre ~150 Hz
    return out.astype(np.float32)


def _vap_from_speaking(speaking: np.ndarray, H: int) -> np.ndarray:
    T = len(speaking)
    vap = np.zeros((T, H), dtype=np.float32)
    for t in range(T):
        chunk = speaking[t + 1:min(t + 1 + H, T)]
        vap[t, :len(chunk)] = chunk
    return vap
