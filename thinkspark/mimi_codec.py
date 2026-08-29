"""
Mimi encode -> cb0 audio tokens + prosody (energy, f0) at 12.5 Hz (Section 4.2, Phase 0).

Phase 0 (offline) converts every wav (open Phase-1 corpora + Soniox Phase-2 user audio)
into three per-frame streams saved to disk:

    cb0     : int64 [T]   Mimi codebook-0 (semantic) token id per 80 ms frame
    energy  : float32 [T] per-frame RMS energy (log-compressed, z-scored later)
    f0      : float32 [T] per-frame fundamental frequency in Hz (0 = unvoiced)

Only codebook 0 is kept (cb0) — it carries language + prosody cues, which is all the
referee needs, and keeps sequences short (Section 4.2). Energy/f0 give the model the
prosodic hooks for endpointing and back-channel timing.

We use the HF `transformers` MimiModel (repo `kyutai/mimi`) so no extra codec dependency
is required. f0 is extracted with torchaudio's pitch detector (a light, dependency-free
fallback is provided if torchaudio is unavailable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thinkspark import vocab

_MIMI_RATE = 24000        # Mimi operates at 24 kHz
_HOP = int(_MIMI_RATE / vocab.FRAME_RATE_HZ)   # 1920 samples per 80 ms frame


@dataclass
class EncodedAudio:
    cb0: np.ndarray        # int64 [T]
    energy: np.ndarray     # float32 [T]
    f0: np.ndarray         # float32 [T]
    num_frames: int

    def save(self, path) -> None:
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, cb0=self.cb0, energy=self.energy, f0=self.f0)

    @staticmethod
    def load(path) -> "EncodedAudio":
        d = np.load(path)
        cb0 = d["cb0"].astype(np.int64)
        return EncodedAudio(cb0=cb0, energy=d["energy"].astype(np.float32),
                            f0=d["f0"].astype(np.float32), num_frames=len(cb0))


class MimiEncoder:
    """Lazy-loaded Mimi model; encode a wav path or waveform to EncodedAudio."""

    def __init__(self, repo: str = "kyutai/mimi", device: str | None = None):
        self.repo = repo
        self._device = device
        self._model = None
        self._fe = None
        self._codebook_size: int | None = None

    # ------------------------------------------------------------------ #
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import os

        import torch
        from transformers import MimiModel, AutoFeatureExtractor

        self._torch = torch
        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = dev
        if dev == "cpu":
            # By default torch uses ALL CPU cores for its own internal intra-op thread
            # pool. If something else in the same process ALSO does CPU-bound torch work
            # concurrently (e.g. scripts/P1_00_pipeline.py's download threads decoding
            # audio via torchcodec, which is torch-based too), that other work can get
            # starved of real CPU cycles by this model's inference alone, independent of
            # — and in addition to — ordinary Python GIL contention. Cap it at half the
            # cores so there's real headroom left for concurrent torch work elsewhere in
            # the same process; encoding a single short clip doesn't need every core.
            torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        self._model = MimiModel.from_pretrained(self.repo).to(dev).eval()
        self._fe = AutoFeatureExtractor.from_pretrained(self.repo)
        # codebook size from config (used to size the model's audio embedding table)
        self._codebook_size = int(getattr(self._model.config, "codebook_size", 2048))

    @property
    def codebook_size(self) -> int:
        self._ensure_loaded()
        return int(self._codebook_size)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    def encode_waveform(self, wav: np.ndarray, sample_rate: int) -> EncodedAudio:
        """Encode a mono float32 waveform in [-1, 1]."""
        self._ensure_loaded()
        torch = self._torch

        wav = _to_mono_float(wav)
        if sample_rate != _MIMI_RATE:
            wav = _resample(wav, sample_rate, _MIMI_RATE)

        inputs = self._fe(raw_audio=wav, sampling_rate=_MIMI_RATE, return_tensors="pt")
        input_values = inputs["input_values"].to(self._device)
        with torch.no_grad():
            enc = self._model.encode(input_values)
        # audio_codes: [B, num_codebooks, T] -> take codebook 0
        codes = enc.audio_codes[0]                 # [num_codebooks, T]
        cb0 = codes[0].detach().cpu().numpy().astype(np.int64)  # [T]

        energy, f0 = _prosody(wav, _MIMI_RATE, num_frames=len(cb0))
        return EncodedAudio(cb0=cb0, energy=energy, f0=f0, num_frames=len(cb0))

    def encode_wav_file(self, wav_path: str) -> EncodedAudio:
        wav, sr = _read_wav(wav_path)
        return self.encode_waveform(wav, sr)


# --------------------------------------------------------------------------- #
# prosody + io helpers (kept torch-free where possible)
# --------------------------------------------------------------------------- #
def _prosody(wav: np.ndarray, sr: int, num_frames: int):
    """Per-frame log-RMS energy and f0 (Hz), aligned to Mimi's frame grid."""
    hop = _HOP
    energy = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        seg = wav[i * hop:(i + 1) * hop]
        if seg.size:
            energy[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-9)
    energy = np.log(energy + 1e-6).astype(np.float32)

    f0 = _estimate_f0(wav, sr, num_frames, hop)
    return energy, f0


def _estimate_f0(wav: np.ndarray, sr: int, num_frames: int, hop: int) -> np.ndarray:
    """Prefer torchaudio's detect_pitch_frequency; fall back to autocorrelation."""
    try:
        import torch
        import torchaudio.functional as AF

        t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        pitch = AF.detect_pitch_frequency(t, sr).squeeze(0).numpy()
        # resample pitch frames to our frame count
        return _resize_1d(pitch.astype(np.float32), num_frames)
    except Exception:
        return _autocorr_f0(wav, sr, num_frames, hop)


def _autocorr_f0(wav, sr, num_frames, hop, fmin=70.0, fmax=400.0):
    f0 = np.zeros(num_frames, dtype=np.float32)
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    for i in range(num_frames):
        seg = wav[i * hop:(i + 1) * hop].astype(np.float64)
        if seg.size < max_lag or np.sqrt(np.mean(seg ** 2) + 1e-9) < 1e-3:
            continue
        seg = seg - seg.mean()
        corr = np.correlate(seg, seg, mode="full")[seg.size - 1:]
        if corr[0] <= 0:
            continue
        region = corr[min_lag:max_lag]
        if region.size == 0:
            continue
        lag = int(np.argmax(region)) + min_lag
        if corr[lag] / corr[0] > 0.3:
            f0[i] = sr / lag
    return f0


def _resize_1d(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    idx = np.linspace(0, len(x) - 1, n)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def _to_mono_float(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav)
    if wav.ndim == 2:
        wav = wav.mean(axis=1 if wav.shape[1] <= wav.shape[0] else 0)
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32768.0
    return wav.astype(np.float32)


def _read_wav(path: str):
    import wave
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        ch = wf.getnchannels()
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def _resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav
    n_out = int(round(len(wav) * sr_out / sr_in))
    return _resize_1d(wav, n_out)
