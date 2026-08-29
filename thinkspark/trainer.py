"""
Shared training loop for Phase-1 (alignment) and Phase-2 (referee).

Kept deliberately small and readable — it wires together the model, dataset, phase loss,
AdamW + cosine-warmup schedule, bf16 autocast, gradient accumulation and checkpointing
exactly to the Section 9.3/9.4 recipe that fits Kaggle 2xT4 (16 GB each).

DDP across 2xT4 is opt-in via `TrainConfig.ddp` + `torchrun` (Section 9.3). Single-GPU
and CPU/MPS also work (slower). Live loss curves are drawn in the terminal with plotext
when available (nice on Kaggle), else a plain running log.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from thinkspark import vocab
from thinkspark.config import TrainConfig
from thinkspark.dataset import ThinkSparkDataset, build_tokenizer, make_collate
from thinkspark.model import ThinkSparkModel, apply_lora
from thinkspark.losses import Phase1Loss, Phase2Loss, class_alpha_from_freq
from thinkspark.mimi_codec import MimiEncoder


# --------------------------------------------------------------------------- #
def _cosine_warmup(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * prog))


def _class_frequency(shard_paths: list[str]) -> dict[str, int]:
    """Count control-flag frames for focal-loss alpha (Section 9.1)."""
    freq: Counter = Counter()
    for p in shard_paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for fid in rec.get("flags", []):
                freq[vocab.ID_TO_CONTROL_FLAG[int(fid)]] += 1
    return dict(freq)


def _codebook_size(cfg: TrainConfig) -> int:
    """Best-effort Mimi codebook size (for the audio embedding table)."""
    try:
        return MimiEncoder(repo="kyutai/mimi").codebook_size
    except Exception:
        return 2048  # Mimi default


# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, cfg: TrainConfig, shard_paths: list[str]):
        self.cfg = cfg
        self.shards = shard_paths
        torch.manual_seed(cfg.seed)

        self.ddp = cfg.ddp and int(os.environ.get("WORLD_SIZE", "1")) > 1
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.ddp:
            torch.distributed.init_process_group("nccl")
            torch.cuda.set_device(self.local_rank)
            self.device = f"cuda:{self.local_rank}"
        else:
            self.device = ("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")

        hf_token = os.environ.get(cfg.hf_token_env)
        self.tok = build_tokenizer(cfg.base_model, hf_token=hf_token)

        model = ThinkSparkModel(
            base_model=cfg.base_model,
            codebook_size=_codebook_size(cfg),
            vap_horizon=cfg.vap_horizon,
            hf_token=hf_token,
            gradient_checkpointing=cfg.grad_checkpointing,
        )
        model.resize_token_embeddings(len(self.tok))
        if cfg.use_lora:
            model = apply_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha,
                               dropout=cfg.lora_dropout)
        self.model = model.to(self.device)
        if self.ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank])

        # phase loss
        if cfg.phase == 1:
            self.loss_fn = Phase1Loss(lambda_vap=cfg.lambda_vap)
        else:
            alpha = class_alpha_from_freq(_class_frequency(shard_paths))
            self.loss_fn = Phase2Loss(alpha=alpha, gamma=cfg.focal_gamma,
                                      l_ctrl=cfg.lambda_ctrl, l_txt=cfg.lambda_txt,
                                      l_vap=cfg.lambda_vap_p2)
        self.loss_fn = self.loss_fn.to(self.device)

        # data
        ds = ThinkSparkDataset(shard_paths, self.tok, phase=cfg.phase,
                               seq_len=cfg.seq_len, vap_horizon=cfg.vap_horizon)
        sampler = (torch.utils.data.distributed.DistributedSampler(ds)
                   if self.ddp else None)
        self.loader = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=(sampler is None),
            sampler=sampler, collate_fn=make_collate(self.tok.pad_token_id, cfg.phase),
            num_workers=2, drop_last=True,
        )

        # optimiser
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.steps_per_epoch = max(1, len(self.loader) // cfg.grad_accum)
        self.total_steps = self.steps_per_epoch * cfg.epochs
        self.warmup = int(self.total_steps * cfg.warmup_ratio)
        self.use_bf16 = (cfg.precision == "bf16" and self.device.startswith("cuda"))

        self.out_dir = Path(cfg.out_dir)
        if self._is_main:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self._loss_hist: list[float] = []

    @property
    def _is_main(self) -> bool:
        return (not self.ddp) or self.rank == 0

    # ------------------------------------------------------------------ #
    def train(self):
        cfg = self.cfg
        step = 0
        self.model.train()
        for epoch in range(cfg.epochs):
            if self.ddp and self.loader.sampler is not None:
                self.loader.sampler.set_epoch(epoch)
            self.opt.zero_grad()
            for it, batch in enumerate(self.loader):
                batch = self._to_device(batch)
                batch = self._frame_drop_aug(batch)

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                    out = self.model(**self._model_inputs(batch))
                    losses = self.loss_fn(out, batch)
                    loss = losses["loss"] / cfg.grad_accum

                loss.backward()
                if (it + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    lr = _cosine_warmup(step, self.total_steps, self.warmup, cfg.lr)
                    for g in self.opt.param_groups:
                        g["lr"] = lr
                    self.opt.step()
                    self.opt.zero_grad()
                    step += 1

                    if self._is_main and step % cfg.log_every == 0:
                        self._log(epoch, step, losses, lr)
                    if self._is_main and step % cfg.save_every == 0:
                        self.save(f"step{step}")
        if self._is_main:
            self.save("final")
        if self.ddp:
            torch.distributed.destroy_process_group()

    # ------------------------------------------------------------------ #
    def _model_inputs(self, batch: dict) -> dict:
        keys = ["text_ids", "text_seg", "text_mask", "cb0", "prosody",
                "agent_state", "audio_mask", "spoken_ids", "spoken_mask"]
        return {k: batch[k] for k in keys if k in batch}

    def _to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch.items()}

    def _frame_drop_aug(self, batch: dict) -> dict:
        """5% random frame drop-augmentation for robustness (Section 9.4)."""
        p = self.cfg.frame_drop_aug
        if p <= 0:
            return batch
        mask = batch["audio_mask"]
        drop = (torch.rand_like(mask.float()) < p) & (mask.bool())
        batch["audio_mask"] = mask.masked_fill(drop, 0)
        return batch

    def _log(self, epoch, step, losses, lr):
        parts = " ".join(f"{k}={v.item():.4f}" for k, v in losses.items()
                         if k != "loss" and hasattr(v, "item"))
        total = losses["loss"].item() * self.cfg.grad_accum
        self._loss_hist.append(total)
        print(f"[P{self.cfg.phase}] epoch {epoch} step {step}/{self.total_steps} "
              f"lr={lr:.2e} loss={total:.4f} {parts}")
        _maybe_plot(self._loss_hist)

    def save(self, tag: str):
        model = self.model.module if self.ddp else self.model
        ckpt_dir = self.out_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        self.tok.save_pretrained(ckpt_dir)
        (ckpt_dir / "config.json").write_text(
            json.dumps(self.cfg.__dict__, indent=2), encoding="utf-8")
        print(f"  saved checkpoint -> {ckpt_dir}")


def _maybe_plot(hist: list[float]):
    try:
        import plotext as plt
        if len(hist) < 2:
            return
        plt.clf()
        plt.plot(hist[-200:])
        plt.title("training loss")
        plt.plotsize(60, 12)
        plt.show()
    except Exception:
        pass
