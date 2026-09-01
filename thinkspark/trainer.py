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
import warnings
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Silence two purely-cosmetic, third-party deprecation warnings that spam the training
# log and aren't actionable from here: torch's own gradient-checkpointing path still
# calls the deprecated `torch.cpu.amp.autocast` internally, and transformers prints a
# torch_dtype deprecation from deep in from_pretrained on some versions. Our code is
# already on the new APIs (see model.py); these come from inside the libraries.
warnings.filterwarnings("ignore", message=r".*torch\.cpu\.amp\.autocast.*")
warnings.filterwarnings("ignore", message=r".*torch_dtype.*deprecated.*")

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

        # TF32 matmuls — a free ~1.3-2x speedup for bf16/fp32 matmuls on every Ampere+
        # GPU (A100/L4/H100/H200/RTX6000/Blackwell); a no-op on older/other hardware.
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

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
            attn_implementation=getattr(cfg, "attn_implementation", "sdpa"),
        )
        model.resize_token_embeddings(len(self.tok))
        if cfg.use_lora:
            model = apply_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha,
                               dropout=cfg.lora_dropout)
        self._eager_model = model.to(self.device)   # kept for the runtime fallback below
        self.model = self._eager_model
        if getattr(cfg, "compile", False):
            # torch.compile: large speedup on H100/H200/Blackwell/RTX6000 (fuses kernels),
            # modest on older cards. Real observed gap this closes: `torch.compile()`
            # itself basically never raises (it's lazy) — the actual failure surfaces
            # later, mid-forward-pass under Dynamo tracing on the first real batch (hit in
            # practice: a transformers/PEFT + output_hidden_states incompatibility raised
            # a `KeyError` deep inside Dynamo's graph-resume, not at compile() time at
            # all). `train()`'s first step now runs through `self._forward_with_fallback`,
            # which catches exactly that and permanently switches back to the eager model
            # instead of crashing the whole run.
            self.model = torch.compile(self._eager_model)
            if self._is_main:
                print("  torch.compile ON (first step compiles + runtime-verified, then runs fast; "
                     "auto-falls back to eager if compilation breaks on the first real batch)")
        if self.ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank])

        # phase loss
        if cfg.phase == 1:
            self.loss_fn = Phase1Loss(lambda_vap=cfg.lambda_vap,
                                      label_smoothing=getattr(cfg, "label_smoothing", 0.0))
        else:
            alpha = class_alpha_from_freq(_class_frequency(shard_paths))
            self.loss_fn = Phase2Loss(alpha=alpha, gamma=cfg.focal_gamma,
                                      l_ctrl=cfg.lambda_ctrl, l_txt=cfg.lambda_txt,
                                      l_vap=cfg.lambda_vap_p2,
                                      label_smoothing=getattr(cfg, "label_smoothing", 0.0))
        self.loss_fn = self.loss_fn.to(self.device)

        # data — split into train / val / test (val + test held OUT of training so their
        # loss is a real generalization signal, not memorized). Deterministic split via a
        # seeded generator so the same held-out sets are used across resumes/DDP ranks.
        full_ds = ThinkSparkDataset(shard_paths, self.tok, phase=cfg.phase,
                                    seq_len=cfg.seq_len, vap_horizon=cfg.vap_horizon)
        n_total = len(full_ds)
        n_val = int(n_total * getattr(cfg, "val_frac", 0.0))
        n_test = int(n_total * getattr(cfg, "test_frac", 0.0))
        n_train = n_total - n_val - n_test
        if n_train <= 0:
            raise SystemExit(f"val_frac+test_frac too large for {n_total} samples")
        gen = torch.Generator().manual_seed(cfg.seed)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val, n_test], generator=gen)
        if self._is_main:
            print(f"data split: train={n_train} val={n_val} test={n_test} "
                  f"(of {n_total} samples)")

        collate = make_collate(self.tok.pad_token_id, cfg.phase)
        nw = getattr(cfg, "num_workers", 2)
        pin = self.device.startswith("cuda")   # pinned host memory -> faster H2D copies
        loader_kw = dict(collate_fn=collate, num_workers=nw, pin_memory=pin,
                        persistent_workers=(nw > 0))
        sampler = (torch.utils.data.distributed.DistributedSampler(train_ds)
                   if self.ddp else None)
        self.loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=(sampler is None),
            sampler=sampler, drop_last=True, **loader_kw,
        )
        # eval loaders run on the main process only (no DDP sampler) — no shuffle,
        # no drop_last so every held-out sample counts.
        self.val_loader = (DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                      drop_last=False, **loader_kw)
                           if n_val > 0 else None)
        self.test_loader = (DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                                       drop_last=False, **loader_kw)
                            if n_test > 0 else None)

        # optimiser — split params into decay / no-decay groups (the standard LLM /
        # Moshi practice): weight decay applies to matmul weights only, NOT to biases,
        # LayerNorm/RMSNorm scales, or embeddings — decaying those hurts more than it
        # regularizes. Betas default to (0.9, 0.95) per cfg (LLM-pretraining default).
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2 or n.endswith(".bias") or "norm" in n.lower() or "embed" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        self.opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr,
            betas=(getattr(cfg, "adam_beta1", 0.9), getattr(cfg, "adam_beta2", 0.95)),
            eps=getattr(cfg, "adam_eps", 1e-8),
        )
        if self._is_main:
            print(f"AdamW: {len(decay)} decay / {len(no_decay)} no-decay param tensors, "
                  f"betas=({getattr(cfg,'adam_beta1',0.9)}, {getattr(cfg,'adam_beta2',0.95)})")
        self.steps_per_epoch = max(1, len(self.loader) // cfg.grad_accum)
        self.total_steps = self.steps_per_epoch * cfg.epochs
        self.warmup = int(self.total_steps * cfg.warmup_ratio)
        self.use_bf16 = (cfg.precision == "bf16" and self.device.startswith("cuda"))

        self.out_dir = Path(cfg.out_dir)
        if self._is_main:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self._loss_hist: list[float] = []

        # Resume + upload hooks (set by the training script; both default to inert so
        # nothing changes for callers that don't use them).
        # on_checkpoint(tag, ckpt_dir) is called right after every save() on the main
        # process — the training script uses it to upload each checkpoint to HF live,
        # during training, so an interrupted run can be resumed from HF.
        self.on_checkpoint = None
        self._start_epoch = 0    # set by load_checkpoint() to resume mid-run
        self._start_step = 0
        self.run_id = None       # set by thinkspark.train_runs.wire_run (used as wandb name)
        self._wandb = None       # lazily initialized in train() on the main process

        # best-val checkpoint + early-stop bookkeeping (see _track_best / config knobs)
        self._best_val = float("inf")
        self._best_step = -1
        self._evals_no_improve = 0
        self._should_stop = False

    # ------------------------------------------------------------------ #
    def _init_wandb(self):
        """Start a Weights & Biases run if configured (main process only). No-op if
        wandb isn't installed or no project is set — training continues either way."""
        cfg = self.cfg
        project = getattr(cfg, "wandb_project", None) or os.environ.get("WANDB_PROJECT")
        if not project or not self._is_main:
            return
        try:
            import wandb
        except ImportError:
            print("  wandb not installed — skipping W&B logging (`pip install wandb`)")
            return
        try:
            self._wandb = wandb.init(
                project=project,
                entity=getattr(cfg, "wandb_entity", None) or os.environ.get("WANDB_ENTITY"),
                name=getattr(cfg, "wandb_run_name", None) or self.run_id,
                config=cfg.__dict__,
                resume="allow",
                id=(self.run_id.replace(":", "-") if self.run_id else None),
            )
            print(f"  W&B logging ON -> project '{project}' run '{self._wandb.name}'")
        except Exception as e:
            print(f"  ! wandb.init failed ({e}) — continuing without W&B logging")
            self._wandb = None

    def _wandb_log(self, metrics: dict, step: int):
        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=step)
            except Exception:
                pass   # never let a logging hiccup interrupt training

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, loader, split: str, step: int) -> dict:
        """Average every loss component over a held-out loader (no grad, eval mode, no
        frame-drop aug). Logs `<split>/<component>` to W&B and returns the metrics."""
        if loader is None:
            return {}
        was_training = self.model.training
        self.model.eval()
        sums: dict[str, float] = {}
        n = 0
        for batch in loader:
            batch = self._to_device(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                out = self.model(**self._model_inputs(batch))
                losses = self.loss_fn(out, batch)
            for k, v in losses.items():
                if hasattr(v, "item"):
                    sums[k] = sums.get(k, 0.0) + float(v.item())
            n += 1
        if was_training:
            self.model.train()
        if n == 0:
            return {}
        avg = {k: v / n for k, v in sums.items()}
        pretty = " ".join(f"{k}={v:.4f}" for k, v in avg.items())
        print(f"  [eval:{split}] step {step}: {pretty}")
        self._wandb_log({f"{split}/{k}": v for k, v in avg.items()}, step)
        return avg

    # ------------------------------------------------------------------ #
    def _track_best(self, val_avg: dict, step: int) -> None:
        """Given a just-computed validation metric dict, (re)write the `best/` checkpoint
        when val loss improves and drive early stopping. Main process only. No-op if the
        val loader was empty (val_avg has no 'loss')."""
        if not self._is_main or not val_avg or "loss" not in val_avg:
            return
        cfg = self.cfg
        val_loss = float(val_avg["loss"])
        min_delta = getattr(cfg, "early_stop_min_delta", 0.0)
        improved = val_loss < (self._best_val - min_delta)
        if improved:
            self._best_val = val_loss
            self._best_step = step
            self._evals_no_improve = 0
            if getattr(cfg, "save_best", True):
                self.save("best")
                print(f"  ↳ new best val loss {val_loss:.4f} @ step {step} — saved best/")
            self._wandb_log({"val/best_loss": val_loss, "val/best_step": step}, step)
        else:
            self._evals_no_improve += 1
            patience = getattr(cfg, "early_stop_patience", 0)
            if patience > 0:
                print(f"  ↳ no val improvement ({self._evals_no_improve}/{patience}); "
                      f"best {self._best_val:.4f} @ step {self._best_step}")
                if self._evals_no_improve >= patience:
                    print(f"  ↳ early stopping: {patience} evals without improvement. "
                          f"Best checkpoint is best/ (step {self._best_step}).")
                    self._should_stop = True

    # ------------------------------------------------------------------ #
    def load_checkpoint(self, ckpt_dir: Path) -> None:
        """Restore model + optimizer weights and the (epoch, step) position from a
        checkpoint dir written by save(), so train() continues from there instead of
        from scratch. Model weights and optimizer state continue exactly; the resumed
        epoch's data ordering restarts (a shuffled streaming loader can't be
        deterministically fast-forwarded), but the LR schedule and global step counter
        continue correctly since they key off the restored global step."""
        model = self.model.module if self.ddp else self.model
        state = torch.load(ckpt_dir / "model.pt", map_location=self.device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        opt_path = ckpt_dir / "opt.pt"
        if opt_path.exists():
            try:
                self.opt.load_state_dict(torch.load(opt_path, map_location=self.device))
            except Exception as e:
                print(f"  ! couldn't restore optimizer state ({e}) — continuing with a "
                      f"fresh optimizer (weights still resumed)")
        ts_path = ckpt_dir / "trainer_state.json"
        if ts_path.exists():
            ts = json.loads(ts_path.read_text())
            self._start_epoch = int(ts.get("epoch", 0))
            self._start_step = int(ts.get("step", 0))
        print(f"  resumed from {ckpt_dir} — epoch {self._start_epoch}, step "
              f"{self._start_step} (missing={len(missing)}, unexpected={len(unexpected)})")

    @property
    def _is_main(self) -> bool:
        return (not self.ddp) or self.rank == 0

    def _forward_with_fallback(self, model_inputs: dict):
        """Runs the model, and — ONLY on the very first call, when `self.model` is a
        torch.compile-wrapped model — catches a Dynamo/compile runtime failure and
        permanently swaps back to the eager model instead of crashing the whole run.
        After the first successful (or recovered) call, this is just `self.model(...)`
        with no extra overhead."""
        if not self._compile_checked and self.model is not self._eager_model:
            self._compile_checked = True
            try:
                return self.model(**model_inputs)
            except Exception as e:
                if self._is_main:
                    print(f"  ! torch.compile broke on the first real batch ({e}) — "
                          f"falling back to eager for the rest of this run")
                self.model = self._eager_model
                self.model.train()
                return self.model(**model_inputs)
        self._compile_checked = True
        return self.model(**model_inputs)

    # ------------------------------------------------------------------ #
    def train(self):
        cfg = self.cfg
        self._init_wandb()
        self._compile_checked = False
        step = self._start_step         # 0 unless resumed via load_checkpoint()
        self._cur_step = step
        self.model.train()
        self._ema_step_ms = None        # EMA of wall-clock time per optimizer step
        t_prev = time.perf_counter()
        for epoch in range(self._start_epoch, cfg.epochs):
            self._cur_epoch = epoch
            if self.ddp and self.loader.sampler is not None:
                self.loader.sampler.set_epoch(epoch)
            self.opt.zero_grad()
            accum: dict[str, float] = {}   # running sum of true per-micro-batch losses
            n_accum = 0                     # over the current grad_accum window
            for it, batch in enumerate(self.loader):
                batch = self._to_device(batch)
                batch = self._frame_drop_aug(batch)

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                    out = self._forward_with_fallback(self._model_inputs(batch))
                    losses = self.loss_fn(out, batch)
                    loss = losses["loss"] / cfg.grad_accum

                loss.backward()
                for k, v in losses.items():
                    if hasattr(v, "item"):
                        accum[k] = accum.get(k, 0.0) + float(v.item())
                n_accum += 1
                if (it + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    lr = _cosine_warmup(step, self.total_steps, self.warmup, cfg.lr)
                    for g in self.opt.param_groups:
                        g["lr"] = lr
                    self.opt.step()
                    self.opt.zero_grad()
                    step += 1
                    self._cur_step = step

                    # wall-clock time for THIS optimizer step (fwd+bwd+step+data wait),
                    # smoothed into an EMA. Kept as pure step time by resetting t_prev
                    # after eval/save below, so their pauses don't inflate it.
                    now = time.perf_counter()
                    dt_ms = (now - t_prev) * 1000.0
                    t_prev = now
                    self._ema_step_ms = (dt_ms if self._ema_step_ms is None
                                         else 0.9 * self._ema_step_ms + 0.1 * dt_ms)

                    if self._is_main and step % cfg.log_every == 0:
                        # TRUE effective-batch loss: mean over all grad_accum micro-batches
                        # (~64 samples), not one 4-sample micro-batch x grad_accum (the old
                        # reporting inflated the number by grad_accum AND was single-batch
                        # noisy — that's why the loss looked like it bounced 10<->29 when it
                        # was really ~0.6<->1.8 and trending down).
                        self._log(epoch, step, {k: v / n_accum for k, v in accum.items()}, lr)
                    accum, n_accum = {}, 0
                    if self._is_main and step % getattr(cfg, "eval_every", 500) == 0:
                        val_avg = self.evaluate(self.val_loader, "val", step)
                        self._track_best(val_avg, step)
                        t_prev = time.perf_counter()   # don't count eval as step time
                    if self._is_main and step % cfg.save_every == 0:
                        self.save(f"step{step}")
                        t_prev = time.perf_counter()   # don't count save/upload as step time
                if self._should_stop:
                    break
            # end-of-epoch validation summary
            if self._is_main:
                val_avg = self.evaluate(self.val_loader, "val", step)
                self._track_best(val_avg, step)
            if self._should_stop:
                break
        if self._is_main:
            self.save("final")
            if self._best_step >= 0:
                print(f"\n  ** BEST checkpoint: {self.out_dir / 'best'} "
                      f"(val loss {self._best_val:.4f} @ step {self._best_step}). "
                      f"Evaluate THIS, not final/. **")
            # final held-out TEST eval — the honest generalization number
            self.evaluate(self.test_loader, "test", self._cur_step)
            self._join_uploads()   # let any background checkpoint uploads finish
            if self._wandb is not None:
                self._wandb.finish()
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

    def _log(self, epoch, step, avg: dict, lr):
        """avg: float dict of the true losses averaged over the last grad_accum window."""
        total = avg.get("loss", 0.0)
        parts = " ".join(f"{k}={v:.4f}" for k, v in avg.items() if k != "loss")
        self._loss_hist.append(total)
        ms = getattr(self, "_ema_step_ms", None)
        tstr = ""
        if ms:
            sps = 1000.0 / ms if ms > 0 else 0.0
            eta_s = (self.total_steps - step) * ms / 1000.0
            tstr = f" {ms:.0f}ms/step ({sps:.2f} it/s, eta {eta_s/60:.0f}m)"
        print(f"[P{self.cfg.phase}] epoch {epoch} step {step}/{self.total_steps} "
              f"lr={lr:.2e} loss={total:.4f} {parts}{tstr}")
        _maybe_plot(self._loss_hist)
        # W&B: true train loss + every component + lr + epoch, charted live.
        metrics = {"train/loss": total, "train/lr": lr, "train/epoch": epoch}
        if ms:
            metrics["train/ms_per_step"] = ms
        for k, v in avg.items():
            if k != "loss":
                metrics[f"train/{k}"] = v
        self._wandb_log(metrics, step)

    def save(self, tag: str):
        model = self.model.module if self.ddp else self.model
        ckpt_dir = self.out_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        # optimizer state (opt.pt, ~2x the model = the biggest file) is ONLY needed to
        # RESUME a run — and you never resume from best/ (you resume from the latest
        # step*/final, you *evaluate*/serve best/). Writing + live-uploading it for best/
        # doubled every best-save's upload for zero benefit, which is what made training
        # pause after each improvement. So skip it for best/.
        save_opt = tag != "best"
        if save_opt:
            torch.save(self.opt.state_dict(), ckpt_dir / "opt.pt")
        (ckpt_dir / "trainer_state.json").write_text(
            json.dumps({"epoch": getattr(self, "_cur_epoch", 0),
                       "step": getattr(self, "_cur_step", 0),
                       "tag": tag}, indent=2), encoding="utf-8")
        self.tok.save_pretrained(ckpt_dir)
        (ckpt_dir / "config.json").write_text(
            json.dumps(self.cfg.__dict__, indent=2), encoding="utf-8")
        print(f"  saved checkpoint -> {ckpt_dir}")
        # Live upload hook (set by the training script) — runs on the main process only
        # (save() is only ever called under `if self._is_main`). A failure here must NOT
        # kill training: the local checkpoint is already safely on disk, and the next
        # save's upload (or a manual push later) will catch up.
        #
        # Runs in a BACKGROUND daemon thread so training does NOT block on the upload (the
        # old synchronous call paused the loop for the full 1.6 GB transfer after every
        # save). A per-tag lock serializes uploads of the SAME tag (so two rapid best/
        # saves can't race the same repo path) while letting training race ahead. On exit,
        # train() joins outstanding uploads so nothing is lost.
        if self.on_checkpoint is not None:
            self._spawn_upload(tag, ckpt_dir)

    def _spawn_upload(self, tag: str, ckpt_dir) -> None:
        import threading
        if not hasattr(self, "_upload_threads"):
            self._upload_threads = []
            self._upload_locks = {}
        lock = self._upload_locks.setdefault(tag, threading.Lock())

        def _worker():
            with lock:   # serialize same-tag uploads; different tags upload concurrently
                try:
                    self.on_checkpoint(tag, ckpt_dir)
                except Exception as e:
                    print(f"  ! checkpoint upload hook failed for {tag} (training "
                          f"continues, local checkpoint is safe): {e}")

        t = threading.Thread(target=_worker, name=f"upload-{tag}", daemon=True)
        t.start()
        # reap finished threads so the list doesn't grow unbounded
        self._upload_threads = [x for x in self._upload_threads if x.is_alive()]
        self._upload_threads.append(t)

    def _join_uploads(self) -> None:
        for t in getattr(self, "_upload_threads", []):
            if t.is_alive():
                print(f"  waiting for background upload {t.name} to finish ...")
                t.join()


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
