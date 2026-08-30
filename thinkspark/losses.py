"""
Loss functions (Section 9.1) — exact forms from the guide.

Phase 1 (modality alignment + turn-taking):
    L_align  = ASR-style cross-entropy on text given audio  (perplexity = e^L_align)
    L_vap    = BCE over H future 80 ms "is-user-speaking" bins
    L_P1     = L_align + lambda_vap * L_vap        (lambda_vap ~= 0.3)

Phase 2 (referee, handles class imbalance):
    L_ctrl   = focal loss on the control head    (gamma=2, alpha_c ~ 1/sqrt(freq_c))
    L_txt    = cross-entropy on spoken tokens, masked to spoken spans
    L_vap    = same VAP BCE auxiliary
    L_P2     = l1*L_ctrl + l2*L_txt + l3*L_vap    (e.g. (1.0, 0.5, 0.2))
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from thinkspark import vocab


# --------------------------------------------------------------------------- #
def class_alpha_from_freq(freq: dict[str, int]) -> torch.Tensor:
    """alpha_c ∝ 1/sqrt(freq_c), normalised to mean 1 (Section 9.1)."""
    alpha = torch.ones(vocab.NUM_CONTROL_FLAGS, dtype=torch.float32)
    for flag, i in vocab.CONTROL_FLAG_TO_ID.items():
        f = max(1, freq.get(flag, 1))
        alpha[i] = 1.0 / math.sqrt(f)
    alpha = alpha / alpha.mean()
    return alpha


def focal_control_loss(
    logits: torch.Tensor,      # [B, T, C]
    targets: torch.Tensor,     # [B, T]
    mask: torch.Tensor,        # [B, T] (1 = valid frame)
    alpha: torch.Tensor | None = None,   # [C]
    gamma: float = 2.0,
) -> torch.Tensor:
    B, T, C = logits.shape
    logits = logits.reshape(-1, C)
    targets = targets.reshape(-1)
    mask = mask.reshape(-1).bool()
    if mask.sum() == 0:
        return logits.sum() * 0.0

    logits = logits[mask]
    targets = targets[mask]

    logp = F.log_softmax(logits, dim=-1)
    logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)   # [N]
    pt = logpt.exp()
    focal = (1.0 - pt) ** gamma * (-logpt)

    if alpha is not None:
        a = alpha.to(logits.device)[targets]
        focal = a * focal
    return focal.mean()


def vap_bce_loss(
    vap_logits: torch.Tensor,  # [B, T, H]
    vap_targets: torch.Tensor, # [B, T, H]
    mask: torch.Tensor,        # [B, T]
) -> torch.Tensor:
    m = mask.unsqueeze(-1).expand_as(vap_targets).bool()
    if m.sum() == 0:
        return vap_logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(
        vap_logits[m], vap_targets[m], reduction="mean"
    )
    return loss


def spoken_ce_loss(
    lm_logits: torch.Tensor,   # [B, L_total, V]
    labels: torch.Tensor,      # [B, L_total]  (-100 where not a spoken target)
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if (labels != -100).sum() == 0:
        return lm_logits.sum() * 0.0
    # standard shifted LM loss
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )


# --------------------------------------------------------------------------- #
class Phase1Loss(nn.Module):
    """L_P1 = L_align + lambda_vap * L_vap."""

    def __init__(self, lambda_vap: float = 0.3, label_smoothing: float = 0.0):
        super().__init__()
        self.lambda_vap = lambda_vap
        self.label_smoothing = label_smoothing

    def forward(self, out, batch) -> dict[str, torch.Tensor]:
        l_align = spoken_ce_loss(out.lm_logits, batch["align_labels"],
                                 label_smoothing=self.label_smoothing)
        l_vap = vap_bce_loss(out.vap_logits, batch["vap"], batch["audio_mask"])
        total = l_align + self.lambda_vap * l_vap
        return {"loss": total, "align": l_align.detach(), "vap": l_vap.detach(),
                "perplexity": l_align.detach().exp()}


class Phase2Loss(nn.Module):
    """L_P2 = l1*L_ctrl + l2*L_txt + l3*L_vap."""

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0,
                 l_ctrl: float = 1.0, l_txt: float = 0.5, l_vap: float = 0.2):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None
                             else torch.ones(vocab.NUM_CONTROL_FLAGS))
        self.gamma = gamma
        self.l_ctrl, self.l_txt, self.l_vap = l_ctrl, l_txt, l_vap

    def forward(self, out, batch) -> dict[str, torch.Tensor]:
        l_ctrl = focal_control_loss(
            out.control_logits, batch["flags"], batch["audio_mask"],
            alpha=self.alpha, gamma=self.gamma,
        )
        l_txt = spoken_ce_loss(out.lm_logits, batch["spoken_labels"])
        l_vap = vap_bce_loss(out.vap_logits, batch["vap"], batch["audio_mask"])
        total = self.l_ctrl * l_ctrl + self.l_txt * l_txt + self.l_vap * l_vap
        return {"loss": total, "ctrl": l_ctrl.detach(),
                "txt": l_txt.detach(), "vap": l_vap.detach()}
