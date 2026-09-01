"""
Phase-1 -> Phase-2 warm-start checkpoint remapping.

WHY THIS EXISTS (a real, silent, run-ruining bug):

Phase 1 trains with LoRA (`use_lora: true` in configs/train_phase1*.yaml), so peft wraps
the backbone and its saved state_dict has PEFT-wrapped keys:

    backbone.base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight
    backbone.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
    backbone.base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight

Phase 2 is a FULL fine-tune (`use_lora: false`), whose backbone keys are plain:

    backbone.model.layers.0.self_attn.q_proj.weight

`load_state_dict(..., strict=False)` matches almost NOTHING between those two naming
schemes and reports it only as counts. The observed symptom was:

    warm-started from .../phase1/.../final/model.pt (missing=237, unexpected=489)

i.e. the entire backbone was dropped on the floor and Phase 2 silently restarted from
base Gemma-3-270M — throwing away every hour of Phase-1 training, with nothing but two
numbers in the log to say so.

`remap_phase1_state_dict` fixes it: strip the PEFT wrapper prefixes, and MERGE each LoRA
adapter into its base weight

    W  <-  W + (lora_alpha / r) * (B @ A)

which is exactly what peft's own `merge_and_unload()` computes — so nothing Phase 1
learned is lost. `report_load` then makes a bad load LOUD instead of silent.
"""

from __future__ import annotations

import torch


def _looks_peft(state: dict) -> bool:
    return any((".lora_A." in k) or (".base_layer." in k) or (".base_model.model." in k)
               for k in state)


def remap_phase1_state_dict(state: dict, lora_alpha: int = 32, lora_r: int = 16,
                            verbose: bool = True) -> dict:
    """Convert a (possibly LoRA/PEFT-wrapped) Phase-1 state_dict into the plain key
    layout a non-LoRA Phase-2 ThinkSparkModel expects, merging LoRA deltas into the base
    weights. A non-PEFT state_dict is returned unchanged."""
    if not _looks_peft(state):
        if verbose:
            print("  warm-start: checkpoint is not LoRA-wrapped — using keys as-is")
        return state

    scaling = float(lora_alpha) / float(lora_r)
    base: dict[str, torch.Tensor] = {}
    lora_A: dict[str, torch.Tensor] = {}
    lora_B: dict[str, torch.Tensor] = {}

    for k, v in state.items():
        if ".lora_A." in k:
            lora_A[k.split(".lora_A.")[0]] = v
        elif ".lora_B." in k:
            lora_B[k.split(".lora_B.")[0]] = v
        elif ".lora_embedding_" in k:
            continue                      # unused for our target modules
        else:
            base[k] = v

    # ---- merge adapters into their base weights -------------------------------------
    merged, skipped = 0, 0
    for mod, A in lora_A.items():
        B = lora_B.get(mod)
        if B is None:
            skipped += 1
            continue
        wkey = f"{mod}.base_layer.weight"
        if wkey not in base:
            wkey = f"{mod}.weight"
        if wkey not in base:
            skipped += 1
            continue
        W = base[wkey]
        delta = (B.to(torch.float32) @ A.to(torch.float32)) * scaling
        if delta.shape != W.shape:
            skipped += 1
            continue
        base[wkey] = (W.to(torch.float32) + delta).to(W.dtype)
        merged += 1

    # ---- strip the PEFT wrapper from the key names ----------------------------------
    out: dict[str, torch.Tensor] = {}
    for k, v in base.items():
        nk = k.replace(".base_layer.", ".")
        nk = nk.replace("backbone.base_model.model.", "backbone.")
        nk = nk.replace(".modules_to_save.default.", ".")
        nk = nk.replace(".default.", ".")
        out[nk] = v

    if verbose:
        print(f"  warm-start: LoRA checkpoint detected — merged {merged} adapters "
              f"(scale {scaling:g}){f', skipped {skipped}' if skipped else ''}; "
              f"remapped {len(out)} tensors to plain Phase-2 keys")
    return out


def report_load(missing: list, unexpected: list, total_model_keys: int,
                *, source: str, fail_threshold: float = 0.5) -> None:
    """Print a load report and make a MOSTLY-FAILED warm-start impossible to miss.

    `strict=False` turns a total mismatch into two quiet numbers; this turns it into a
    loud, actionable message (the previous run trained from scratch for hours because
    nobody read `missing=237`)."""
    loaded = total_model_keys - len(missing)
    frac = loaded / max(1, total_model_keys)
    print(f"  warm-start from {source}: loaded {loaded}/{total_model_keys} tensors "
          f"({frac:.1%}), missing={len(missing)}, unexpected={len(unexpected)}")
    if frac < fail_threshold:
        print("  " + "!" * 72)
        print(f"  !! WARM-START FAILED: only {frac:.1%} of the model's tensors were "
              f"loaded.")
        print("  !! Training would start from the BASE model, discarding Phase 1.")
        print("  !! First few unmatched model keys:")
        for k in list(missing)[:5]:
            print(f"  !!    missing  {k}")
        for k in list(unexpected)[:5]:
            print(f"  !!    unexpect {k}")
        print("  " + "!" * 72)
        raise SystemExit(
            "Refusing to train on a failed warm-start. Check that --init points at a "
            "Phase-1 checkpoint for this same architecture, and that lora_r/lora_alpha "
            "match the Phase-1 config (pass --init-lora-r / --init-lora-alpha)."
        )
