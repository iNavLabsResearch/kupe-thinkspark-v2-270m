"""
ThinkSpark-v2-350M model (Section 4.2, 5).

Gemma-3-270M backbone + a thin multi-modal front-end + three output heads:

    control_head : hidden -> 11 control flags, one per 80 ms audio frame (internal)
    spoken_head  : the Gemma LM head, reused -> plain multilingual back-channel text
    vap_head     : hidden -> H future "is-user-speaking" bins (Phase-1/2 auxiliary)

Front-end (per Section 4.2 "Gemma-3-270M + Mimi audio-token embeddings + prosody
projection (energy,f0) + segment embeddings"):

    text stream   : [<|sys|> system prompt] [<|agent|> rolling agent text]
                    [<|stt|> optional user STT partials]      -> Gemma token embeddings
    audio stream  : per frame  audio_embed(cb0) + prosody_proj(energy,f0)
                    + state_embed(agent_state) + seg_embed(AUDIO)

The two streams are concatenated into one `inputs_embeds` sequence and run through the
backbone with a KV-cache-friendly causal mask. The control/vap heads read the hidden
states at the audio-frame positions; the spoken head reads all positions (masked to
spoken spans in the loss).

Why Gemma-3-270M: 256K-token embedding table reads Devanagari/Gujarati natively while
the active transformer stack is small (~100M), so per-frame decode is cheap (Section 4.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from thinkspark import vocab

# Segment ids for the segment-embedding table.
SEG_SYS, SEG_AGENT, SEG_STT, SEG_AUDIO = 0, 1, 2, 3
NUM_SEGMENTS = 4


@dataclass
class ModelOutputs:
    control_logits: torch.Tensor      # [B, T, num_flags]
    vap_logits: torch.Tensor          # [B, T, H]
    lm_logits: torch.Tensor           # [B, L_total, vocab]  (spoken head)
    hidden: torch.Tensor              # [B, L_total, hidden]
    audio_start: int                  # index where audio frames begin in the sequence
    spoken_start: int                 # index where the spoken tail begins (== L_total if none)


class ThinkSparkModel(nn.Module):
    def __init__(
        self,
        base_model: str = "google/gemma-3-270m",
        codebook_size: int = 2048,
        vap_horizon: int = 25,
        hf_token: str | None = None,
        gradient_checkpointing: bool = True,
        extra_special_tokens: list[str] | None = None,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model, token=hf_token, torch_dtype=torch.bfloat16
        )
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # resize embeddings if the tokenizer added special tokens (done by the caller)
        if extra_special_tokens:
            # caller resizes via tie to tokenizer; kept here for documentation.
            pass

        hidden = self.backbone.config.hidden_size
        self.hidden_size = hidden
        self.vap_horizon = vap_horizon

        # reuse Gemma's token embedding + lm head (spoken head)
        self.embed_tokens = self.backbone.get_input_embeddings()

        # multi-modal front-end
        self.audio_embed = nn.Embedding(codebook_size, hidden)
        self.prosody_proj = nn.Linear(2, hidden)
        self.state_embed = nn.Embedding(vocab.NUM_AGENT_STATES, hidden)
        self.seg_embed = nn.Embedding(NUM_SEGMENTS, hidden)

        # heads
        self.control_head = nn.Linear(hidden, vocab.NUM_CONTROL_FLAGS)
        self.vap_head = nn.Linear(hidden, vap_horizon)

        self._init_new_params()

    # ------------------------------------------------------------------ #
    def _init_new_params(self):
        for m in (self.audio_embed, self.prosody_proj, self.state_embed,
                  self.seg_embed, self.control_head, self.vap_head):
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, mean=0.0, std=0.02)
                else:
                    nn.init.zeros_(p)

    def resize_token_embeddings(self, new_num_tokens: int):
        self.backbone.resize_token_embeddings(new_num_tokens)
        self.embed_tokens = self.backbone.get_input_embeddings()

    # ------------------------------------------------------------------ #
    def _audio_frame_embeds(self, cb0, prosody, agent_state):
        """Per-frame audio embedding: token + prosody + state + AUDIO segment."""
        emb = self.audio_embed(cb0)                        # [B, T, H]
        emb = emb + self.prosody_proj(prosody)             # [B, T, H]
        emb = emb + self.state_embed(agent_state)          # [B, T, H]
        seg = torch.full(cb0.shape, SEG_AUDIO, dtype=torch.long, device=cb0.device)
        emb = emb + self.seg_embed(seg)
        return emb

    def _text_embeds(self, text_ids, seg_ids):
        emb = self.embed_tokens(text_ids)                  # [B, L_text, H]
        emb = emb + self.seg_embed(seg_ids)
        return emb

    # ------------------------------------------------------------------ #
    def forward(
        self,
        text_ids: torch.Tensor,        # [B, L_text]
        text_seg: torch.Tensor,        # [B, L_text]  (SEG_SYS/AGENT/STT per token)
        text_mask: torch.Tensor,       # [B, L_text]  (1 = real, 0 = pad)
        cb0: torch.Tensor,             # [B, T]
        prosody: torch.Tensor,         # [B, T, 2]
        agent_state: torch.Tensor,     # [B, T]
        audio_mask: torch.Tensor,      # [B, T]
        spoken_ids: torch.Tensor | None = None,   # [B, S] teacher-forced spoken tail
        spoken_mask: torch.Tensor | None = None,  # [B, S]
    ) -> ModelOutputs:
        text_emb = self._text_embeds(text_ids, text_seg)          # [B, L_text, H]
        audio_emb = self._audio_frame_embeds(cb0, prosody, agent_state)  # [B, T, H]

        parts = [text_emb, audio_emb]
        masks = [text_mask, audio_mask]
        T = audio_emb.shape[1]
        audio_start = text_emb.shape[1]
        spoken_start = audio_start + T

        # optional spoken tail (Phase-2 back-channel / thinking text), same segment as
        # agent text so the spoken head predicts plain words autoregressively.
        if spoken_ids is not None:
            spoken_seg = torch.full(spoken_ids.shape, SEG_AGENT,
                                    dtype=torch.long, device=spoken_ids.device)
            spoken_emb = self._text_embeds(spoken_ids, spoken_seg)  # [B, S, H]
            parts.append(spoken_emb)
            masks.append(spoken_mask if spoken_mask is not None
                         else torch.ones_like(spoken_ids))

        inputs_embeds = torch.cat(parts, dim=1)                   # [B, L_total, H]
        attn = torch.cat(masks, dim=1)                            # [B, L_total]

        # Call the FULL backbone with output_hidden_states=True rather than reaching into
        # `self.backbone.model` for a base-model output. Real observed break: on this
        # transformers version `self.backbone.model(...)` returns a CausalLMOutputWithPast
        # (which has `.logits`, NOT `.last_hidden_state`), so the old `out.last_hidden_state`
        # raised AttributeError. `out.hidden_states[-1]` is the final post-norm hidden
        # state (== what `.last_hidden_state` used to give) on every transformers version,
        # and `out.logits` is exactly `lm_head(that hidden)` — so we reuse it instead of
        # re-running the lm head, which also drops a dependency on the exact `.lm_head`
        # attribute location (differs between Gemma3ForCausalLM and the conditional-gen
        # wrapper).
        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            use_cache=False,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]                            # [B, L_total, H]

        audio_hidden = hidden[:, audio_start:audio_start + T, :]  # [B, T, H]
        control_logits = self.control_head(audio_hidden)          # [B, T, num_flags]
        vap_logits = self.vap_head(audio_hidden)                  # [B, T, H_vap]
        lm_logits = out.logits                                    # [B, L_total, vocab]

        return ModelOutputs(
            control_logits=control_logits,
            vap_logits=vap_logits,
            lm_logits=lm_logits,
            hidden=hidden,
            audio_start=audio_start,
            spoken_start=spoken_start,
        )

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step_logits(self, **batch) -> ModelOutputs:
        """Single-frame streaming inference helper (used by inference.py)."""
        self.eval()
        return self.forward(**batch)


def apply_lora(model: ThinkSparkModel, r=16, alpha=32, dropout=0.05):
    """Wrap the backbone in LoRA adapters for Phase-1 (peft is optional)."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model.backbone = get_peft_model(model.backbone, cfg)
    return model
