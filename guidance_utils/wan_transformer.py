"""Wan2.1 counterpart of `guidance_utils/custom_transformer.py`.

The CogVideoX `ControlledTransformer` exists to (a) expose an optimisable
positional encoding and (b) let the guidance pass stop early at the block whose
attention we read. This does the same for `WanTransformer3DModel`, with three
structural differences that fall out of Wan's architecture:

  * Wan has **no learned absolute position embedding**, so there is no
    `pos_embedding` optimisation target. Only RoPE and the latent itself can be
    optimised (`--opt_mode emb` -> rope, `--opt_mode latent` -> latent).
  * Wan blocks return a **single tensor**; text conditioning is applied in
    `attn2` (cross-attention) and never enters the self-attention sequence.
  * Early exit is a real `return` rather than the forward-monkeypatching that
    `Guidance.change_mode` does for CogVideoX, so `norm_out`/`proj_out`/
    unpatchify are skipped without touching module state.

RoPE tensor convention used throughout this port:

    init_rope      (2, 1, N, 1, D)        stack([freqs_cos, freqs_sin])
    trainable rope (2, 2, 1, N, 1, D)     stack([rope_q, rope_k])

so `rope[0]` drives the query rotation and `rope[1]` the key rotation, mirroring
the `q_rope, k_rope` split the CogVideoX-5B path uses.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers


class ControlledWanTransformer(WanTransformer3DModel):
    # NOTE: deliberately NO __init__ override.
    #
    # diffusers' ConfigMixin.extract_init_dict picks which config keys to pass to
    # the constructor by reading `inspect.signature(cls.__init__)`. An override
    # declared as (*args, **kwargs) exposes no named parameters, so EVERY config
    # key gets dropped and from_pretrained silently builds the model from
    # WanTransformer3DModel's own defaults -- which are the 14B ones. Loading a
    # 1.3B checkpoint into that shape then dies with:
    #     blocks.0.attn1.norm_k.weight expected [5120], got [1536]
    # Class-level defaults avoid the override entirely; assigning to an instance
    # shadows them as usual.
    init_rope: Optional[torch.Tensor] = None
    trainable_rope: Optional[torch.Tensor] = None
    # Block index to stop after during guidance passes (None = full forward).
    stop_after_block: Optional[int] = None
    _checked_rope_processors: bool = False

    # ------------------------------------------------------------------ #
    def default_rope(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Baseline RoPE for this latent shape, as (2, 1, N, 1, D)."""
        freqs_cos, freqs_sin = self.rope(hidden_states)
        return torch.stack([freqs_cos, freqs_sin], dim=0)

    @staticmethod
    def _split_qk_rope(rope: torch.Tensor):
        """(2, 2, ...) -> ((cos_q, sin_q), (cos_k, sin_k))."""
        return (rope[0][0], rope[0][1]), (rope[1][0], rope[1][1])

    def _resolve_rope(self, hidden_states: torch.Tensor, rope: Optional[torch.Tensor]):
        if rope is None:
            # Plain (cos, sin) so the model still runs under the stock
            # WanAttnProcessor when no guidance processors are installed.
            base = self.init_rope if self.init_rope is not None else self.default_rope(hidden_states)
            return (base[0], base[1])
        # Explicit rope implies an optimised q/k split, which only
        # WanInjectionProcessor understands. Check once and fail loudly rather
        # than letting the stock processor raise an opaque tuple-index error.
        if not self._checked_rope_processors:
            from .wan_modules import WanInjectionProcessor

            stock = [i for i, b in enumerate(self.blocks) if not isinstance(b.attn1.processor, WanInjectionProcessor)]
            if stock:
                raise RuntimeError(
                    f"rope= needs WanInjectionProcessor on every block, but blocks {stock[:8]}"
                    f"{'...' if len(stock) > 8 else ''} still use the stock WanAttnProcessor, which "
                    "cannot consume a q/k-split rope. Call register_attention_processor() first."
                )
            self._checked_rope_processors = True
        return self._split_qk_rope(rope)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        rope: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. Rotary embeddings -- optimisable, unlike the stock forward.
        rotary_emb = self._resolve_rope(hidden_states, rope)

        # 2. Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # 3. Conditioning
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        use_checkpoint = torch.is_grad_enabled() and self.gradient_checkpointing
        for i, block in enumerate(self.blocks):
            if use_checkpoint:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

            if self.stop_after_block is not None and i >= self.stop_after_block:
                # Guidance pass: everything downstream is dead compute. The
                # caller reads features/QK off the hooked block, not the return.
                if USE_PEFT_BACKEND:
                    unscale_lora_layers(self, lora_scale)
                if not return_dict:
                    return (hidden_states,)
                return Transformer2DModelOutput(sample=hidden_states)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
