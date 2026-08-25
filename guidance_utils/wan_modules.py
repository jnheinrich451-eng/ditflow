"""Wan2.1 counterpart of `guidance_utils/custom_modules.py`.

Two things differ from the CogVideoX versions and both are simplifications:

  * `WanTransformerBlock.attn1` is **video-only self-attention** -- text goes
    through `attn2`. Every `text_seq_length` split in `InjectionProcessor`
    disappears, and so does the `226:` slice in the AMF computation.
  * A Wan block returns a single tensor, not `(hidden, encoder_hidden)`.

The rotary application here is an out-of-place rewrite of the one in
`diffusers.models.transformers.transformer_wan.WanAttnProcessor`. It is
numerically identical (o1 lands on even indices, o2 on odd) but avoids the
in-place `torch.empty_like` writes, which keeps gradients flowing cleanly back
to an optimised RoPE tensor.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from diffusers.models.transformers.transformer_wan import _get_added_kv_projections, _get_qkv_projections


def apply_rotary_emb(hidden_states: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2).type_as(hidden_states)


def _as_qk_rope(rotary_emb):
    """Accept either (cos, sin) or ((cos_q, sin_q), (cos_k, sin_k))."""
    first = rotary_emb[0]
    if isinstance(first, (tuple, list)):
        return rotary_emb[0], rotary_emb[1]
    return rotary_emb, rotary_emb


class WanInjectionProcessor:
    r"""`WanAttnProcessor` with QK capture and KV injection for guidance."""

    def __init__(self, block_name: str):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanInjectionProcessor requires PyTorch 2.0 or higher.")

        self.block_name = block_name
        self.inject_kv = False
        self.copy_kv = False

        self.query = None
        self.key = None
        self.value = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb=None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # I2V: CLIP image tokens are prepended to the text stream in attn2.
            # 512 is the umT5 context length, hardcoded upstream too.
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # (B, S, heads, head_dim) -- note Wan does not transpose to (B, heads, S, D).
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            q_rope, k_rope = _as_qk_rope(rotary_emb)
            query = apply_rotary_emb(query, *q_rope)
            key = apply_rotary_emb(key, *k_rope)

        # Guidance hooks. Self-attention only -- attn2 carries no motion signal
        # and its key/value come from text, so injecting there is meaningless.
        if not attn.is_cross_attention:
            if self.inject_kv and self.key is not None:
                key = torch.cat([key[:-1], self.key[-1:]], dim=0) if key.shape[0] > 1 else self.key[-1:]
                value = torch.cat([value[:-1], self.value[-1:]], dim=0) if value.shape[0] > 1 else self.value[-1:]
            elif self.copy_kv:
                self.query = query[-1:]
                self.key = key[-1:]
                self.value = value[-1:]

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key_img.transpose(1, 2),
                value_img.transpose(1, 2),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def clear(self):
        self.query = None
        self.key = None
        self.value = None


class WanModuleWithGuidance(torch.nn.Module):
    """Wraps a `WanTransformerBlock` so its output tokens can be read as t d h w."""

    def __init__(self, module, h, w, p, block_name, num_frames):
        super().__init__()
        self.module = module
        self.attn1 = module.attn1
        self.attn2 = module.attn2

        self.starting_shape = "(t h w) d"
        self.h = h
        self.w = w
        self.p = p
        self.block_name = block_name
        self.num_frames = num_frames
        self.saved_features = None

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)

        p_h = self.h // self.p
        p_w = self.w // self.p
        self.saved_features = rearrange(
            out[-1], f"{self.starting_shape} -> t d h w", t=self.num_frames, h=p_h, w=p_w
        )
        return out
