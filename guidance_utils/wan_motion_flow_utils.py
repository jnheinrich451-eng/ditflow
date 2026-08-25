"""Attention Motion Flow for Wan2.1 -- memory-frugal reformulation.

Same quantity as `guidance_utils/motion_flow_utils.compute_motion_flow`, but it
never materialises the joint attention map. Three exact algebraic rewrites do
the work; none of them is an approximation.

1. **Head fusion.** The reference averages *pre-softmax logits* over heads:

       mean_h( q_h @ k_h^T ) / sqrt(D)  ==  (q_flat @ k_flat^T) / (H * sqrt(D))

   where `q_flat` is the heads axis folded into features, (S, H*D). The
   reference builds the full `(H, S, S)` tensor first -- for CogVideoX-5B that
   is 30 x 8100 x 8100, ~3.9 GB in bf16 -- purely to average it away. Folding
   removes the H factor entirely.

2. **Frame-pair chunking.** The softmax in the reference is already applied per
   `(source token, target frame)` block over that frame's `hw` positions, so
   the map decomposes exactly into `nframes^2` independent `(hw, hw)` blocks.
   Computing them one at a time is bit-comparable to slicing them out of the
   joint map, and each block is ~7 MB instead of ~4 GB.

3. **Displacement without the relative-coordinate grids.** Because each block is
   row-stochastic after softmax,

       sum_j A[i,j] * (x[j] - x[i])  ==  (A @ x)[i] - x[i]

   so the two `(hw, hw)` relative-coordinate grids and their elementwise
   products -- all of which the reference keeps alive in the autograd graph --
   collapse to a matrix-vector product.

Output ordering matches the reference exactly: `(nframes^2, hw, 2)`, iterated
source-frame-major, `[dx, dy]` last.

Note on a reference-implementation bug this port does not inherit: upstream
`load_attn_features` calls `compute_motion_flow` without `nframes`, silently
falling back to the default of 6. That is only correct for `--video_length 24`.
Here `nframes` is a required argument.
"""

from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint


def _flatten_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, S, heads, D) or (B, heads, S, D) -> (S, heads*D) for the last batch item."""
    if x.ndim != 4:
        raise ValueError(f"expected a 4D q/k tensor, got shape {tuple(x.shape)}")
    return x[-1].flatten(1)


def _pair_flow(
    q_i: torch.Tensor,
    k_j: torch.Tensor,
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    scale: float,
    temp: float,
    argmax: bool,
    w: int,
    softmax_fp32: bool,
) -> torch.Tensor:
    """Displacement field from source frame i to target frame j -> (hw, 2)."""
    logits = (q_i @ k_j.transpose(-1, -2)) * scale

    if argmax:
        # Reference AMF only; not differentiable, never used on the target side.
        matches = logits.argmax(dim=-1)
        dx = (matches % w).to(x_coords.dtype) - x_coords
        dy = torch.div(matches, w, rounding_mode="floor").to(y_coords.dtype) - y_coords
        return torch.stack((dx, dy), dim=-1)

    attn = logits.float() if softmax_fp32 else logits
    attn = F.softmax(attn * temp, dim=-1)
    attn = attn.to(x_coords.dtype)

    # Rows sum to 1, so the expected absolute position minus the source position
    # is the expected displacement.
    dx = attn @ x_coords - x_coords
    dy = attn @ y_coords - y_coords
    return torch.stack((dx, dy), dim=-1)


def compute_motion_flow(
    q: torch.Tensor,
    k: torch.Tensor,
    h: int,
    w: int,
    nframes: int,
    temp: float = 2.0,
    argmax: bool = False,
    checkpoint_pairs: bool = False,
    softmax_fp32: bool = True,
    head_dim: Optional[int] = None,
) -> torch.Tensor:
    """Compute Attention Motion Flow (AMF) -> (nframes^2, h*w, 2).

    Args:
        q, k: self-attention query/key, (B, S, heads, D) as Wan produces them.
              S must equal nframes * h * w -- Wan's self-attention carries no
              text prefix, so unlike CogVideoX there is nothing to slice off.
        checkpoint_pairs: recompute each frame-pair block during backward
              instead of retaining its softmax. Trades ~30% extra compute for
              O(S) instead of O(S^2) retained activations; needed above roughly
              33 frames at 480p.
    """
    hw = h * w
    if head_dim is None:
        head_dim = q.shape[-1]

    q_flat = _flatten_heads(q)
    k_flat = _flatten_heads(k)

    seq_len = q_flat.shape[0]
    if seq_len != nframes * hw:
        raise ValueError(
            f"sequence length {seq_len} != nframes*h*w = {nframes}*{h}*{w} = {nframes * hw}. "
            "Check that h/w are the post-patch grid dims and nframes the latent frame count."
        )

    num_heads = q_flat.shape[-1] // head_dim
    scale = 1.0 / (num_heads * (head_dim**0.5))

    q_frames = q_flat.view(nframes, hw, -1)
    k_frames = k_flat.view(nframes, hw, -1)

    device = q_flat.device
    # fp32 is plenty for a coordinate grid; fp64 is honoured so the equivalence
    # test against the reference implementation can assert exactness.
    coord_dtype = torch.float64 if q_flat.dtype == torch.float64 else torch.float32
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=coord_dtype),
        torch.arange(w, device=device, dtype=coord_dtype),
        indexing="ij",
    )
    x_coords = xx.flatten()
    y_coords = yy.flatten()

    flows = []
    for frame_i in range(nframes):
        q_i = q_frames[frame_i]
        for frame_j in range(nframes):
            k_j = k_frames[frame_j]
            if checkpoint_pairs and not argmax and torch.is_grad_enabled():
                flow = torch.utils.checkpoint.checkpoint(
                    _pair_flow,
                    q_i,
                    k_j,
                    x_coords,
                    y_coords,
                    scale,
                    temp,
                    argmax,
                    w,
                    softmax_fp32,
                    use_reentrant=False,
                )
            else:
                flow = _pair_flow(q_i, k_j, x_coords, y_coords, scale, temp, argmax, w, softmax_fp32)
            flows.append(flow)

    return torch.stack(flows, dim=0)
