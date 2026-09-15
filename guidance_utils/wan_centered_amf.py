"""Experimental centered target readout; native Wan attention is unchanged."""
import math

import torch


def centered_pair_flow(query, key, height, width, multiplier=8.):
    """Normalized post-RoPE (spatial, heads, dim) Q/K -> differentiable XY flow.

    Center over ALL source patches. Do not detach that mean or restrict it to
    an evaluation ROI. Reference extraction must not call this target helper.
    """
    if query.ndim != 3 or query.shape != key.shape or query.shape[0] != height * width:
        raise ValueError('Expected matching (height*width, heads, dim) Q/K')
    heads, dim = query.shape[-2:]
    logits = (query.flatten(1) @ key.flatten(1).T) * (1 / (heads * math.sqrt(dim)))
    logits = logits.float() if query.dtype != torch.float64 else logits
    centered = logits - logits.mean(dim=0, keepdim=True)
    probability = (centered * multiplier).softmax(dim=-1)
    yy, xx = torch.meshgrid(torch.arange(height, device=query.device),
                            torch.arange(width, device=query.device), indexing='ij')
    coords = torch.stack((xx.flatten(), yy.flatten()), dim=-1).to(probability.dtype)
    return probability @ coords - coords
