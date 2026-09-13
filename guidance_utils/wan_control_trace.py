"""Opt-in, observational captures of Wan's actual denoising path."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def temporal_attention_mass(query, key, frames, head, chunk=128):
    """Reconstruct unsharpened attention in FP32, without a full S x S map.

    Input Q/K are the native post-normalization/post-RoPE tensors. This is a
    diagnostic reconstruction, not the probabilities inside the fused kernel.
    Return one probability mass per source token and destination frame.
    """
    q, k = query[-1, :, head].float(), key[-1, :, head].float()
    if len(q) != len(k) or len(q) % frames:
        raise ValueError('Native attention tokens do not match the video grid')
    result = []
    with torch.autocast(device_type=q.device.type, enabled=False):
        for start in range(0, len(q), chunk):
            logits = (q[start:start+chunk] @ k.T) / q.shape[-1]**.5
            mass = logits.softmax(-1).reshape(-1, frames, len(k)//frames).sum(-1)
            result.append(mass.cpu())
    return torch.cat(result).numpy()


@torch.no_grad()
def projected_head_stats(concatenated, projected, linear, head, head_dim, chunk=128):
    """Measure the observed head's additive contribution before dropout.

    Uses the actual native SDPA output, not a value output reconstructed from
    diagnostic probabilities. The ratio is not a fraction of total influence:
    head contributions can reinforce or cancel each other.
    """
    selected = concatenated[-1, :, head*head_dim:(head+1)*head_dim].float()
    weight = linear.weight[:, head*head_dim:(head+1)*head_dim].float()
    sum_square, full_square, selected_square = 0., 0., 0.
    full = projected[-1]
    with torch.autocast(device_type=selected.device.type, enabled=False):
        for start in range(0, len(selected), chunk):
            part = selected[start:start+chunk]
            contribution = F.linear(part, weight)
            sum_square += float(contribution.double().square().sum())
            selected_square += float(part.double().square().sum())
            full_part = full[start:start+chunk].float()
            if linear.bias is not None:
                full_part = full_part - linear.bias.float()
            full_square += float(full_part.double().square().sum())
    rms = (sum_square / full.numel())**.5
    full_rms = (full_square / full.numel())**.5
    return dict(head_output_rms=(selected_square / selected.numel())**.5,
                projected_head_rms=rms, projected_all_heads_rms=full_rms,
                projected_rms_ratio=rms/full_rms if full_rms else None)


class ControlRecorder:
    """Capture existing full forwards, without adding or replacing predictions.

    Attach only to the separate control experiment. Loss/gradient forwards are
    excluded. The normal sampler and its CFG calculation remain unchanged.
    """
    def __init__(self, owner, block=30, head=30):
        self.owner, self.block, self.head = owner, block, head
        self.path = Path(owner.output_path) / 'control'
        self.path.mkdir()
        self.active = None
        self.pending_attention = None
        self.handles = [owner.transformer.register_forward_pre_hook(self._start_forward),
                        owner.transformer.register_forward_hook(self._end_forward)]
        self.attn = owner.transformer.blocks[block].attn1
        self.handles += [self.attn.register_forward_pre_hook(self._start_attention),
                         self.attn.register_forward_hook(self._end_attention, always_call=True),
                         self.attn.to_out[0].register_forward_hook(self._projection)]

    def _start_forward(self, module, inputs):
        if self.active is None or module.stop_after_block is not None:
            return
        count = self.active['calls']
        if count >= 2:
            raise RuntimeError('Expected exactly conditional and unconditional full forwards')
        self.active['branch'] = ('cond', 'uncond')[count]
        self.active['calls'] += 1

    def _end_forward(self, module, inputs, output):
        if self.active is None or module.stop_after_block is not None:
            return
        self.active['arrays'][self.active['branch']+'_velocity'] = self.array(output[0])
        self.active['branch'] = None

    def _start_attention(self, module, inputs):
        if self.active is None or self.active['branch'] is None:
            return
        proc = module.processor
        if proc.inject_kv:
            raise RuntimeError('Control experiment must not inject KV')
        self.pending_attention = (proc.copy_kv, proc.query, proc.key, proc.value)
        proc.copy_kv = True

    def _restore_attention(self):
        if self.pending_attention is not None:
            copy, query, key, value = self.pending_attention
            proc = self.attn.processor
            proc.copy_kv = copy
            if not copy:
                proc.query, proc.key, proc.value = query, key, value
            self.pending_attention = None

    def _end_attention(self, module, inputs, output):
        if self.pending_attention is None:
            return
        try:
            if output is not None:
                proc = module.processor
                mass = temporal_attention_mass(proc.query, proc.key, self.owner.latent_num_frames, self.head)
                self.active['arrays'][self.active['branch']+'_frame_mass'] = mass
        finally:
            self._restore_attention()

    def _projection(self, module, inputs, output):
        if self.active is None or self.active['branch'] is None:
            return
        self.active['native'][self.active['branch']] = projected_head_stats(
            inputs[0], output, module, self.head, self.owner.transformer.config.attention_head_dim)

    @staticmethod
    def array(tensor):
        return tensor.detach().float().cpu().numpy().copy()

    @contextmanager
    def capture(self, label, step, latent):
        from benchmark.wan_head_pilot import write_json
        if self.active is not None:
            raise RuntimeError('Nested control capture')
        destination = self.path / f'{label}_{step:02d}'
        if destination.with_suffix('.npz').exists():
            raise ValueError('Duplicate control capture')
        state = dict(calls=0, branch=None, arrays={'latent': self.array(latent)}, native={})
        self.active = state
        try:
            yield state['arrays']
            if state['calls'] != 2 or set(state['native']) != {'cond', 'uncond'}:
                raise RuntimeError('Incomplete native denoising capture')
            arrays = state['arrays']
            arrays['cfg_velocity'] = arrays['uncond_velocity'] + self.owner.guidance_scale * (
                arrays['cond_velocity'] - arrays['uncond_velocity'])
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise RuntimeError('Nonfinite control capture')
            if not np.array_equal(arrays['latent'], self.array(latent)):
                raise RuntimeError('Captured input latent changed in place')
            np.savez_compressed(destination.with_suffix('.npz'), **arrays)
            write_json(destination.with_suffix('.json'), dict(label=label, step=step,
                timestep=float(self.owner.timesteps[step]), sigma=float(self.owner.scheduler.sigmas[step]),
                sigma_next=float(self.owner.scheduler.sigmas[step+1]), guidance_scale=self.owner.guidance_scale,
                block=self.block, head=self.head, native=state['native'],
                mass_precision='FP32 reconstruction from native Q/K; no AMF sharpening'))
        finally:
            self._restore_attention()
            self.active = None

    def close(self):
        self._restore_attention()
        for handle in self.handles:
            handle.remove()
        self.handles = []
