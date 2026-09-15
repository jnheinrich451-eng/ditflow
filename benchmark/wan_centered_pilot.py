"""Opt-in centered-AMF pilot: shared prefix, one guided step, three suffixes.

Nothing in this module changes WanGuidance's production defaults.
"""
import copy
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from benchmark.inspect_wan_static_matches import sha
from benchmark.wan_port_acceptance import acceptance_config, difference, write_json
from guidance_utils.wan_centered_amf import centered_pair_flow
from guidance_utils.wan_reference_diagnostics import reference_images
from motion_guidance_wan import WanGuidance
from probe_wan_response import decode, tensor_hash


class DeferredReferenceGuidance(WanGuidance):
    def load_latent(self):
        return torch.zeros_like(self.init_latents, dtype=self.dtype)

    def load_attn_features(self):
        return {}


def preflight(acceptance, gpu_report):
    """Read-only compatibility gate before downloading or loading weights."""
    acceptance = Path(acceptance)
    meta = json.loads((acceptance / 'manifest.json').read_text())
    gate = json.loads(Path(gpu_report).read_text())
    if not gate['stage_a_passed'] or not json.loads((acceptance / 'parity.json').read_text())['passed']:
        raise ValueError('Passing cached-GPU readout and saved native parity are required')
    if not json.loads((acceptance / 'vanilla_review.json').read_text()).get('credible_motion'):
        raise ValueError('Saved vanilla video must have credible reviewed motion')
    root = Path(__file__).resolve().parents[1]
    if gate['kernel_sha256'] != sha(root / 'guidance_utils/wan_centered_amf.py'):
        raise ValueError('Candidate kernel changed after the GPU test')
    # Allow only line-ending conversion when reusing the established parity.
    core = ('motion_guidance_wan.py', 'guidance_utils/wan_transformer.py',
            'guidance_utils/wan_modules.py', 'guidance_utils/wan_motion_flow_utils.py',
            'guidance_utils/wan_guidance_schedule.py', 'guidance_utils/motion_probe.py',
            'configs/guidance_config_wan.yaml')
    for name in core:
        raw = (root / name).read_bytes()
        lf = raw.replace(b'\r\n', b'\n')
        possible = {hashlib.sha256(b).hexdigest() for b in (raw, lf, lf.replace(b'\n', b'\r\n'))}
        if meta['source_sha256'][name] not in possible:
            raise ValueError(f'Core source changed; cannot reuse native parity: {name}')
    for package, expected in meta['packages'].items():
        if version(package) != expected:
            raise ValueError(f'Runtime drift: {package}: expected {expected}, found {version(package)}')
    revision = '38ec498cb3208fb688890f8cc7e94ede2cbd7f68'
    if meta['checkpoint_revision'] != revision or meta['mode'] != 'T2V':
        raise ValueError('This pilot is pinned to the tested T2V checkpoint')
    frames = reference_images(acceptance / 'forward/reference', 21, (832, 480))
    if [hashlib.sha256(f.tobytes()).hexdigest() for f in frames] != meta['reference_rgb_sha256']:
        raise ValueError('Reference RGB differs from the accepted seed-1 case')
    return meta, frames


def load_pilot(acceptance, gpu_report, output):
    from huggingface_hub import snapshot_download
    meta, frames = preflight(acceptance, gpu_report)
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 35 * 2**30:
        raise RuntimeError('Full-model pilot requires A100 40 GB/high RAM or an 80 GB GPU')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    print('SETUP: one pinned 14B model load; zero reference forwards during setup.', flush=True)
    checkpoint = snapshot_download(meta['model'], revision=meta['checkpoint_revision'])
    config = acceptance_config(checkpoint, Path(acceptance) / 'forward/reference', output / 'setup',
        meta['config']['target_prompt'], seed=1,
        cpu_offload=torch.cuda.get_device_properties(0).total_memory < 70 * 2**30)
    g = DeferredReferenceGuidance(config)
    checks = dict(initial_latent_sha256=tensor_hash(g.init_latents),
                  conditioning_sha256=tensor_hash(g.guidance_embeds),
                  source_conditioning_sha256=tensor_hash(g.source_embeds),
                  rope_sha256=tensor_hash(g.transformer.init_rope))
    for name, actual in checks.items():
        if actual != meta[name]:
            raise ValueError(f'Frozen baseline mismatch: {name}')
    if g.scheduler.sigmas.tolist() != meta['sigmas'] or g.timesteps.cpu().tolist() != meta['timesteps']:
        raise ValueError('Scheduler mismatch')
    manifest = dict(baseline_manifest_sha256=sha(Path(acceptance) / 'manifest.json'),
                    gpu_report_sha256=sha(gpu_report), baseline=meta, checks=checks,
                    experiment=dict(target_readout='centered logits, multiplier 8', reference_readout='unchanged hard argmax, multiplier 2',
                        pairs='five adjacent latent pairs', guidance_index=39, guidance_timestep=float(g.timesteps[39]),
                        guidance_sigma=float(g.scheduler.sigmas[39]), optimization_steps=5, optimizer='Adam', lr=.001,
                        injection=False, guidance_blocks=[20], head_aggregation='all 40, mean logits',
                        batch='single positive target for AMF; separate positive/negative passes for CFG',
                        model_parameter_dtypes={name:str(p.dtype) for name,p in g.transformer.named_parameters()
                            if 'time_embedder' in name or name == 'patch_embedding.weight'},
                        latent_dtype=str(g.init_latents.dtype), token_grid=[6,30,52]),
                    kernel_sha256=sha(Path(__file__).resolve().parents[1] / 'guidance_utils/wan_centered_amf.py'))
    existing = output / 'pilot_manifest.json'
    if existing.exists() and json.loads(existing.read_text()) != manifest:
        raise ValueError('Pilot provenance changed; do not mix revisions in one output')
    write_json(existing, manifest)
    return g, frames


def pack_state(value):
    """Save CPU payloads while remembering which schedule tensors stay on CPU."""
    if torch.is_tensor(value):
        return ('__wan_saved_tensor__', str(value.device), value.detach().cpu().clone())
    if isinstance(value, dict):
        return type(value)((k, pack_state(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return type(value)(pack_state(v) for v in value)
    return copy.deepcopy(value)


def restore_state(value, device):
    if isinstance(value, tuple) and len(value) == 3 and isinstance(value[0], str) and value[0] == '__wan_saved_tensor__':
        target = device if value[1].startswith('cuda') else value[1]
        return value[2].to(target).clone()
    if isinstance(value, dict):
        return type(value)((k, restore_state(v, device)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return type(value)(restore_state(v, device) for v in value)
    return copy.deepcopy(value)


def clear(g):
    blocks = list(range(len(g.transformer.blocks)))
    g._set_kv_mode(blocks, inject=False, copy=False)
    g._clear_kv(blocks)
    g.transformer.trainable_rope = None


@torch.no_grad()
def step(g, latent, index):
    clear(g)
    predictions = []
    hook = g.transformer.register_forward_hook(lambda m, a, out: predictions.append(out[0].detach()))
    try:
        nxt = g.denoise_step(latent, index, g.guidance_embeds)
    finally:
        hook.remove()
    if len(predictions) != 2:
        raise RuntimeError('Expected fresh conditional and unconditional predictions')
    cond, uncond = predictions
    return nxt, uncond + g.guidance_scale * (cond - uncond)


def candidate_loss(g, latent, index):
    """Target-only experimental readout; clean reference and mask stay native."""
    clear(g)
    blocks = list(g.config.guidance_blocks)
    g._set_kv_mode(blocks, inject=False, copy=True)
    g._forward_transformer(latent, g.guidance_embeds[1:2], g.timesteps[index].expand(latent.shape[0]))
    f, h, w = g.latent_num_frames, g.patches_height, g.patches_width
    pair_indices = torch.arange(f-1, device=g.device) * (f+1) + 1
    losses = []
    for block in blocks:
        proc = g.transformer.blocks[block].attn1.processor
        if proc.query.shape[0] != 1:
            raise ValueError('Pilot requires explicit single conditional batch')
        q = proc.query[0].reshape(f, h*w, *proc.query.shape[-2:])
        k = proc.key[0].reshape(f, h*w, *proc.key.shape[-2:])
        flows = torch.stack([centered_pair_flow(q[i], k[i+1], h, w) for i in range(f-1)])
        ref = g.motion_attn_features[proc.block_name][pair_indices].detach().to(flows.dtype)
        mask = g.motion_attn_masks[proc.block_name][pair_indices]
        if not mask.any():
            raise ValueError('No valid adjacent reference entries')
        losses.append((flows[mask] - ref[mask]).square().mean())
    g._clear_kv(blocks)
    return torch.stack(losses).mean()


def trace_pilot(g, frames, output, index=39, updates=5, lr=.001):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    marker = output / 'trace_started.json'
    if marker.exists():
        raise RuntimeError('Trace budget already started; inspect saved artifacts instead of repeating')
    write_json(marker, dict(index=index, updates=updates, lr=lr, max_prefixes=1, max_reference_encodes=2,
        max_guided_updates=2*updates, max_post_update_loss_reads=2, max_step_branches=3,
        hypothesis='Candidate gradients reach the latent and propagate through a fresh prediction and scheduler step.',
        expected='Finite gradients and nonzero post-cast, prediction and next-state differences; no motion success claim.'))
    print('HYPOTHESIS: centered target AMF propagates through actual sampling. '
          f'LIMIT: one prefix to index {index}, two references, {updates} updates each, three step branches; no full videos.', flush=True)
    g.transformer.trainable_rope = None
    latent = g.init_latents.detach().clone()
    for i in range(index):
        latent, _ = step(g, latent, i)
        if i % 5 == 0:
            print(f'Common unguided prefix: {i+1}/{index}', flush=True)
    prefix = latent.detach().clone()
    solver = copy.deepcopy(g.scheduler)
    torch.save(dict(latent=prefix.cpu(), scheduler_state=pack_state(solver.__dict__), index=index), output / 'prefix.pt')
    g.scheduler = copy.deepcopy(solver)
    off_next, off_prediction = step(g, prefix, index)
    torch.save(dict(latent=off_next.cpu(), scheduler_state=pack_state(g.scheduler.__dict__), index=index+1), output / 'off.pt')
    traces = []
    for arm, sequence in (('forward', frames), ('reverse', frames[::-1])):
        folder = output / arm; folder.mkdir()
        reference = folder / 'reference'; reference.mkdir()
        for i, frame in enumerate(sequence):
            Image.fromarray(frame).save(reference / f'{i:05d}.png')
        g.output_path = str(folder); g.config.video_path = str(reference)
        # Bypass the setup-only deferred methods, preserving original reference normalization/readout.
        g.motion_latent = WanGuidance.load_latent(g)
        g.motion_attn_features = WanGuidance.load_attn_features(g)
        np.savez_compressed(folder / 'reference_amf.npz', **{
            name: value.detach().cpu().numpy() for name, value in g.motion_attn_features.items()}, **{
            name + '_mask': value.detach().cpu().numpy() for name, value in g.motion_attn_masks.items()})
        g.scheduler = copy.deepcopy(solver)
        x = prefix.clone().float().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=lr)
        losses, gradients = [], []
        for j in range(updates):
            optimizer.zero_grad(set_to_none=True)
            loss = candidate_loss(g, x, index)
            if not torch.isfinite(loss):
                raise RuntimeError(f'{arm}: nonfinite loss')
            loss.backward()
            if x.grad is None or not torch.isfinite(x.grad).all() or float(x.grad.norm()) == 0:
                raise RuntimeError(f'{arm}: absent, zero or nonfinite latent gradient')
            losses.append(float(loss.detach())); gradients.append(float(x.grad.norm()))
            optimizer.step()
            print(f'{arm}: update {j+1}/{updates}, loss={losses[-1]:.6f}, gradient={gradients[-1]:.6f}', flush=True)
            write_json(folder / 'optimization.json', dict(losses=losses, gradient_norms=gradients))
        after = x.detach()
        with torch.no_grad():
            final_loss = float(candidate_loss(g, after, index))
        nxt, prediction = step(g, after, index)
        trace = dict(arm=arm, index=index, losses_before_updates=losses, loss_after_updates=final_loss,
            latent_gradient_norms=gradients, update_fp32=difference(after, prefix),
            update_after_input_cast=difference(after.to(g.dtype), prefix.to(g.dtype)),
            recomputed_cfg_prediction=difference(prediction, off_prediction),
            next_scheduler_state=difference(nxt, off_next), same_solver_state=True,
            actual_sampler_received_optimized_latent=True, source_conditioning_sha256=tensor_hash(g.source_embeds),
            target_conditioning_sha256=tensor_hash(g.guidance_embeds))
        trace['passed'] = bool(np.isfinite(final_loss) and all(np.isfinite(trace[key]['rms']) and trace[key]['rms'] > 0 for key in
            ('update_fp32', 'update_after_input_cast', 'recomputed_cfg_prediction', 'next_scheduler_state')))
        write_json(folder / 'step_trace.json', trace)
        torch.save(dict(latent=nxt.cpu(), scheduler_state=pack_state(g.scheduler.__dict__), index=index+1), output / f'{arm}.pt')
        np.savez_compressed(folder / 'step_trace.npz', before=prefix.cpu().numpy(), after=after.cpu().numpy(),
                            before_prediction=off_prediction.float().cpu().numpy(), after_prediction=prediction.float().cpu().numpy(),
                            before_next=off_next.cpu().numpy(), after_next=nxt.cpu().numpy())
        traces.append(trace)
        if not trace['passed']:
            raise RuntimeError(f'{arm}: intervention failed to propagate; do not decode guided arms')
    write_json(output / 'trace_gate.json', dict(passed=all(r['passed'] for r in traces), traces=traces, port_success=False))
    return traces


@torch.no_grad()
def decode_pilot(g, output):
    output = Path(output)
    if not json.loads((output / 'trace_gate.json').read_text())['passed']:
        raise RuntimeError('Real-step propagation gate must pass')
    marker = output / 'decode_started.json'
    if marker.exists():
        raise RuntimeError('Three-suffix decode budget already started')
    write_json(marker, dict(max_suffixes=3, max_additional_guidance_updates=0,
        hypothesis='Opposite references at the shared sampling step produce opposite decoded subject travel.',
        expected='Independent decoded subject trajectories follow reference direction; image changes alone fail.'))
    print('HYPOTHESIS: opposite references change decoded subject travel. '
          'LIMIT: three suffixes (off/forward/reverse), no further guidance or parameter changes.', flush=True)
    for arm in ('off', 'forward', 'reverse'):
        # These are checkpoints written by trace_pilot, including trusted scheduler state.
        state = torch.load(output / f'{arm}.pt', map_location='cpu', weights_only=False)
        latent = state['latent'].to(g.device)
        g.scheduler.__dict__ = restore_state(state['scheduler_state'], g.device)
        for i in range(state['index'], len(g.timesteps)):
            latent, _ = step(g, latent, i)
        (output / arm).mkdir(exist_ok=True)
        decode(g, latent, output / arm / 'final.mp4')
    import imageio.v3 as iio
    import imageio.v2 as writer
    videos = [iio.imread(output / a / 'final.mp4', plugin='FFMPEG') for a in ('off', 'forward', 'reverse')]
    panels = []
    for i in range(g.num_frames):
        panel = Image.new('RGB', (416*3, 266), 'white')
        for col, (video, label) in enumerate(zip(videos, ('Guidance off', 'Original reference', 'Reversed reference'))):
            panel.paste(Image.fromarray(video[i]).resize((416, 240)), (416*col, 26))
            ImageDraw.Draw(panel).text((416*col+5, 5), label, fill='black')
        panels.append(np.asarray(panel))
    writer.mimwrite(output / 'comparison.mp4', panels, fps=16, codec='libx264', macro_block_size=2)
    import csv
    # Empty image-space annotations are evidence to collect, never inferred from AMF.
    with (output / 'trajectory_annotations.csv').open('w', newline='', encoding='utf-8') as handle:
        fields = ['video', 'frame', 'subject_x', 'subject_y', 'landmark1_x', 'landmark1_y',
                  'landmark2_x', 'landmark2_y', 'landmark3_x', 'landmark3_y', 'quality_notes']
        writer_csv = csv.DictWriter(handle, fieldnames=fields)
        writer_csv.writeheader()
        for video in ('off/final.mp4', 'forward/final.mp4', 'reverse/final.mp4', 'forward/original.mp4', 'reverse/original.mp4'):
            for frame in sorted(set(min(i, g.num_frames-1) for i in (1,4,8,12,16,20))):
                writer_csv.writerow(dict(video=video, frame=frame))
    write_json(output / 'status.json', dict(status='Decoded videos ready for independent trajectory evaluation',
                                           port_success=False, automatic_success=False))
