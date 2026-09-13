"""Isolated step-9 visual experiment; production Wan defaults are unchanged."""
import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from benchmark import wan_head_pilot as head
from benchmark import wan_noised_reference_pilot as pilot
from benchmark.wan_head_visual_pilot import require_matched_starts
from probe_wan_head_visual import capture_estimate
from probe_wan_response import tensor_hash


def matched_reference_input(clean, scheduler, step, seed, output=None):
    """Independent CPU RNG; neither consume nor reuse generation RNG/noise."""
    if step!=9: raise ValueError('This pilot validates only sampling index 9')
    noise = torch.randn(clean.shape,generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
    sigma = float(scheduler.sigmas[step])
    value = ((1-sigma)*clean.float()+sigma*noise.to(clean.device)).to(clean.dtype)
    if output is not None:
        np.savez_compressed(output,clean=clean.float().cpu().numpy(),noise=noise.numpy(),
                            model_input=value.float().cpu().numpy())
    return value,dict(mode='matched_noised',step=step,sigma=sigma,
        timestep=float(scheduler.timesteps[step]),noise_seed=seed,noise_generator='dedicated CPU torch.Generator',
        clean_sha256=tensor_hash(clean),noise_sha256=tensor_hash(noise),input_sha256=tensor_hash(value),
        dtype=str(value.dtype))


class NoisedReferenceMixin:
    """Opt-in reference intervention; clean latent and time are restored on failure too."""
    @torch.no_grad()
    def load_attn_features(self):
        if not self.config.guidance_blocks:
            self.motion_attn_masks = {}; self.motion_regions = None
            head.write_json(Path(self.output_path)/'reference_protocol.json',dict(mode='unused'))
            return {}
        step = self.config.reference_noise_step
        clean, old_time = self.motion_latent, self.motion_timestep
        try:
            if step is None:
                provenance = dict(mode='clean',sigma=0.,timestep=float(old_time[0]),clean_sha256=tensor_hash(clean))
            else:
                self.motion_latent, provenance = matched_reference_input(clean,self.scheduler,step,self.config.reference_noise_seed,
                    Path(self.output_path)/'reference_inputs.npz')
                self.motion_timestep = self.timesteps[step].reshape(1)
            # Same production hard all-pair readout, masks and selected head as the target loss.
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                result = super().load_attn_features()
            head.write_json(Path(self.output_path)/'reference_protocol.json',provenance)
            return result
        finally:
            self.motion_latent, self.motion_timestep = clean, old_time

    def _amf(self, processor):
        flow = super()._amf(processor)
        # Save precisely the tensor entering production loss, not detached-FP32 QK recomputation.
        if self.probe.enabled:
            self.full_pair_serial = getattr(self,'full_pair_serial',0)+1
            filename = f'full_pair_{self.full_pair_serial:03d}_{processor.block_name}.npz'
            self.last_full_pair_file = filename
            np.savez_compressed(self.probe.path/filename,flow=flow.detach().float().cpu().numpy())
            self.probe.emit('full_pair_target',block=processor.block_name,file=filename,
                q_dtype=str(processor.query.dtype),k_dtype=str(processor.key.dtype),
                flow_head=self.config.flow_head,shape=list(flow.shape))
        return flow

    def compute_motion_flow_loss(self,x,ts,rope=None):
        loss = super().compute_motion_flow_loss(x,ts,rope=rope)
        if self.probe.enabled:
            block = f'block_{self.config.guidance_blocks[0]}_attn1_processor'
            self.probe.emit('full_pair_loss',block=block,file=self.last_full_pair_file,loss=float(loss.detach()))
        return loss


def initial_state(g):
    return dict(latent_sha256=tensor_hash(g.init_latents),reference_latent_sha256=tensor_hash(g.motion_latent),
        conditioning_sha256=tensor_hash(g.guidance_embeds),source_conditioning_sha256=tensor_hash(g.source_embeds),
        rope_sha256=tensor_hash(g.transformer.init_rope),timesteps=g.timesteps.cpu().tolist(),
        sigmas=g.scheduler.sigmas.cpu().tolist(),model_revision=getattr(g.transformer.config,'_commit_hash',None),
        guidance_blocks=list(g.config.guidance_blocks),flow_head=g.config.flow_head)


def check_step9_start(g, latent, plan_root):
    state = dict(latent_sha256=tensor_hash(latent),step=9,scheduler_index=g.scheduler.step_index)
    for arm in pilot.ARMS:
        marker = plan_root/f'{arm}_done.json'
        if marker.is_file():
            previous = plan_root/pilot.read_json(marker)['directory']
            if pilot.read_json(previous/'step9_state.json')!=state:
                raise ValueError('Step-9 starting state differs between arms before guidance')
    head.write_json(Path(g.output_path)/'step9_state.json',state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--arm',choices=pilot.ARMS,required=True)
    parser.add_argument('--output_path',type=Path,required=True)
    args = parser.parse_args()
    plan_root = args.plan.resolve().parent
    plan = pilot.checked_plan(plan_root)
    if args.arm=='candidate':
        confirmation = plan_root/pilot.read_json(plan_root/'confirmation_done.json')['directory']
        if not pilot.validate_confirmation(confirmation,plan_root,plan)['gate']['passed']:
            raise ValueError('Fixed candidate did not pass independent step-9 confirmation')
    root = args.output_path.resolve()
    if root.exists() and any(root.iterdir()): parser.error('Use a fresh attempt directory')
    root.mkdir(parents=True,exist_ok=True)
    from motion_guidance_wan import WanGuidance, WAN_NEGATIVE_PROMPT

    class VisualWan(NoisedReferenceMixin,WanGuidance):
        def guidance_step(self,x,i,t,mode,loss_type):
            if i!=9: raise ValueError('Unexpected guidance index')
            check_step9_start(self,x,plan_root)
            capture_estimate(self,x,i,'before')
            result = super().guidance_step(x,i,t,mode,loss_type)
            capture_estimate(self,result[0],i,'after')
            return result

        def denoise_step(self,latents,i,prompt_embeds,rope=None):
            if i==9 and not self.config.guidance_blocks:
                check_step9_start(self,latents,plan_root)
                capture_estimate(self,latents,i,'before')
            return super().denoise_step(latents,i,prompt_embeds,rope=rope)

    config = OmegaConf.merge(OmegaConf.load('configs/guidance_config_wan.yaml'),
        pilot.fixed_config(plan,args.arm,root),dict(negative_prompt=WAN_NEGATIVE_PROMPT))
    OmegaConf.save(config,root/'suite_config.yaml')
    g = VisualWan(config)
    initial = initial_state(g)
    head.write_json(root/'initial_state.json',initial)
    states = [initial]
    for arm in pilot.ARMS:
        marker = plan_root/f'{arm}_done.json'
        if marker.is_file():
            previous = plan_root/pilot.read_json(marker)['directory']
            states.append(pilot.read_json(previous/'initial_state.json'))
    require_matched_starts(states)
    g.run(custom_name='final')
    unchanged = tensor_hash(g.transformer.init_rope)==initial['rope_sha256'] and g.transformer.trainable_rope is None
    frozen = all(not p.requires_grad and p.grad is None for p in g.transformer.parameters())
    head.write_json(root/'complete.json',dict(arm=args.arm,native_rope_unchanged=unchanged,frozen_weights=frozen))
    print('Completed actual 50-step generation:',root/'final.mp4',flush=True)


if __name__=='__main__':
    main()
