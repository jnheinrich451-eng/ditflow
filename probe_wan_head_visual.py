"""One fresh production generation arm with state-preserving step-9 estimates."""
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from benchmark import wan_head_pilot as head
from benchmark.wan_head_visual_pilot import ARMS, fixed_config, require_matched_starts
from guidance_utils.motion_probe import without_injection
from probe_wan_response import decode, predict_clean, tensor_hash


def capture_estimate(g, latent, step, label):
    """Leave latent, RNG, native RoPE, scheduler and attention caches unchanged."""
    processors = [b.attn1.processor for b in g.transformer.blocks]
    before = (tensor_hash(latent),g.scheduler.step_index,tensor_hash(g.transformer.init_rope))
    devices = list(range(torch.cuda.device_count()))
    with torch.no_grad(),torch.random.fork_rng(devices=devices),without_injection(processors):
        estimate = predict_clean(g,latent,step)
        decode(g,estimate,Path(g.output_path)/f'estimated_clean_{label}.mp4')
        if g.config.guidance_blocks:
            g._set_kv_mode(g.config.guidance_blocks,inject=False,copy=True)
            with g.probe.phase('visual_endpoint',step,g.timesteps[step]):
                loss = g.compute_motion_flow_loss(latent,g.timesteps[step])
                g.probe.emit('visual_endpoint_loss',label=label,loss=float(loss))
    after = (tensor_hash(latent),g.scheduler.step_index,tensor_hash(g.transformer.init_rope))
    if before != after: raise RuntimeError('Estimate instrumentation changed sampling state')
    g.probe.emit('visual_estimate',step=step,label=label,latent_sha256=before[0],state_preserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--output_path',type=Path,required=True)
    args = parser.parse_args()
    plan = head.load_plan(args.plan.parent)
    if plan['stage']!='visual': parser.error('Requires a gated visual plan')
    root = args.output_path.resolve()
    if root.exists() and any(root.iterdir()): parser.error('Use a fresh attempt directory')
    root.mkdir(parents=True,exist_ok=True)
    from motion_guidance_wan import WanGuidance, WAN_NEGATIVE_PROMPT

    class VisualWan(WanGuidance):
        def guidance_step(self,x,i,t,mode,loss_type):
            if i==9: capture_estimate(self,x,i,'before')
            result = super().guidance_step(x,i,t,mode,loss_type)
            if i==9: capture_estimate(self,result[0],i,'after')
            return result

        def denoise_step(self,latents,i,prompt_embeds,rope=None):
            if i==9 and not self.config.guidance_blocks: capture_estimate(self,latents,i,'before')
            return super().denoise_step(latents,i,prompt_embeds,rope=rope)

    config = OmegaConf.merge(OmegaConf.load('configs/guidance_config_wan.yaml'),
        fixed_config(plan,args.arm,root),dict(negative_prompt=WAN_NEGATIVE_PROMPT))
    OmegaConf.save(config,root/'suite_config.yaml')
    g = VisualWan(config)
    initial = dict(latent_sha256=tensor_hash(g.init_latents),reference_latent_sha256=tensor_hash(g.motion_latent),
        conditioning_sha256=tensor_hash(g.guidance_embeds),source_conditioning_sha256=tensor_hash(g.source_embeds),
        rope_sha256=tensor_hash(g.transformer.init_rope),timesteps=g.timesteps.cpu().tolist(),
        sigmas=g.scheduler.sigmas.cpu().tolist(),model_revision=getattr(g.transformer.config,'_commit_hash',None),
        guidance_blocks=list(config.guidance_blocks),flow_head=config.flow_head)
    head.write_json(root/'initial_state.json',initial)
    # Check against completed arms before paying for another full generation.
    states = [initial]
    for arm in ARMS:
        marker = args.plan.parent/f'{arm}_done.json'
        if marker.is_file():
            completed = args.plan.parent/json.loads(marker.read_text(encoding='utf-8'))['directory']
            states.append(json.loads((completed/'initial_state.json').read_text(encoding='utf-8')))
    require_matched_starts(states)
    result = g.run(custom_name='final')
    unchanged = (tensor_hash(g.transformer.init_rope)==initial['rope_sha256']
                 and g.transformer.trainable_rope is None)
    head.write_json(root/'complete.json',dict(arm=args.arm,result=Path(result).name,native_rope_unchanged=unchanged))
    print('Completed production generation:',result,flush=True)


if __name__=='__main__':
    main()
