"""One new generation: identical head/reference/step/budget, restricted loss pairs."""
import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from benchmark import wan_head_pilot as head
from benchmark import wan_pair_pilot as pilot
from probe_wan_noised_reference import NoisedReferenceMixin, initial_state
from probe_wan_head_visual import capture_estimate
from probe_wan_response import tensor_hash


class ForwardAdjacentMixin:
    """Filter the production validity mask; reference and target share its indices."""
    @torch.no_grad()
    def load_attn_features(self):
        if self.config.flow_pair_mode!=pilot.PAIR_MODE:
            raise ValueError('This experiment requires forward-adjacent pair selection')
        features = super().load_attn_features()
        indices = pilot.adjacent_pairs(self.latent_num_frames).tolist()
        for block_id in self.config.guidance_blocks:
            block = f'block_{block_id}_attn1_processor'
            original = self.motion_attn_masks[block]
            selected = torch.zeros_like(original)
            selected[indices] = original[indices]
            if not selected.any(): raise ValueError('No forward-adjacent entries survive the reference mask')
            self.motion_attn_masks[block] = selected
            filename = f'selected_pair_reference_{block}.npz'
            np.savez_compressed(self.probe.path/filename,flow=features[block].detach().float().cpu().numpy(),
                                mask=selected.cpu().numpy())
            self.probe.emit('selected_pair_reference',block=block,file=filename,pair_indices=indices,
                valid_positions=int(selected.sum()),base_valid_positions=int(original.sum()))
        return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output_path',type=Path,required=True)
    args = parser.parse_args()
    plan_root = args.plan.resolve().parent
    plan = pilot.checked_plan(plan_root)
    previous = plan_root/'previous'
    pilot.require_reusable_environment(pilot.read_json(previous/'environment.json'),head.environment_snapshot())
    root = args.output_path.resolve()
    if root.exists() and any(root.iterdir()): parser.error('Use a fresh attempt directory')
    root.mkdir(parents=True,exist_ok=True)
    from motion_guidance_wan import WanGuidance, WAN_NEGATIVE_PROMPT

    class PairWan(ForwardAdjacentMixin,NoisedReferenceMixin,WanGuidance):
        def guidance_step(self,x,i,t,mode,loss_type):
            if i!=9: raise ValueError('Unexpected guidance index')
            state = dict(latent_sha256=tensor_hash(x),step=9,scheduler_index=self.scheduler.step_index)
            head.write_json(root/'step9_state.json',state)
            capture_estimate(self,x,i,'before')
            pilot.require_before_update_match(root,previous)
            print('Matched previous reference, latent, estimate and all pre-update pair fields.',flush=True)
            result = super().guidance_step(x,i,t,mode,loss_type)
            capture_estimate(self,result[0],i,'after')
            return result

    config = OmegaConf.merge(OmegaConf.load('configs/guidance_config_wan.yaml'),
        pilot.fixed_config(plan,root),dict(negative_prompt=WAN_NEGATIVE_PROMPT))
    OmegaConf.save(config,root/'suite_config.yaml')
    g = PairWan(config)
    initial = initial_state(g)
    head.write_json(root/'initial_state.json',initial)
    pilot.require_reference_match(root,previous)
    g.run(custom_name='final')
    unchanged = tensor_hash(g.transformer.init_rope)==initial['rope_sha256'] and g.transformer.trainable_rope is None
    frozen = all(not p.requires_grad and p.grad is None for p in g.transformer.parameters())
    head.write_json(root/'complete.json',dict(arm='forward_adjacent',native_rope_unchanged=unchanged,frozen_weights=frozen))
    print('Completed forward-adjacent generation:',root/'final.mp4',flush=True)


if __name__=='__main__': main()
