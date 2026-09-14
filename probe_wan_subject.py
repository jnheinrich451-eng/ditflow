"""Run one isolated subject-alignment arm; reuse the production latent optimizer."""
import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from benchmark import wan_head_pilot as head
from benchmark import wan_subject_pilot as pilot
from guidance_utils.wan_control_trace import ControlRecorder
from guidance_utils.wan_subject_alignment import alignment_loss
from probe_wan_control import ControlGuidanceMixin
from probe_wan_noised_reference import NoisedReferenceMixin, initial_state
from probe_wan_pairs import ForwardAdjacentMixin
from probe_wan_response import tensor_hash


class SubjectLossMixin:
    @torch.no_grad()
    def load_attn_features(self):
        features = super().load_attn_features()
        bundle = pilot.arrays(self.config.alignment_target)
        block = f'block_{self.config.guidance_blocks[0]}_attn1_processor'
        if (not np.array_equal(bundle['source_flow'], features[block].float().cpu().numpy())
                or not np.array_equal(bundle['source_mask'], self.motion_attn_masks[block].cpu().numpy())):
            raise ValueError('Fresh reference readout differs from frozen alignment input')
        self.alignment = {key: torch.as_tensor(bundle[key], device=self.device) for key in
                          ('flow', 'subject', 'background')}
        np.savez_compressed(self.probe.path/'aligned_reference.npz', **bundle)
        self.probe.emit('aligned_reference', block=block, file='aligned_reference.npz',
                        mode=self.config.alignment_mode)
        return features

    def compute_motion_flow_loss(self, x, ts, rope=None):
        if len(self.config.guidance_blocks) != 1:
            raise ValueError('Subject experiment requires exactly one frozen readout block')
        self._forward_transformer(x, self.guidance_embeds[1:2], ts.expand(x.shape[0]).to(self.device), rope=rope)
        proc = self.transformer.blocks[self.config.guidance_blocks[0]].attn1.processor
        flow = self._amf(proc)
        target = self.alignment['flow'].to(flow.dtype)
        subject, background = self.alignment['subject'], self.alignment['background']
        total, fg, bg, weight = alignment_loss(flow, target, subject, background, self.config.alignment_mode)
        if (torch.is_grad_enabled() and x.requires_grad
                and self.config.get('record_region_gradients', True)):
            # autograd.grad returns component derivatives without filling x.grad or model .grad.
            # The unchanged production optimizer subsequently backpropagates the actual total.
            fg_grad = torch.autograd.grad(fg, x, retain_graph=True)[0]
            bg_grad = torch.autograd.grad(bg, x, retain_graph=True)[0]
            # Ordinary probe.context is deliberately absent on middle iterations.
            # Count differentiable loss calls without enabling extra ordinary captures.
            iteration = getattr(self, 'subject_gradient_iteration', 0)
            self.subject_gradient_iteration = iteration+1
            if iteration >= self.config.optimization_steps:
                raise ValueError('Unexpected extra region-gradient iteration')
            path = Path(self.output_path)/'region_gradients'
            path.mkdir(exist_ok=True)
            filename = path/f'iteration_{iteration:02d}.npz'
            if filename.exists():
                raise ValueError('Duplicate region-gradient capture')
            np.savez_compressed(filename, latent=ControlRecorder.array(x),
                subject_gradient=ControlRecorder.array(fg_grad), background_gradient=ControlRecorder.array(bg_grad))
            self.probe.emit('region_gradient', file=filename.relative_to(self.output_path).as_posix(),
                            step=9, iteration=iteration, subject_weight=weight, background_weight=1-weight)
        self.probe.training_flow(proc.block_name, flow, target, subject | background, total)
        self.probe.emit('full_pair_loss', block=proc.block_name, file=self.last_full_pair_file, loss=float(total.detach()))
        self.probe.emit('aligned_loss', block=proc.block_name, file=self.last_full_pair_file,
            subject_mse=float(fg.detach()), background_mse=float(bg.detach()), total=float(total.detach()),
            subject_weight=weight, mode=self.config.alignment_mode)
        self._clear_kv(self.config.guidance_blocks)
        return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--arm', choices=pilot.ARMS, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    args = parser.parse_args()
    plan_root, root = args.plan.resolve().parent, args.output_path.resolve()
    plan = pilot.checked_plan(plan_root)
    head.timing.require_same_environment(pilot.read_json(plan_root/'environment.json'), pilot.environment_snapshot())
    if root.exists() and any(root.iterdir()):
        parser.error('Use a fresh attempt directory')
    root.mkdir(parents=True, exist_ok=True)
    from motion_guidance_wan import WanGuidance, WAN_NEGATIVE_PROMPT

    class SubjectWan(ControlGuidanceMixin, SubjectLossMixin, ForwardAdjacentMixin, NoisedReferenceMixin, WanGuidance):
        def check_control_start(self):
            pilot.require_previous_start(root, plan_root, args.arm)

    config = OmegaConf.merge(OmegaConf.load('configs/guidance_config_wan.yaml'),
        pilot.fixed_config(plan, args.arm, root), dict(negative_prompt=WAN_NEGATIVE_PROMPT))
    OmegaConf.save(config, root/'suite_config.yaml')
    g = SubjectWan(config)
    initial = initial_state(g)
    head.write_json(root/'initial_state.json', initial)
    pilot.require_previous_reference(root, plan_root, args.arm)
    g.control = ControlRecorder(g)
    try:
        g.run(custom_name='final')
    finally:
        g.control.close()
    unchanged = tensor_hash(g.transformer.init_rope) == initial['rope_sha256'] and g.transformer.trainable_rope is None
    frozen = all(not p.requires_grad and p.grad is None for p in g.transformer.parameters())
    head.write_json(root/'complete.json', dict(arm=args.arm, native_rope_unchanged=unchanged, frozen_weights=frozen))
    print('Completed aligned-subject generation:', root/'final.mp4', flush=True)


if __name__ == '__main__':
    main()
