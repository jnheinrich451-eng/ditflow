"""Two subject-only arms with the archived mapping and production latent Adam loop."""
import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from benchmark import wan_subject_only_pilot as pilot
from benchmark import wan_head_pilot as head
from guidance_utils.wan_control_trace import ControlRecorder
from guidance_utils.wan_subject_alignment import alignment_loss
from probe_wan_subject import SubjectLossMixin
from probe_wan_control import ControlGuidanceMixin
from probe_wan_pairs import ForwardAdjacentMixin
from probe_wan_noised_reference import NoisedReferenceMixin, initial_state
from probe_wan_response import tensor_hash


def subject_only_loss(prediction, reference, subject, background):
    # Reuse the validated region definition/reductions. Only fg enters backward.
    _, fg, bg, _ = alignment_loss(prediction, reference, subject, background, 'balanced')
    return fg, fg, bg, 1.0


class SubjectOnlyLossMixin(SubjectLossMixin):
    """Keep extraction unchanged; return the subject mean as the optimized loss."""
    def compute_motion_flow_loss(self, x, ts, rope=None):
        if len(self.config.guidance_blocks) != 1 or self.config.alignment_mode != 'subject_only':
            raise ValueError('Subject-only experiment requires one frozen block and subject_only loss')
        self._forward_transformer(x, self.guidance_embeds[1:2], ts.expand(x.shape[0]).to(self.device), rope=rope)
        proc = self.transformer.blocks[self.config.guidance_blocks[0]].attn1.processor
        flow = self._amf(proc)
        target = self.alignment['flow'].to(flow.dtype)
        subject, background = self.alignment['subject'], self.alignment['background']
        total, fg, bg, weight = subject_only_loss(flow, target, subject, background)
        if torch.is_grad_enabled() and x.requires_grad and self.config.get('record_region_gradients', True):
            fg_grad = torch.autograd.grad(fg, x, retain_graph=True)[0]
            bg_grad = torch.autograd.grad(bg, x, retain_graph=True)[0]
            iteration = getattr(self, 'subject_gradient_iteration', 0)
            self.subject_gradient_iteration = iteration+1
            if iteration >= self.config.optimization_steps:
                raise ValueError('Unexpected extra region-gradient iteration')
            path = Path(self.output_path)/'region_gradients'; path.mkdir(exist_ok=True)
            filename = path/f'iteration_{iteration:02d}.npz'
            if filename.exists():
                raise ValueError('Duplicate region-gradient capture')
            np.savez_compressed(filename, latent=ControlRecorder.array(x),
                subject_gradient=ControlRecorder.array(fg_grad), background_gradient=ControlRecorder.array(bg_grad))
            self.probe.emit('region_gradient', file=filename.relative_to(self.output_path).as_posix(),
                            step=9, iteration=iteration, subject_weight=weight, background_weight=0.)
        self.probe.training_flow(proc.block_name, flow, target, subject, total)
        self.probe.emit('full_pair_loss', block=proc.block_name, file=self.last_full_pair_file, loss=float(total.detach()))
        self.probe.emit('aligned_loss', block=proc.block_name, file=self.last_full_pair_file,
            subject_mse=float(fg.detach()), background_mse=float(bg.detach()), total=float(total.detach()),
            subject_weight=weight, mode='subject_only')
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
    pilot.require_same_environment(pilot.read_json(plan_root/'environment.json'), pilot.environment_snapshot())
    if root.exists() and any(root.iterdir()):
        parser.error('Use a fresh attempt directory')
    root.mkdir(parents=True, exist_ok=True)
    from motion_guidance_wan import WanGuidance, WAN_NEGATIVE_PROMPT

    class SubjectOnlyWan(ControlGuidanceMixin, SubjectOnlyLossMixin, ForwardAdjacentMixin, NoisedReferenceMixin, WanGuidance):
        def check_control_start(self):
            pilot.require_previous_start(root, plan_root, args.arm)

    config = OmegaConf.merge(OmegaConf.load('configs/guidance_config_wan.yaml'),
        pilot.fixed_config(plan, args.arm, root), dict(negative_prompt=WAN_NEGATIVE_PROMPT))
    OmegaConf.save(config, root/'suite_config.yaml')
    g = SubjectOnlyWan(config)
    initial = initial_state(g); head.write_json(root/'initial_state.json', initial)
    pilot.require_previous_reference(root, plan_root, args.arm)
    g.control = ControlRecorder(g)
    try:
        g.run(custom_name='final')
    finally:
        g.control.close()
    unchanged = tensor_hash(g.transformer.init_rope) == initial['rope_sha256'] and g.transformer.trainable_rope is None
    frozen = all(not p.requires_grad and p.grad is None for p in g.transformer.parameters())
    head.write_json(root/'complete.json', dict(arm=args.arm, native_rope_unchanged=unchanged, frozen_weights=frozen))
    print('Completed subject-only generation:', root/'final.mp4', flush=True)


if __name__ == '__main__':
    main()
