"""Run one fresh Wan arm with forward-adjacent guidance and sampler captures."""
import argparse
from pathlib import Path

from omegaconf import OmegaConf

from benchmark import wan_control_pilot as pilot
from benchmark import wan_head_pilot as head
from benchmark import wan_pair_pilot as pairs
from guidance_utils.wan_control_trace import ControlRecorder
from probe_wan_head_visual import capture_estimate
from probe_wan_noised_reference import NoisedReferenceMixin, initial_state
from probe_wan_pairs import ForwardAdjacentMixin
from probe_wan_response import tensor_hash


class ControlGuidanceMixin:
    """Observe production guidance/sampling; do not replace its predictions."""
    def before_control(self, latent, step):
        state = dict(latent_sha256=tensor_hash(latent), step=step, scheduler_index=self.scheduler.step_index)
        head.write_json(Path(self.output_path)/'step9_state.json', state)
        with self.control.capture('before', step, latent):
            capture_estimate(self, latent, step, 'before')
        self.check_control_start()

    def check_control_start(self):
        """Runner supplies archive/paired-start checks; tiny-model tests override nothing."""

    def guidance_step(self, x, i, t, mode, loss_type):
        if i != 9:
            raise ValueError('Control experiment only guides at index 9')
        self.before_control(x, i)
        result = super().guidance_step(x, i, t, mode, loss_type)
        with self.control.capture('after', i, result[0]):
            capture_estimate(self, result[0], i, 'after')
        return result

    def denoise_step(self, latents, i, prompt_embeds, rope=None):
        if i == 9 and not self.config.guidance_blocks:
            self.before_control(latents, i)
        if i not in pilot.CHECKPOINTS:
            return super().denoise_step(latents, i, prompt_embeds, rope=rope)
        with self.control.capture('denoise', i, latents) as record:
            result = super().denoise_step(latents, i, prompt_embeds, rope=rope)
            record['next_latent'] = self.control.array(result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--arm', choices=pilot.ARMS, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    args = parser.parse_args()
    plan_root, root = args.plan.resolve().parent, args.output_path.resolve()
    plan = pilot.checked_plan(plan_root)
    head.timing.require_same_environment(pilot.read_json(plan_root/'environment.json'), head.environment_snapshot())
    if root.exists() and any(root.iterdir()):
        parser.error('Use a fresh attempt directory')
    root.mkdir(parents=True, exist_ok=True)
    from motion_guidance_wan import WanGuidance, WAN_NEGATIVE_PROMPT

    class ControlWan(ControlGuidanceMixin, ForwardAdjacentMixin, NoisedReferenceMixin, WanGuidance):
        def check_control_start(self):
            if args.arm == 'forward':
                pairs.require_before_update_match(root, plan_root/'previous')
            if args.arm != 'off':
                off = pairs.completed(plan_root, 'off')
                pilot.require_same_capture(root/'control/before_09.npz', off/'control/before_09.npz')

    config = OmegaConf.merge(OmegaConf.load('configs/guidance_config_wan.yaml'),
        pilot.fixed_config(plan, args.arm, root), dict(negative_prompt=WAN_NEGATIVE_PROMPT))
    OmegaConf.save(config, root/'suite_config.yaml')
    g = ControlWan(config)
    initial = initial_state(g)
    head.write_json(root/'initial_state.json', initial)
    if args.arm == 'forward':
        pairs.require_reference_match(root, plan_root/'previous')
    if args.arm != 'off':
        off = pairs.completed(plan_root, 'off')
        pilot.require_common_start(initial, pilot.read_json(off/'initial_state.json'))
    if args.arm == 'reverse':
        pilot.require_directional_targets(root, pairs.completed(plan_root, 'forward'))
    g.control = ControlRecorder(g)
    try:
        g.run(custom_name='final')
    finally:
        g.control.close()
    unchanged = tensor_hash(g.transformer.init_rope) == initial['rope_sha256'] and g.transformer.trainable_rope is None
    frozen = all(not p.requires_grad and p.grad is None for p in g.transformer.parameters())
    head.write_json(root/'complete.json', dict(arm=args.arm, native_rope_unchanged=unchanged, frozen_weights=frozen))
    print('Completed directional-control generation:', root/'final.mp4', flush=True)


if __name__ == '__main__':
    main()
