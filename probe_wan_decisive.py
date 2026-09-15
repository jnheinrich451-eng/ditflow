"""Run one arm of the decisive Wan test. Launched by benchmark/wan_decisive_pilot.py::run."""
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from benchmark import wan_decisive_pilot as pilot


class DecisiveMixin:
    """Adds the random and SDEdit arms and records what guidance did. AMF arms run production code unchanged."""

    def _records(self):
        if '_decisive' not in self.__dict__:
            self.__dict__['_decisive'] = dict(update_rms={}, losses={}, step=None)
        return self.__dict__['_decisive']

    def load_attn_features(self):
        if self.config.get('decisive_kind') == 'random':
            return {}  # Random arms never evaluate an AMF loss, so no reference field is needed.
        return super().load_attn_features()

    def compute_motion_flow_loss(self, x, ts, rope=None):
        loss = super().compute_motion_flow_loss(x, ts, rope=rope)
        records = self._records()
        records['losses'].setdefault(str(records['step']), []).append(float(loss.detach()))
        return loss

    def guidance_step(self, x, i, t, mode, loss_type):
        kind, records = self.config.get('decisive_kind'), self._records()
        if i not in pilot.GUIDANCE_STEPS:
            raise ValueError(f'Guidance is frozen to sampling indices {pilot.GUIDANCE_STEPS}, got {i}')
        before = x.detach().float().clone()
        if kind == 'amf':
            records['step'] = i
            result, rope = super().guidance_step(x, i, t, mode, loss_type)
        elif kind == 'random':
            result = pilot.random_perturbation(before, self.config.random_dose[str(i)], self.config.random_seed, i)
            rope = None
        else:
            raise ValueError(f'The {kind!r} arm must not guide')
        records['update_rms'][str(i)] = pilot.rms(result.float() - before)
        return result.detach(), rope

    def run(self, rope=None, custom_name=None):
        if self.config.get('decisive_kind') == 'sdedit':
            start = int(self.config.sdedit_start)
            sigma = float(self.scheduler.sigmas[start])
            # Same initial noise as every other arm, mixed with the reference at this sampling index.
            self.init_latents = pilot.sdedit_start_latent(self.motion_latent, self.init_latents, sigma)
            self.timesteps = self.timesteps[start:]
            self.scheduler.set_begin_index(start)
            self._records().update(sdedit_start=start, sdedit_sigma=sigma)
        return super().run(rope=rope, custom_name=custom_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--arm', choices=pilot.ARMS, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    args = parser.parse_args()
    plan_root, root = args.plan.resolve().parent, args.output_path.resolve()
    plan = pilot.checked_plan(plan_root)
    pilot.timing.require_same_environment(pilot.read_json(plan_root / 'environment.json'), pilot.environment_snapshot())
    if root.exists() and any(root.iterdir()):
        parser.error('Use a fresh attempt directory')
    root.mkdir(parents=True, exist_ok=True)
    kind = pilot.KIND[args.arm]
    dose = pilot.matched_dose(plan_root) if kind == 'random' else None

    from motion_guidance_wan import WAN_NEGATIVE_PROMPT, WanGuidance

    class DecisiveWan(DecisiveMixin, WanGuidance):
        pass

    config = OmegaConf.merge(OmegaConf.load('configs/guidance_config_wan.yaml'),
                             pilot.fixed_config(plan, args.arm, root, dose), dict(negative_prompt=WAN_NEGATIVE_PROMPT))
    OmegaConf.save(config, root / 'suite_config.yaml')
    g = DecisiveWan(config)
    initial = pilot.initial_state(g)
    pilot.write_json(root / 'initial_state.json', initial)
    if args.arm != 'off':
        pilot.require_common_start(initial, pilot.read_json(pilot.completed(plan_root, 'off') / 'initial_state.json'))
    native_rope = g.transformer.init_rope.detach().clone()
    g.run(custom_name='final')
    records = g._records()
    pilot.write_json(root / 'dose.json', dict(arm=args.arm, kind=kind, update_rms=records['update_rms'],
                                              losses=records['losses'], sdedit_start=records.get('sdedit_start'),
                                              sdedit_sigma=records.get('sdedit_sigma')))
    unchanged = torch.equal(native_rope, g.transformer.init_rope) and g.transformer.trainable_rope is None
    frozen = all(not p.requires_grad and p.grad is None for p in g.transformer.parameters())
    pilot.write_json(root / 'complete.json', dict(arm=args.arm, native_rope_unchanged=unchanged, frozen_weights=frozen))
    print('Completed decisive arm:', args.arm, root / 'final.mp4', flush=True)


if __name__ == '__main__':
    main()
