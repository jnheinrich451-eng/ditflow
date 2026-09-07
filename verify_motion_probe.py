"""Weight-free checks (plus a CUDA integration check when available)."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from guidance_utils.motion_probe import MotionProbe, adjacent_attention, flow_summary, without_injection
from guidance_utils.wan_motion_flow_utils import compute_motion_flow


def owner_for(model, root, family="wan", enabled=True):
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(2)
    return SimpleNamespace(
        config=OmegaConf.create(dict(probe=enabled, probe_blocks=[0, 1], probe_steps=[0], loss_type="flow",
                                     guidance_blocks=[1], motion_temp=2, model_key="tiny", seed=0)),
        transformer=model, output_path=root, patches_height=2, patches_width=3, latent_num_frames=2,
        scheduler=scheduler, timesteps=scheduler.timesteps)


class ProbeTests(unittest.TestCase):
    def test_direction_and_dense_equivalence(self):
        h, w, f, n = 2, 3, 2, 6
        q = torch.eye(n).repeat(f, 1)[None, :, None, :] * 10
        k = q.clone()
        k[:, n:] = q[:, :n][:, (torch.arange(n) - 1) % n]
        data = adjacent_attention(q, k, h, w, f, 2)
        # Ignore row-boundary wraparound in this synthetic permutation.
        valid = np.arange(n) % w != w - 1
        np.testing.assert_array_equal(data["hard"][0, valid], np.tile([1, 0], (valid.sum(), 1)))
        dense = compute_motion_flow(q, k, h, w, f, temp=2, argmax=False)
        np.testing.assert_allclose(data["soft"][0], dense[1].numpy(), atol=1e-6)
        vertical = np.tile([0, 1], (1, n, 1))
        stats = flow_summary(vertical, h, w)[0]
        self.assertEqual(stats["mean_dx_fraction"], 0)
        self.assertEqual(stats["mean_dy_fraction"], 1 / h)
        self.assertEqual(stats["horizontal_fraction_of_moving"], 0)

    def test_uniform_attention_is_not_zero_flow(self):
        q = torch.zeros(1, 12, 2, 4)
        data = adjacent_attention(q, q, 2, 3, 2, 2)
        np.testing.assert_allclose(data["confidence"], 1 / 6)
        np.testing.assert_allclose(data["entropy"], 1, atol=1e-6)
        np.testing.assert_allclose(data["soft"][0, 0], [1, 0.5], atol=1e-6)

    def test_cache_restoration_on_error(self):
        p = SimpleNamespace(inject_kv=True, copy_kv=False, query=object(), key=object(), value=object())
        before = p.__dict__.copy()
        with self.assertRaises(RuntimeError):
            with without_injection([p]):
                self.assertFalse(p.inject_kv)
                p.key = None
                raise RuntimeError("test")
        self.assertEqual(p.__dict__, before)

    def test_wan_stock_parity_and_observer_invariance(self):
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
        from guidance_utils.wan_transformer import ControlledWanTransformer
        from guidance_utils.wan_modules import WanInjectionProcessor
        tiny = dict(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8,
                    in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=32,
                    num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
                    eps=1e-6, rope_max_seq_len=64)
        torch.manual_seed(17)
        stock = WanTransformer3DModel(**tiny).eval().requires_grad_(False)
        model = ControlledWanTransformer(**tiny).eval().requires_grad_(False)
        model.load_state_dict(stock.state_dict())
        for i, block in enumerate(model.blocks):
            block.attn1.set_processor(WanInjectionProcessor(f"block_{i}_attn1_processor"))
        x = torch.randn(1, 4, 2, 4, 6)
        text = torch.randn(1, 5, 16)
        kwargs = dict(timestep=torch.tensor([500]), encoder_hidden_states=text, return_dict=False)
        torch.testing.assert_close(model(x, **kwargs)[0], stock(x, **kwargs)[0], rtol=1e-5, atol=1e-6)
        baseline_x = x.clone().requires_grad_()
        baseline = model(baseline_x, **kwargs)[0]
        baseline.square().mean().backward()
        with tempfile.TemporaryDirectory() as root:
            owner = owner_for(model, root)
            owner.config.probe_rope = True
            probe = MotionProbe(owner, "wan")
            for checkpoint in (False, True):
                if checkpoint:
                    model.enable_gradient_checkpointing()
                xg = x.clone().requires_grad_()
                rng = torch.get_rng_state().clone()
                with probe.phase("reference"):
                    output = model(xg, **kwargs)[0]
                count = len((probe.path / "events.jsonl").read_text().splitlines())
                output.square().mean().backward()
                self.assertEqual(count, len((probe.path / "events.jsonl").read_text().splitlines()))
                torch.testing.assert_close(output, baseline, rtol=0, atol=0)
                torch.testing.assert_close(xg.grad, baseline_x.grad, rtol=0, atol=0)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            with probe.phase("denoise_cond", 0, 500):
                model(x, **kwargs)
            with probe.phase("final_latent"):
                model(x, **kwargs)
            probe.optimization(0, 500, 0, torch.tensor(1.), x, x + 0.01, {"rms": 0.1}, 0.002)
            probe.sampling(0, 500, x, x + 0.01, x + 0.1, owner_for(model, root).scheduler)
            from probe_report import make_report, plot_capture
            report = make_report([root], Path(root) / "report")
            self.assertIn("data:image/png;base64", report.read_text(encoding="utf-8"))
            import matplotlib.pyplot as plt
            plt.close(plot_capture(root, 1, "denoise_cond", step=0, kind="soft"))

    def test_cog_processor_invariance_and_prefix(self):
        from guidance_utils.custom_modules import InjectionProcessor
        torch.manual_seed(3)
        d = 8
        attn = SimpleNamespace(heads=2, norm_q=None, norm_k=None, is_cross_attention=False,
                               to_q=torch.nn.Linear(d, d), to_k=torch.nn.Linear(d, d),
                               to_v=torch.nn.Linear(d, d), to_out=[torch.nn.Linear(d, d), torch.nn.Identity()])
        proc = InjectionProcessor("block_0_attn1_processor")
        x = torch.randn(1, 12, d, requires_grad=True)
        txt = torch.randn(1, 226, d)
        baseline = proc(attn, x, txt)[0]
        baseline.square().mean().backward()
        gradient = x.grad.clone()
        with tempfile.TemporaryDirectory() as root:
            model = SimpleNamespace(transformer_blocks=[SimpleNamespace(attn1=SimpleNamespace(processor=proc))])
            owner = owner_for(model, root)
            owner.config.probe_blocks = [0]
            probe = MotionProbe(owner, "cogvideox")
            x.grad = None
            with probe.phase("reference"):
                output = proc(attn, x, txt)[0]
            output.square().mean().backward()
            torch.testing.assert_close(output, baseline, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, gradient, rtol=0, atol=0)
            event = json.loads((probe.path / "events.jsonl").read_text().splitlines()[0])
            self.assertEqual(event["query"]["shape"], [1, 12, 2, 4])

    def test_disabled_probe_no_output(self):
        with tempfile.TemporaryDirectory() as root:
            probe = MotionProbe(owner_for(None, root, enabled=False), "wan")
            with probe.phase("reference"):
                self.assertIsNone(probe.context)
            probe.sampling(0, 0, None, None, None, None)
            self.assertEqual(list(Path(root).iterdir()), [])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required by WanGuidance's forward")
    def test_wan_guidance_and_sampling_integration(self):
        """Exercise real bf16 guidance/CFG methods without downloading weights."""
        from diffusers import FlowMatchEulerDiscreteScheduler
        from motion_guidance_wan import WanGuidance
        from guidance_utils.wan_transformer import ControlledWanTransformer
        outputs = []
        for enabled in (False, True):
            torch.manual_seed(11)
            with tempfile.TemporaryDirectory() as root:
                g = WanGuidance.__new__(WanGuidance)
                torch.nn.Module.__init__(g)
                g.config = OmegaConf.create(dict(probe=enabled, probe_blocks=[0, 1, 2], probe_steps=[0],
                                                probe_rope=enabled,
                                                guidance_blocks=[1], injection_blocks=[0], loss_type="flow",
                                                motion_temp=2, softmax_fp32=True, argmax_motion_flow=True,
                                                threshloss=False, optimization_steps=2, verbose=False,
                                                save_embeds=False, flow_loss="mse"))
                g.device, g.dtype = torch.device("cuda"), torch.bfloat16
                g.transformer = ControlledWanTransformer(
                    patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8,
                    in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=32,
                    num_layers=3, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
                    eps=1e-6, rope_max_seq_len=64).to(device=g.device, dtype=g.dtype).eval().requires_grad_(False)
                g.transformer.enable_gradient_checkpointing()
                g.latent_height, g.latent_width, g.patch_size = 4, 6, 2
                g.patches_height, g.patches_width, g.latent_num_frames = 2, 3, 2
                g.checkpoint_amf, g._guidance_scale = False, 5
                g.scheduler = FlowMatchEulerDiscreteScheduler(shift=3)
                g.scheduler.set_timesteps(2, device=g.device)
                g.timesteps = g.scheduler.timesteps
                g.lr_range = np.array([0.002])
                g.lr_by_step = {0: 0.002}
                g.output_path = root
                g.register_guidance([1])
                g.register_attention_processor([0, 1, 2])
                g.probe = MotionProbe(g, "wan")
                g.motion_latent = torch.randn(1, 4, 2, 4, 6, device=g.device, dtype=g.dtype)
                g.source_embeds = torch.randn(1, 5, 16, device=g.device, dtype=g.dtype)
                g.guidance_embeds = torch.randn(2, 5, 16, device=g.device, dtype=g.dtype)
                g.motion_timestep = torch.tensor([0], device=g.device)
                g.motion_attn_features = g.load_attn_features()
                x = torch.randn_like(g.motion_latent).float()
                t = g.timesteps[0]
                g._set_kv_mode([0], inject=False, copy=True)
                with torch.no_grad():
                    g._forward_transformer(g._add_noise(g.motion_latent.float(), x, t),
                                           g.guidance_embeds[1:2], t[None], stop_at="injection")
                g._set_kv_mode([0], inject=True, copy=False)
                optimized, rope = g.guidance_step(x, 0, t, "latent", "flow")
                result = g.denoise_step(optimized, 0, g.guidance_embeds, rope)
                g.probe.sampling(0, t, x, optimized, result, g.scheduler)
                outputs.append(result.cpu())
                if enabled:
                    events = [json.loads(s) for s in (g.probe.path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(sum(e["kind"] == "optimization" for e in events), 2)
                    self.assertEqual(sum(e["kind"] == "training_flow" for e in events), 2)
                    self.assertTrue(any(e.get("stage") == "reference" and e.get("block") == "block_2_attn1_processor" for e in events))
                    self.assertFalse(any(e.get("stage") == "guidance" and e.get("block") == "block_2_attn1_processor" for e in events))
                    self.assertTrue(any(e.get("stage") == "denoise_cond" and e.get("injected") for e in events))
                    self.assertTrue(any(e['kind'] == 'rope_attention' and e['actual_attention_injected']
                                        and e['key_source'] == 'native_before_injection' for e in events))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
