"""Bounded pretrained Wan acceptance: native parity, vanilla review, two references.

Used by wan_port_acceptance.ipynb. No automatic success decision from AMF.
"""
import copy
import hashlib
import json
import platform
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from diffusers.models.transformers.transformer_wan import WanAttnProcessor, WanTransformer3DModel
from huggingface_hub import HfApi, snapshot_download
from omegaconf import OmegaConf
from PIL import Image

from motion_guidance_wan import WanGuidance, WAN_NEGATIVE_PROMPT, save_video
from guidance_utils.motion_probe import without_injection
from guidance_utils.wan_reference_diagnostics import reference_images
from probe_wan_response import decode, tensor_hash


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def difference(a, b):
    delta = a.detach().float() - b.detach().float()
    return dict(rms=float(delta.square().mean().sqrt()), max_abs=float(delta.abs().max()),
                changed_fraction=float((delta != 0).float().mean()))


@contextmanager
def native_transformer(model):
    """Use the installed native forward/attention with the SAME loaded weights."""
    # Accelerate offload wraps forward and stores the implementation in
    # _old_forward. Keep that wrapper active during the native parity pass.
    forward_attribute = '_old_forward' if hasattr(model, '_hf_hook') else 'forward'
    blocks, forward = model.blocks, getattr(model, forward_attribute)
    processors = [b.attn1.processor for b in blocks]
    try:
        model.blocks = torch.nn.ModuleList([getattr(b, 'module', b) for b in blocks])
        for block in model.blocks:
            block.attn1.set_processor(WanAttnProcessor())
        setattr(model, forward_attribute, MethodType(WanTransformer3DModel.forward, model))
        yield
    finally:
        model.blocks = blocks
        setattr(model, forward_attribute, forward)
        for block, processor in zip(blocks, processors):
            block.attn1.set_processor(processor)


class AcceptanceRun:
    """One model load. Four full trajectories maximum: native/off/forward/reverse.

    The native/off pair checks the port, then the two guided arms reuse exactly
    that off baseline. Call compare() only after reviewing the vanilla video.
    """
    def __init__(self, output, video, prompt, seed=1, revision=None, model=None):
        self.root = Path(output).resolve()
        if self.root.exists():
            raise ValueError('Use a fresh output directory; never mix source/environment revisions.')
        self.root.mkdir(parents=True)
        self._parity_started = self._compare_started = False
        if model is not None:
            # Dependency injection for a tiny-model integration test only.
            self.g = model
            self.config = OmegaConf.to_container(model.config, resolve=True)
            self._finish_setup('tiny random test fixture', None)
            return
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 35 * 2**30:
            raise RuntimeError('Use an A100 40 GB or an 80 GB A100/H100 with high host RAM.')
        cpu_offload = torch.cuda.get_device_properties(0).total_memory < 70 * 2**30
        print(f'MEMORY: model CPU offload={cpu_offload}; keep high-RAM enabled.', flush=True)
        repo = 'Wan-AI/Wan2.1-T2V-14B-Diffusers'
        commit = HfApi().model_info(repo, revision=revision).sha
        print('SETUP: one pinned checkpoint load and one reference feature extraction; no generation yet.', flush=True)
        model_path = snapshot_download(repo, revision=commit)
        config = OmegaConf.merge(OmegaConf.load('configs/guidance_config_wan.yaml'), dict(
            model_key=model_path, video_path=str(Path(video).resolve()), output_path=str(self.root/'setup'),
            target_prompt=prompt, negative_prompt=WAN_NEGATIVE_PROMPT, source_prompt='', seed=seed,
            height=480, width=832, num_frames=21, num_inference_steps=50,
            scheduler='unipc', flow_shift=3., guidance_scale=5.,
            guidance_blocks=[20], injection_blocks=[], guidance_timestep_range=[50,40],
            lr=[.002,.001], optimization_steps=5, motion_temp=2., flow_head=None,
            flow_max_disp=None, flow_min_conf=None, flow_region_masks=None,
            loss_type='flow', flow_loss='mse', threshloss=True, argmax_motion_flow=True,
            opt_mode='latent', guidance_mode='latent', inject_embeds=False, save_embeds=False,
            save_format='mp4', verbose=False, reference_only=False, probe=False, probe_rope=False,
            enable_model_cpu_offload=cpu_offload, enable_gradient_checkpointing=True))
        self.config = OmegaConf.to_container(config, resolve=True)
        self.g = WanGuidance(config)
        self._finish_setup(repo, commit)

    def _finish_setup(self, repo, revision):
        g = self.g
        self.guidance_blocks = list(g.config.guidance_blocks)
        self.scheduler_template = copy.deepcopy(g.scheduler)
        self.frames = reference_images(g.config.video_path, g.num_frames, g.resolution)
        self.sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path('motion_guidance_wan.py'), Path(__file__),
                                  *Path('guidance_utils').glob('*.py'), Path('configs/guidance_config_wan.yaml')]}
        self.manifest = dict(
            model=repo, checkpoint_revision=revision, mode='T2V', image_conditioning=None,
            backend='Diffusers WanPipeline / PyTorch SDPA, no TeaCache or compilation',
            packages={p: version(p) for p in ('torch','diffusers','transformers','huggingface-hub','ftfy','accelerate')},
            python=platform.python_version(), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            parameter_dtypes={name: str(p.dtype) for name,p in g.transformer.named_parameters()
                              if 'time_embedder' in name or name == 'patch_embedding.weight'},
            latent_dtype=str(g.init_latents.dtype), latent_shape=list(g.init_latents.shape),
            token_grid=[g.latent_num_frames,g.patches_height,g.patches_width],
            valid_video_tokens=g.latent_num_frames*g.patches_height*g.patches_width,
            batch_selection='B=1 per forward; source prompt for reference; positive target for AMF; separate positive/negative CFG passes',
            initial_latent_sha256=tensor_hash(g.init_latents), conditioning_sha256=tensor_hash(g.guidance_embeds),
            source_conditioning_sha256=tensor_hash(g.source_embeds), rope_sha256=tensor_hash(g.transformer.init_rope),
            reference_rgb_sha256=[hashlib.sha256(f.tobytes()).hexdigest() for f in self.frames],
            scheduler_class=type(g.scheduler).__name__, scheduler_config=dict(g.scheduler.config),
            transformer_config=dict(g.transformer.config), vae_config=dict(g.vae.config),
            sigmas=g.scheduler.sigmas.tolist(), timesteps=g.timesteps.cpu().tolist(),
            config=self.config, source_sha256=self.sources)
        source_root = Path(__file__).resolve().parents[1]
        if (source_root/'.git').exists():
            from benchmark.wan_acceptance_runtime import git_identity
            self.manifest['source_git'] = git_identity(source_root)
        write_json(self.root/'manifest.json', self.manifest)
        OmegaConf.save(g.config, self.root/'configuration.yaml')

    def _reset(self, arm):
        g = self.g
        for file, expected in self.sources.items():
            if hashlib.sha256(Path(file).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Source changed during acceptance run: '+file)
        if tensor_hash(g.init_latents) != self.manifest['initial_latent_sha256']:
            raise RuntimeError('Initial noise changed')
        if tensor_hash(g.guidance_embeds) != self.manifest['conditioning_sha256']:
            raise RuntimeError('Conditioning changed')
        g.scheduler = copy.deepcopy(self.scheduler_template)
        g.pipe.scheduler = g.scheduler
        g.transformer.trainable_rope = None
        g.transformer.stop_after_block = None
        g.output_path = str(self.root/arm)
        Path(g.output_path).mkdir()
        g.config.output_path = g.output_path
        for b in g.transformer.blocks:
            b.attn1.processor.inject_kv = b.attn1.processor.copy_kv = False
            b.attn1.processor.clear()

    @torch.no_grad()
    def parity(self, atol=1e-5, rtol=1e-5):
        if self._parity_started:
            raise RuntimeError('Parity budget already used; inspect the saved result before starting a new run.')
        self._parity_started = True
        print('HYPOTHESIS: guidance-off matches native WanPipeline with identical loaded weights and inputs. '
              'EXPECTED: every next latent agrees within atol=rtol=1e-5. LIMIT: 2 full unguided trajectories.', flush=True)
        g = self.g
        self._reset('native')
        native_states = []
        def capture(pipe, i, t, values):
            native_states.append(values['latents'].detach().cpu())
            return values
        with native_transformer(g.transformer):
            result = g.pipe(prompt_embeds=g.guidance_embeds[1:2], negative_prompt_embeds=g.guidance_embeds[:1],
                            latents=g.init_latents.clone(), height=g.config.height, width=g.config.width,
                            num_frames=g.num_frames, num_inference_steps=g.num_inference_steps,
                            guidance_scale=g.guidance_scale, output_type='latent', callback_on_step_end=capture).frames
        decode(g, result, self.root/'native/final.mp4')
        self._reset('off')
        g.config.guidance_blocks = []
        rows = []
        original = g.denoise_step
        def observe(latents, i, embeds, rope=None):
            result = original(latents, i, embeds, rope)
            expected = native_states[i].to(result.device)
            rows.append(dict(step=i, **difference(result, expected),
                             passed=bool(torch.allclose(result.float(), expected.float(), atol=atol, rtol=rtol))))
            write_json(self.root/'parity.json', dict(atol=atol, rtol=rtol, steps=rows, passed=False))
            return result
        g.denoise_step = observe
        try:
            g.run(custom_name='final')
        finally:
            g.denoise_step = original
            g.config.guidance_blocks = self.guidance_blocks
        passed = all(row['passed'] for row in rows) and len(rows) == g.num_inference_steps
        write_json(self.root/'parity.json', dict(atol=atol, rtol=rtol, steps=rows, passed=passed))
        if not passed:
            raise RuntimeError('Native parity failed. No guided generations allowed; inspect parity.json.')
        print('PARITY PASSED. Review off/final.mp4 for coherent motion before compare().', flush=True)
        return rows

    @contextmanager
    def _trace_step(self, index=9):
        """One counterfactual CFG forward; actual post-update forward is captured unchanged."""
        g = self.g
        guidance, denoise = g.guidance_step, g.denoise_step
        pending, losses = {}, []
        loss_method = g.compute_motion_flow_loss
        def observe_loss(*args, **kwargs):
            value = loss_method(*args, **kwargs)
            losses.append(float(value.detach()))
            return value
        def sample(latent, i, embeds, rope=None):
            predictions = []
            handle = g.transformer.register_forward_hook(lambda m, a, out: predictions.append(out[0].detach()))
            try:
                nxt = denoise(latent, i, embeds, rope)
            finally:
                handle.remove()
            if len(predictions) != 2:
                raise RuntimeError('Expected fresh conditional and unconditional predictions')
            cond, uncond = predictions
            return nxt, uncond + g.guidance_scale*(cond-uncond)
        def guided(x, i, t, mode, loss_type):
            if i != index:
                return guidance(x, i, t, mode, loss_type)
            live = g.scheduler
            torch.save(copy.deepcopy(live.__dict__), self.root/'forward/solver_state_before.pt')
            g.scheduler = copy.deepcopy(live)
            try:
                with without_injection([b.attn1.processor for b in g.transformer.blocks]):
                    pending['before_next'], pending['before_prediction'] = sample(x, i, g.guidance_embeds)
            finally:
                g.scheduler = live
            pending['before'] = x.detach().clone()
            g.compute_motion_flow_loss = observe_loss
            try:
                result = guidance(x, i, t, mode, loss_type)
            finally:
                g.compute_motion_flow_loss = loss_method
            pending['after'] = result[0].detach().clone()
            return result
        def sampled(x, i, embeds, rope=None):
            if i != index:
                return denoise(x, i, embeds, rope)
            nxt, prediction = sample(x, i, embeds, rope)
            torch.testing.assert_close(x, pending['after'], rtol=0, atol=0)
            report = dict(step=i, losses_before_updates=losses,
                          update_fp32=difference(pending['after'],pending['before']),
                          update_after_input_cast=difference(pending['after'].to(g.dtype),pending['before'].to(g.dtype)),
                          recomputed_cfg_prediction=difference(prediction,pending['before_prediction']),
                          next_scheduler_state=difference(nxt,pending['before_next']),
                          same_solver_state=True, actual_sampler_received_optimized_latent=True)
            write_json(self.root/'forward/step_trace.json',report)
            arrays={**pending,'after_prediction':prediction,'after_next':nxt}
            np.savez_compressed(self.root/'forward/step_trace.npz',
                                **{k:v.detach().float().cpu().numpy() for k,v in arrays.items()})
            return nxt
        g.guidance_step, g.denoise_step = guided, sampled
        try:
            yield
        finally:
            g.guidance_step, g.denoise_step, g.compute_motion_flow_loss = guidance, denoise, loss_method

    def compare(self, vanilla_review):
        if self._compare_started:
            raise RuntimeError('Two-arm generation budget already used.')
        if not json.loads((self.root/'parity.json').read_text())['passed']:
            raise RuntimeError('Pretrained native parity must pass first')
        if not vanilla_review or not vanilla_review.get('credible_motion') or not vanilla_review.get('notes'):
            raise ValueError('Record the observed vanilla subject motion and quality before generating guided arms.')
        self._compare_started = True
        write_json(self.root/'vanilla_review.json',vanilla_review)
        print('HYPOTHESIS: changing only reference order changes decoded subject trajectory. '
              'EXPECTED: opposing reference-dependent travel, beyond appearance differences. '
              'LIMIT: 2 guided generations, no KV, fixed block/head aggregation/timing/LR; '
              'one extra pre-update CFG forward for the step-9 trace.', flush=True)
        g = self.g
        for arm, frames in (('forward',self.frames),('reverse',self.frames[::-1])):
            self._reset(arm)
            folder=self.root/arm/'reference'; folder.mkdir()
            for i, frame in enumerate(frames): Image.fromarray(frame).save(folder/f'{i:05d}.png')
            g.config.video_path=str(folder)
            g.config.guidance_blocks=self.guidance_blocks
            g.motion_latent=g.load_latent()
            g.motion_attn_features=g.load_attn_features()
            if arm=='forward':
                with self._trace_step(): g.run(custom_name='final')
            else:
                g.run(custom_name='final')
        from colab_utils import show_videos
        import imageio.v3 as iio
        import imageio.v2 as iio_writer
        from PIL import ImageDraw
        paths=[self.root/a/'final.mp4' for a in ('off','forward','reverse')]
        paths += [self.root/a/'original.mp4' for a in ('forward','reverse')]
        labels=['Guidance off','Intended reference','Opposite reference','Reference','Reversed reference']
        videos=[iio.imread(path,plugin='FFMPEG') for path in paths]
        panels=[]
        for i in range(g.num_frames):
            panel=Image.new('RGB',(416*5,266),'white')
            for col,(frames,label) in enumerate(zip(videos,labels)):
                panel.paste(Image.fromarray(frames[i]).resize((416,240)),(416*col,26))
                ImageDraw.Draw(panel).text((416*col+5,5),label,fill='black')
            panels.append(np.asarray(panel))
        iio_writer.mimwrite(self.root/'comparison.mp4',panels,fps=16,codec='libx264',macro_block_size=2)
        page=show_videos([self.root/a/'final.mp4' for a in ('off','forward','reverse')]
                        +[self.root/a/'original.mp4' for a in ('forward','reverse')],
                        ['Guidance off','Intended reference','Opposite reference','Reference','Reversed reference'])
        (self.root/'comparison.html').write_text(page.data,encoding='utf-8')
        write_json(self.root/'status.json',dict(status='decoded videos ready for independent subject trajectory review',
                   success=False, next='Annotate decoded subject tracks and review videos; AMF is not an acceptance metric.'))
        return page
