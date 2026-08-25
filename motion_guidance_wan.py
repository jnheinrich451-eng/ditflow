"""DiTFlow motion transfer for Wan2.1 T2V.

Port of `motion_guidance.py` (CogVideoX) to Wan2.1. The original is left
untouched so the paper baseline stays runnable side by side.

    python motion_guidance_wan.py \
        --video_path ./assets/bmx-trees.mp4 \
        --prompt "Leopard running up a snowy hill in a forest"

What differs from the CogVideoX implementation, and why:

  * Wan is a **rectified-flow** model. UniPC/FlowMatchEuler replace DDIM/DPM,
    `scale_model_input` is a no-op and drops out, and `add_noise` interpolates
    as `(1-s)*x0 + s*eps` rather than the DDPM alpha/sigma form.
  * Wan's VAE normalises with **per-channel** `latents_mean`/`latents_std`, not
    a single `scaling_factor`, and its latents are `(B, C, F, H, W)` -- no
    permute to `(B, F, C, H, W)`.
  * Wan has no learned absolute position embedding, so `--opt_mode emb`
    optimises RoPE. There is no `posemb` mode.
  * Text never enters self-attention, so AMF needs no text-prefix slice.
  * `stop_after_block` on the transformer replaces the forward-monkeypatching
    `change_mode` does upstream.

Known upstream bug NOT carried over: `load_attn_features` in `motion_guidance.py`
calls `compute_motion_flow` without `nframes`, so the reference AMF is always
built with the default of 6 latent frames. That is only correct at
`--video_length 24`. Here `nframes` is passed explicitly everywhere.
"""

import argparse
import gc
import math
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Union

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchvision.io import read_video, write_video
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from transformers import logging

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler, WanPipeline
from diffusers.utils import export_to_video

from guidance_utils.wan_motion_flow_utils import compute_motion_flow
from guidance_utils.wan_modules import WanInjectionProcessor, WanModuleWithGuidance
from guidance_utils.wan_transformer import ControlledWanTransformer

logging.set_verbosity_error()

MODEL_IDS = {
    "1.3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "14b": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
}

# Wan's reference negative prompt. Materially affects image quality -- do not
# swap in the CogVideoX one.
WAN_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def save_video(video, path, fps=16):
    write_video(path, video, fps=fps, video_codec="libx264", options={"crf": "17", "preset": "slow"})


def get_timesteps(timesteps, guidance_timestep_range, skip_timesteps=1):
    """Unchanged from the CogVideoX implementation -- indexes into the timestep list."""
    max_guidance_timestep, min_guidance_timestep = guidance_timestep_range
    num_inference_steps = len(timesteps)
    init_timestep = min(max_guidance_timestep, num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    t_end = min_guidance_timestep
    if t_end > 0:
        return timesteps[t_start:-t_end:skip_timesteps]
    return timesteps[t_start::skip_timesteps]


class WanGuidance(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(config["device"])
        self.dtype = torch.bfloat16
        self.batch_size = 1
        self.num_inference_steps = config["num_inference_steps"]
        self._guidance_scale = config.guidance_scale

        print(f"Loading {config.model_key}")
        # Load the controlled transformer straight from the hub rather than
        # building a second copy and copying weights across, as the CogVideoX
        # path does -- that pattern needs 2x the weights resident, which is
        # ~64GB of host RAM for the 14B.
        transformer = ControlledWanTransformer.from_pretrained(
            config.model_key, subfolder="transformer", torch_dtype=self.dtype
        )
        vae = AutoencoderKLWan.from_pretrained(config.model_key, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanPipeline.from_pretrained(
            config.model_key, transformer=transformer, vae=vae, torch_dtype=self.dtype
        )

        if config.scheduler == "unipc":
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.pipe.scheduler.config, flow_shift=config.flow_shift
            )
        elif config.scheduler == "flowmatch":
            # First-order: no multistep history to be corrupted by latent-mode
            # guidance edits between steps. Worth trying if `latent` mode drifts.
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config, shift=config.flow_shift
            )
        else:
            raise ValueError(f"Unknown scheduler {config.scheduler!r}, expected 'unipc' or 'flowmatch'")

        self.pipe.to(self.device)
        if config.enable_model_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        if hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

        self.vae = self.pipe.vae
        self.transformer = self.pipe.transformer
        self.scheduler = self.pipe.scheduler

        if config.enable_gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        print("video model loaded")

        self.generator = torch.Generator(device="cuda").manual_seed(config.seed)

        # ---- Geometry --------------------------------------------------- #
        height, width = config.height, config.width
        if height % 16 or width % 16:
            raise ValueError(f"height/width must be multiples of 16, got {height}x{width}")
        num_frames = config.num_frames
        if (num_frames - 1) % self.pipe.vae_scale_factor_temporal:
            raise ValueError(
                f"num_frames must be 4k+1 for Wan's causal VAE, got {num_frames}. "
                f"Try {((num_frames - 1) // 4) * 4 + 1}."
            )

        self.resolution = (width, height)
        self.num_frames = num_frames
        self.latent_num_frames = (num_frames - 1) // self.pipe.vae_scale_factor_temporal + 1
        self.latent_height = height // self.pipe.vae_scale_factor_spatial
        self.latent_width = width // self.pipe.vae_scale_factor_spatial
        _, p_h, p_w = self.transformer.config.patch_size
        self.patch_size = p_h
        self.patches_height = self.latent_height // p_h
        self.patches_width = self.latent_width // p_w

        seq_len = self.latent_num_frames * self.patches_height * self.patches_width
        print(
            f"latent {self.latent_num_frames}x{self.latent_height}x{self.latent_width} -> "
            f"{self.latent_num_frames} x {self.patches_height}x{self.patches_width} patches, "
            f"seq_len={seq_len}"
        )

        if str(config.checkpoint_amf).lower() == "auto":
            self.checkpoint_amf = self.latent_num_frames > 7
        else:
            self.checkpoint_amf = bool(config.checkpoint_amf)
        print(f"AMF frame-pair checkpointing: {self.checkpoint_amf}")

        # ---- Prompts ------------------------------------------------------ #
        with torch.no_grad():
            self.source_embeds, _ = self.pipe.encode_prompt(
                prompt=config["source_prompt"],
                negative_prompt=config["negative_prompt"],
                do_classifier_free_guidance=True,
                num_videos_per_prompt=1,
                max_sequence_length=512,
                device=self.device,
            )
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt=config["target_prompt"],
                negative_prompt=config["negative_prompt"],
                do_classifier_free_guidance=True,
                num_videos_per_prompt=1,
                max_sequence_length=512,
                device=self.device,
            )
            self.guidance_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(self.dtype)
            self.source_embeds = self.source_embeds.to(self.dtype)

        # umT5-XXL is ~11GB in bf16 and is finished with. Evicting it matters on
        # anything under 24GB and costs nothing on an A100.
        if not config.enable_model_cpu_offload:
            self.pipe.text_encoder.to("cpu")
            clean_memory()

        # ---- Latents ------------------------------------------------------ #
        self.init_latents = self.pipe.prepare_latents(
            batch_size=self.batch_size,
            num_channels_latents=self.transformer.config.in_channels,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=self.device,
            generator=self.generator,
        )

        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        self.timesteps = self.scheduler.timesteps
        self.guidance_schedule = get_timesteps(self.timesteps, config.guidance_timestep_range)

        self.transformer.init_rope = self.transformer.default_rope(self.init_latents.to(self.device))
        self.transformer.guidance_blocks = config.guidance_blocks

        # ---- Output paths -------------------------------------------------- #
        self.output_path = config["output_path"]
        os.makedirs(self.output_path, exist_ok=True)
        embeds_path = os.path.join(self.output_path, "embeds")
        if config.inject_embeds:
            if not os.path.exists(embeds_path):
                raise FileNotFoundError(
                    f"Embeds folder not found at {embeds_path}. Run once without --inject_embeds "
                    "so the optimised RoPE is saved, then inject it with a new prompt."
                )
        else:
            if os.path.exists(embeds_path):
                shutil.rmtree(embeds_path)
            os.makedirs(embeds_path, exist_ok=True)

        # ---- Guidance setup ------------------------------------------------ #
        self.motion_timestep = torch.tensor([0], device=self.device)
        self.register_guidance(config.guidance_blocks)
        self.register_attention_processor(list(range(len(self.transformer.blocks))))

        num_guidance_steps = config.guidance_timestep_range[0] - config.guidance_timestep_range[1] + 1
        self.lr_range = np.linspace(config.lr[0], config.lr[1], num_guidance_steps)

        print("Loading features from motion video")
        self.motion_latent = self.load_latent()
        if config.loss_type == "flow":
            self.motion_attn_features = self.load_attn_features()
        elif config.loss_type == "smm":
            self.motion_orig_features = self.load_features()
        elif config.loss_type == "moft":
            self.motion_orig_features, self.motion_channels = self.load_features(moft=True)

    # ---------------------------------------------------------------- #
    def register_guidance(self, block_idxs):
        for out_i in block_idxs:
            self.transformer.blocks[out_i] = WanModuleWithGuidance(
                self.transformer.blocks[out_i],
                self.latent_height,
                self.latent_width,
                self.patch_size,
                block_name=f"block_{out_i}",
                num_frames=self.latent_num_frames,
            )

    def register_attention_processor(self, block_idxs):
        for out_i in block_idxs:
            processor = WanInjectionProcessor(block_name=f"block_{out_i}_attn1_processor")
            self.transformer.blocks[out_i].attn1.set_processor(processor)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    def _set_kv_mode(self, block_idxs, inject: bool, copy: bool):
        for block_id in block_idxs:
            proc = self.transformer.blocks[block_id].attn1.processor
            proc.inject_kv = inject
            proc.copy_kv = copy

    def _clear_kv(self, block_idxs):
        for block_id in block_idxs:
            self.transformer.blocks[block_id].attn1.processor.clear()

    def _amf(self, processor):
        return compute_motion_flow(
            processor.query,
            processor.key,
            h=self.patches_height,
            w=self.patches_width,
            nframes=self.latent_num_frames,
            temp=self.config.motion_temp,
            argmax=False,
            checkpoint_pairs=self.checkpoint_amf,
            softmax_fp32=self.config.softmax_fp32,
            head_dim=self.transformer.config.attention_head_dim,
        )

    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def load_latent(self):
        """Load the reference video and encode it with Wan's VAE."""
        data_path = self.config.video_path

        if data_path.endswith(".mp4"):
            video = read_video(data_path, pts_unit="sec")[0].permute(0, 3, 1, 2).cuda() / 255
            video = [ToPILImage()(video[i]).resize(self.resolution) for i in range(video.shape[0])]
        else:
            images = list(Path(data_path).glob("*.png")) + list(Path(data_path).glob("*.jpg"))
            images = sorted(images, key=lambda x: int(x.stem.split("f")[-1]))
            video = [Image.open(img).resize(self.resolution).convert("RGB") for img in images]

        video = video[: self.num_frames]
        if len(video) < self.num_frames:
            raise ValueError(
                f"Reference video has {len(video)} frames but num_frames={self.num_frames} was requested."
            )
        save_video([np.array(img) for img in video], str(Path(self.output_path) / "original.mp4"))

        video = self.pipe.video_processor.preprocess_video(video)
        video = video.to(device=self.device, dtype=self.vae.dtype)

        # Wan normalises per channel, not with a single scaling_factor. Use the
        # distribution mode (not a sample) so the reference AMF is deterministic.
        latents = self.vae.encode(video).latent_dist.mode()
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        latents = (latents - latents_mean) / latents_std

        # (B, C, F, H, W) -- no permute, unlike CogVideoX.
        return latents.to(self.dtype)

    def _forward_transformer(self, hidden_states, encoder_hidden_states, timestep, rope=None, stop_at="guidance"):
        """Run the transformer for its side effects (captured QK / saved features).

        `stop_at` picks the early-exit block: "guidance" for loss passes,
        "injection" for the reference-KV cache pass (block 0 by default, so this
        skips ~97% of the network), or None for a full forward.
        """
        if stop_at == "guidance":
            blocks = self.config.guidance_blocks
        elif stop_at == "injection":
            blocks = self.config.injection_blocks
        else:
            blocks = None
        self.transformer.stop_after_block = max(blocks) if blocks else None
        try:
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                self.transformer(
                    hidden_states=hidden_states.to(self.dtype),
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    rope=rope,
                    return_dict=False,
                )
        finally:
            self.transformer.stop_after_block = None

    @torch.no_grad()
    def load_features(self, moft=False):
        motion_features, motion_channels = {}, {}
        self._forward_transformer(self.motion_latent, self.source_embeds, self.motion_timestep)

        for block_id in self.config.guidance_blocks:
            module = self.transformer.blocks[block_id]
            orig_features = module.saved_features
            motion_features[module.block_name] = orig_features

            if moft:
                orig_norm = orig_features - torch.mean(orig_features, axis=0)[None]
                num_frames, c, h, w = orig_norm.shape
                channels = orig_norm.permute(0, 2, 3, 1).reshape(-1, c)
                _, _, Vt = torch.linalg.svd(channels.to(torch.float32), full_matrices=False)
                top_n = list(torch.argsort(torch.abs(Vt[0]), descending=True)[: int(self.config.prop_motion * c)])
                motion_channels[module.block_name] = top_n

        if moft:
            return motion_features, motion_channels
        return motion_features

    @torch.no_grad()
    def load_attn_features(self):
        """AMF extraction from the reference video."""
        self._set_kv_mode(self.config.guidance_blocks, inject=False, copy=True)
        self._forward_transformer(self.motion_latent, self.source_embeds, self.motion_timestep)

        attn_features = {}
        for block_id in self.config.guidance_blocks:
            proc = self.transformer.blocks[block_id].attn1.processor
            attn_features[proc.block_name] = compute_motion_flow(
                proc.query,
                proc.key,
                h=self.patches_height,
                w=self.patches_width,
                nframes=self.latent_num_frames,
                temp=self.config.motion_temp,
                argmax=self.config.argmax_motion_flow,
                softmax_fp32=self.config.softmax_fp32,
                head_dim=self.transformer.config.attention_head_dim,
            )

        self._set_kv_mode(self.config.guidance_blocks, inject=False, copy=False)
        self._clear_kv(self.config.guidance_blocks)
        return attn_features

    # ------------------------- LOSSES -------------------------------- #
    def compute_motion_flow_loss(self, x, ts, rope=None):
        self._forward_transformer(x, self.guidance_embeds[1:2], ts.expand(x.shape[0]).to(self.device), rope=rope)

        total_loss = 0
        for block_id in self.config.guidance_blocks:
            proc = self.transformer.blocks[block_id].attn1.processor
            motion_flow = self._amf(proc)
            ref_motion_flow = self.motion_attn_features[proc.block_name].detach().to(motion_flow.dtype)

            if self.config.threshloss:
                idxs = torch.norm(ref_motion_flow, dim=-1) > 0
                attn_loss = F.mse_loss(ref_motion_flow[idxs], motion_flow[idxs])
            else:
                attn_loss = F.mse_loss(ref_motion_flow, motion_flow)
            total_loss = total_loss + attn_loss

        if self.config.guidance_blocks:
            total_loss = total_loss / len(self.config.guidance_blocks)
        self._clear_kv(self.config.guidance_blocks)
        return total_loss

    def compute_moft_loss(self, x, ts, rope=None):
        def compute_MOFT(orig, target, motion_channels):
            orig_norm = orig - torch.mean(orig, axis=0)[None]
            target_norm = target - torch.mean(target, axis=0)[None]
            loss = 0
            for f in range(orig_norm.shape[0]):
                loss += 1 - F.cosine_similarity(
                    target_norm[f, motion_channels], orig_norm[f, motion_channels].detach(), dim=0
                ).mean()
            return loss / orig_norm.shape[0]

        self._forward_transformer(x, self.guidance_embeds[1:2], ts.expand(x.shape[0]).to(self.device), rope=rope)

        total_loss = 0
        for block_id in self.config.guidance_blocks:
            module = self.transformer.blocks[block_id]
            total_loss = total_loss + compute_MOFT(
                self.motion_orig_features[module.block_name].detach(),
                module.saved_features,
                self.motion_channels[module.block_name],
            )
        if self.config.guidance_blocks:
            total_loss = total_loss / len(self.config.guidance_blocks)
        return total_loss

    def compute_smm_loss(self, x, ts, rope=None):
        def compute_SMM(orig, target):
            orig_smm = orig.mean(dim=(-1, -2), keepdim=True)
            target_smm = target.mean(dim=(-1, -2), keepdim=True)
            loss = 0
            for f in range(orig_smm.shape[0]):
                orig_diffs = orig_smm - orig_smm[f]
                target_diffs = target_smm - orig_smm[f]
                loss += 1 - F.cosine_similarity(target_diffs, orig_diffs.detach(), dim=0).mean()
            return loss / orig_smm.shape[0]

        self._forward_transformer(x, self.guidance_embeds[1:2], ts.expand(x.shape[0]).to(self.device), rope=rope)

        total_loss = 0
        for block_id in self.config.guidance_blocks:
            module = self.transformer.blocks[block_id]
            total_loss = total_loss + compute_SMM(
                self.motion_orig_features[module.block_name].detach(), module.saved_features
            )
        if self.config.guidance_blocks:
            total_loss = total_loss / len(self.config.guidance_blocks)
        return total_loss

    # --------------------- GUIDED DENOISING -------------------------- #
    def guidance_step(self, x, i, t, mode, loss_type):
        """Optimise `mode` (latent | rope) against `loss_type` at one denoising step."""
        self._set_kv_mode(self.config.guidance_blocks, inject=False, copy=True)

        lr = self.lr_range[i]
        loss_method = {
            "flow": self.compute_motion_flow_loss,
            "moft": self.compute_moft_loss,
            "smm": self.compute_smm_loss,
        }[loss_type]

        optimized_rope = None
        optimized_x = x

        if mode == "rope":
            if self.transformer.trainable_rope is None:
                base = torch.stack([self.transformer.init_rope, self.transformer.init_rope], dim=0)
            else:
                base = self.transformer.trainable_rope
            optimized_rope = base.clone().detach().to(dtype=torch.float32, device=self.device).requires_grad_(True)
            optimizer = torch.optim.Adam([optimized_rope], lr=lr)

            for _ in tqdm(range(self.config.optimization_steps), desc=f"opt t={int(t)}", leave=False):
                optimizer.zero_grad()
                total_loss = loss_method(x, t, rope=optimized_rope)
                if self.config.verbose:
                    print(f"Loss t={t}: {total_loss.item()}")
                total_loss.backward()
                optimizer.step()
                clean_memory()

            self.transformer.trainable_rope = optimized_rope.detach()
            if self.config.save_embeds:
                torch.save(optimized_rope.detach(), os.path.join(self.output_path, "embeds", f"rope_{t}.pt"))

        elif mode == "latent":
            optimized_x = x.clone().detach().to(dtype=torch.float32).requires_grad_(True)
            optimizer = torch.optim.Adam([optimized_x], lr=lr)

            for _ in tqdm(range(self.config.optimization_steps), desc=f"opt t={int(t)}", leave=False):
                optimizer.zero_grad()
                total_loss = loss_method(optimized_x, t)
                if self.config.verbose:
                    print(f"Loss t={t}: {total_loss.item()}")
                total_loss.backward()
                optimizer.step()

            if self.config.save_embeds:
                torch.save(optimized_x.detach(), os.path.join(self.output_path, "embeds", f"latent_{t}.pt"))
        else:
            raise ValueError(f"Unknown guidance mode {mode!r}. Wan supports 'latent' and 'rope' only.")

        return optimized_x.detach(), optimized_rope

    @torch.no_grad()
    def denoise_step(self, latents, i, prompt_embeds, rope=None):
        t = self.timesteps[i]
        latent_model_input = latents.to(self.dtype)
        ts = t.expand(latent_model_input.shape[0]).to(self.device)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            noise_pred_text = self.transformer(
                hidden_states=latent_model_input,
                timestep=ts,
                encoder_hidden_states=prompt_embeds[1:2],
                rope=rope,
                return_dict=False,
            )[0].float()

            noise_pred_uncond = self.transformer(
                hidden_states=latent_model_input,
                timestep=ts,
                encoder_hidden_states=prompt_embeds[:1],
                rope=rope,
                return_dict=False,
            )[0].float()

        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        return self.scheduler.step(noise_pred, t, latents.float(), return_dict=False)[0]

    def _add_noise(self, sample, noise, t):
        """Flow-matching interpolation: (1-sigma)*x0 + sigma*eps."""
        if hasattr(self.scheduler, "add_noise"):
            return self.scheduler.add_noise(sample, noise, t.reshape(1))
        return self.scheduler.scale_noise(sample, t.reshape(1), noise)

    @torch.no_grad()
    def run(self, rope=None, custom_name=None):
        clean_memory()
        latents = self.init_latents
        injection_blocks = list(self.config.injection_blocks)

        for i, t in enumerate(tqdm(self.timesteps, desc="Sampling")):
            is_guidance_step = bool((self.guidance_schedule == t).any())
            if not is_guidance_step:
                rope = None

            self._set_kv_mode(injection_blocks, inject=False, copy=True)

            if is_guidance_step and injection_blocks:
                # Cache KV from the reference video at this noise level.
                noisy_latent = self._add_noise(self.motion_latent.float(), self.init_latents, t)
                self._forward_transformer(
                    noisy_latent,
                    self.guidance_embeds[1:2],
                    t.expand(noisy_latent.shape[0]).to(self.device),
                    stop_at="injection",
                )
                self._set_kv_mode(injection_blocks, inject=True, copy=False)

            with torch.enable_grad():
                if is_guidance_step and self.config.guidance_blocks:
                    if not self.config.inject_embeds:
                        latents, rope = self.guidance_step(
                            latents, i, t, mode=self.config.guidance_mode, loss_type=self.config.loss_type
                        )
                    else:
                        embeds_path = os.path.join(self.output_path, "embeds")
                        if self.config.guidance_mode == "rope":
                            rope = torch.load(os.path.join(embeds_path, f"rope_{t}.pt")).to(self.device)
                        elif self.config.guidance_mode == "latent":
                            latents = torch.load(os.path.join(embeds_path, f"latent_{t}.pt")).to(self.device)

            latents = self.denoise_step(latents, i, self.guidance_embeds, rope=rope)

        # ---- Decode ------------------------------------------------------- #
        with torch.no_grad():
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1)
            latents = latents.to(self.vae.dtype)
            latents_mean = latents_mean.to(latents.device, latents.dtype)
            latents_std = latents_std.to(latents.device, latents.dtype)
            frames = self.vae.decode(latents * latents_std + latents_mean, return_dict=False)[0]
        video = self.pipe.video_processor.postprocess_video(frames, output_type="pil")[0]

        if custom_name:
            result_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in custom_name)[:50]
        else:
            result_name = "results"
        if self.config.inject_embeds:
            result_name += "_inject_embeds"

        out_dir = Path(self.output_path)
        if self.config.save_format == "frames":
            Path(out_dir, result_name).mkdir(parents=True, exist_ok=True)
            for j, frame in enumerate(video):
                frame.save(Path(out_dir, result_name, f"{j:04d}.png"))
        elif self.config.save_format == "gif":
            imageio.mimsave(str(out_dir / f"{result_name}.gif"), video)
        else:
            export_to_video(video, str(out_dir / f"{result_name}.mp4"), fps=16)
        return str(out_dir / f"{result_name}.{self.config.save_format}")


def main():
    parser = argparse.ArgumentParser(description="DiTFlow motion transfer for Wan2.1 T2V")
    parser.add_argument("-v", "--video_path", type=str, required=True, help="Reference video (.mp4 or dir of frames)")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Prompt for the new generation")

    parser.add_argument("--model", type=str, default="1.3b", choices=list(MODEL_IDS))
    parser.add_argument("-n", "--num_frames", type=int, default=None, help="Frames to generate (must be 4k+1)")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--negative_prompt", type=str, default=WAN_NEGATIVE_PROMPT)
    parser.add_argument("--loss_type", type=str, default="flow", choices=["flow", "moft", "smm"])
    parser.add_argument("--opt_mode", type=str, default="latent", choices=["latent", "emb"])
    parser.add_argument("--scheduler", type=str, default=None, choices=["unipc", "flowmatch"])
    parser.add_argument("--flow_shift", type=float, default=None)
    parser.add_argument("--guidance_blocks", type=int, nargs="+", default=None, help="Override guidance block indices")
    parser.add_argument("--motion_temp", type=float, default=None)
    parser.add_argument("--lr", type=float, nargs=2, default=None, metavar=("HI", "LO"))
    parser.add_argument("--optimization_steps", type=int, default=None)
    parser.add_argument("--guidance_timestep_range", type=int, nargs=2, default=None, metavar=("MAX", "MIN"))
    parser.add_argument("--no_guidance", action="store_true")
    parser.add_argument("--no_injection", action="store_true")
    parser.add_argument("--inject_embeds", action="store_true")
    parser.add_argument("--low_vram", action="store_true", help="Enable model CPU offload")
    parser.add_argument("--output_path", type=str, default="./results_wan")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_format", type=str, default="mp4", choices=["mp4", "gif", "frames"])
    parser.add_argument("--verbose", action="store_true")
    opt = parser.parse_args()

    config = OmegaConf.load("configs/guidance_config_wan.yaml")
    if opt.no_injection:
        config.injection_blocks = []

    overrides = {
        "model_key": MODEL_IDS[opt.model],
        "video_path": opt.video_path,
        "target_prompt": opt.prompt,
        "negative_prompt": opt.negative_prompt,
        "output_path": opt.output_path,
        "seed": opt.seed,
        "opt_mode": opt.opt_mode,
        "loss_type": opt.loss_type,
        "save_format": opt.save_format,
        "save_embeds": True,
        "inject_embeds": opt.inject_embeds,
        "verbose": opt.verbose,
    }
    for key, value in [
        ("num_frames", opt.num_frames),
        ("height", opt.height),
        ("width", opt.width),
        ("scheduler", opt.scheduler),
        ("flow_shift", opt.flow_shift),
        ("motion_temp", opt.motion_temp),
        ("lr", list(opt.lr) if opt.lr else None),
        ("optimization_steps", opt.optimization_steps),
        ("guidance_timestep_range", list(opt.guidance_timestep_range) if opt.guidance_timestep_range else None),
    ]:
        if value is not None:
            overrides[key] = value
    if opt.low_vram:
        overrides["enable_model_cpu_offload"] = True
    config = OmegaConf.merge(config, overrides)

    block_key = "guidance_blocks_1_3b" if opt.model == "1.3b" else "guidance_blocks_14b"
    config["guidance_blocks"] = list(opt.guidance_blocks) if opt.guidance_blocks else list(config[block_key])
    if opt.no_guidance:
        config["guidance_blocks"] = []

    # Wan has no learned absolute position embedding, so `emb` means RoPE.
    config.guidance_mode = "latent" if opt.opt_mode == "latent" else "rope"

    Path(config["output_path"]).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, Path(config["output_path"]) / "config.yaml")

    guidance = WanGuidance(config)

    print(f"[*] Starting inference for {opt.video_path}...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    start = time.time()
    out = guidance.run(custom_name=opt.prompt)
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    print(f"[*] Finished in {elapsed:.2f}s | peak VRAM {peak:.2f}GB | {out}")


if __name__ == "__main__":
    main()
