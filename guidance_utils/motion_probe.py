"""Opt-in, detached AMF diagnostics shared by Wan and CogVideoX.

No RNG draws, model updates, or scheduler steps. Attention diagnostics use fp32
and adjacent latent-frame pairs; the real training loss is logged separately.
"""

import json
import hashlib
import math
import platform
import subprocess
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def add_probe_arguments(parser):
    parser.add_argument("--probe", action="store_true", help="Save detached motion diagnostics")
    parser.add_argument("--probe_blocks", type=int, nargs="+", default=None,
                        help="Observed blocks only; does not change guidance/injection blocks")
    parser.add_argument("--probe_steps", type=int, nargs="+", default=[0, 1, 4, 9, 10, 19, 29, 39, 49],
                        help="Zero-based sampling steps to capture AMF")


def probe_config(args):
    return {name: getattr(args, name) for name in ("probe", "probe_blocks", "probe_steps")}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def tensor_stats(x):
    if x is None:
        return {"missing": True}
    x = x.detach().float()
    finite = torch.isfinite(x)
    values = x[finite]
    return {"shape": list(x.shape), "finite_fraction": finite.float().mean().item(),
            "rms": values.square().mean().sqrt().item() if values.numel() else None,
            "abs_max": values.abs().max().item() if values.numel() else None}


@torch.no_grad()
def adjacent_attention(q, k, h, w, frames, temperature):
    """B,S,heads,D -> small CPU arrays. Never allocate heads*S*S attention."""
    if q.shape != k.shape or q.ndim != 4 or q.shape[1] != frames * h * w:
        raise ValueError(f"Invalid probe Q/K geometry: {q.shape}, {k.shape}, {frames}x{h}x{w}")
    heads, dim = q.shape[-2:]
    # Explicitly leave the caller's autocast so diagnostics have the same
    # numeric convention for both models. This does not modify model tensors.
    with torch.autocast(device_type=q.device.type, enabled=False):
        qf = q[-1].detach().float().reshape(frames, h * w, -1)
        kf = k[-1].detach().float().reshape(frames, h * w, -1)
        yy, xx = torch.meshgrid(torch.arange(h, device=q.device), torch.arange(w, device=q.device), indexing="ij")
        xy = torch.stack((xx.flatten(), yy.flatten()), -1).float()
        fields = {name: [] for name in ("soft", "hard", "confidence", "entropy", "logit_std")}
        for i in range(frames - 1):
            logits = (qf[i] @ kf[i + 1].T) / (heads * math.sqrt(dim))
            p = (logits * temperature).softmax(-1)
            fields["soft"].append((p @ xy - xy).cpu().numpy())
            fields["hard"].append((xy[logits.argmax(-1)] - xy).cpu().numpy())
            fields["confidence"].append(p.max(-1).values.cpu().numpy())
            entropy = -(p * p.clamp_min(1e-30).log()).sum(-1) / math.log(max(h * w, 2))
            fields["entropy"].append(entropy.cpu().numpy())
            fields["logit_std"].append(logits.std(-1, unbiased=False).cpu().numpy())
    return {name: np.stack(values) for name, values in fields.items()}


def flow_summary(flow, h, w):
    """Per forward-adjacent pair, avoiding cancellation with reverse pairs."""
    norm = np.asarray(flow, dtype=np.float64) / np.array([w, h])
    result = []
    for i, field in enumerate(norm):
        valid = np.isfinite(field).all(-1)
        a = field[valid]
        moving = np.linalg.norm(a, axis=-1) > 1e-8
        result.append({"pair": [i, i + 1], "finite_fraction": float(valid.mean()),
                       "mean_dx_fraction": float(a[:, 0].mean()) if len(a) else None,
                       "mean_dy_fraction": float(a[:, 1].mean()) if len(a) else None,
                       "moving_fraction": float(moving.mean()) if len(a) else None,
                       "right_fraction_of_moving": float((a[moving, 0] > 0).mean()) if moving.any() else None,
                       "horizontal_fraction_of_moving": float((np.abs(a[moving, 0]) > np.abs(a[moving, 1])).mean()) if moving.any() else None})
    return result


class MotionProbe:
    def __init__(self, owner, model):
        self.enabled = bool(owner.config.get("probe", False))
        self.context = None
        self.blocks = []
        if not self.enabled:
            return
        self.model = model
        if owner.config.loss_type != "flow":
            raise ValueError("Motion probes currently support --loss_type flow")
        self.h, self.w, self.frames = owner.patches_height, owner.patches_width, owner.latent_num_frames
        if self.frames < 2:
            raise ValueError("Motion probes require at least two latent frames")
        modules = owner.transformer.blocks if model == "wan" else owner.transformer.transformer_blocks
        self.blocks = sorted(set(owner.config.get("probe_blocks") or owner.config.guidance_blocks or [len(modules) // 2]))
        if any(b < 0 or b >= len(modules) for b in self.blocks):
            raise ValueError(f"probe_blocks must be between 0 and {len(modules) - 1}")
        self.steps = set(owner.config.get("probe_steps", [0, 1, 4, 9, 10, 19, 29, 39, 49]))
        self.temperature = float(owner.config.motion_temp)
        # A fresh subdirectory avoids mixing traces when an output path is reused.
        from datetime import datetime, timezone
        self.path = Path(owner.output_path) / "probes" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.path.mkdir(parents=True)
        self.serial = 0
        self.references = {}
        packages = {}
        for package in ("torch", "diffusers", "transformers", "numpy"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        metadata = {"schema_version": 1, "model": model, "config": OmegaConf.to_container(owner.config, resolve=True),
                    "scheduler": type(owner.scheduler).__name__, "scheduler_config": dict(owner.scheduler.config),
                    "timesteps": owner.timesteps.detach().cpu().tolist(), "blocks": self.blocks,
                    "grid": [self.frames, self.h, self.w], "packages": packages, "python": platform.python_version(),
                    "git_commit": commit, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    "diagnostic_precision": "fp32 QK, mean logits over heads, forward-adjacent pairs",
                    "warning": "AMF is a correspondence proxy, not measured optical flow or camera pose."}
        root = Path(__file__).resolve().parent.parent
        source_files = [root / "motion_guidance.py", root / "motion_guidance_wan.py", *sorted((root / "guidance_utils").glob("*.py"))]
        metadata["source_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
        metadata["model_dtype"] = str(owner.dtype) if hasattr(owner, "dtype") else None
        metadata["reference_conditioning"] = owner.config.get("source_prompt", "")
        metadata["target_conditioning"] = owner.config.get("target_prompt", "")
        metadata["guidance_timesteps"] = owner.guidance_schedule.detach().cpu().tolist() if hasattr(owner, "guidance_schedule") else None
        if hasattr(owner, "lr_range"):
            metadata["lr_range"] = list(owner.lr_range)
        (self.path / "metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
        for b in self.blocks:
            modules[b].attn1.processor.motion_probe = self
        print(f"[probe] {self.path} | blocks={self.blocks} | sampling steps={sorted(self.steps)}")

    @contextmanager
    def phase(self, stage, step=-1, timestep=0, iteration=None, capture=True):
        previous = self.context
        active = self.enabled and capture and (step < 0 or step in self.steps)
        self.context = ({"stage": stage, "step": int(step), "timestep": float(timestep),
                         "iteration": iteration, "seen": set()} if active else None)
        try:
            yield
        finally:
            self.context = previous

    def emit(self, kind, **values):
        if not self.enabled:
            return
        record = {"kind": kind, **values}
        if self.context:
            record = {**{k: v for k, v in self.context.items() if k != "seen"}, **record}
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(record), allow_nan=False) + "\n")

    @torch.no_grad()
    def attention(self, block_name, query, key, injected=False, text_prefix=0):
        if self.context is None or block_name in self.context["seen"]:
            return
        self.context["seen"].add(block_name)
        if self.model == "cogvideox":
            query = query[:, :, text_prefix:].transpose(1, 2)
            key = key[:, :, text_prefix:].transpose(1, 2)
        data = adjacent_attention(query, key, self.h, self.w, self.frames, self.temperature)
        self.serial += 1
        filename = f"amf_{self.serial:05d}_{block_name}.npz"
        np.savez_compressed(self.path / filename, **data)
        comparisons = {}
        if self.context["stage"] == "reference":
            self.references[block_name] = data
        elif block_name in self.references:
            ref = self.references[block_name]["hard"]
            delta = (data["soft"] - ref) / np.array([self.w, self.h])
            comparisons["mse_to_reference_hard_normalized"] = float(np.mean(delta ** 2))
        self.emit("attention", block=block_name, file=filename, injected=bool(injected),
                  query=tensor_stats(query), key=tensor_stats(key),
                  mean_confidence=float(data["confidence"].mean()), mean_entropy=float(data["entropy"].mean()),
                  mean_logit_std=float(data["logit_std"].mean()),
                  soft_pairs=flow_summary(data["soft"], self.h, self.w),
                  hard_pairs=flow_summary(data["hard"], self.h, self.w), **comparisons)

    def actual_reference(self, block, flow, mask):
        if self.enabled:
            filename = f"training_reference_{block}.npz"
            np.savez_compressed(self.path / filename, flow=flow.detach().float().cpu().numpy(),
                                mask=mask.detach().cpu().numpy())
            self.emit("training_reference", block=block, file=filename, kept_fraction=mask.float().mean().item())

    @torch.no_grad()
    def training_flow(self, block, flow, reference, mask, loss):
        """Record the actual loss-path AMF, separate from fp32 diagnostics."""
        if self.context is None:
            return
        indices = torch.arange(self.frames - 1, device=flow.device) * (self.frames + 1) + 1
        if mask is None:
            mask = torch.ones_like(flow[..., 0], dtype=torch.bool)
        self.serial += 1
        filename = f"loss_amf_{self.serial:05d}_{block}.npz"
        np.savez_compressed(self.path / filename, flow=flow[indices].detach().float().cpu().numpy(),
                            reference=reference[indices].detach().float().cpu().numpy(),
                            mask=mask[indices].detach().cpu().numpy())
        self.emit("training_flow", block=block, file=filename, loss=float(loss.detach()),
                  kept_fraction=mask.float().mean().item())

    def prediction(self, step, timestep, conditional, unconditional, scale):
        if self.enabled:
            self.emit("prediction", step=step, timestep=float(timestep), cfg_scale=float(scale),
                      conditional=tensor_stats(conditional), unconditional=tensor_stats(unconditional),
                      cfg_difference=tensor_stats(conditional - unconditional))

    def optimization(self, step, timestep, iteration, loss, before, after, gradient, lr):
        if self.enabled:
            self.emit("optimization", step=step, timestep=float(timestep), iteration=iteration,
                      loss_before_update=float(loss.detach()), lr=float(lr), gradient=gradient,
                      variable=tensor_stats(after), update=tensor_stats(after.detach() - before))

    @torch.no_grad()
    def sampling(self, step, timestep, before_guidance, after_guidance, after_sampling, scheduler):
        if not self.enabled:
            return
        guidance_delta = after_guidance.float() - before_guidance.float()
        sampling_delta = after_sampling.float() - after_guidance.float()
        denominator = guidance_delta.square().sum()
        projection = ((sampling_delta * guidance_delta).sum() / denominator).item() if denominator > 0 else None
        sigma = None
        if hasattr(scheduler, "sigmas") and step < len(scheduler.sigmas):
            sigma = float(scheduler.sigmas[step])
        self.emit("sampling", step=step, timestep=float(timestep), sigma=sigma,
                  latent=tensor_stats(after_sampling), guidance_update=tensor_stats(guidance_delta),
                  sampling_update=tensor_stats(sampling_delta), sampling_projection_on_guidance=projection)


@contextmanager
def without_injection(processors):
    """Isolate the final diagnostic forward and restore caches even on failure."""
    states = [(p, p.inject_kv, p.copy_kv, p.query, p.key, p.value) for p in processors]
    try:
        for p in processors:
            p.inject_kv = p.copy_kv = False
        yield
    finally:
        for p, inject, copy, q, k, v in states:
            p.inject_kv, p.copy_kv, p.query, p.key, p.value = inject, copy, q, k, v
