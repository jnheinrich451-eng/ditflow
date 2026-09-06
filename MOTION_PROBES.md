# Investigating the Wan DiTFlow port

Use `motion_probes.ipynb` in the existing Colab environment, or the commands
below. Probes are opt-in on both entry points. They leave the guidance
objective, layers, injection flags, optimizer, and scheduler algorithm as-is.
The existing `notebook.ipynb` and its outputs are not rewritten.

The purpose is to compare a working CogVideoX example with the Wan adaptation
at the same stages, before concluding that AMF itself is unsuitable. Follow-
Your-Motion includes DiTFlow among its baselines ([paper, section 4.3](https://arxiv.org/html/2506.05207v2#S4.SS3)).
These probes do not reproduce that paper's unpublished baseline settings.

## Start here

```bash
python verify_motion_probe.py
python verify_wan_port.py

# Wan Euler: preserve the ordinary AMF objective, without the optional filters.
python motion_guidance_wan.py -v assets/lucia.mp4 -p "Cat walks in a city lane" --scheduler flowmatch --seed 1 --probe --probe_blocks 0 10 15 20 --output_path probe_runs/lucia_wan_euler

# Same Wan settings with UniPC, to investigate scheduler effects.
python motion_guidance_wan.py -v assets/lucia.mp4 -p "Cat walks in a city lane" --scheduler unipc --seed 1 --probe --probe_blocks 0 10 15 20 --output_path probe_runs/lucia_wan_unipc

# Original CogVideoX-5B configuration, with the same instrumentation.
python motion_guidance.py -v assets/lucia.mp4 -p "Cat walks in a city lane" --seed 1 --probe --probe_blocks 0 15 20 25 --output_path probe_runs/lucia_cog

python probe_report.py probe_runs/lucia_wan_euler probe_runs/lucia_wan_unipc probe_runs/lucia_cog --output_dir probe_comparison
```

Open `probe_comparison/comparison.html`. It embeds its images, so the HTML can
be copied independently. PNG figures and `attention_pairs.csv` are also saved.
Archive the run's `probes/` folder for the complete evidence. Reusing an output
directory creates a fresh timestamped trace; the report chooses the latest one
unless passed a timestamped trace path explicitly. An interrupted run can still
be reported; absent final captures are identified.

To reproduce the exact earlier filtered Lucia example, add `--flow_max_disp 10`
to the Wan command and use a separate output directory. Do not attribute the
difference between filtered Wan and unfiltered CogVideoX solely to the backbone.
Repeat the same comparison on BMX with its original leopard prompt after Lucia.

## What is observed

`--probe_blocks` selects **observation** blocks; it never changes guidance or
injection placement. Wan-1.3B still guides at block 15 and injects at block 0.
CogVideoX-5B still guides at block 20 and injects at block 0. All indices are
zero-based. `--probe_steps` selects zero-based sampling indices; its default is
`0 1 4 9 10 19 29 39 49`.

| Event/stage | Measurement |
|---|---|
| `reference` | Clean reference latent at t=0, source prompt, no KV injection. Hard and soft AMF at every requested block. |
| `guidance` | First and last **pre-update** optimization forwards at selected steps; actual injection remains active. |
| `training_flow` | Adjacent slices of the **actual** loss-path flow, reference and mask, plus that block's configured loss. |
| `optimization` | Every optimization iteration: actual loss before the update, LR, unscaled gradient RMS, variable RMS, update RMS and finite fractions. |
| `denoise_cond` | Conditional full forward after the last optimizer update, before the scheduler step. |
| `denoise_uncond` | Unconditional full forward; also reveals the inherited injection into both CFG branches. |
| `prediction` | Conditional/unconditional prediction norms, their difference and actual CFG scale at every sampling step. |
| `sampling` | Latent RMS, net guidance update, scheduler update, timestep and scheduler sigma when available. |
| `final_latent` | One extra forward of the finished latent at t=0, target prompt, no injected KV or optimized RoPE. This is not optical flow of the decoded output. |

Reference and final observation forwards may extend past the usual guidance
early-exit block in Wan. During optimization, blocks beyond the guidance block
remain skipped, so they have no `guidance` capture. They are observed during the
full denoising forwards. No extra denoising or scheduler steps are performed.

Probes calculate detached, fp32 attention logits averaged over heads. They save
forward-adjacent latent pairs only (0→1, 1→2, etc.), avoiding cancellation from
averaging forward and reverse motion. Each capture contains:

- Argmax displacement and soft expected displacement, in patch units.
- Peak softmax probability and normalized entropy, per source patch.
- Per-row logit standard deviation, before temperature multiplication.
- Per-pair direction summaries in fractions of image width/height.
- All-patch normalized MSE of soft target flow against hard reference flow.

The fp32 diagnostics use a consistent numerical convention across backbones;
they are **not** the original bf16 loss computation. `training_reference_*.npz`
stores the actual all-frame-pair reference flow and training mask, including the
same-frame pairs. `loss_amf_*.npz` stores the actual forward-adjacent loss-path
arrays at selected iterations. `events.jsonl` distinguishes these quantities.
No Q/K tensors or full attention matrices are written to disk.

## How to interpret the comparison

1. **Reference direction differs:** inspect blocks before touching the optimizer.
   Are the background arrows predominantly horizontal in the correct direction?
   Is the subject region plausible? High confidence alone does not prove that a
   correspondence is physically correct; RoPE can also favor same-position matches.
2. **Reference looks plausible, guidance barely changes it:** inspect gradient
   finite fractions, gradient/update RMS, actual loss, and before/after fields.
   Compare `guidance` iteration 0 with `denoise_cond` at the same step, at the
   guidance block. This includes all five Adam updates.
3. **Guidance improves AMF, then it deteriorates:** inspect `denoise_cond` at steps
   9, 10, 19, 29, 39, 49 and `final_latent`. This distinguishes progress during
   optimization from persistence after guidance ends.
4. **Euler and UniPC diverge:** compare per-step updates and AMF trajectories.
   `sampling_projection_on_guidance` is dot(sampling_delta, guidance_delta) /
   ||guidance_delta||². Negative means the sampling update has an opposing
   component. This is descriptive, **not** a causal measure of guidance removal;
   legitimate denoising can also oppose that direction.
5. **Final AMF looks right, decoded motion is wrong:** the internal descriptor
   may be satisfied without the intended visible correspondence. At that point,
   check decoded-video tracking rather than concluding from loss alone.

For an individual intermediate capture in a notebook:

```python
from probe_report import plot_capture
plot_capture("probe_runs/lucia_wan_euler", block=15,
             stage="guidance", step=9, iteration=0, kind="soft")
plot_capture("probe_runs/lucia_wan_euler", block=15,
             stage="denoise_cond", step=9, kind="soft")
```

Arrow scale is fixed in normalized image coordinates, with x increasing right
and y increasing down. Gray backgrounds show peak attention probability, not
video frames. No automatic foreground/background labels are assigned.

## Comparison limits and overhead

- The default Cog run uses 24 frames at 720×480; Wan uses 21 at 832×480. Both have
  six latent frames, but their VAE temporal support differs. Equal seeds do not
  produce equal noise tensors across different shapes. Normalize displacement,
  but do not treat this as exact paired latent geometry or equivalent noise levels.
- Raw losses, confidence, or gradient magnitudes are not calibrated across models.
  Compare direction patterns and within-run trends first. Source prompt differs
  from target prompt for reference/final extraction; metadata records both.
- Probe timing/VRAM is not benchmark timing/VRAM: detached fp32 QK calculations,
  device synchronization, file writes and one final forward add overhead. Reduce
  observed blocks/steps if needed. Start with short six-latent-frame clips.
- The existing loss, mask, guidance-window LR behavior and solver settings are
  deliberately preserved so these runs diagnose the current implementation.
  Avoid moving the guidance window later until the previously identified LR
  indexing issue is fixed. Empty training masks are recorded, not repaired here.
- Metadata records resolved run/scheduler configs, schedules, package versions,
  commit hash and hashes of local guidance source files. It does not resolve the
  downloaded checkpoint to an immutable Hub commit.

## Validation

`verify_motion_probe.py` checks a known rightward correspondence, normalized
vertical displacement, uniform-attention behavior, equivalence with the existing
AMF utility, stock-Wan forward equivalence on identical tiny weights, output and
gradient invariance with probes (including checkpointing), Cog processor prefix
handling and invariance, disabled-probe behavior, exception-safe cache restoration,
and report generation. When CUDA is available, an additional test exercises the
real Wan guidance and CFG sampling methods on a tiny bf16 model, including KV
injection and a probe block beyond the guidance early exit. All tests use random
weights without downloads; pretrained video runs are still required to diagnose
Lucia.
