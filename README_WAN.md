# DiTFlow for Wan2.1 T2V

A port of DiTFlow to Wan2.1. **Nothing in the original CogVideoX implementation is
modified** — `motion_guidance.py`, `guidance_utils/custom_*.py` and
`configs/guidance_config.yaml` are untouched, so the paper baseline stays runnable
side by side for comparison.

| New file | Role | CogVideoX counterpart |
|---|---|---|
| `motion_guidance_wan.py` | Entry point | `motion_guidance.py` |
| `guidance_utils/wan_transformer.py` | Optimisable RoPE + early exit | `custom_transformer.py` |
| `guidance_utils/wan_modules.py` | QK capture, KV injection, feature hook | `custom_modules.py` |
| `guidance_utils/wan_motion_flow_utils.py` | AMF | `motion_flow_utils.py` |
| `configs/guidance_config_wan.yaml` | Guidance params | `configs/guidance_config.yaml` |

## Environment

Wan needs **diffusers >= 0.33** (0.36 verified). The `diffusers==0.30.2` pin in
`requirements.txt` exists only because the CogVideoX path forks that version's
transformer forward — it does not apply here.

```
pip install "diffusers>=0.33" "transformers>=4.44,<5" accelerate ftfy imageio imageio-ffmpeg omegaconf einops
```

`transformers>=5` breaks diffusers 0.36's model imports. Pin below 5.

## Verify first

```bash
python verify_wan_port.py
```

Weight-free, CPU, seconds. Checks the port's AMF against **this repo's own**
`guidance_utils/motion_flow_utils.py` (the published DiTFlow implementation) at
f64, plus the transformer chain: rotary equivalence, Q/K capture, gradient flow
to latent and RoPE, gradient-checkpointing invariance, and KV injection. Run it
before trusting a sweep; if it fails, nothing downstream is trustworthy.

## Run

The bundled `assets/*.mp4` are **24 frames at 720x480**, so `num_frames` must be
a 4k+1 value <= 24. The config defaults to 21 (-> 6 latent frames, the same
working point DiTFlow used on CogVideoX). A longer reference clip lifts that
ceiling: 33 -> 9 latent frames, 81 -> 21.

```bash
# DiTFlow (-z_t): optimise the latent  -- the paper's headline setting
python motion_guidance_wan.py \
    --video_path ./assets/bmx-trees.mp4 \
    --prompt "Leopard running up a snowy hill in a forest"

# DiTFlow (-rho_t): optimise RoPE, reusable for zero-shot injection
python motion_guidance_wan.py -v ./assets/bmx-trees.mp4 -p "..." --opt_mode emb

# Zero-shot injection with a new prompt (after an --opt_mode emb run)
python motion_guidance_wan.py -v ./assets/bmx-trees.mp4 \
    -p "Polar bear walking up a snowy hill in a forest" --opt_mode emb --inject_embeds

# Baselines
python motion_guidance_wan.py -v ... -p ... --loss_type smm      # SMM
python motion_guidance_wan.py -v ... -p ... --loss_type moft     # MOFT
python motion_guidance_wan.py -v ... -p ... --no_guidance                 # injection only
python motion_guidance_wan.py -v ... -p ... --no_guidance --no_injection  # backbone
```

Evaluate exactly as DiTFlow does — `eval/motion_fidelity_score.py` and
`eval/clip_score.py` are model-agnostic and need no changes.

## What changed from CogVideoX, and why

* **Rectified flow.** UniPC (default) or FlowMatchEuler replaces DDIM/DPM.
  `scale_model_input` is a no-op and drops out; `add_noise` interpolates
  `(1-s)·x₀ + s·ε`. If `--opt_mode latent` drifts, try `--scheduler flowmatch`:
  it is first-order, so editing the latent between steps cannot corrupt
  multistep solver history the way it can with UniPC.
* **VAE.** Per-channel `latents_mean`/`latents_std`, not a single
  `scaling_factor`. Latents are `(B, C, F, H, W)` — no permute. The VAE is kept
  in fp32 (bf16 produces artifacts).
* **No absolute position embedding.** `--opt_mode emb` optimises RoPE. There is
  no `posemb` mode.
* **No text in self-attention.** AMF needs no text-prefix slice.
* **`num_frames` must be 4k+1** (Wan's causal VAE). Default 33 → 9 latent frames.

### AMF is reformulated, not reimplemented

`wan_motion_flow_utils.compute_motion_flow` computes the same quantity as the
original but never materialises the joint attention map. Three exact algebraic
rewrites (head fusion, frame-pair chunking, displacement without the
relative-coordinate grids) are documented in the module docstring. Verified
equal to the reference implementation to fp64 round-off (~2e-15), with the
argmax path bit-identical.

The practical effect: the reference builds an `(H, S, S)` tensor — ~3.9 GB in
bf16 for CogVideoX-5B, and ~44 GB for Wan at 33 frames — purely to average the
head axis away. Head fusion removes the `H` factor outright. `checkpoint_amf`
(default `auto`, on above 7 latent frames) additionally recomputes frame-pair
blocks in backward, trading ~30% compute for O(S) instead of O(S²) retained
activations.

## Sweeping

`sweep_wan.py` runs a grid and scores every output with the paper's own metrics
(`eval/motion_fidelity_score.py` for MF, `eval/clip_score.py` for CLIP):

```bash
python sweep_wan.py \
    --video_path ./assets/bmx-trees.mp4 \
    --prompt "Leopard running up a snowy hill in a forest" \
    --guidance_blocks 12 --guidance_blocks 15 --guidance_blocks 18 \
    --motion_temp 1 2 4 \
    --include_baselines \
    --output_root ./sweeps/bmx
```

Writes `results.md` (ranked by MF), `results.csv` and `results.json`. Add
`--dry_run` to print the grid first.

* Each grid point runs as a **subprocess** — Wan is torn down between runs, so
  VRAM doesn't accumulate and peak-VRAM numbers stay comparable.
* **Resumable**: a point whose output already exists is skipped, so a sweep
  killed by a dying Colab VM picks up where it stopped. `--force` re-runs.
* Results are written **after every run**, so a crashed sweep still leaves a
  usable table.
* `--include_baselines` adds the backbone and injection-only reference rows,
  which is what tells you whether guidance is actually buying MF.

MF requires the `cotracker` package (pulled via `torch.hub`) and CLIP requires
`clip`; a missing one is recorded as `n/a` and the sweep continues rather than
discarding the generations. `--skip_mf` / `--skip_clip` opt out explicitly.

## Tuning

The defaults are **starting points transplanted from CogVideoX, not ported
optima.** Expect to sweep, in roughly this order of leverage:

1. `motion_temp` — Wan applies RMSNorm to q/k (`qk_norm=rms_norm_across_heads`),
   so raw attention logits are on a different scale than CogVideoX's. This
   directly sets how sharp the AMF softmax is. Tune first.
2. `guidance_blocks` — 15/30 (1.3B) and 20/40 (14B) just match CogVideoX's ~50%
   depth. Which block encodes motion is an empirical property of each model:
   `--guidance_blocks 10 12 15 18 20`.
3. `guidance_timestep_range` — flow-matching models resolve structure at
   different points in the trajectory than DDPM ones, so `[50,40]` may not be
   the right window.
4. `lr`, `optimization_steps`.

## Hardware

* **Wan2.1-T2V-1.3B** fits comfortably; also the practical choice under 24 GB
  (add `--low_vram` for model CPU offload).
* **Wan2.1-T2V-14B** needs an A100/H100 for guidance — backprop runs through
  blocks 0…20 of a 14B model.

## Known upstream bug not carried over

`load_attn_features` in `motion_guidance.py` calls `compute_motion_flow` without
`nframes`, so the reference AMF always uses the default of 6 latent frames. That
is correct only at `--video_length 24`; any other length silently mismatches the
target AMF. Here `nframes` is a required argument.
