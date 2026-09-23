# DiTFlow for Wan2.1 T2V

## Active workflow: T2V 14B port

Start with [CURRENT_PLAN.md](CURRENT_PLAN.md) for the goal, evidence, open questions,
and next decision; [AGENTS.md](AGENTS.md) defines the working rules. Current scope is
the DiTFlow port to **Wan2.1 T2V 14B**, an intermediate platform toward the user's
final I2V 14B target. This workflow does not validate I2V.

The active implementation is [motion_guidance_wan.py](motion_guidance_wan.py), loading
[configs/guidance_config_wan.yaml](configs/guidance_config_wan.yaml). Specify
`--model 14b`: the unchanged CLI default selects 1.3B. The existing configuration
enables K/V injection; `--no_injection` disables it, and a native-backbone comparison
requires both `--no_guidance --no_injection`. See the plan for the exact defaults.

The port has saved compatibility evidence, but decoded motion-transfer acceptance
remains unmet. Lower internal loss does not establish success. The current plan
names a prepared fourfold-LR original-AMF comparison; GPU execution remains pending.
Start it with [wan_lr4_colab.ipynb](wan_lr4_colab.ipynb) through the VS Code Colab
extension: Git clone/pull, 80 GB A100/H100, and direct Drive results/logs/review ZIP.
See [setup, baseline reuse, and paired commands](docs/WAN_LR4_CANDIDATE.md).
The notebook defaults to preparation only and uses the saved runtime pins rather
than the general installation example below.
Read the relevant saved evidence before changing the implementation. Reuse applicable
checks; resolving every historical diagnostic discrepancy is not a prerequisite.

Prior notebook workflows, protocols labelled historical, report recommendations,
and their "next run" or publish instructions are historical, not active requirements.
This includes `notebook.ipynb`, `decisive.ipynb`, `wan_port_acceptance.ipynb`, and the
centered-AMF notebooks. Their implementation, outputs, and tests are retained.
Earlier workflow, sweep, and tuning instructions are in the
[historical workflow archive](docs/WAN_WORKFLOW_HISTORY.md).
The following technical reference documents existing behavior; its examples are not
a queued experiment or an instruction to change the environment during this cleanup.

## Port implementation reference

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

## Repository layout

| Path | Contents |
|---|---|
| `motion_guidance.py`, `motion_guidance_wan.py` | Current generation entry points (CogVideoX baseline, Wan port). Relocation requires updating callers and retaining traceability to archived run paths. |
| `guidance_utils/` | Importable library: transformers, attention processors, AMF, probes and diagnostics for both backbones. |
| `configs/` | Guidance parameters per backbone. |
| `probe_wan_affine.py`, `probe_cog_affine.py`, `probe_wan_rope.py` | Known-motion readout suites; run from the root. |
| `probe_report.py`, `probe_temporal_report.py`, `sweep_wan.py`, `colab_utils.py` | Reports, sweeps and notebook helpers, imported by the notebooks by these names. |
| `benchmark/` | Benchmark harness, manifests, analysis and table generation. |
| `eval/` | The paper's MF and CLIP metrics, unchanged. |
| `tests/` | Preserved `verify_*.py` checks. Select checks relevant to an implementation change; the whole diagnostic collection is not an active prerequisite. |
| `docs/` | Working notes: probe guides, benchmark design, decisions and results. |
| `assets/` | Reference clips and teaser media. |
| `probe_runs/`, `probe_comparison/`, `results*/`, `sweeps/` | Generated outputs and offline reviews; git-ignored. |

## Environment

Wan needs **diffusers >= 0.33** (0.36 verified). The `diffusers==0.30.2` pin in
`requirements.txt` exists only because the CogVideoX path forks that version's
transformer forward — it does not apply here.

```
pip install "diffusers>=0.33" "transformers>=4.44,<5" "huggingface-hub>=0.34,<1" accelerate ftfy imageio imageio-ffmpeg omegaconf einops
```

`transformers>=5` breaks diffusers 0.36's model imports. Pin below 5.

Transformers 4.57.x also requires `huggingface-hub>=0.34,<1`; installing Hub 1.x
separately can cause an import error before Wan loads. To repair that specific
error in an existing notebook kernel without upgrading the model libraries:

```python
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'huggingface-hub>=0.34,<1'])
subprocess.check_call([sys.executable, '-c',
    'import transformers, diffusers, huggingface_hub; '
    'print(transformers.__version__, diffusers.__version__, huggingface_hub.__version__)'])
```

Then rerun the failed generation cell: it launches a fresh Python process and
keeps the existing experiment plan. If using model libraries directly in the
notebook process, restart the kernel after changing installed packages.

## Port compatibility check (when relevant)

```bash
python tests/verify_wan_port.py
```

Weight-free, CPU, seconds. Checks the port's AMF against **this repo's own**
`guidance_utils/motion_flow_utils.py` (the published DiTFlow implementation) at
f64, plus the transformer chain: rotary equivalence, Q/K capture, gradient flow
to latent and RoPE, gradient-checkpointing invariance, and KV injection. Use it for
relevant implementation changes; a pass does not establish decoded motion transfer.

## CLI examples (explicit 14B; not a scheduled run)

The bundled `assets/*.mp4` are **24 frames at 720x480**, so `num_frames` must be
a 4k+1 value <= 24. The config defaults to 21 (-> 6 latent frames, the same
working point DiTFlow used on CogVideoX). A longer reference clip lifts that
ceiling: 33 -> 9 latent frames, 81 -> 21.

```bash
# DiTFlow (-z_t): optimise the latent  -- the paper's headline setting
python motion_guidance_wan.py --model 14b \
    --video_path ./assets/bmx-trees.mp4 \
    --prompt "Leopard running up a snowy hill in a forest"

# DiTFlow (-rho_t): optimise RoPE, reusable for zero-shot injection
python motion_guidance_wan.py --model 14b -v ./assets/bmx-trees.mp4 -p "..." --opt_mode emb

# Zero-shot injection with a new prompt (after an --opt_mode emb run)
python motion_guidance_wan.py --model 14b -v ./assets/bmx-trees.mp4 \
    -p "Polar bear walking up a snowy hill in a forest" --opt_mode emb --inject_embeds

# Baselines
python motion_guidance_wan.py --model 14b -v ... -p ... --loss_type smm      # SMM
python motion_guidance_wan.py --model 14b -v ... -p ... --loss_type moft     # MOFT
python motion_guidance_wan.py --model 14b -v ... -p ... --no_guidance                 # injection only
python motion_guidance_wan.py --model 14b -v ... -p ... --no_guidance --no_injection  # backbone
```

Evaluate exactly as DiTFlow does — `eval/motion_fidelity_score.py` and
`eval/clip_score.py` are model-agnostic and need no changes.

## What changed from CogVideoX, and why

* **Rectified flow.** UniPC (default) or FlowMatchEuler replaces DDIM/DPM.
  `scale_model_input` is a no-op and drops out; `add_noise` interpolates
  `(1-s)·x₀ + s·ε`. FlowMatch is a first-order alternative; scheduler choice is
  part of each saved configuration, not an automatic remedy for guidance drift.
* **VAE.** Per-channel `latents_mean`/`latents_std`, not a single
  `scaling_factor`. Latents are `(B, C, F, H, W)` — no permute. The VAE is kept
  in fp32 (bf16 produces artifacts).
* **No absolute position embedding.** `--opt_mode emb` optimises RoPE. There is
  no `posemb` mode.
* **No text in self-attention.** AMF needs no text-prefix slice.
* **`num_frames` must be 4k+1** (Wan's causal VAE). Default 21 → 6 latent frames.

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

## Hardware

* **Wan2.1-T2V-1.3B** fits comfortably; also the practical choice under 24 GB
  (add `--low_vram` for model CPU offload). This historical option is outside the
  current 14B scope; limited hardware does not select a different target.
* **Wan2.1-T2V-14B** needs an A100/H100 for guidance — backprop runs through
  blocks 0…20 of a 14B model.

## Known upstream bug not carried over

`load_attn_features` in `motion_guidance.py` calls `compute_motion_flow` without
`nframes`, so the reference AMF always uses the default of 6 latent frames. That
is correct only at `--video_length 24`; any other length silently mismatches the
target AMF. Here `nframes` is a required argument.
