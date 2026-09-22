# Historical Wan workflows and tuning notes

Archived from README_WAN.md on 2026-09-22. These instructions preserve earlier
experiment context; their "next run", stop, and tuning directives are not active
requirements. Read [CURRENT_PLAN.md](../CURRENT_PLAN.md) for the active goal,
evidence, and next decision, or [README_WAN.md](../README_WAN.md) for installation
and the current implementation reference.

The original text is retained below, with relative Markdown links adjusted for
this location. Commands and inline paths are still relative to the repository
root. This archive does not select a run or authorize compute.

## Archived workflow instructions

The original instructions below are retained for provenance. None selects the next run.

**Next run: [decisive.ipynb](../decisive.ipynb).** One clip, seven generations, and a stop rule fixed
before the run: does AMF guidance at full DiTFlow strength steer decoded motion at all? It is scored
only on decoded video. If it fails on camel, AMF guidance on Wan stops. See
[the decisive protocol](../docs/WAN_DECISIVE_TEST.md). The subject-only section in `notebook.ipynb`
is on hold until that verdict.

Use the lightweight [notebook.ipynb](../notebook.ipynb) for the current remote GPU run.
The setup cell is the second code cell, under **2. Experiment setup**. After syncing
the checkout and restarting the kernel, follow runtime preparation -> setup ->
generate/resume -> view results -> archive. Historical experiments and their saved
outputs are preserved in [tests.ipynb](../tests.ipynb); they do not need to run first.

After `wan_subject_14b_20260914T222436877557Z`, run **Wan 14B: subject-only guidance**
in the main notebook. It generates two new forward/reverse videos with background
loss weight zero, reusing the exact frozen mapping and completed balanced controls.
Head, LR, timestep, prompt and update budget stay fixed; background diagnostics
remain recorded. See [the subject-only protocol](../docs/WAN_SUBJECT_ONLY.md).

The preceding four-generation **subject alignment and loss balance** workflow
and its saved outputs are archived in `tests.ipynb`. See
[its protocol](../docs/WAN_SUBJECT_ALIGNMENT.md).

After `wan_noised_reference_14b_20260913T213928341143Z`, the preceding
**Wan 14B: AMF control and retention** experiment generates matched off/forward/reversed
controls with forward-adjacent guidance, actual sampler captures and native-head
measurements. Its completed workflow is archived in `tests.ipynb`.
See [the control protocol](../docs/WAN_CONTROL_EXPERIMENT.md).
It includes the preceding [pair-selection intervention](../docs/WAN_PAIR_COMPARISON.md);
that smaller one-generation section does not need to run first.

The earlier test after the crossover is the archived section **Wan 14B: noised
reference and real generation**. It confirms fixed block 30/head 30 on camel/seed 29,
then generates playable vanilla / baseline-readout / noised-reference-candidate
videos with a matched step-9 intervention. Its cells are in `tests.ipynb`.
See [the fixed protocol](../docs/WAN_NOISED_REFERENCE_PILOT.md).
The candidate is experimental; the original full clean-reference screen still failed.

The earlier 14B diagnostic is **Wan 14B: individual heads, then visual validation**
in `tests.ipynb`. It screens heads in blocks 20/30 at fixed
sharpening 2, confirms one frozen candidate on another texture/noise seed, then
permits a matched AMF-off / historical baseline / candidate generation pilot.
See [the fixed protocol](../docs/WAN_HEAD_DIAGNOSTIC.md). The new `--flow_head`
option changes only the experimental AMF readout; its default preserves mean logits.
Local checks do not establish pretrained generation quality.

The archived `tests.ipynb` section **Next generation test: early versus later AMF guidance**
runs four paired generations: car-turn/camel, each with sampling indices 0–9 or
20–29. It uses uniform AMF at block 10, cap 100, seed 1, no KV injection, and five
Adam updates per active step. Both arms share the same ten-step LR sequence and
50-update budget. The only generation-setting difference is
`--guidance_timestep_range 50 40` versus `--guidance_timestep_range 30 20`, with
`--lr_decay_steps 10` for both. The sampler still runs all 50 denoising steps.

Run its setup, generate, compare and archive cells. Existing reference inputs are
reused, but both generation arms run fresh. Do not update packages/code between
arms: the setup and each run compare fresh-process environment/source fingerprints.
Rerunning only generate resumes the plan. The comparison cell audits actual
optimizer counts, LR values, guided sigmas and paired reference images, saving
`timing_audit.json`. AMF probes cover both windows; intermediate decoded video
estimates are not collected. Baseline defaults and generation code are unchanged.

The archived `tests.ipynb` section **Next diagnostic: rotation, expansion and attention heads**
tests AMF extraction without generating target videos. It reuses the
`wan_reference_inputs` bundle for car-turn and camel and compares all individual
heads with averaged logits at block 10 on known image-plane transforms.
Run setup, probe, review and archive; rerun only probe to resume completed suites.

Standalone equivalent for one clip:

```bash
python probe_wan_affine.py -v probe_runs/wan_reference_inputs/clips/car-turn \
  --output_path probe_runs/car_affine_fresh --blocks 10 --noise_steps 0 9 29
```

The seven controls are static, left/right translation, clockwise/counterclockwise
rotation, expansion and contraction. Clean plus three noise states means 28
observed forwards per clip. The model loads once. Text stays blank throughout;
optional `--prompt` changes it for every state, so use a separate output folder.
Native RoPE remains fixed and no guidance/injection is performed. These are
forward-noised controls, not actual generated intermediate videos.

`affine_report.html` is self-contained; `metrics.csv` contains hard/soft endpoint
errors, direction and amplitude, support counts and anchor sensitivity. Per-head
NPZ files retain every adjacent pair, while `truth.npz` stores known flow and
evaluation masks. `base_frame.png` and `control.json` reproduce the lossless
synthetic inputs; the MP4s are visual previews. Metadata records package versions,
source hashes, shared noise seed/hash and actual sigmas. Do not interpret scores
on identical pure-noise inputs as recovered motion or select a production head
from these two diagnostic clips alone. Passing 2D transforms does not establish
correct 3D turns, gait or motion transfer.


## Historical sweep interface (not the active workflow)

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

## Historical tuning suggestions (not active requirements)

The defaults are **starting points transplanted from CogVideoX, not ported
optima.** The original proposed tuning order is retained below; no sweep is selected
by the current plan:

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
