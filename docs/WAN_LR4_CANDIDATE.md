# Prepared candidate: fourfold latent LR, original AMF

Active only through [CURRENT_PLAN.md](../CURRENT_PLAN.md). Status: **prepared, not
executed**. Current authorization is preparation/CPU checking, with zero GPU runs.

## Rationale and limits

The completed early-window experiment failed opposite-direction control, but its
forward/reverse displacement ordering (-86/-122 px; off -102 px) is compatible with
a weak useful response. It is not strong evidence given coarse endpoint uncertainty.
AMF measures image-grid displacement, not subject motion relative to a fence.
The references combine subject and camera motion; this is not a subject-only test.

Hypothesis: the original AMF objective has useful directional influence, but the
latent update size is insufficient in this fixed Wan configuration. Test **only**
LR `.008 -> .004` against saved `.002 -> .001`. Fourfold is a chosen trial magnitude,
not an inferred optimum or a promise of fourfold effective latent/motion change.
No centering, Huber, NLL, masking, head, layer, scheduler, or timing change is combined.

The local search on 2026-09-22 inspected 21 saved configuration/plan/manifest files
under `probe_runs/` and `probe_comparison/`, plus experiment source/protocol text.
No equivalent stronger early original-AMF comparison was found. Centered variants
use a different objective/state; the historical decisive protocol uses the original
LR and FlowMatch, not this fourfold UniPC candidate. Remote-only archives were not checked.

## Frozen comparison

Profile: [configs/wan_lr4_camel_s1.json](../configs/wan_lr4_camel_s1.json).
Baseline: `probe_runs/wan_port_acceptance_camel_s1_v2_retry`; restore its complete
directory locally or pass its Drive location with `--baseline`. Use its existing
lossless `forward/reference` and `reverse/reference` PNG directories, not recompressed
MP4 inputs or latent reversal. Reuse all three completed baseline videos.

Keep T2V 14B revision `38ec498cb3208fb688890f8cc7e94ede2cbd7f68`, seed 1,
the exact saved prompt/negative/source conditioning and initial noise, 21 frames at
832x480, UniPC/shift 3/50 steps/CFG 5, block 20/mean logits, original MSE and
nonzero-reference mask, sharpening 2, clean hard reference, all frame pairs,
indices 0-9, five Adam updates each, and K/V injection off.

The [launcher](../scripts/run_wan_lr4.py) delegates to the existing entry point and
its `--lr 0.008 0.004` override. It pins the snapshot through the process-local
model mapping because the CLI has no revision flag, disables extra embedding-file
saving to match acceptance, and checks all saved configuration fields before loading.
It reuses existing source/runtime validators and checks initial-noise, conditioning,
RoPE, and scheduler identity after ordinary setup and before sampling. These are
reuse checks, not new parity or propagation experiments. No optimizer or sampler is
implemented in the launcher, and production source/configuration remain unchanged.

The user's execution hardware is an **80 GB A100 or H100** on Colab. The launcher
accepts either and records actual device/memory/runtime details; keep both arms on
the same server. The saved baseline used A100-SXM4-40GB, so this is a documented
hardware difference: setup hashes do not prove identical sampling trajectories, and
small differences from the old baseline cannot be attributed solely to LR. CPU
offload remains enabled. Match Python 3.13.15, CUDA 12.8, and package versions in the
baseline manifest; the supplementary tested tokenizer pin is 0.22.2. Runtime/source/
setup mismatch stops the comparison. Do not relax hashes or generate a replacement
baseline automatically. Compatibility on the execution runtime remains unverified.

## VS Code Colab entry point

Use [wan_lr4_colab.ipynb](../wan_lr4_colab.ipynb), with a working notebook copy outside
the remote checkout. Publish the prepared files to `docs/wan-active-instructions`
before syncing: notebook, launcher, JSON profile, this document, `CURRENT_PLAN.md`,
`README_WAN.md`, and `tests/verify_wan_lr4_colab.py`. The notebook clones that branch to `/content/ditflow_lr4`, or
uses `git pull --ff-only` after checking its remote, branch, and clean working tree.
It records the source commit. It does not publish local changes for you.

Mount Drive using VS Code's **Colab: Mount Google Drive to Server...** command and
execute the inserted cell ([official extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide#mounting-google-drive)).
Default locations, editable in the notebook:

- Baseline folder: `/content/drive/MyDrive/ditflow_results/wan_port_acceptance_camel_s1_v2_retry`.
  Alternatively, place its original ZIP beside it; the archive must have
  `manifest.json` at its root. Baseline evidence is ignored by Git, so clone/pull
  alone does not restore it.
- Results: `/content/drive/MyDrive/ditflow_results/wan_lr4_camel_s1/`.
  Logs stream directly to `logs/`; videos and records go into each arm directory.
- Review export: a timestamped `wan_lr4_camel_s1_review_*.zip` beside the results,
  including candidate results, baseline clips/reference PNGs, and configuration.
  Download it from Drive; browser-specific `files.download()` is not required.

The notebook defaults to `RUN_GENERATIONS = False`. It checks the saved Python/Torch
runtime, installs missing/differing non-Torch dependencies, and requests a kernel
restart if needed. It does not replace Python or Torch. A runtime mismatch is a
specific compatibility blocker to report, not a reason to rerun scientific probes.
Prepare both arms first; deliberately enable forward execution, inspect its full
video, and only then run reverse. Do not use Run All with generation enabled.

## Commands and proposed compute cap

Preparation only, safe under the current zero-GPU budget, from the repository root:

```bash
python scripts/run_wan_lr4.py --arm forward
python scripts/run_wan_lr4.py --arm reverse
```

After GPU execution is authorized, the paired commands are:

```bash
python scripts/run_wan_lr4.py --arm forward --execute
python scripts/run_wan_lr4.py --arm reverse --execute
```

Outside the notebook, add both `--baseline /path/to/saved/baseline` and
`--output-root /content/drive/MyDrive/ditflow_results/wan_lr4_camel_s1` to each
command to read/write Drive directly. Omitting them retains the local defaults.

Proposed maximum: **two new guided generations**, one per reference, each 50 sampling
steps and 50 Adam updates, plus their normal model/reference setup. No off/native
generation, propagation trace, automatic retry, or sweep. Run forward first; inspect
failure/quality before reverse. The launcher refuses existing arm directories and
requires forward completion before reverse. If setup fails or quality is already
unacceptable, preserve the output and stop. Current GPU compute used: zero.

Outputs: `probe_runs/wan_lr4_camel_s1/{forward,reverse}/`, with effective config,
baseline/setup/source identity, completion/video hash and elapsed time, or failure
record. The generated filename comes from the target prompt; `completion.json`
identifies it. Preserve logs and all outputs. Completion is not motion acceptance.

## Evaluation and stopping rule

Measure subject screen dx, fence dx, and their difference separately on frames 5-20,
using visually checked tracks and the same coordinate conventions as the
[baseline review](../probe_comparison/port_acceptance_20260922_early_review/REVIEW.md).
Choose corresponding visible landmarks per video; do not reuse old pixel coordinates
blindly when composition changes. Inspect all 21 frames for opening corruption,
anatomy, coherence, and quality relative to the saved off and guided arms. If a track
fails, report uncertainty rather than selecting whichever metric looks favorable.

| Observed result | Decision |
|---|---|
| Useful opposite-direction subject motion with acceptable quality | Advance to a separately budgeted independent seed/motion confirmation; no general success claim yet. |
| Credibly larger reference-ordered separation, both still leftward | Strength affects response, but directional acceptance still fails. Stop this comparison without automatic escalation. |
| No meaningful improvement or substantial degradation | Reject this fourfold candidate; stop increasing strength automatically. |
| Unresolved measurement uncertainty or incompatible setup | Do not accept or make a scientific failure claim; identify the precise limitation before deciding further work. |

Scalar loss reduction cannot pass. Subject/fence motion must remain distinguishable;
a larger camera pan alone is not proof of intended subject control. The existing
32-pixel manual review allowance is not a statistical confidence interval or an
automatic significance threshold for small between-arm changes.

Preparation validation: both preview commands passed source and RGB-reference checks.
The real entry point's CLI parsing/config merge was exercised on CPU, stopping before
model construction: only LR and reference/output locations differed from the saved
configuration after matching embedding-file saving. Both arms retain 50 configured
updates. Launcher syntax passed. No model/Torch import, CUDA initialization, download,
or candidate output directory was created by these preparation checks.

Colab adapter validation: `python tests/verify_wan_lr4_colab.py` passes four CPU
checks covering hardware/arm consistency, notebook syntax and default budget,
persistent failure logs, and partial-result ZIP export. Both launcher previews also
pass with explicit baseline/output overrides against the saved local evidence.
The VS Code extension, Drive mount, package installation, and pretrained generation
have not been exercised on a live Colab server in this preparation task.
