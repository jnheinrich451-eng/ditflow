# Current plan: DiTFlow port to Wan2.1 T2V 14B

Updated 2026-09-22. This is the active handoff; historical "next run" instructions
are superseded as task assignments, while their evidence and procedures remain intact.

## Goal and present scope

Faithfully port CogVideoX DiTFlow to **Wan2.1 T2V 14B** for testing and as the user's
model base. Retain native Wan behavior and the original DiTFlow comparison.
Motion-transfer acceptance requires reference changes to produce intended decoded
subject-motion changes with useful quality. Per the latest supplied instructions,
the final target is **I2V 14B**; this task remains on the intermediate **T2V 14B**
platform and does not establish I2V support. Current task: prepare one fourfold-LR original-AMF comparison,
following the completed saved-output review. Budget now: **zero GPU runs**; only
configuration/command preparation and CPU checks. Stop when the paired experiment
is ready for review. Preparation HEAD: `84c8bae`, with pre-existing modified/untracked work.

## Active implementation and configuration

- Entry: [motion_guidance_wan.py](motion_guidance_wan.py), explicitly `--model 14b`
  (`Wan-AI/Wan2.1-T2V-14B-Diffusers`). The unchanged CLI default is **1.3b**.
- Configuration: [configs/guidance_config_wan.yaml](configs/guidance_config_wan.yaml),
  loaded by the entry point. Existing latent/flow-MSE defaults: block 20, mean-head
  logits, clean hard reference, all ordered frame pairs, sharpening 2; five Adam
  updates at sampling indices 0-9, LR .002 -> .001; UniPC, shift 3, 50 steps,
  CFG 5, 21 frames at 480x832. These are starting settings, not validated optima.
- K/V injection defaults to block 0 in the guidance window. `--no_injection`
  disables it; native-backbone comparison needs **both** `--no_guidance --no_injection`.
  Saved acceptance runs disabled injection, so they are not default-command results.
- Integration: [transformer](guidance_utils/wan_transformer.py),
  [attention](guidance_utils/wan_modules.py), [AMF](guidance_utils/wan_motion_flow_utils.py).
  Original baseline: [motion_guidance.py](motion_guidance.py) and
  [CogVideoX config](configs/guidance_config.yaml). Experimental notebooks/runners
  remain available; the prepared candidate below is the only named next comparison.

## Established evidence and limits

The early-window result below was reviewed from decoded frames and independent motion
measurements on 2026-09-22; other entries summarize inspected saved reports. No model
inference was run. Generated evidence is local/git-ignored and must be retained.

- **Native/off parity:** the saved 14B [parity record](probe_runs/wan_port_acceptance_camel_s1_v2_retry/parity.json)
  reports zero maximum difference at all 50 steps. Its [manifest](probe_runs/wan_port_acceptance_camel_s1_v2_retry/manifest.json)
  pins checkpoint revision `38ec498cb3208fb688890f8cc7e94ede2cbd7f68`; its
  [configuration](probe_runs/wan_port_acceptance_camel_s1_v2_retry/configuration.yaml)
  identifies the tested setup. This is bounded Diffusers-path evidence, not universal
  parity across versions or official Wan backends; see the [audit](docs/WAN_PORT_ACCEPTANCE.md).
- **Early original-AMF comparison completed and failed:** the [saved-output review](probe_comparison/port_acceptance_20260922_early_review/REVIEW.md)
  verifies off/forward/reverse videos, both 50-step completion logs, and the original
  AMF/MSE configuration (indices 0-9, five updates each, block 20, no injection).
  Across clean RGB frames 5-20, fence-relative hump displacements are approximately
  **-102/-86/-122 px**, versus **+127/-135 px** in the references. Both guided animals
  still travel left. Existing hash-identical off/reference tracks were reused; only
  missing guided motion was measured on CPU. [Metrics and limits](probe_comparison/port_acceptance_20260922_early_review/motion_review.json),
  [visual review](probe_comparison/port_acceptance_20260922_early_review/landmark_review.jpg).
  Arm differences do not establish zero influence, but fail intended directional control.
  Core source/config hashes match; the saved runner was recovered exactly from Git.
  The original run's full commit is unrecorded. Opening corruption limits quality.
- **Separate late centered-AMF failure:** the [completed centered-AMF review](probe_runs/wan_centered_pilot_camel_s1/review/REVIEW.md)
  reports about -134 px fence-relative camel displacement in off/forward/reverse arms,
  versus opposing references. Loss and sampler state changed, but motion did not respond
  as intended. This tests only the specified late index-39 intervention, five updates,
  seed 1; it neither validates the port's motion transfer nor rejects every schedule.
- **Localization is diagnostic:** the [block-input review](probe_runs/wan_centered_block_input_camel_s1/review/REVIEW.md)
  reports reference-gradient cosine .292 at block-20 input versus .975 at the archived
  latent for the experimental torso NLL objective. It does not identify a port defect.
- **Replay discrepancy is not a mandatory detour:** the [repeatability review](probe_runs/wan_centered_repeatability_camel_s1/review/REVIEW.md)
  reports .9816% relative RMS difference even between two unlogged same-reference
  backwards, failing the old gate. A separate hook effect remains unresolved; thresholds
  were not revised. Two original pilot trace NPZs are absent locally. No decoded
  torso-NLL acceptance comparison is established by these diagnostics.

## Named candidate and next decision

Whether correspondence quality, intervention timing/strength, head/block choice, or
the earlier transformer path limits reference response remains unresolved. Centering,
sharpening, regional support, Huber, and destination NLL are historical experimental
variants, not adopted replacements for the DiTFlow port.

**Decision reached:** the existing early-window comparison is complete and ineffective
for the intended direction test; no missing arm or basic propagation/parity rerun is
needed. This result is distinct from the late centered-AMF failure.

**Prepared candidate: original AMF with fourfold latent LR**, `.008 -> .004` instead
of `.002 -> .001`. Hypothesis: a weak useful response exists but update size is
insufficient. The -86/-122 px ordering around the -102 px off baseline is compatible
with this hypothesis, not proof. References combine subject and camera motion;
AMF represents image-grid displacement, not fence-relative subject motion.

[Profile](configs/wan_lr4_camel_s1.json), [paired commands and reuse requirements](docs/WAN_LR4_CANDIDATE.md),
[launcher](scripts/run_wan_lr4.py), [VS Code Colab notebook](wan_lr4_colab.ipynb).
The notebook clones/pulls `docs/wan-active-instructions` and writes directly to
`MyDrive/ditflow_results/wan_lr4_camel_s1`, with logs and a downloadable review ZIP.
Use an 80 GB A100/H100, retaining CPU offload and the saved runtime pins. Hardware
differs from the saved A100 40 GB baseline; small cross-hardware differences cannot
be assigned solely to LR. Both new arms must use matching hardware/runtime. Keep checkpoint/input/noise/conditioning, original
MSE/masks, block 20/mean heads, indices 0-9, five updates, UniPC, and no injection fixed.
The local 21-record search found no equivalent stronger run; remote-only evidence is
unverified. No production defaults or guidance method changed. Preparation checks
pass; pretrained execution and baseline setup compatibility remain untested.

**Proposed cap, not executed:** two new guided generations (forward/reverse), 50
sampling steps and 50 Adam updates each; reuse existing off and ordinary-LR outputs.
Measure subject screen, fence, and relative motion separately; assess quality over
all frames, including opening corruption. Opposite-direction motion with acceptable
quality advances to independent confirmation. Larger credible separation with both
still left is partial response, not acceptance. No improvement/degradation rejects
the candidate; uncertainty is reported. Stop without automatic strength escalation,
parity/propagation reruns, or the decisive suite. GPU execution is not authorized yet.

Remaining contradictions are preserved with context: the [I2V target](docs/WAN_RESEARCH_TARGET.md)
is the final goal per the latest supplied instructions, while this comparison and
older repository scope wording concern T2V; the [acceptance audit](docs/WAN_PORT_ACCEPTANCE.md)
mixes an early "not run" status with later completed results; older reports name
different next runs. CLI 1.3b and injection defaults remain unchanged and need explicit
interpretation. Historical environment/scheduler variants are not interchangeable.
