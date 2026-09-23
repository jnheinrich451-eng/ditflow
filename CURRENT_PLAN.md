# Current plan: DiTFlow port to Wan2.1 T2V 14B

Updated 2026-09-22. This is the active handoff; historical "next run" instructions
are superseded as task assignments, while their evidence and procedures remain intact.

## Goal and present scope

Faithfully port CogVideoX DiTFlow to **Wan2.1 T2V 14B** for testing and as the user's
model base. Retain native Wan behavior and the original DiTFlow comparison.
Motion-transfer acceptance requires reference changes to produce intended decoded
subject-motion changes with useful quality. I2V is optional future work, not the
current target. Current task: evaluate the proposed port next step using saved
early-window acceptance outputs before implementation. Budget: **zero GPU runs**;
reuse motion measurements and fill only missing decoded-motion evidence on CPU.
Stop after deciding whether that comparison completed and met the motion criterion.
Review checkout HEAD: `aec1915`, with pre-existing modified/untracked work.

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
  remain available; none is designated the next run.

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

## Unresolved hypotheses and next decision

Whether correspondence quality, intervention timing/strength, head/block choice, or
the earlier transformer path limits reference response remains unresolved. Centering,
sharpening, regional support, Huber, and destination NLL are historical experimental
variants, not adopted replacements for the DiTFlow port.

**Decision reached:** the existing early-window comparison is complete and ineffective
for the intended direction test; no missing arm or basic propagation/parity rerun is
needed. This result is distinct from the late centered-AMF failure.

**Next implementation choice remains unresolved.** Failure alone does not distinguish
a port defect from target/readout, placement, or strength limitations. Evaluate a
specific hypothesis using existing captures before selecting a bounded change; do
not automatically adopt Huber, destination NLL, another layer/schedule, or the historical
decisive suite. This review selects no replacement method and implements no inference
change. Any later experiment must state its decision, budget, and stopping condition
and remain within authorized scope; uncertainty does not require a chain of diagnostics.

Remaining contradictions are preserved with context: the old [I2V target](docs/WAN_RESEARCH_TARGET.md)
is superseded by current T2V scope; the [acceptance audit](docs/WAN_PORT_ACCEPTANCE.md)
mixes an early "not run" status with later completed results; older reports name
different next runs. CLI 1.3b and injection defaults remain unchanged and need explicit
interpretation. Historical environment/scheduler variants are not interchangeable.
