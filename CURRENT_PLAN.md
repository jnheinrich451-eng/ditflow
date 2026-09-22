# Current plan: DiTFlow port to Wan2.1 T2V 14B

Updated 2026-09-22. This is the active handoff; historical "next run" instructions
are superseded as task assignments, while their evidence and procedures remain intact.

## Goal and present scope

Faithfully port CogVideoX DiTFlow to **Wan2.1 T2V 14B** for testing and as the user's
model base. Retain native Wan behavior and the original DiTFlow comparison.
Motion-transfer acceptance requires reference changes to produce intended decoded
subject-motion changes with useful quality. I2V is optional future work, not the
current target. This task is documentation only: **zero GPU runs, no inference or
method changes**. Starting HEAD: `4a2edc7`, with pre-existing modified/untracked work;
that commit alone does not identify the full experimental checkout.

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

Sources below were read during cleanup; no inference, video re-evaluation, or metric
recomputation was performed. Generated evidence is local/git-ignored and must be retained.

- **Native/off parity:** the saved 14B [parity record](probe_runs/wan_port_acceptance_camel_s1_v2_retry/parity.json)
  reports zero maximum difference at all 50 steps. Its [manifest](probe_runs/wan_port_acceptance_camel_s1_v2_retry/manifest.json)
  pins checkpoint revision `38ec498cb3208fb688890f8cc7e94ede2cbd7f68`; its
  [configuration](probe_runs/wan_port_acceptance_camel_s1_v2_retry/configuration.yaml)
  identifies the tested setup. This is bounded Diffusers-path evidence, not universal
  parity across versions or official Wan backends; see the [audit](docs/WAN_PORT_ACCEPTANCE.md).
- **Decoded failure despite propagation:** the [completed centered-AMF review](probe_runs/wan_centered_pilot_camel_s1/review/REVIEW.md)
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

**Next decision: unresolved.** Available evidence does not select a specific port
repair or the next bounded motion-transfer comparison. Continue from the entry point
and configuration above when substantive port work resumes; do not automatically run
the old decisive, subject-only, prefix-localization, or timing proposals. No new
experiment or scientific method is selected by this cleanup.

Remaining contradictions are preserved with context: the old [I2V target](docs/WAN_RESEARCH_TARGET.md)
is superseded by current T2V scope; the [acceptance audit](docs/WAN_PORT_ACCEPTANCE.md)
mixes an early "not run" status with later completed results; older reports name
different next runs. CLI 1.3b and injection defaults remain unchanged and need explicit
interpretation. Historical environment/scheduler variants are not interchangeable.
