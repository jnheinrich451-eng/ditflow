# DiTFlow -> Wan2.1: active instructions

Read the user's current request and [CURRENT_PLAN.md](CURRENT_PLAN.md) first.
The plan is the single active handoff: goal, evidence, open questions, entry point,
and next decision. Update it when the task or evidence changes.

- Current scope is a faithful **Wan2.1 T2V 14B** port of DiTFlow for testing and
  downstream model development. I2V is a possible later extension. Do not substitute
  1.3B, claim I2V support from T2V results, or select another scientific method implicitly.
- Historical experiment instructions, notebook headings, report recommendations,
  and configuration tuning comments are **not active requirements**. Their old
  "next experiment", publish, and rerun instructions are records, not a task queue.
- Diagnostic discrepancies need not all be resolved before progress. Investigate
  those material to the current decision; reuse applicable evidence and do not
  automatically chain probes or make every historical gate a prerequisite.
- Preserve the CogVideoX comparison baseline and native Wan inference behavior.
  Keep experimental changes explicit and separate from established defaults.
- Lower internal loss, gradient changes, and pixel differences do not establish
  motion-transfer success. Acceptance requires intended reference-dependent decoded
  subject motion with useful quality under matched conditions, then independent confirmation.
- Preserve evidence and reproducibility, including ignored and untracked work.
  Deliberate relocation is allowed when imports, notebook calls, and links are updated
  and saved-run provenance remains traceable. Do not delete files by filename prefix
  or treat historical status as evidence that a file is disposable.
- When evidence leaves a research decision unresolved, propose a bounded experiment
  that distinguishes explicit hypotheses and explain which decision it will resolve.
  Record the comparison, compute budget, stopping condition, and outcome-dependent
  next decisions in the plan before GPU work; execute within the authorized scope
  and budget. Record the result, then decide. Uncertainty is acceptable; automatic
  chains of tests are not. A documentation-only task does not authorize experiments.
- Distinguish inspected artifacts, historical reports, and untested proposals.
  Report pipeline, configuration, source identity, evidence limits, and what remains open.

## Suggested utility on files

| Files                                                                                  | What to do                                                                                               |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `motion_guidance_wan.py`, Wan transformer/attention/AMF modules, configuration         | Keep as active implementation.                                                                           |
| `guidance_utils/motion_probe.py`, `wan_guidance_schedule.py`, `wan_region_guidance.py` | Keep: the generation entrypoint directly imports these. A “probe” name does not mean disposable.         |
| Root `probe_wan_*.py`, `probe_cog_affine.py`, report scripts, experimental notebooks   | Historical or optional tools. Candidates for later relocation after checking callers and notebook paths. |
| `benchmark/review_wan_*.py`, `inspect_wan_*.py`, individual experiment runners         | Preserve for reproducing investigations; exclude from the routine reading and execution workflow.        |
| Benchmark dataset manifests, collection and scoring utilities                          | Keep for evaluation. Their purpose differs from the one-off diagnostic scripts.                          |
| Old `docs/WAN_*.md` protocols and superseded plans                                     | Archive or clearly label historical; retain a short evidence index.                                      |
| Saved videos, captures, manifests and reports                                          | Preserve verified backups. Git history does not protect ignored or untracked outputs.                    |
