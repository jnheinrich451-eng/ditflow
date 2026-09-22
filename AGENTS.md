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
- Preserve implementation files, tests, diagnostic scripts, notebooks, raw evidence,
  and their paths, including untracked work. Do not clean by deleting research history.
- Before authorized GPU work, record the question, smallest comparison, budget,
  stopping condition, and resulting decision in the plan. Do not invent a new
  experiment when the evidence leaves the decision unresolved.
- Distinguish inspected artifacts, historical reports, and untested proposals.
  Report pipeline, configuration, source identity, evidence limits, and what remains open.
