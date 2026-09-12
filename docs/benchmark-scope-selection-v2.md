# Benchmark Scope & Method Selection — v2

**Purpose.** Fixes which methods are *tested* (run, scored, tabulated) versus *characterized*
(read, taxonomised, cited in Related Work, not run), with a stated justification for every
untested paper. This document is the agreed scope; methods not listed here are out of scope
and additions require an explicit scope amendment.

**Selection principle.** Methods are selected to span the space of **control-signal
representations**, not to maximise count. Within a family, we test representatives whose
motion representation differs from one another. Papers sharing a representation with a tested
representative are characterized in §5 and not re-run, because the hypothesis under test
concerns the representation, not the implementation.

---

## 0. What changed from v1, and why

v1 committed to 22 rows (A1-A4, B1-B14, C1-C4), 14 of them separate external repositories. Three changes.

**(a) Port cost is now the tier boundary, and it collapsed.** DiTFlow's codebase implements
SMM and MOFT as alternative losses inside the *same* guidance loop — verified: both
`motion_guidance.py` and `motion_guidance_wan.py` define `compute_motion_flow_loss`,
`compute_moft_loss` and `compute_smm_loss`, selected by `--loss_type`. So porting DiTFlow to a
backbone ports **three methods** to that backbone, sharing one harness, one input contract and
one set of hyperparameters. v1 costed these as separate native-backbone runs. They are free.

**(b) External ports cut from ~14 to 3.** The ladder in §1 requires occupied *rungs*, not
methods per rung. One representative per rung tests the slope prediction; two or three are
only needed for a Pareto frontier, which is a secondary claim. Everything cut moves to §5 with
its justification, which is what §5 is for.

**(c) A reproduction anchor was added (§2).** v1 had no row whose published numbers could be
checked against. The controls calibrate the *metrics*; nothing validated the *pipeline*.
Without an anchor, a reader has no reason to believe this harness's MF means what DiTFlow's MF
means, and every number in the chapter floats.

**Evidence for the cut.** Bringing up one method — DiTFlow, whose code is in this repository,
on the backbone it was written for — surfaced two hard blockers (`torchvision.io.read_video`
removed upstream; a pinned `diffusers==0.30.2` against a current torch build) before a single
generation completed. That is the realistic per-method cost, and unfamiliar external repos are
worse, not better. Fourteen of them at three days each is two months before any result exists.

---

## 1. The representation ladder

The organising axis is *how decomposed the control signal already is when the method receives it.*

| Rung | Control input | Camera/object separated? | Predicted parallax behaviour |
|---|---|---|---|
| **R0** | RGB reference video | No — entangled in pixels | Degrades with parallax |
| **R1** | 2D image-space trajectory | **No** — sparse, still entangled | **Degrades with parallax** (key prediction) |
| **R2** | Camera pose only | Camera given; object unconstrained | Camera accurate; object dynamics suppressed |
| **R3** | 3D world-frame trajectory + camera pose | Yes | Flat in parallax |
| **RJ** | Joint camera + object control | Claimed yes — audit required | To be determined |

**Load-bearing prediction:** R1 degrades like R0 despite being marketed as "explicit control."
A 2D image-space trajectory under a translating camera still satisfies
`u_obs = u_cam(1/Z) + u_obj`. Sparsifying the evidence does not disentangle it.

**The claim is about slope, not level.** Rungs are occupied by methods on different backbones,
so absolute scores are not comparable across rungs — §6's tier rule forbids exactly that. What
*is* comparable is each method's **degradation gradient against parallax**, measured within its
own row and normalised to its own low-parallax band. A backbone difference shifts a row's
intercept; it does not explain a shared slope. Every cross-rung claim in this chapter is a
statement about slopes, and must be written that way.

---

## 2. The reproduction anchor

Runs before anything in §3 and §4. **Not a contribution — a calibration of trust.**

| Item | Value |
|---|---|
| Method | DiTFlow, as released, CogVideoX (2B full sweep; 5B subset) |
| Protocol | DiTFlow's own: DAVIS, Subject prompt class, MF + IQ |
| Deliverable | Δ from published, and the **rank correlation** of method ordering |
| Status | harness built and validated; manifests frozen |

**Why ordering, not absolute values.** Our DAVIS subset is *a* 50 of the 90 sequences, not the
50 DiTFlow used — the paper does not list them, so the exact set is unrecoverable. Absolute
values are therefore not expected to match. Method *ordering* is robust to subsetting, and
ordering is what a benchmark asserts. Report Spearman ρ against the published ordering; report
absolute Δ as context, not as a claim.

The same run yields SMM and MOFT rows for free (§0a), so the anchor also confirms that the
three-mechanism R0 block behaves sensibly before it is carried to another backbone.

---

## 3. Control rows (not methods)

These calibrate every metric before any method is admitted.

| ID | Construction | Bounds | Blocking? |
|---|---|---|---|
| C1 | Reference image duplicated to N frames | Degenerate floor — flags broken metrics | Yes |
| C2 | Wan2.1 I2V, reference image + caption, no motion signal | Backbone-prior floor | Yes |
| C3 | Reference video through Wan2.1 VAE encode→decode | Codec ceiling — see note | No |
| C4 | Extraction pass run twice on the reference video | **Instrument noise floor → sets parallax band thresholds** | Yes |

**Gate:** any metric on which C1 scores within noise of the best method is dropped from the
main table and reported only in an appendix with an explicit note. Band thresholds are set
from C4 and frozen in `calibration.lock.yaml` before any method runs.

**Note on C3.** This is a ceiling on what the VAE preserves, not on motion transfer — the task
requires the content to *change*, which C3 does not do. Label it "codec ceiling" and do not
present it as an achievable target for any tested row.

---

## 4. Tested methods

Target: 10 method rows (7 Tier A + 3 Tier B) plus the 4 controls in §3 and the anchor in §2. Run order is by **risk**, not family order — the RJ block runs first because
it is the block most likely to change the plan.

### Tier A — one port, seven rows (Wan2.1-14B I2V)

Porting DiTFlow's guidance loop to a backbone brings SMM, MOFT and both ablation controls with
it, since they are `--loss_type` and `--no_guidance` switches on the same loop. One port
therefore yields a **complete R0 block on the target backbone, at one input contract**.

| # | Row | Representation | Port cost |
|---|---|---|---|
| A1 | DiTFlow (AMF) | Per-patch displacement field — **most local** | the port |
| A2 | SMM | Global spatio-temporal feature descriptor — **most global** | free (`--loss_type smm`) |
| A3 | MOFT | Motion-channel decomposition — third distinct mechanism | free (`--loss_type moft`) |
| A4 | Injection only | KV injection, no guidance | free (`--no_guidance`) |
| A5 | Backbone | No guidance, no injection — lower bound | free (`--no_guidance --no_injection`) |
| A6 | DiTFlow, object-masked | AMF with object support removed | derived from A1 |
| A7 | Follow-Your-Motion | Spatial/temporal decoupled LoRA | none *if* native Wan2.1 **I2V** — verify |

A1–A3 span the locality axis end to end (per-patch → channel → global) with one free variable:
the motion field source. That is the contrast that makes A1 interpretable, and v1 spent three
external ports on it.

**Ports required, honestly counted.** `motion_guidance_wan.py` targets Wan2.1 **T2V**
(`MODEL_IDS` holds T2V-1.3B and T2V-14B only). Tier A specifies **I2V**. So:

1. **T2V → I2V on Wan2.1** — the real port, and the only one Tier A needs.
2. The existing **T2V-14B** path is already working and is not wasted: it becomes the
   **matched-contract row**. An I2V method compared against T2V baselines has seen the
   reference's first frame and wins identity metrics for trivial reasons. Running the same
   methods at T2V-14B gives one block where the input contract is equal.
3. Validate the port at **T2V-1.3B** first (`grid_wan_1_3b.yaml`). The Wan `guidance_blocks`
   and `motion_temp` are documented in-repo as starting points, not ported optima; find that
   out on cheap hardware, not at 14B on A100 hours.

The "CogVideoX-5B-I2V cross-check" in v1 is dropped: DiTFlow is T2V on CogVideoX too, so it
was a third port bought for a cross-check the anchor (§2) already provides.

### Tier B — three external ports, one per unoccupied rung (R1, R2, RJ)

R0 is fully covered by Tier A. Tier B exists only to occupy R1, R2 and RJ.

| # | Method | Rung | Why this one |
|---|---|---|---|
| B1 | ATI | R1 | Trajectory instruction on the **Wan2.1 backbone** — same family as Tier A at near-zero port cost, so the R1-vs-R0 slope comparison is least confounded here |
| B2 | AC3D | R2 | The dynamics-recovery representative; documents the RE10K suppression mechanism explicitly, which is the audit in §7 |
| B3 | SymphoMotion | RJ | Direct competitor, checkpoint confirmed available; compares against VidCRAFT3 in its own paper, establishing the family-comparison precedent |

**Promotion rules, in order.** If a row fails to run, promote its named substitute rather than
inventing a replacement: R1 → Tora, then Motion Prompting. R2 → CameraCtrl, then CameraCtrl II.
RJ → Uni3C, then LeviTor. Record which substitution fired and why.

**Conditional additions**, in priority order, only if the three above are running and budget
remains: OrthoMotion (positioning threat — claims disentanglement by construction, worth
disproportionate effort *if released*), then MotionClone (attention-based R0 mechanism; the
DiTFlow paper adapted it but the release exposes only flow/moft/smm, so it is a genuine port),
then a second R2 point to make the trade-off a frontier rather than two points.

---

## 5. Characterized, not tested — with justification

Every entry needs its justification sentence to appear in §2 of the chapter. Silence is the
failure mode.

### R0 family

| Paper | Justification for not testing |
|---|---|
| MotionDirector | Spatial/temporal LoRA separation is **subsumed by FYM's STD LoRA** (A7). Testing both benchmarks a method against its own ablation. |
| MotionInversion | Tuning-based single-video customisation; the tuning-based class is already represented by A7 at lower cost and on the target backbone. |
| Motion Consistency Loss | Loss-level regulariser rather than a distinct motion representation; orthogonal to and composable with tested methods. |
| MotionClone | Distinct attention-based mechanism and a genuine port. Deferred to the conditional list rather than cut — R0 already has three mechanisms via A1–A3. |
| Motionshop, MotionCrafter | No public release. |
| Let Your Image Move with Your Motion | **Flag: closest published task match** (I2V + reference video, multi-object). Highest examiner-question risk in the document. Promote to tested if release quality permits; otherwise the justification must be written more carefully than any other entry here. |

### R1 — all share the image-space trajectory representation tested at B1

| Paper | Justification |
|---|---|
| DragNUWA | Trajectory + text + image; same image-space representation. |
| MagicMotion | Dense-to-sparse trajectory; a density variant of the same representation. |
| TrackGo | Mask + arrow → attention; same image-space signal, different encoder. |
| Boximator | Bounding box — strictly coarser than point trajectory; same plane. |
| FreeTraj | Training-free box trajectory; representation subsumed by Boximator. |
| Tora | Named R1 substitute if B1 fails; DiT architecture class matching Tier A. |
| Motion Prompting | Second R1 substitute. |
| DragAnything | Entity-level trajectory; same plane, entity granularity. |

### R2 family

| Paper | Justification |
|---|---|
| CameraCtrl | Named R2 substitute if B2 fails. RE10K-trained representative. |
| CameraCtrl II | Second substitute; dynamic-scene camera data, same corrective direction as B2. |
| CamCo | Epipolar-constrained; RE10K-class training data, frontier position predicted by B2. |
| MotionCtrl (camera branch) | Separately trained camera and object branches; conditional third R2 point if budget allows. |
| ViewCrafter | **Different task** — novel view synthesis on static scenes. No object dynamics to preserve, so dynamics-retention is undefined, not merely low. |

### R3 / RJ family

| Paper | Justification |
|---|---|
| VidCRAFT3 | **No public release** (confirmed directly). Characterized; cited as B3's own comparison point. |
| Uni3C | Named RJ substitute; 3D-conditioned world-frame control. |
| LeviTor | Second RJ substitute; depth-ordered 3D trajectory, closest to M3's control interface. |
| OrthoMotion | Conditional on release — see §4. Positioning threat, not a coverage row. |
| ActCam | Conditional on release. |
| Perception-as-Control | 3D-aware representation, closest to Uni3C/LeviTor. |
| FreeForm Motion Control | **Dataset paper**, not a method. Evaluate as a clip source for the high-parallax band alongside Kubric. |
| Physical Simulator In-the-Loop | Belongs to Aim 2 (physical plausibility), not the decomposition benchmark. Cite in Aim 2 positioning. |

### Instrument components — not benchmark rows

ViPE, D4RT / Open-D4RT, SpatialTrackerV2, VGGT, Shape of Motion, SAM2. Components of M1 or of
the evaluation harness. They appear in the methods chapter and in M1 validation (Kubric
metric-ATE), **never as rows in the generation benchmark**. Stating this explicitly pre-empts
the category confusion.

---

## 6. Comparison rules

1. **Never compare absolute scores across backbones.** Rows on different backbones go in
   separate table blocks. A Wan number and a CogVideoX number differ by backbone capacity
   before any method effect.
2. **Cross-rung claims are slope claims** (§1). Normalise each row to its own low-parallax band
   and compare gradients.
3. **Never compare across input contracts.** T2V and I2V rows sit in separate blocks; an I2V
   method has seen the reference's first frame. The matched-contract T2V-14B block (§4) exists
   for this reason.
4. **Re-implementations are labelled.** A2/A3 are DiTFlow's SMM and MOFT, not the originals;
   they compare fairly against each other and against A1, never against published SMM/MOFT
   numbers, which were measured on different backbones entirely.

---

## 7. Audits attached to specific tested rows

Not extra runs — analyses of runs already scheduled.

| Row | Audit | Claim it tests |
|---|---|---|
| A1–A3 | Locality sweep: per-patch → channel → global, one backbone, one contract | Whether the motion-field *representation* drives parallax degradation, or the backbone does |
| A1 vs A6 | Object-masked ablation | How much of A1's score is camera-only contribution |
| B1 vs A1–A3 | Parallax degradation **slope** | R1 degrades like R0 → "explicit control" ≠ decomposed control |
| B2 | Camera accuracy vs dynamics retention | Camera control and object dynamics are traded, not jointly solved |
| B3 | Is the object trajectory world-frame or camera-frame? | Camera-frame joint control inherits the same non-identifiability |
| All | C1 comparison | Which reported metrics have any dynamic range at all |

Dropped from v1: the MotionClone disjoint-partition audit (row not tested), and the
three-point R2 frontier (one R2 point; frontier is conditional per §4).

---

## 8. Cell availability

Not every family is scoreable on every metric. Mark N/A; never impute.

| Family | Camera accuracy | Object trajectory error | Dynamics retention | Identity | Quality |
|---|---|---|---|---|---|
| R0 | ✓ | ✓ | ✓ | ✓* | ✓ |
| R1 | drift only | ✓ | ✓ | ✓* | ✓ |
| R2 | ✓ | N/A | ✓ | ✓ | ✓ |
| R3 / RJ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Controls C1–C4 | ✓ | ✓ | ✓ | ✓ | ✓ |

`drift only` = camera behaviour is emergent, not controlled; report as drift, labelled, never
as a control-accuracy score.
`✓*` = only defined for methods accepting a reference image; T2V-only rows either receive it as
the I2V first frame or are marked N/A. State which, per row.

**The N/A pattern is itself a result.** No existing method fills every cell. Render it as a
figure in the gap statement rather than asserting the gap in prose.

---

## 9. Open decisions

1. **Follow-Your-Motion (A7)** — confirm it is native Wan2.1 **I2V**, not T2V. If T2V, it is a
   port and Tier A's "one port" claim needs restating.
2. **Let Your Image Move with Your Motion** — promote to tested? Closest task match, highest
   examiner-question risk.
3. **OrthoMotion / ActCam release status** — check before finalising; both conditional.
4. **Tier A pinned frame count** — blocked on the DiTFlow VRAM spike. If 81f does not fit, all
   Tier A rows re-run at the reduced config and the deviation from Wan2.1's native envelope is
   stated. Note the in-repo `nframes` defect: reference and target AMF agree only at 6 latent
   frames (21–24 pixel frames); any other length mismatches silently.
5. **High-parallax clip source** — Kubric render sweep vs FreeForm Motion Control vs MiraData
   subset. Blocked on the parallax screening pass. Note MiraData clips now carry measured
   camera trajectories via RealCam-Vid, so parallax banding is available without estimation.
6. **Seeds per clip** — 3 assumed for budgeting. Confirm against total GPU hours once
   per-method runtime is measured. Prefer 1 seed across the full grid plus 3 seeds on a fixed
   subset reported once as run-to-run variance.
