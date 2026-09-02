# Benchmark Scope & Method Selection — v1

**Purpose.** Fixes which methods are *tested* (run, scored, tabulated) versus *characterized*
(read, taxonomised, cited in Related Work, not run), with a stated justification for every
untested paper. This document is the agreed scope; methods not listed here are out of scope
and additions require an explicit scope amendment.

**Selection principle.** Methods are selected to span the space of **control-signal
representations**, not to maximise count. Within a family, we test representatives whose
motion representation differs from one another. Papers sharing a representation with a tested
representative are characterized in §2 and not re-run, because the hypothesis under test
concerns the representation, not the implementation.

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

---

## 2. Control rows (not methods)

These are run first. They calibrate every metric before any method is admitted.

| ID | Construction | Bounds | Blocking? |
|---|---|---|---|
| C1 | Reference image duplicated to N frames | Degenerate floor — flags broken metrics | Yes |
| C2 | Wan2.1 I2V, reference image + caption, no motion signal | Backbone-prior floor | Yes |
| C3 | Reference video through Wan2.1 VAE encode→decode | Achievable ceiling | No |
| C4 | Extraction pass run twice on the reference video | **Instrument noise floor → sets parallax band thresholds** | Yes |

**Gate:** any metric on which C1 scores within noise of the best method is dropped from the
main table and reported only in an appendix with an explicit note. Band thresholds are set
from C4 and frozen in `calibration.lock.yaml` before any method runs.

---

## 3. Tested methods

Target: ~14 rows. Run order is by **risk**, not by family order — the RJ block runs first
because it is the block most likely to change the plan.

### Tier A — ported, controlled (Wan2.1-14B I2V, with CogVideoX-5B-I2V cross-check)

| # | Method | Representation | Port burden | Role |
|---|---|---|---|---|
| A1 | DiTFlow | AMF, per-patch displacement — **most local** | Port required | Primary baseline; single free variable = motion field source |
| A2 | DiTFlow, object-masked | AMF with object support removed | Derived from A1 | Honesty row — isolates camera-only contribution |
| A3 | Follow-Your-Motion | Spatial/temporal decoupled LoRA | None (native Wan2.1) | Same-backbone secondary; tuning-based reference point |
| A4 | *(M3, later)* | 3D track guidance | — | Own method — excluded at this stage per supervisor guidance |

### Tier B — as published, native backbone, within-row contrasts only

**R0 — reference-video motion transfer**

| # | Method | Representation | Why this one |
|---|---|---|---|
| B1 | SMM / DMT (Yatim et al.) | Global spatio-temporal feature descriptor | The **global** end of the locality axis — the contrast that makes A1 interpretable |
| B2 | MotionClone | Temporal-attention sparse guidance | Attention-based mechanism, distinct from flow and from features; also carries the disjoint-partition audit (§5) |
| B3 | MOFT | Motion feature channel decomposition | Third distinct R0 mechanism; explicit motion/appearance channel split |

**R1 — 2D image-space trajectory**

| # | Method | Representation | Why this one |
|---|---|---|---|
| B4 | Tora | Trajectory→motion patches, DiT | DiT architecture class matching Tier A; trained trajectory encoder |
| B5 | ATI | Trajectory instruction, **Wan2.1 backbone** | Same backbone as Tier A at zero port cost — partially collapses the tier boundary |

**R2 — camera pose only**

| # | Method | Training data | Why this one |
|---|---|---|---|
| B6 | CameraCtrl | RealEstate10K | The RE10K-trained representative; expected high camera accuracy / low dynamics retention |
| B7 | AC3D | RE10K + 20K curated dynamic-scene/static-camera | The dynamics-recovery representative; documents the suppression mechanism explicitly |
| B8 | MotionCtrl (camera branch) | RealEstate10K | Third frontier point; also the only method with separately trained camera and object branches |

Three points minimum: two do not define a Pareto frontier.

**R3 / RJ — 3D trajectory and joint control** *(highest priority — run first)*

| # | Method | Claim | Why this one |
|---|---|---|---|
| B9 | SymphoMotion | Joint camera + object dynamics | Direct competitor; compares against B10 in its own paper, establishing the family comparison precedent |
| B10 | VidCRAFT3 | Camera + object + lighting | Direct competitor; separately trained camera and object control datasets |
| B11 | Uni3C | 3D-enhanced camera + human motion | 3D-conditioned; world-frame control signal |
| B12 | LeviTor | Depth-ordered 3D trajectory | The R3 representative closest to M3's control interface |
| B13* | OrthoMotion | Proves non-identifiability; claims disentanglement by construction | **Positioning threat — test if code releases.** Worth disproportionate effort |
| B14* | ActCam | Zero-shot joint camera + 3D motion | Test if released |

`*` = conditional on public release. If unavailable, move to characterized with the
FYM precedent as justification (they excluded Motionshop and MotionCrafter on the same grounds).

---

## 4. Characterized, not tested — with justification

Every entry below needs its justification sentence to appear in §2. Silence is the failure mode.

### R0 family

| Paper | Justification for not testing |
|---|---|
| MotionDirector | Spatial/temporal LoRA separation is **subsumed by FYM's STD LoRA**, which is tested (A3). Testing both benchmarks a method against its own ablation. |
| MotionInversion | Tuning-based single-video customisation; motion embedding representation, but tuning-based class already represented by A3 at lower cost and on the target backbone. |
| Motion Consistency Loss (training-free, temporal consistency) | Loss-level regulariser rather than a distinct motion representation; orthogonal to and composable with tested methods. |
| Motionshop, MotionCrafter | No public release. |
| Let Your Image Move with Your Motion | **Flag: closest published task match (I2V + reference video, multi-object).** Currently characterized; promote to tested if release quality permits. Justify carefully — this is the paper an examiner is most likely to name. |

### R1 family — all share the image-space trajectory representation tested at B4/B5

| Paper | Justification |
|---|---|
| DragNUWA | Trajectory + text + image; same image-space representation as B4. |
| MagicMotion | Dense-to-sparse trajectory; a density variant of the same representation. |
| TrackGo | Mask + arrow → attention; same image-space signal, different encoder. |
| Boximator | Bounding box — strictly coarser than point trajectory; same plane. |
| FreeTraj | Training-free box trajectory; representation subsumed by Boximator, method class by B2. |
| Motion Prompting | Track-conditioned; closest R1 alternative to B4. Second-choice substitute if Tora fails to run. |
| DragAnything | Entity-level trajectory; same plane, entity granularity. |

### R2 family

| Paper | Justification |
|---|---|
| CameraCtrl II | Dynamic-scene camera dataset; **same corrective direction as B7**, one representative suffices. Promote if B7 fails to run. |
| CamCo | Epipolar-constrained camera control; RE10K-class training data, frontier position predicted by B6/B8. |
| ViewCrafter | **Different task** — novel view synthesis on static scenes. No object dynamics to preserve, so the dynamics-retention metric is undefined. |

### R3 / RJ family

| Paper | Justification |
|---|---|
| Perception-as-Control | 3D-aware motion representation, closest to B11/B12. Promote if either fails. |
| FreeForm Motion Control | **Dataset paper**, not a method. Evaluate as a potential clip source for the high-parallax band alongside Kubric. |
| Physical Simulator In-the-Loop | Simulator-conditioned generation; belongs to Aim 2 (physical plausibility), not the decomposition benchmark. Cite in Aim 2 positioning. |

### Instrument components — not benchmark rows

ViPE, D4RT / Open-D4RT, SpatialTrackerV2, VGGT, Shape of Motion, SAM2.
These are components of M1 or of the evaluation harness. They appear in the methods chapter
and in the M1 validation section (Kubric metric-ATE), **never as rows in the generation
benchmark**. Stating this explicitly pre-empts the category confusion.

---

## 5. Audits attached to specific tested rows

Not extra runs — analyses of runs already scheduled.

| Row | Audit | Claim it tests |
|---|---|---|
| B2 MotionClone | Run one hyperparameter config across **both** camera and object clips | Their 15-camera / 25-object disjoint partitions with per-partition configs undermine the joint-competence claim |
| B6/B7/B8 | Camera accuracy vs. dynamics retention scatter | Camera control and object dynamics are traded, not jointly solved (RE10K suppression, documented by AC3D) |
| B4/B5 vs B1–B3 | Parallax degradation slope | R1 degrades like R0 → "explicit control" ≠ decomposed control |
| B9–B12 | Is the object trajectory world-frame or camera-frame? | Camera-frame joint control inherits the same non-identifiability |
| All | C1 comparison | Which reported metrics have any dynamic range at all |

---

## 6. Cell availability

Not every family is scoreable on every metric. Mark N/A; never impute.

| Family | Camera accuracy | Object trajectory error | Dynamics retention | Identity | Quality |
|---|---|---|---|---|---|
| R0 | ✓ | ✓ | ✓ | ✓* | ✓ |
| R1 | drift only | ✓ | ✓ | ✓* | ✓ |
| R2 | ✓ | N/A | ✓ | ✓ | ✓ |
| R3 / RJ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Controls C1–C4 | ✓ | ✓ | ✓ | ✓ | ✓ |

`drift only` = camera behaviour is emergent, not controlled; report as drift, labelled, never as a control-accuracy score.
`✓*` = only defined for methods accepting a reference image; T2V-only methods either receive it as the I2V first frame (Tier A) or are marked N/A. State which, per row.

**The N/A pattern is itself a result.** No existing method fills every cell. Render it as a
figure in the gap statement rather than asserting the gap in prose.

---

## 7. Open decisions

1. **Let Your Image Move with Your Motion** — promote to tested? Closest task match; highest examiner-question risk.
2. **OrthoMotion / ActCam release status** — check before finalising; both are conditional rows.
3. **Tier A pinned frame count** — blocked on the DiTFlow VRAM spike. If 81f does not fit, all Tier A rows including A3 re-run at the reduced config, and the deviation from Wan2.1's native envelope is stated.
4. **High-parallax clip source** — Kubric render sweep vs. FreeForm Motion Control vs. MiraData subset. Blocked on the parallax screening pass.
5. **Seeds per clip** — 3 assumed for budgeting. Confirm against total GPU hours once per-method runtime is measured.
