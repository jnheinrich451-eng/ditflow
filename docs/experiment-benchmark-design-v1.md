# Experiment Chapter — Design Plan v1
### Stage: unified benchmark of existing methods (own method excluded)

**One-line statement of what this stage delivers.**
A working evaluation apparatus, plus numbers from existing methods re-run under a single
frozen protocol, on axes that include camera–object separation. The apparatus is the
deliverable. The method drops in later without changing anything.

**Falsification sentence for the whole stage.** If published numbers reproduce cleanly under
a unified protocol, and if standard metrics respond to camera-induced motion the way they
respond to object motion, then the evaluation gap this thesis claims does not exist.

---

## 0. Decisions to freeze before anything runs

Fill these in, then do not change them. If a number later looks wrong, that is a finding,
not a reason to edit this table.

| # | Decision | Value | Rationale |
|---|---|---|---|
| D1 | Evaluation window `T_eval` | 2.5 s | binding constraint = shortest method in Tier 1 |
| D2 | Evaluation fps `fps_eval` | 8 | common divisor of 16 (Wan) and 8 (CogVideoX) |
| D3 | Evaluation frame count `n` | 20 | = D1 × D2 |
| D4 | Resolution for all metrics | 480×832 (or 512×512) | must match lowest common capability |
| D5 | Seeds per (clip, method) | 3 | report mean ± std |
| D6 | Clip count, real split | ~30 | stratified by parallax; see §2 |
| D7 | Clip count, Kubric split | ~20 | ground truth available |
| D8 | Pose estimator for camera metrics | ViPE (pinned hash) | one estimator for all methods |
| D9 | Tracker for motion metrics | pin one, hash it | one tracker for all methods |
| D10 | Translation error convention | unsquared L2 (CamCo) | note the CameraCtrl discrepancy in a footnote |
| D11 | Alignment | Umeyama sim(3), full trajectory | report fitted scale `s` |
| D12 | Primary reporting form | per-frame mean | sum in parentheses for comparability with literature |

**Compute budget check.** Do this arithmetic before Stage 5:

```
GPU-hours ≈ (clips × methods_tier1 × seeds × minutes_per_video) / 60
```

Example: 30 clips × 5 methods × 3 seeds × 8 min = 60 GPU-hours per full sweep, plus
re-runs. On one A100 40GB that is ~3 days of pure generation. If over budget, cut clips
(D6), never seeds (D5).

---

## 1. Method selection

### 1.1 The selection principle

Do not select for count. Select for **span across the axis the thesis is about**.
Seven papers that all do 2D motion transfer produce a longer table with no more
information. The set must contain methods that can and cannot separate camera from object
motion, so the table has contrast.

### 1.2 Cluster grid

| Cluster | What it accepts | Candidates (all in project folder) |
|---|---|---|
| **A** 2D transfer, training-free | ref video | MOFT, MotionClone, SMM, DiTFlow, Motion Consistency Loss |
| **B** 2D transfer, tuning-based | ref video | MotionDirector, Motion Inversion, Follow-Your-Motion |
| **C** Camera-conditioned | explicit trajectory | CameraCtrl, CameraCtrl II, CamCo, AC3D, ViewCrafter, Uni3C |
| **D** Joint camera + object | trajectory + object signal | MotionCtrl, SymphoMotion, OrthoMotion, ActCam, FMC, VidCRAFT3, Perception-as-Control |
| **E** Trajectory-conditioned object | 2D/3D drag or box | DragNUWA, Tora, TrackGo, LeviTor, MagicMotion, Boximator, FreeTraj, ATI, Motion Prompting |

Current gap: the existing 7-paper set is entirely A + B. **Clusters C and D are the ones
that give the chapter a camera axis, and cluster D contains the actual competition.**

### 1.3 Tiering rule

- **Tier 1 (re-run, full numbers).** Public code + public weights + portable to the chosen
  backbone without changing what the method claims. Target: 4–6 methods. Must include at
  least one from cluster A/B and at least one from C/D, or the table has no contrast.
- **Tier 2 (quoted numbers, marked †, separate table).** Published, not portable. Protocol
  disclosed in a dedicated column. Target: 6–10 methods.
- **Tier 3 (capability matrix only).** No code, no weights, or an input contract that makes
  the comparison meaningless. Everything in cluster E lands here.

### 1.4 Verification checklist, per candidate

Record all of these before assigning a tier. A "no" in either of the last two columns
demotes to Tier 2 or 3.

| Field | Notes |
|---|---|
| Repo URL | |
| Weights released (not just inference code) | |
| Released checkpoint == paper checkpoint? | Y / N / unknown — the Open-D4RT lesson |
| License permits evaluation + publication | |
| Native backbone | |
| Native frame count / fps / resolution | |
| Can produce ≥ `T_eval` seconds | if no → Tier 2 |
| Accepts explicit camera trajectory | determines camera-axis eligibility |
| Accepts reference image identity | determines identity-axis eligibility |

---

## 2. Protocol P

Written once, cited by every table caption as "Protocol P."

### 2.1 Task
Inputs: reference video `V_ref`, reference image `I_ref`, text prompt `c`, and (where the
method supports it) target camera trajectory `Π_tgt`. Output: generated video.
Methods that do not accept one of these inputs receive `–` in the corresponding table
cells, with a footnote stating why, not a poor score.

### 2.2 Data
- **Real split.** ~30 clips, stratified into three parallax bands (low / medium / high
  camera translation relative to scene depth) and three motion types (rigid single object,
  articulated, multi-object). Record source and license per clip.
- **Synthetic split.** Kubric MOVi-F, ~20 scenes, full ground truth for camera pose, object
  pose, depth, and segmentation. This split is what makes Stages 4 and 6 possible.

### 2.3 Temporal alignment — the single most important implementation rule

Every method generates at its **native** length. Never force a backbone off its trained
frame count; the resulting quality drop measures your configuration choice, not the method.
Then every video — generated, reference, and ground truth alike — passes through one
function before any metric touches it.

```python
import numpy as np

def eval_indices(n_src, fps_src, t_eval=2.5, fps_eval=8):
    """Nearest-neighbour indices covering [0, t_eval) at fps_eval.
    Applied identically to generated, reference, and GT video."""
    n_eval = int(round(t_eval * fps_eval))
    times  = np.arange(n_eval) / fps_eval
    idx    = np.rint(times * fps_src).astype(int)
    if idx.max() >= n_src:
        raise ValueError(f"{t_eval}s exceeds source ({n_src} frames @ {fps_src} fps)")
    return idx

# Wan2.1     : 81 @ 16 fps -> 20 frames, indices 0,2,4,...,38
# CogVideoX  : 49 @  8 fps -> 20 frames, indices 0,1,2,...,19
# AnimateDiff: 16 @  8 fps -> FAILS -> Tier 2
```

Rules:
1. **Nearest-neighbour index selection only.** No interpolation or frame blending — blending
   creates ghosting that corrupts LPIPS, FVD, and any tracker run on the output.
2. **`fps_eval` must divide every native fps.** Non-integer decimation produces uneven time
   steps, which appear directly as jitter in velocity estimates.
3. **All physical quantities use elapsed time, not frame index.** Velocity is `ΔX / Δt` with
   `Δt = 1/fps_eval`. Per-frame displacement is not a velocity.
4. **One call site.** The function lives in the harness, not in per-method scripts.

**What this buys and does not buy.** It buys within-backbone comparability and
reference-vs-generated comparability. It does **not** buy cross-backbone comparability — a
Wan number and a CogVideoX number still differ by backbone capacity. Keep those in separate
tables. Claiming otherwise is protocol merging in a new costume.

### 2.4 Environment and parameters
Two tables in §5.1.4 of the chapter:
- **Environment.** GPU, CUDA/driver, PyTorch, diffusers, OS, Python; checkpoint names with
  SHA/revision hashes; dtypes (bf16 transformer / fp32 VAE); a note that bf16 attention
  kernels are not bitwise deterministic.
- **Parameters.** One row per hyperparameter, one column per method, plus a `source` column
  recording *paper default / repo default / tuned by me (on which split)*. Key rows:
  resolution, frames, fps, denoising steps, scheduler + shift, CFG scale, guidance window
  **expressed in SNR not step index**, optimisation steps, LR schedule, injection block,
  temperature τ, LoRA rank, seeds.

---

## 3. Metrics — three layers

### Layer 1: instrument metrics (on intermediate artifacts, Kubric only)
Not video metrics. They exist to answer the circularity objection: *"you used the same
estimator to build the guidance signal and to score the output."*
- Camera pose: ATE, RPE against Kubric GT
- Metric depth: AbsRel, δ₁
- 3D tracks: 3D-EPE, track survival rate
- Masks: J&F
- Reported in their own subsection. **Never in the comparison table** — baselines have no
  equivalent module, so the numbers are not comparable to anything.

### Layer 2: system metrics (on generated video)
Four blocks:
- **Block 0 — capability matrix.** No numbers. Rows = methods, columns = accepts ref video /
  accepts trajectory / accepts ref image / separates by construction / training-free /
  backbone. Shows the comparison space is smaller than it looks and pre-empts "why not X?"
- **Block 1 — motion transfer.** MF, Text Sim, Temporal Consistency, FVD (report N), Time(s)
  with GPU stated.
- **Block 2 — camera.** CamTransErr, CamRotErr, fitted scale `s`, per-frame mean primary.
- **Block 3 — identity.** DINO-I / CLIP-I on subject region, LPIPS on static background.
  *Absent from FYM's table; required here because your setting conditions on a reference
  image, and shape-support leakage is your main known distortion mode.*

### Layer 3: bounds
Every metric gets a floor and a ceiling, or its numbers are uninterpretable.
- **Lower bound row.** Backbone + prompt + `I_ref` only, no motion guidance.
- **Upper bound / oracle row.** Reconstruct the reference video itself.
- **Noise floor.** Run the pose estimator and tracker on GT videos and report the error they
  produce against known GT. Any improvement smaller than this floor is not an improvement.

### Metric caveats to state in the text
- CLIPSIM is near-saturated; across seven methods FYM's Text Sim spans 0.279–0.380. Do not
  let a 0.005 gap carry an argument.
- FVD is strongly biased below ~2000 samples. Report N or drop it.
- Camera errors as defined are **sums**, hence length-dependent. Fixing `n = 20` (D3) makes
  them internally comparable; published sums at n = 14 or 32 are not.
- CameraCtrl's TransErr (squared norm) and CamCo's (unsquared) disagree. Pick one (D10),
  footnote the discrepancy.

---

## 4. The five studies

Each gets: question → design → falsification sentence → result → interpretation.

| # | Study | Needs GPU? | Output |
|---|---|---|---|
| S1 | **Protocol audit** | No | Table: per paper — dataset, clips, prompts, frames, resolution, backbone, metrics, seeds, re-run vs quoted. Expected finding: no two papers share a protocol, and several quote across protocol boundaries. Justifies the whole re-benchmark. |
| S2 | **Instrument validation** | Light | Layer-1 metrics on Kubric. Establishes noise floors. Kills the circularity objection. |
| S3 | **Reproduction** | Heavy | Tier-1 methods under Protocol P; report Δ from published numbers. Highest value per GPU-hour. Also de-risks the DiTFlow → Wan2.1 port before anything depends on it. |
| S4 | **Metric sensitivity / null study** | Medium | *The study that makes the thesis.* On Kubric, construct pairs where world-frame object motion is **identical** and only camera trajectory differs. Show MF, Text Sim, Temporal Consistency barely move. Falsification: if they move as much as they do for genuine object-motion changes, the evaluation gap is not real. |
| S5 | **Component analysis of others' methods** | Medium | Toggle *their* published components on a new backbone — e.g. DiTFlow with/without KV injection on Wan2.1. Their paper justifies it on CogVideoX; does it hold? This is the "analysis on their methods" your senior meant. |

**Note on S4.** This is where the evaluation gap stops being an assertion and becomes a
measurement. It is cheap, it needs none of your own method, and it converts the new
disentanglement metric from a preference into a necessity.

**Note on ablations.** At this stage "ablation" means two things, neither of which is your
method: (i) sensitivity of the metrics themselves — does MF change when the window is 20 vs
32, when the tracker changes, across seeds? (ii) component ablation of baselines, per S5.
Title the section *Sensitivity and component analysis*. Method ablations get added later.

---

## 5. Chapter skeleton

Construction order is the reverse of reading order: build metrics → protocol → studies,
but present setup → metrics → studies.

```
5.1 Experimental setup
    5.1.1 Task and Protocol P          <- the contract; every caption cites it
    5.1.2 Benchmark data               <- real + Kubric splits, licensing
    5.1.3 Methods under test           <- cluster grid + tiering rule
    5.1.4 Implementation and environment
    5.1.5 Temporal and spatial alignment  <- the 81-vs-49 problem, solved once
5.2 Metrics
    5.2.1 Instrument metrics (Layer 1)
    5.2.2 System metrics (Layer 2)
    5.2.3 Bounds and noise floors (Layer 3)
5.3 Protocol audit (S1)
5.4 Instrument validation (S2)
5.5 Reproduction under a unified protocol (S3)
5.6 Metric sensitivity: are standard metrics blind to camera? (S4)
5.7 Component analysis of existing methods (S5)
5.8 Capability matrix and main comparison tables
5.9 Sensitivity and component analysis  <- "ablation" at this stage
5.10 Failure analysis and error budget
5.11 Threats to validity
```

**Two composition habits.**
1. Write every table caption *before* you have the numbers. If you cannot write a caption
   saying what the reader should conclude, the table has no job and should be cut.
2. Give every study a falsification sentence. That one sentence per study is the difference
   between an experiment chapter and a results dump.

**Caption convention, stated once in §5.1.1:**
> A number enters this table only if (a) it was produced under Protocol P in this work, or
> (b) it is quoted from a single source paper, marked †, with that paper's protocol given in
> Appendix X. Rows of type (a) and (b) are never compared directly.

---

## 6. Build order and deliverables

| Stage | Work | Depends on | Deliverable | GPU |
|---|---|---|---|---|
| **B0** | Freeze §0 decision table; fill §1.4 checklist per candidate; assign tiers | — | `protocol.lock.yaml` | none |
| **B1** | Harness: `eval_indices`, camera metrics with Umeyama + noise-floor harness, MF, identity metrics; unit tests on synthetic SE(3) trajectories | B0 | `benchmark/` package, tests green | none |
| **B2** | Protocol audit (S1) — paper reading only | — (**parallel with B1**) | audit table | none |
| **B3** | Instrument validation (S2) on Kubric; establish all floors and ceilings | B1 | Layer-1 table, `noise_floor.json` | light |
| **B4** | Tier-1 reproduction runs (S3) | B1, B3 | Block 1–3 tables | heavy |
| **B5** | Metric sensitivity (S4) + component analysis (S5) | B3, B4 | the two studies that carry the argument | medium |
| **B6** | Assemble tables, write 5.3–5.11 | all | chapter draft | none |

**Critical path note.** B2 needs no compute and can run entirely in parallel with B1. Start
both this week. B4 is the long pole; its risk is the DiTFlow → Wan2.1 port, so treat the
port as a B1 sub-task with its own test (does KV injection actually fire? is the guidance
window matched by SNR rather than step index?) rather than discovering the problem inside
a 60-GPU-hour sweep.

**Governance.** Thresholds and protocol values live in `protocol.lock.yaml`, frozen after
B0. Add a hard stop in any SKILL file handed to Claude Code: *no parameter in
`protocol.lock.yaml` may be modified to make a gate pass.* This is the documented
threshold-tuning failure mode and the lock file is the structural fix.
