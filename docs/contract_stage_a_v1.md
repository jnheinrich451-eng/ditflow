# Stage A output contract — schema **v1** (FROZEN 2026-08-03)

Input spec for every downstream consumer (Stage A.5 bake-off, Stage B motion
proposal). Consumers read **this contract**, never Stage A's implementation
(CLAUDE.md §3.5). Contract changes require a new schema version + human
sign-off — never an in-place edit.

Frozen after human approval of the review packages for the calibration set
(SKILL §11). Produced by `scripts/run_stage_a.py`; validated on load by
`src.preprocess.bundle.read_bundle`, which refuses malformed bundles and
bundles whose config hash differs from the current config.

---

## 1. Files per clip

```
<cache_dir>/<clip_id>/
    bundle.npz          arrays below
    provenance.json     model/checkpoint/GPU/conventions/resize factors/config hash
    qc.json             gate outcomes, thresholds used, sim3 scale curve
```

## 2. Arrays (`bundle.npz`)

| Field | Shape / dtype | Semantics |
|---|---|---|
| `K` | `[F,3,3] f32` | intrinsics, **process-resolution** (256×256) coordinates |
| `G` | `[F,4,4] f32` | `G_{0←t}`: camera-t → camera-0 (direction measured, probe P2) |
| `D` | `[F,H,W] f16` | **z-depth** (measured, probe P3), model scale, process resolution |
| `anchors` | `[N,3] f32` | `(u_i, v_i, s_i)`, process-resolution px; `s_i` int-valued spawn frame |
| `X_local` | `[N,F,3] f32` | per-target-camera coordinates (`t_cam = t`) |
| `X0` | `[N,F,3] f32` | shared camera-0 coordinates — **queried** (`t_cam = 0`), not transformed |
| `P` | `[N,F,2] f32` | image projections, process resolution |
| `V`, `C` | `[N,F] f32 ∈ [0,1]` | visibility, confidence (both `sigmoid` of raw heads) |
| `Q` | `[N,F] bool` | four-factor validity: `V ≥ τ_v`, `C ≥ τ_c`, cheirality `z > 0`, in-frame |

All arrays finite (hard assert in the writer). Fields Stage A does **not**
produce — clusters, `b_t`, masks, `t_c` — are **absent**, never placeholder-filled.

## 3. Conventions (measured, never assumed — CLAUDE.md §3.3)

| Property | Value | Evidence |
|---|---|---|
| Camera axes | OpenCV: x right, y **down**, z forward | probe P5 self-consistency |
| Extrinsic direction | `G_{0←t}`; camera centre `o_t = G[t][:3,3]`, `o_0 ≈ 0` (asserted) | P2: 0.149 vs 0.514 err both ways |
| Depth semantics | z-depth (image-plane-parallel) | P3: 1.1% vs 12.9% rel err vs ray-length |
| Index base | 0-based; model silently accepts out-of-range → adapter range-asserts | P4 |
| Intrinsics frame | process resolution 256×256; native→process resize factors in provenance | P5, SKILL §5 rule |
| Geometry source | single: OpenD4RT `OpenD4RT_48CLIP_9Mix_NoCropAUG` (A-G8) | provenance |

`K` and `G` are **derived**, not model outputs — OpenD4RT exposes no camera
heads. Per the repo's own `geometry_decoding` spec: `G` from rigid Umeyama
between `t_cam=t` and `t_cam=0` grid queries; `K` from a least-squares pinhole
fit (measured better than the repo's median-fx/fy estimator: 1.88 vs 2.14 px).

## 4. Quality gates (`qc.json`)

Thresholds live in `configs/calibration.lock.yaml` (read-only to Claude Code,
CLAUDE.md §4), frozen 2026-08-03 from the calibration table over the three
calibration clips (`worst-clip + 3×MAD`).

| Gate | Statistic | Frozen threshold |
|---|---|---|
| A-G2 | median relative 3D residual `‖X0 − G·X_local‖ / max(Z, Z_min)`, `Z_min` = 5% of clip median depth | `< 0.15` |
| A-G2 (2nd) | inlier fraction: share of valid entries within 5× the clip median | `≥ 0.814` |
| A-G2b | median `‖project(K, X_local(t=s_i)) − (u_i,v_i)‖` | `< 4.0` px |
| A-CH | fraction of `Q=1` entries with `z ≤ 0` | exactly 0 |
| A-COV | fraction of frames with ≥ 200 valid tracks | `> 0.9` |

A failed gate emits no bundle (`qc_failed.json` instead). The **pixel** A-G2
median is reported for visualization only — pixel error scales with `f/Z` and
is not clip-invariant.

## 5. Measured limitations — REQUIRED READING for consumers

Numbers below are measured on the calibration set; no assumed values appear
anywhere (CLAUDE.md §9.12).

1. **`X0` degrades with temporal distance.** Relative residual grows from
   ~0.004 at `|t − s_i| = 1` to ~0.15 at 40+ frames. Causes: monocular scale
   drift plus per-point incoherence. **`X_local`, `D`, `K` are not affected** —
   they come from same-frame queries and are the strongest fields in the bundle.
2. **Scale drift is recorded, not corrected.** `qc.json.sim3_scale_per_frame`
   holds the per-frame similarity scale between `X_local` and `X0` grid fits.
   Measured range: camera-following clip 1.00 → 1.08; static clip stays within
   ±0.006. Drift correction (`b_t`) is Stage B's job (SKILL §9). Stage A.5
   measures the drift rate independently against ViPE.
3. **`V` is not a calibrated occlusion probability.** This checkpoint's
   visibility head collapses cross-frame (≈1.0 at the source frame → ≈0.05 a
   few frames away) and does **not** correlate with geometric quality — higher
   `V` measured *worse* residuals on the following clip. `τ_v = 0.005` is
   therefore an outlier filter, not an occlusion gate. For occlusion, prefer a
   z-buffer test against `D` (as `inspect_bundle.py` does for visualization).
4. **`C` saturates.** Raw confidence ∈ [6.6, 10.1] → `sigmoid` ∈ [0.9986, 1.0].
   `τ_c = 0.5` is deliberately permissive; `C` is useful for *ranking*, not
   thresholding.
5. **Window stitching leaves residual per-point steps.** Clips longer than the
   48-frame model window are processed in quantized encode windows, each
   aligned to a reference window by overlap sim3 (recorded in
   `provenance.x0_window_stitch`). This removes the global frame shift — camera
   trajectories are smooth — but per-point disagreement at window boundaries
   remains visible in `x0_static_drift.png`. Overlap blending is the known
   remedy if `b_t` cannot absorb it.
6. **Statistic is duration-sensitive.** A-G2 is invariant to depth and focal
   length by construction, but not to clip length: 24-frame clips measured
   median 0.011, 81-frame clips 0.044. Compare clips of similar length.
7. **Mid-entry (`s_i > 0`) is uncalibrated.** The dataset contains no mid-clip
   object-entry content, so the calibration set covers static-camera and
   camera-following only (human decision 2026-08-02). A future mid-entry clip
   failing gates is new calibration data, not a pipeline bug.
8. **Object points are much worse than background.** Residual decomposition on
   the following clip: object proxy 0.552 vs background proxy 0.106 (5.2×).
   Movers are where the shared frame is least trustworthy — directly relevant
   to Stage B's dynamic scoring.

## Addendum (2026-08-07, human-signed — additive, schema still v1)

`qc.json` gains **`window_boundaries: [t₁, t₂, …]`** — the frames at which the
encode-window block changes (empty for clips within one 48-frame window; e.g.
`[35, 47, 59]` for 81-frame clips). These are the scale re-gauge points
measured in the A.5 σ̂_t curves (limitation #5); Stage B's drift model treats
them as discontinuities and must read them **from qc.json only**, never
re-derive the schedule. Source of truth: `d4rt_adapter.window_boundaries`
(pure function shared by the writer and `scripts/refresh_qc_boundaries.py`).
All other fields unchanged; existing bundles refreshed in place (writer-only,
no geometry recomputation, config hash unaffected).

## 6. Out of scope for Stage A

Not produced, by design: static/dynamic labels, object clusters, drift
correction `b_t`, segmentation masks, contact times `t_c`, retargeting, any
Phase-2 artifact. The static-track *proxy* used by A.5 is a heuristic ranking
by excursion — explicitly **not** a classifier.
