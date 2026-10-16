# Poster-first experiment plan — v1

> **Superseded 2026-10-17** for the thesis by `mixture-motion-transfer/docs/plan_experiment_v1.md` (the reviewer's D1–D6). Kept as the record of the poster plan; scoring rules live in `poster-scoring-rules-v1.md` (amended).

**What this is.** The experiment plan for the poster (§3 "What we found", two panels), sized to
what can be produced before the deadline, and written so that every poster number reappears in
the thesis chapter unchanged. It replans across both repos: `ditflow` (benchmark, instrument,
scoring) and `mixture-motion-transfer` (the method: Recover → Retarget → Guide).

Written 2026-10-14 from a survey of both repos and the literature folders. Facts about the method
repo are from `mixture-motion-transfer/docs/report_m3_s2_first.md` (last dated 2026-10-13) and
`measurements/m3_s4/a0_motion.md`; they are quoted, not re-measured.

---

## 1. Where everything stands (one table)

| Piece | State | Consequence for the poster |
|---|---|---|
| Instrument (extract-v1) | ported, pinned, bit-exact transfer check, deterministic; 58/79 pool clips extracted; 19 refusals from one uncalibrated gate (A-G2b) | usable now for scoring generated videos; the refusal question is a thesis matter, not a poster one |
| Camera metrics | `benchmark/camera_metrics.py`, 13 tests; scale-only + sim(3); RealCam agreement median 2.3° on 23 clips | the poster's camera number exists |
| Object metric | not built (only "centroid measured vs expected" columns in `a0_motion.md`) | **must build** (`benchmark/score_rows.py`, §4) |
| DiTFlow baseline | CogVideoX-5B works; DAVIS block done (41 clips); **not run on MiraData** | run DiTFlow on the poster clips (§4, step 5); 4–6 min/clip |
| Wan DiTFlow port | failed; stopped | no same-backbone "entangled" row for the poster |
| Method, Stage I/II | closed; **5 clips** have frozen Stage II handoffs (035.1.003, 265.2.003, 265.2.005 ×2, 370.1.008, 404.1.004); 035's camera fails the gate | **N ≤ 4 clips** (5 handoffs) without new Stage II work |
| Method, Stage III | generates videos on Wan2.1 I2V (33 f, 832×480); mechanism switched to render-target A3 on 10-11; bulk of videos on **265.2.005** only | the poster rests mostly on one clip unless A3 is run on the other 3 |
| Sampler fault | text-padding bug degraded every A0/A3 run; fixed 10-13; **parity retest not run; all baselines flagged "re-base"** | **nothing from the method is quotable until the retest passes** — this is step 1 |
| Headline so far | background follows the geometric target 0.92–0.93 (guided) vs 0.06–0.11 (unguided); source reads 0.75; *fidelity to the source camera withheld* pending the bridge check | the poster may claim **adherence**; "loyal to the reference" needs the measurement in §3 |
| Known artefact | A3 freezes non-rigid texture (water residual 0.10 vs source 0.49); `_nodyn` variant proposed under "constrain geometry, free dynamics" | pre-register the variant before scoring (§4, step 2) |
| Kubric / Q1 calibration | renderer works, 48-render batch not started; pipeline changes (a)/(b) pending | **paused** for the poster unless told otherwise |

---

## 2. The poster claim, sized to the evidence

Three candidate sentences, strongest first. Which one the poster carries is decided by the
measurements in §4, **not chosen now** — write all three, delete two.

| Level | Finding 1 (frames) | Finding 2 (chart) | Needs |
|---|---|---|---|
| **A. Fidelity** | "The generated camera and subject follow the *reference's* camera and subject paths." | CamRotErr and object error of ours vs guidance-off vs DiTFlow, each against the **reference** | §3 measurement passes on ≥ 3 of 4 clips |
| **B. Adherence** | "The generated camera and subject follow the *geometric target* the method constructed." | same chart, against the **constructed target** | retest passes; §3 fails |
| **C. Control** | "Switching the guidance on changes the generated camera and subject paths in the intended direction; nothing else changes." | guided vs guidance-off only | retest passes |

A is what you believe you saw. The method repo's report explicitly does not yet claim it — it
claims B and withholds A pending a bridge check. §3 is the direct way to settle it.

**What stays off the poster:** MF (direction-only, non-predictive: ρ = 0.23 vs humans on FYM's
own table) except as a one-line "why not MF"; any ranking; any Kubric number; any cross-backbone
*level* claim stated as such (DiTFlow is on CogVideoX-5B, ours on Wan I2V — the caption says so).

**The quality question.** Report it, don't hide it, don't lead with it. One number in the chart
caption — subject CLIP-I against the anchor (identity kept) and a frame of the worst case in
Finding 1's "where to look" line. The poster's claim is about *motion control of a frozen model*;
texture quality is a stated limitation, and the literature's own comparisons (SymphoMotion vs
ViewCrafter FID 303 / FVD 1690) show the field is not clean on it either.

---

## 3. The "loyal to the reference" measurement — the trick that makes it visible

The reader cannot judge motion fidelity from texture. The thing that *can* be judged is the
motion itself, drawn on the frames and measured with the same instrument. Both come from one
run of `extract.py` on the generated video:

```
reference clip ──extract──▶ Ĉ_ref (camera path), X̂_ref (subject root path, world frame)
generated video ─extract──▶ Ĉ_gen,                X̂_gen
                              │                      │
             CamRotErr(Ĉ_gen, Ĉ_ref)  scale-only   ObjErr(X̂_gen, X̂_ref) after camera compensation
```

- **Camera:** `camera_errors(E_gen, E_ref, align="scale_only")` → RotErr in degrees (scale-free,
  the reliable number; RealCam agreement showed translation scale is unstable), RPE per frame.
- **Object:** subject root trajectory from `derived.json` `instances[salient].world_traj`,
  camera-compensated, aligned to the reference's with the camera's scale; endpoint error per
  frame, normalised by the reference's median background depth so it is unitless. Also the
  **2D version** (reprojected into the reference camera, px) — this is the literature's ObjMC and
  what the poster shows, because pixels are readable.
- **Floor and ceiling on the same chart:** guidance-off (A0, re-based) is the floor; the
  **reference re-extracted from its own VAE round-trip** (C3) is the ceiling — what a perfect
  transfer would score given instrument noise. Without both, a bar means nothing.

**Finding 1 overlay (the visual trick).** Three frames (start / middle / end) per row; on each,
draw the extracted subject root path up to that frame (coloured arrow chain) and a small camera
inset showing Ĉ up to that frame. Three rows, matching the poster template: Reference,
Existing method (DiTFlow), Ours — *same overlay code for all*. Guidance-off appears in the
chart only, not as a frame row. The viewer compares paths, not textures. "Where to look": "the car's
path against the road edge; the camera inset should trace the same arc as the reference's."
This is exactly the figure the method repo already prototyped as projections of saved geometry
(`measurements/poster_retarget_*`), but drawn from *measured* output, which is what makes it
evidence.

**Honesty rule for the overlay:** the overlay is drawn from `extract.py` on the generated video,
never from the method's own internal target. If the two differ, that difference *is* the result.

---

## 4. Build and run order (poster critical path)

| # | Step | Repo | GPU | Blocks |
|---|---|---|---|---|
| 1 | **Sampler parity retest** after the text-padding fix (registered: step-0 diff 0; A0 matches stock: man walks, car drives, horse intact). Re-base A0. | method | A100, ~1 h | everything |
| 2 | **Pre-register the A3 variant** for the poster (proposal: `_nodyn`, per "constrain geometry, free dynamics") and the seeds (0, 1). Written down before any scoring. | method | — | 3, 6 |
| 3 | **Generate** A3-variant + A0 on the 4 gated clips × 2 seeds = 16 videos (265.2.005 has most already; 265.2.003, 370, 404 need A3). | method | A100, ~8 h | 6 |
| 4 | **Canonicalise generated videos**: 33 f @ 832×480 → 24 f @ 480×720. Needs a temporal rule: nearest-neighbour `eval_indices` (design doc D1–D3), **never interpolation**. `canonicalize.py --eval-indices` to add. Same rule applied to the 24-frame references (identity). | ditflow | — | 6 |
| 5 | **DiTFlow rows** on the same 4 clips: `benchmark/sweep.py` with `grid_5b.yaml` on the `miradata.csv` Subject rows for those clip ids, seeds 0, 1 → 8 videos. Native CogVideoX-5B, no port. | ditflow | A100, ~1 h | 6 |
| 6 | **Extract** all generated videos + C3 round-trips with `extract.py` (Colab L4; `q1_extract_colab.ipynb` pattern, 32 videos ≈ 1.5 h). Unique file names per cell. | ditflow | L4 | 7 |
| 7 | **Score**: `benchmark/score_rows.py` (to build) — intended-target table per row, CamRotErr, ObjErr 2D/3D, CLIP-I; paired over clips × seeds; bootstrap CI; writes `results/poster_rows.csv`. | ditflow | — | 8 |
| 8 | **Figures**: `benchmark/poster_figs.py` — overlay frames (Finding 1) and the two-panel chart (Finding 2). | ditflow | — | poster |

Steps 4, 5 (prep), 7 and 8 can be built now, before step 1 lands. Step 1 is the method repo's and
nothing downstream is quotable before it.

**Budget check.** 16 method videos + 8 DiTFlow + 8 round-trips = 32 extractions. One A100 day
for generation, one L4 session for extraction. If the deadline forces a cut: drop seed 1 before
dropping a clip, and drop DiTFlow before dropping guidance-off — the floor is the comparison
that carries claim C, and C is the fallback.

---

## 5. Finding 2 — the chart, specified before the numbers

**One chart, three methods, no table** (revised 2026-10-14 to the poster template and rubric:
"one score, 2–3 methods"; results 30% and layout 20% both reward a chart a non-expert reads in
seconds over a table). Two small groups sharing one axis style, three bars each:

```
   How far the generated motion is from the reference's   (lower is better)

   Camera direction error (°)            Subject path error (px)
   ┌──────────────────────────┐          ┌──────────────────────────┐
   │ No guidance  ▇▇▇▇▇▇▇▇▇▇ 9.1│          │ No guidance  ▇▇▇▇▇▇▇▇▇▇ 41│
   │ DiTFlow†     ▇▇▇▇▇▇  5.4   │          │ DiTFlow†     ▇▇▇▇▇▇▇ 28   │
   │ Ours         ▇▇▇  2.6      │          │ Ours         ▇▇▇ 12       │
   │ - - - - - - - instrument limit        │ - - - - - - - instrument limit
   └──────────────────────────┘          └──────────────────────────┘
   N = 4 clips × 2 seeds; dots = clips; bars = median.  † runs on a different video model
   (numbers above are layout placeholders, not results)
```

- **Value printed on each bar** — the only numbers on the poster. Enough for a marker who
  wants a number; no table to parse.
- **Per-clip dots** over the bars: N is small, and dots show that honestly in one glance.
- **The ceiling is a dashed line labelled "instrument limit"**, not a bar: it tells the reader
  how low a bar *can* go without adding a fourth method.
- **Lower is better**, stated in the title. The template's "higher is better" is a placeholder;
  do not invert an error into a made-up "agreement %" to fit it.
- Caption, plain words: what each error measures in one sentence, N, the dagger, CLIP-I as the
  one quality number. The full table (CI, RPE, 3D error, per clip) goes behind the QR code and
  into the thesis.

**Pre-written outcomes.** Ours below floor on both panels with CI clear → claim A or B (per §3).
Ours below floor on camera only → the camera claim stands; the object claim becomes C. Ours not
below floor → the poster leads with the Recover/Retarget geometry (which is real and measured)
and reports Guide as in progress. All three are honest posters; only the sentence changes.

---

## 6. How this folds back into the thesis

The poster rows **are** 4.10 Level 1 at N = 4: same clips (a subset of the frozen MiraData pool),
same instrument, same intended-target rule, same statistics. Nothing is thrown away; the chapter
adds clips, the parallax bands (once Kubric settles the gate), the Entangled rung if the Wan port
is ever revived, and DiTFlow's DAVIS block. The "why not MF" line expands into 4.5.

What the poster does *not* license and the chapter must not inherit: a cross-backbone level
claim (DiTFlow vs ours is slope-only in the thesis; on the poster it is two bars with a dagger),
and anything on 035 (camera gate failed; stays a qualitative demo at most).

---

## 7. Decisions needed from you

1. **Deadline and format** — sets N and whether seed 1 survives.
2. **A3 variant for the poster** — `_nodyn` is my recommendation; must be written down before
   scoring (step 2).
3. **Pause Kubric/Q1 until after the poster** — my recommendation; say otherwise if you want the
   batch running in the background meanwhile.
4. **Who runs step 1** (the parity retest) and when — it is the method repo's registered test and
   the single blocker for every method number.
