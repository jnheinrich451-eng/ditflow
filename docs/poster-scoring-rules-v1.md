# Poster scoring rules — v1 (the grader's rules, in writing)

Written 2026-10-15. Applies to the poster's "What we found" table and chart. Anyone recomputing
the table from `poster_pairs.csv` applies exactly these rules; the result must equal §6.

Source of truth: Drive `MyDrive/ditflow_extract/results/poster_pairs.csv` (one row per scored
video), produced by `poster_score_colab.ipynb` with `benchmark/score_pair.py`.

---

## 1. Pairs (inclusion is decided by rule, before scoring)

| Pair | Source frames (30 fps) | In the table? | Why |
|---|---|---|---|
| 000000000265.2.003 / vehicle | 47–67 | **yes** | passes the method's Stage I camera gate |
| 000000000265.2.005 / vehicle | 0–20 | **yes** | passes |
| 000000000265.2.005 / human | 0–20 | **yes** | passes |
| 000000000370.1.008 / human | 0–20 | **yes** | passes. **Not excluded.** It is the method's worst pair; dropping it would be a result-based exclusion |
| 000000000404.1.004 / vehicle | 47–67 | **yes** | passes |
| 000000000035.1.003 / vehicle | 4–24 | **no** | reference camera fails the Stage I camera gate (coverage 0.62 → 0.02, render shears from frame 20), so no guided (A3) run exists. Recorded as `camera_gate_pass: false` in `pairs/refs/…ref21.json`. Caption: "a sixth pair was excluded because the camera could not be recovered from its reference clip" |

Matching: generated frame *i* ↔ reference frame *i* ↔ source frame `start + i`, *i* = 0–20.

## 2. Rows (`label` column in the CSV)

| `label` | Table name | Model | Run |
|---|---|---|---|
| `no_guidance` | No guidance (CogVideoX) | CogVideoX-5B, guidance and injection off | DiTFlow repo, `--no_guidance --no_injection` |
| `ditflow` | DiTFlow (baseline) | CogVideoX-5B | DiTFlow headline config (flow loss, latent optimisation) |
| `ours_noguide` | No guidance (Wan 2.1 I2V) | Wan2.1 I2V, same anchor, Stage III off | method arm A0 |
| `ours` | Our method | Wan2.1 I2V | method arm A3, variant per pair (see §3) |
| `smm` | SMM (re-impl., CogVideoX) | CogVideoX-5B | DiTFlow repo's re-implementation of SMM's loss, `--loss_type smm`; same loop, reference and prompt as `ditflow` |
| `moft` | MOFT (re-impl., CogVideoX) | CogVideoX-5B | DiTFlow repo's re-implementation of MOFT's loss, `--loss_type moft` |

`smm` and `moft` are comparable to `ditflow` and to CogVideoX's no-guidance row, never to the published SMM (ZeroScope) or MOFT (AnimateDiff) numbers. Added 2026-10-17.

Other labels in the CSV (e.g. `vae_roundtrip`) are not table rows.

## 3. Seed rule: one run per model per pair

- **CogVideoX rows (`ditflow`, `no_guidance`, `smm`, `moft`): seed 0 only.** Seed-1 DiTFlow videos exist for some
  pairs (generated before the switch to one seed); they are ignored. Rule:
  drop rows where `label in {ditflow, no_guidance, smm, moft, ditflow_rho}` and `gen` does not end in `seed0`.
  All CogVideoX outputs are trimmed 24 -> 21 frames (keep 3..23) before extraction.
- **Method rows (`ours`, `ours_noguide`): the single run supplied per pair**, whatever its seed:

  | Pair | `ours` (A3) | `ours_noguide` (A0) | A0/A3 seed match |
  |---|---|---|---|
  | 265.2.003 vehicle | seed 1, `obj_ow0.75_sm_sky` | seed 1 | yes |
  | 265.2.005 vehicle | seed 1, `coff` | seed 0 | **no** |
  | 265.2.005 human | seed 1, `sm_sky` | seed 0 | **no** |
  | 370.1.008 human | seed 0, plain | seed 1 | **no** |
  | 404.1.004 vehicle | seed 1, `obj_ow0.75_sm_sky` | seed 1 | yes |

  Disclose: "Guidance-off and guided runs share seeds on 2 of 5 pairs." The A3 variant differs per
  pair; disclose as "per-pair settings".
- Seed numbers are not matched across models (different models; a shared seed number means nothing).

## 4. Aggregation

- **Pairs are fixed by §1, never by results:** the table always uses the 5 gated pairs.
  A row appears only if it has a scored video on **all 5**; a row missing anywhere is left out of
  the table and listed as such (revised 2026-10-17 from "drop the pair", which let a missing
  baseline video shrink everyone's sample).
- **Cell value = median over those pairs** (one run per pair, so no seed averaging remains).
  Mean is reported in the thesis, not on the poster.
- **No confidence intervals** at N = 5. Per-pair values (`poster_chart.csv`) are the dots.
- **Per-pair wins** are reported alongside medians where they disagree (§6).

## 5. Flags (instrument check, disclosed, never dropped)

Stage A's A-G2b consistency check (threshold 4 px, marked *provisional — pending calibration*)
was run **report-only** for every video: its value and verdict are recorded, it does not block
scoring. All other gates stayed hard. Videos over 4 px (scored, flagged `*`):

| Video | A-G2b |
|---|---|
| 265.2.003 vehicle `ditflow` seed 0 | 11.76 px |
| 404.1.004 vehicle `ditflow` seed 0 | 17.62 px |
| 265.2.003 vehicle `no_guidance` seed 0 | 6.69 px |
| 265.2.005 vehicle `no_guidance` seed 0 | 10.24 px |
| 404.1.004 vehicle `no_guidance` seed 0 | 5.98 px |

All `ours` videos pass (1.27–3.84 px). `ours_noguide`: see the next scoring run. Caption line:
"N baseline videos failed the tracker's consistency check and were scored with a flag; all of ours
passed."

## 6. The numbers the page must reproduce (3-row table, 2026-10-15 run, before A0 was added)

Median over 5 pairs, one seed each; lower is better except Box-IoU.

| Method | CamRotErr (°) | RPE (°/frame) | RotErr (CameraCtrl) | TransErr (CameraCtrl) | Box-IoU | ObjMC (px) | ObjMC motion-only (px) | ObjErr3D |
|---|---|---|---|---|---|---|---|---|
| No guidance (CogVideoX) | 2.363 | 0.345 | 0.866 | 1.215 | 0.251 | 86.156 | 20.688 | 0.016 |
| DiTFlow (baseline) | 1.081 | 0.195 | 0.396 | 0.499 | 0.305 | 63.693 | 8.468 | 0.037 |
| Our method | 0.704 | 0.107 | 0.258 | 0.437 | 0.223 | 111.891 | 12.475 | 0.018 |

Per pair (`*` = flagged):

| Pair | CamRotErr NoG / DiTFlow / Ours | ObjMC motion-only NoG / DiTFlow / Ours | ObjErr3D NoG / DiTFlow / Ours |
|---|---|---|---|
| 265.2.003 vehicle | 3.171* / 1.081* / **0.704** | 53.019* / **2.564*** / 4.309 | 0.105* / **0.037*** / 0.074 |
| 265.2.005 human | 2.925 / 0.496 / **0.148** | 18.808 / 21.824 / **14.827** | 0.018 / **0.007** / 0.025 |
| 265.2.005 vehicle | 2.363* / 2.126 / **2.021** | 59.432* / 8.468 / **7.436** | **0.016*** / 0.044 / 0.018 |
| 370.1.008 human | 0.898 / **0.284** / 3.358 | 6.200 / **5.983** / 75.105 | 0.010 / **0.007** / 0.012 |
| 404.1.004 vehicle | 1.573* / 2.125* / **0.368** | 20.688* / 14.757* / **12.475** | **0.007*** / 0.067* / 0.015 |

Wins, ours vs DiTFlow: CamRotErr 4/5, RPE 4/5, ObjMC motion-only 3/5, ObjErr3D 2/5.
On the 3 pairs where neither video is flagged: CamRotErr 2/3 (265.2.005 vehicle within 0.11°).

When the A0 row is scored, §6 is extended, not edited: the 3-row table above stays as the dated
record of what the poster first showed.

## 7. Recompute (Python, from the CSV alone)

```python
import csv, statistics as st
ROWS = ['no_guidance', 'ditflow', 'ours']             # add 'ours_noguide' once scored
GATED = {'000000000265.2.003__vehicle', '000000000265.2.005__vehicle', '000000000265.2.005__human',
         '000000000370.1.008__human', '000000000404.1.004__vehicle'}
rows = [r for r in csv.DictReader(open('poster_pairs.csv'))
        if r['ref'].replace('__ref21', '') in GATED and r['label'] in ROWS
        and not (r['label'] in ('ditflow', 'no_guidance') and not r['gen'].endswith('seed0'))]
val = {(r['ref'].replace('__ref21', ''), r['label']): r for r in rows}
common = [p for p in sorted(GATED) if all((p, x) in val for x in ROWS)]
for x in ROWS:
    print(x, {m: round(st.median(float(val[(p, x)][m]) for p in common), 3)
              for m in ('CamRotErr_deg', 'RPE_rot_deg', 'BoxIoU', 'ObjMC_px', 'ObjMC_motion_px', 'ObjErr3D')})
```
