# Follow-Your-Motion on the MiraData pairs — design v1

Written 2026-10-18. Design only: nothing here changes a running notebook. Implementation is two new
files (§6), built after the current runs finish.

## 1. Goal

One FYM row and one FYM-floor row on **every gated row of `benchmark/pairs_manifest.csv`** (today: the
method's 5 pairs + the 19 census clips), scored by the same notebook, the same instrument and the same
rules as every other row. Same reference frames, same subject prompt, seed 0.

| Row label | Table name | Model |
|---|---|---|
| `fym` | Follow-Your-Motion (Wan 2.1 T2V 1.3B) | per-clip spatial + temporal LoRA, FYM release recipe |
| `fym_noguide` | No guidance (Wan 2.1 T2V 1.3B) | the same checkpoint, no LoRA — FYM's own floor |

Each method sits next to its own floor: CogVideoX rows vs CogVideoX no-guidance, FYM vs Wan T2V
no-LoRA, ours vs Wan I2V A0. Cross-backbone comparisons are reported, not claimed (plan D4).

## 2. What is reused unchanged

`fym_bench/run_davis.py` already implements and records the whole FYM procedure, and its DAVIS run is
the record of the recipe:

- `prepare_tuning`: data_process → Collect_attn_map → head classification (fails cleanly when a layer has
  heads of one type only: 3 of 21 DAVIS clips, all layer 13).
- `tune`: spatial LoRA then temporal LoRA, `RECIPE` = 3 epochs × 500 steps, LR 1e-4, rank 16, alpha 16,
  weight decay 0.1 / 0.5, tune seed 0. Measured cost: **median 3,041 s (51 min) per clip** on A100-40GB.
- `fym_bench/generate.py`: generation, ~50 s per clip.
- Resumability: a finished `tune/tune.json` is reused; `done.json` written last; `failed.json` with a
  reason code on failure.

## 3. What changes — the input side only

| | DAVIS runner | Pairs runner |
|---|---|---|
| Reference | first 21 JPEGs of the DAVIS cut, sha256 checked against davis50 | **the pair's `pairs/refs/<pair>__ref21.mp4`** — the exact frames every other method is scored against (source frames `start…start+20` of the chosen source file, centre-cropped 720×480). Its sha256 is recorded in `done.json` |
| Tuning text | the clip's `caption` row | the **reference's own caption**: `miradata.csv` caption row for census clips; RealCam-Vid `short_caption` for 265.2.003, 370.1.008, 404.1.004 (not in `miradata.csv`); 265.2.005 → `miradata.csv`. Stored as a new manifest column `caption` |
| Generation prompt | the clip's `subject` row | the pair's `prompt` — identical to every other row of that pair |
| Seed | 1 | **0** (plan D3); the DAVIS split keeps seed 1 |
| fps | cut's effective fps | `source_fps` from the ref json |
| Output | `runs/fym_wan/davis/…/results.mp4` | `<pair>__fym__seed0.mp4` and `<pair>__fym_noguide__seed0.mp4`, **converted** (§4), + `.json`, written into `pairs/inbox/` — the scoring notebook files and scores them with no change |

Why tune on the caption and generate with the pair's prompt: that is FYM's protocol (the LoRAs learn
the reference *as described by its own caption*; the new prompt then changes the content). Using the
pair's prompt for tuning would teach the LoRAs a description that does not match the reference.

## 4. Geometry and frames

- **Size:** keep the recorded recipe, 832×480 with FYM's whole-frame resize (as on DAVIS), so FYM runs the
  same way on both splits. The 720×480 reference is stretched to 832×480 in x; the output is mapped back
  with `canonicalize.from_full_frame_resize(frames, src_hw=(480, 720), crop_ltrb=[0, 0, 720, 480], 480, 720)`:
  the exact inverse, so output frame *i* lands on the reference's pixels. (Alternative: generate at
  720×480 directly — divisible by 16, no mapping — but off the recipe FYM ran on DAVIS. Not chosen.)
- **Frames:** Wan decodes exactly 21 frames for 21 requested (4k+1), so frame *i* = reference frame *i*;
  **no trim** (unlike CogVideoX's 24). Asserted per video.
- Recorded per video: the mapping, the source ref sha256, tune seconds, recipe, checkpoint revision,
  FYM repo commit.

## 5. Failure handling and budget

- A clip whose attention step finds a single-type layer fails with `empty_head_group`, recorded, never
  retried with altered settings. Its `fym` row is missing for that pair; the scoring notebook's table (B)
  shows FYM's n, table (A) restricts every row to ours' subset — so a FYM failure is visible, not hidden.
  `fym_noguide` does not depend on tuning and exists for every pair.
- **Budget: ~52 min per pair** (51 tuning + 1 generation) + ~1 min per floor video.
  5 method pairs ≈ 4.5 h; 19 census clips ≈ 17 h; all 24 ≈ 21 h of A100 time. Resumable across sessions.
- **Run order (fixed now):** the 5 method pairs first (they already have every other row, so the FYM
  column is complete there soonest), then the 19 census clips in manifest order.

## 6. Implementation (after the current runs)

1. `FollowYourMotion/fym_bench/run_pairs.py`: imports `prepare_tuning`, `tune`, `run_stage`, `RECIPE`
   from `run_davis.py`; new `prepare_reference_from_ref21` (reads the ref mp4, resizes to 832×480,
   records sha256 and fps); writes converted outputs + json to `pairs/inbox/`. ~150 lines.
2. `ditflow/fym_pairs_colab.ipynb`: clone FYM, its environment, Wan2.1-T2V-1.3B weights at the recorded
   revision, run `run_pairs.py` over the manifest. Separate runtime from DiTFlow's and the extractor's.
3. `benchmark/pairs_manifest.csv`: add the `caption` column (values per §3).
4. Scoring: add `'fym_noguide': 'No guidance (Wan 2.1 T2V 1.3B)'` and `'fym': 'Follow-Your-Motion (Wan T2V 1.3B)'`
   to the table rows. Nothing else changes.

## 7. Decisions to confirm before implementation

1. The ~21 A100-hours (or ~4.5 h for the 5 method pairs only).
2. 832×480 with the mapping (recommended, recipe-consistent) vs 720×480 direct.
3. Tuning caption from the reference's own caption (recommended, FYM's protocol).
4. The frame rate decision for the census clips (15 vs 30 fps) applies to FYM too: FYM reads the same
   ref21, so it inherits whatever the references are cut at.
