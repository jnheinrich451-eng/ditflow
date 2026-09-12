2. Structural fixes to the skeleton

Four problems in the current §5, in order of severity.

(a) 5.6/5.7 and 5.9 are the same section written twice. S4 is metric sensitivity, S5 is component analysis of baselines — and then 5.9 is titled "Sensitivity and component analysis." Delete 5.9. The studies are the sensitivity analysis.

(b) There is no statistical decision rule. This is the biggest hole. Your §3 warns "do not let a 0.005 gap carry an argument" but gives no rule for when a gap does carry one. You need a frozen inference plan, and it has to be frozen before numbers land or it is post-hoc:

Unit of analysis is the clip, not the run. Seeds are nested within clip; average over seeds first, then compare methods across clips.
Comparison is paired (same clips through every method) — so Wilcoxon signed-rank or a bootstrap over clips, not an unpaired t-test.
Report median paired difference + 95% bootstrap CI, not just two means.
Declare a difference real only when the CI excludes zero and the magnitude exceeds the Layer-3 noise floor. Two gates, both stated in advance.
Seed std is reported separately as run-to-run variability. It is not the comparison's error bar.
Multiple comparisons: 5 methods × 4 metrics × 3 parallax bands ≈ 60 tests. Pre-declare one primary endpoint per study (e.g. S3's primary endpoint is MF under the subject prompt, pooled across bands) and mark everything else exploratory. That is cleaner than a Bonferroni correction and it is what preregistration conventions expect.

This becomes a new 5.2.4 Statistical treatment, and it is pure Class A — write it this week.

(c) There is no figure protocol. Which clips appear in qualitative figures must be chosen before you look at outputs. Otherwise the qualitative section is cherry-picked and an examiner will say so in one sentence. Freeze a rule now, e.g. "the clip nearest the median parallax of each band, seed 0, plus one mandatory failure case per method." Put it in protocol.lock.yaml.

(d) The human-evaluation decision is unmade, and it has lead time. You cut FYM's user-study columns. But you also know MF is non-predictive (ρ ≈ 0.23, p = 0.66 on FYM's own table). If MF doesn't predict human ranking and you have no human data, nothing in the chapter anchors the automatic metrics to perception. Two defensible routes:

A small pre-registered forced-choice study — 2AFC, ~15 raters, ~20 pairs, reporting agreement and a per-metric Spearman ρ against the ranking. This costs weeks of lead time, so decide now or lose the option.
Declare explicitly that no human study is run, and let S4 carry construct validity instead — S4 is a validity test of MF, which is arguably a stronger contribution than another user study. Then the absence goes in threats to validity as a stated choice, not a gap.

Route 2 is coherent with your thesis and cheaper. But it must be a written decision, not a silence.

One smaller call: with ~30 clips × 3 seeds you have N ≈ 90 for FVD, far below the bias threshold you already flagged. Keep the column if your supervisor wants template parity with FYM's Table 1, print N in the caption, and state in the text that no claim in the chapter rests on it.

3. The technique that makes Class B writable: pre-committed outcome branches

For each study, write the interpretation paragraph for every outcome the study could produce, before you see any. When numbers land you delete the branches that didn't happen. Three things follow: you can't rationalise post-hoc, you discover now whether a study is even diagnostic (if two outcomes lead to the same paragraph, the study measures nothing), and the chapter is 90% drafted the day the runs finish.

Worked example, S4 (the null study — the one that carries your argument):

**Question.** On Kubric pairs where world-frame object motion is identical and only Θ_cam differs, do MF, Text Sim, and Temporal Consistency respond?

**Falsification.** If they move as much as they do under a genuine object-motion change of comparable magnitude, the evaluation gap this thesis claims does not exist.

**Outcome A — Δ below noise floor.** Standard metrics are camera-blind. A method could get the camera entirely wrong and score identically. This converts the disentanglement metric from a preference into a necessity, and explains why no table in the surveyed literature has a camera column: the metrics could not have shown one.

**Outcome B — Δ comparable to object-motion Δ.** The claim is falsified as stated. The thesis repositions: metrics can see camera error, and existing methods nonetheless fail on it — which is a weaker but still real finding, relocating the gap from evaluation to method. (Write this paragraph. Most students only write A, and that is exactly the tell.)

**Outcome C — Δ nonzero but non-monotonic in parallax.** MF is unstable rather than blind. Different claim, weaker: the metric is unreliable rather than insensitive, and the chapter's recommendation becomes "report with CIs" rather than "replace."

Do this for S2, S3, S5 as well. It takes an afternoon each and it is the highest-leverage writing you can do while a GPU is busy.

1. What your runs are destroying right now

This is the part I'd act on today. Several chapter-required fields exist only at run time and cannot be reconstructed afterwards:

Exclusion reasons. Every OOM, crash, NaN, degenerate output, or manual restart, with clip ID, method, seed, and reason code. At the end you need a run-accounting table (attempted / completed / excluded, by reason). Reconstructed from memory it is guesswork; logged live it is evidence.
Wall-clock per video and the GPU it ran on — your Time(s) column. Unrecoverable later.
Peak VRAM per method. Cheap to log (torch.cuda.max_memory_allocated()), impossible to backfill, and it justifies your tiering decisions.
Commit hash + checkpoint SHA at time of run. If you update a repo mid-sweep, early and late runs are different methods.
Any manual intervention. "Reduced steps to 30 after OOM on clip 17" must be in the log or the parameter table is a fiction.

If your harness isn't emitting these, patch it before the next batch — it's a twenty-minute change that protects two sections.

5. Revised skeleton
5.1 Experimental setup
    5.1.1 Task and Protocol P                        [A]
    5.1.2 Benchmark data: splits, curation, parallax  [A]
    5.1.3 Methods under test + capability matrix      [A]  <- Block 0 moves here
    5.1.4 Implementation, environment, parameters     [A]
    5.1.5 Temporal and spatial alignment              [A]
5.2 Metrics
    5.2.1 Instrument metrics (Layer 1)                [A]
    5.2.2 System metrics (Layer 2)                    [A]
    5.2.3 Bounds and noise floors (Layer 3)           [A]
    5.2.4 Statistical treatment                       [A]  <- NEW
5.3 Protocol audit (S1)                               [A]  <- fully writable, no GPU
5.4 Instrument validation (S2)                        [B]
5.5 Reproduction under a unified protocol (S3)        [B]  <- main tables live here
5.6 Metric sensitivity: are metrics camera-blind (S4) [B]
5.7 Component analysis of existing methods (S5)       [B]
5.8 Failure analysis and error budget                 [C]  <- taxonomy pre-registered
5.9 Threats to validity                               [A]

Note on 5.8: you can't pre-write the analysis, but you can pre-register the failure taxonomy — a fixed code list (shape-support leakage, camera drift, identity collapse, temporal flicker, object dropout, background tearing) that you tag outputs against as they come in. Counting against a fixed taxonomy is analysis; inventing categories after seeing failures is anecdote.