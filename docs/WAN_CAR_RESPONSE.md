# Car-turn 14B diagnostic

Use the six new cells at the end of `notebook.ipynb`, under **Wan2.1 14B: car-turn readout and update response**, tag `wan-car-response-14b`. The existing `wan_reference_inputs.zip` is sufficient. The setup validates the bundle, but only car-turn is passed to the model. No camel generation is scheduled.

## Stage 1: motion readout

`probe_wan_affine.py` now accepts `--model 14b --low_vram --mean_only`. Existing 1.3B commands retain their historical behavior. The notebook explicitly selects 14B, block 10, five controls (static, right/left translation, expansion/contraction), and clean/step-0/step-9/step-29 states: 20 observed forwards. No head sweep, optimization, target generation or RoPE edits.

The first car frame supplies texture for synthetic image-plane transforms. Known geometric truth, texture support, and temporal-anchor offsets are retained. Blank text matches reference extraction; target-prompt dependence is not isolated in this stage. Pure-noise fields must agree across controls. Check static false motion and expansion/contraction amplitude as well as direction cosine. No model-specific bias correction is applied.

## Stage 2: direction-selective update and retention

This optional stage loads 14B once. It uses the aligned car prompt from the previous direction test, seed 1, CFG 5, flow-match Euler with shift 3, 50 steps, 21 frames at 480x832. Block 10, temperature 2, cap 100, nonzero reference mask and native MSE are preserved. No KV injection, reference-region balancing or trainable RoPE.

An unguided text-conditioned prefix produces the actual sampled latent immediately before step 9. Three branches clone that latent and the complete scheduler state:

| Branch | Intervention at step 9 | Remaining generation |
| --- | --- | --- |
| off | none | steps 9–49 |
| forward | five production Adam updates toward the reference AMF | steps 9–49, no more guidance |
| reverse | five production Adam updates toward the reversed-reference AMF | steps 9–49, no more guidance |

The learning rate is 0.001, matching step 9 of the prior ten-step schedule. Total optimizer updates are 10 across the two guided branches. RNG states, initial latent, prompt and scheduler are paired. Reversal is performed on the first 21 decoded reference frames before causal VAE encoding. It is not a latent-time flip or vector negation. Reference hard AMFs are extracted independently, so changing reference order can also change confidence and valid support. Native training masks are retained, while target preference is additionally measured on their intersection.

This is an **impulse response** experiment, not a reproduction of the prior 50-update guidance window. A weak response to this small intervention is inconclusive; it does not prove AMF cannot affect motion. Forward-noised controls from Stage 1 are not substituted for sampled generation latents here.

Saved evidence includes actual optimizer gradients/updates, all-pair loss-path flow immediately before/after, both reference targets/masks, adjacent attention captures through denoising, final loss-path flow, and three final videos. The report recomputes scores from the arrays. Positive `forward_preference` means the current AMF is closer to the forward target than the reversed target on common reference positions. It is not a heading metric or segmentation of the generated car.

`estimated_clean_before.mp4` and each branch's `estimated_clean_after.mp4` decode the instantaneous flow-matching estimate `x0_hat = z_t - sigma * velocity_CFG`. These use full conditional/unconditional model forwards without advancing the scheduler. They expose the predicted immediate visual effect; at high noise, estimates may be poor and must not be treated as true clean frames or final video quality. Compare them alongside the completed trajectories.

Interpretation:
- Wrong known-motion readout identifies a measurement problem to investigate; it does not establish the direction of the optimization gradient.
- Own loss reduction without a direction-selective response or useful decoded effect points toward limitations of the objective/update, subject to the intervention's small size.
- An initially useful predicted effect that is absent in the final video motivates retention/timing work. A high-noise clean estimate alone is not sufficient proof.
- Final improvement is judged visually and independently of the AMF objective; no automatic loss threshold selects a fix.

## Running, recovery and export

Run setup, then Stage 1. To minimize Colab time, archive immediately and share that report before deciding whether Stage 2 is needed. Alternatively execute Stage 2 and compare. Do not rerun setup to resume: restore the printed `RESPONSE_OUT` after a kernel restart and rerun a stage cell. Completed stages are audited and skipped. Failed stages receive fresh attempt directories, retaining their logs. Stage 2 requires a completed Stage 1. Source/package changes require a new experiment.

The archive cell can run after Stage 1 alone. It includes logs, configuration, metadata, arrays, reports and videos, excluding large `.pt`/`.pth` files and embeddings. `export_probe_archive` persists it to `My Drive/ditflow_probes` in Colab and provides its existing browser download option. It works with the existing VS Code-to-Colab Drive workflow.

The experiment does not implement I2V or change the production AMF loss. Local CPU tests validate geometry, readout equivalence, scheduler isolation, score semantics and rejection of invalid traces. They do not validate pretrained 14B behavior or GPU memory usage; those require the Colab run.

## Follow-up: compare 14B blocks 10, 20 and 30

The notebook section **Wan2.1 14B: car-turn block readout comparison (10 / 20 / 30)**, tag `wan-car-block-readout-14b`, adds five cells: setup, one shared readout command, summary, and Drive export after the explanatory cell. It uses the same retained car-turn inputs. Restore `BLOCK_OUT` to resume. Output prefix: `wan_car_blocks_14b_`.

The plan selects `readout_blocks=[10,20,30]` and `readout_only=True`; it contains no response-generation command and rejects attempts to launch that stage. All blocks are observed in the same 20 forwards, using the same encoded controls, noise tensor, prompt and schedule. Each pass runs through block 30, so it costs more than the earlier block-10-only pass. It does not load three models or run three separate generation suites. Observed blocks do not change the generator's guidance block configuration.

The audit requires all 720 metric rows and every selected block's capture. Pure-noise fields must match across controls **within each block**; different blocks may have different noise biases. The report retains hard/soft fields, geometric and textured supports, and anchor offsets. `block_summary.csv` displays nominal soft-AMF static error, both translation cosines and both scale amplitudes together, without selecting a winner.

A static control repeats the first decoded car frame 21 times. There is no true image-plane movement. The noisy tests add independent-frame Gaussian noise after VAE encoding; static EPE is the mean length of the spurious AMF vectors on the selected patches, not actual car motion. At sigma 0.93047, large false displacement demonstrates an unreliable motion readout in that setting but does not by itself establish an indexing error. A good block must be checked on both opposite directions and static error, then validated with actual guidance and ultimately I2V; clean or pure-noise scores alone cannot establish transfer quality.
