# Wan research target and local probe retention

Decision recorded 2026-09-13: the intended final model is Wan2.1 I2V 14B. Use Wan2.1 **14B** for new pretrained motion-guidance experiments and model-specific decisions. Earlier 1.3B runs remain historical comparisons. Existing experiment plans and their defaults are preserved for reproducibility; new plans must explicitly select 14B.

## Current implementation versus final target

The existing T2V entrypoint supports both 1.3B and 14B through the same ControlledWanTransformer, attention processor, and AMF implementation. Checkpoint configuration supplies dimensions. The 1.3B transformer has 30 blocks and 12 heads; T2V 14B has 40 blocks and 40 heads. Both have head dimension 128, patch size (1,2,2), and 16 input/output latent channels. A block index or tuned readout is not automatically equivalent across checkpoints.

Wan2.1 I2V 14B uses the same transformer family and 40-block/40-head backbone, but has 36 input channels, image embeddings, and additional image cross-attention projections. Its output remains 16 latent channels. It requires an image-to-video pipeline and image-conditioning preparation, not merely a new model ID. Current motion_guidance_wan.py loads WanPipeline and only lists T2V checkpoints. Lower-level image-conditioning branches exist, but do not establish an operational or validated I2V port.

For an I2V extension, preserve native image conditioning in every reference, guidance, conditional, and unconditional forward; optimize only the generated noisy latent, keeping conditioning tensors fixed. Record the motion-reference clip and target starting image separately. Compare AMF off/on with the same target image, prompt, seed, scheduler, and environment. First verify the guidance-disabled custom path against the stock I2V pipeline. T2V results cannot establish I2V behavior.

Sources checked on the decision date:
- https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/raw/main/transformer/config.json
- https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/blob/main/transformer/config.json
- https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers/raw/main/transformer/config.json
- https://raw.githubusercontent.com/huggingface/diffusers/v0.39.0/src/diffusers/pipelines/wan/pipeline_wan_i2v.py

## Probe folders

The user reports Drive backups. Their completeness was not verified from this checkout. No probe files were deleted.

Recommended local working set:
- `probe_runs/wan_reference_inputs`: current two-clip inputs, masks, and manifest; required by the current direction test and regional analysis.
- `probe_runs/wan_davis_pilot_inputs`: broader pilot inputs used by earlier notebook sections and useful for later holdouts; cheap to retain.
- `probe_runs/wan_direction_amf_14b_20260912T184130167398Z`: current 14B baseline comparison.
- `probe_runs/car_cog_affine`: raw control arrays used in the current readout investigation.

Older `wan_direction_amf_20260912T173918232186Z` and `wan_timing_20260912T123714676348Z` results can be kept only on Drive if local space is needed; restoring them is necessary to rerun their detailed analyses. Other older output folders are not prerequisites for the next 14B generation. Removing them means historical notebook display/audit cells that point at those paths need restoration before use.

Retain `probe_comparison` reports and analysis scripts locally. Reports are summaries, not substitutes for backed-up raw NPZ captures, metadata, event logs, plans, environment snapshots, and videos. This retention advice does not apply to source code or tests.
