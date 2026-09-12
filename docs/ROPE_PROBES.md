# Wan RoPE and motion extraction probes

The existing AMF port matches the original DiTFlow formula, but that does not
establish that Wan's attention gives equally useful motion correspondences.
The next experiment isolates this block's positional rotation and uses inputs
with known pixel motion. It does not modify the motion loss or disable RoPE in
the actual transformer.

In `notebook.ipynb`, run **Wan RoPE: known-motion reference controls** after the
Colab installation/restart. The section is self-contained and initially selects
Lucia. It loads pretrained Wan once, extracts five references, then builds an
HTML report and downloadable evidence ZIP. It performs no denoising generation.

The archive cell saves the ZIP to **My Drive / ditflow_probes** by default,
mounting Google Drive if needed. It also displays a **Download ZIP** button and
an **Open Google Drive** link. Set `SAVE_ROPE_ZIP_TO_DRIVE = False` for browser
download only. Repeated exports keep existing Drive copies and choose a new
filename. Drive storage is private to your account; the link opens My Drive,
not a public shared-file URL. Only the archive cell needs rerunning to export
an already completed suite.

```sh
python verify_wan_rope.py
python verify_motion_probe.py
python probe_wan_rope.py -v assets/lucia.mp4 --output_path probe_runs/lucia_rope_test
```

Use a fresh output folder each time. The four synthetic controls derive from
the first reference frame at the model's input resolution and are lossless PNGs:

| Input | Known pixel motion |
|---|---|
| Real reference | Unknown numerical ground truth; compare visible patterns |
| Static | Exactly identical pixels across all frames |
| Pan right | Whole image shifts right by 4 pixels per decoded frame |
| Pan left | Whole image shifts left by 4 pixels per decoded frame |
| Patch right | A unique textured rectangle shifts right over a fixed background |

Whole-image translation is a screen-space control, not a physical camera-pose
simulation. Wrapped image borders are excluded from evaluation. The patch's
evaluation regions intersect its boxes over a temporal neighborhood and erode
one token at each edge. They are conservative approximations, not exact VAE
receptive fields. Known input direction is evaluated; exact latent displacement
is not assumed because the causal VAE mixes time.

## What each variant means

- `pre_rope`: normalized Q/K before rotation in the observed block.
- `temporal_only`: rotate that block's temporal channel group only.
- `spatial_only`: rotate its height/width channel groups only.
- `full_rope`: normal rotation, using native keys before any KV injection.

Earlier transformer blocks still used full RoPE in every variant. In particular,
pre-RoPE does not mean position-free features. These detached computations do
not feed back into the model and are not candidate generation configurations.

For sampling probes, add `--probe_rope` to `motion_guidance_wan.py`; it enables
`--probe` automatically and uses the selected probe blocks/steps. Native-key
variants are explicitly labeled, since comparing pre-RoPE native keys with
injected reference keys would mix two different effects. The standalone report
currently displays reference-stage variants; sampling variants remain in traces.

## Reading the evidence

1. Check whether right/left controls recover the appropriate horizontal sign,
   at each block and in each RoPE variant. Many zeros can mean a position-biased
   correspondence estimate rather than successful motion extraction.
2. If pre-RoPE recovers the known direction but full RoPE does not, the current
   block's rotation changes the useful correspondence signal. That supports an
   AMF-extraction experiment, not removing RoPE throughout Wan.
3. If both fail, inspect the static control and earlier blocks. The recorded
   VAE adjacent-latent RMS changes reveal temporal differences already present
   before the transformer. A causal VAE may encode identical pixels differently
   across latent time; nonzero RMS is not automatically a bug.
4. Compare global translation with the moving-patch control. Success on one does
   not imply that object and camera motion both transfer to generated video.

The CSV includes the evaluated patch count, zero-match fraction, median dx/dy,
and expected-direction fraction. Expected-direction fraction includes zero
matches in its denominator; it is a controlled correspondence diagnostic,
not a real-video motion-transfer score.

## Checks performed locally

The RoPE oracle independently constructs complex phases for time, row and column
on a 6 x 30 x 52 token grid with head dimension 128. It checks the port's rotation,
norm preservation, inverse rotation, and known content-translation sign. The
existing tiny stock-Wan comparison also checks output and gradient invariance
with these probes enabled, including checkpointing, RNG state and KV injection.
Pretrained control results must be collected in Colab; local tests do not prove
that pretrained reference extraction or video transfer succeeds.

Source references: [original Wan implementation](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py)
and [Diffusers 0.39 Wan implementation](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/transformers/transformer_wan.py).
