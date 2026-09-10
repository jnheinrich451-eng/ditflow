"""Reference AMF versus independent image motion; masks never enter generation."""
import csv
import html
from pathlib import Path

import numpy as np
from PIL import Image


def reference_images(path, count, size):
    path = Path(path)
    if path.is_dir():
        files = sorted([*path.glob('*.jpg'), *path.glob('*.png')], key=lambda p: int(p.stem.split('f')[-1]))
        frames = [np.asarray(Image.open(p).convert('RGB').resize(size)) for p in files[:count]]
    else:
        import imageio.v3 as iio
        frames = []
        for frame in iio.imiter(path, plugin='FFMPEG'):
            frames.append(np.asarray(Image.fromarray(frame).convert('RGB').resize(size)))
            if len(frames) == count:
                break
    if len(frames) != count:
        raise ValueError(f'{path}: expected at least {count} frames, got {len(frames)}')
    return frames


def image_motion(first, second, grid):
    """Farneback estimate in patch units with forward/backward validity checks."""
    import cv2
    h, w = grid
    size = (w * 8, h * 8)
    gray = [cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_RGB2GRAY) for frame in (first, second)]
    args = dict(pyr_scale=.5, levels=5, winsize=25, iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
    forward = cv2.calcOpticalFlowFarneback(gray[0], gray[1], None, **args)
    backward = cv2.calcOpticalFlowFarneback(gray[1], gray[0], None, **args)
    yy, xx = np.indices(gray[0].shape, dtype=np.float32)
    ex, ey = xx + forward[..., 0], yy + forward[..., 1]
    sampled = cv2.remap(backward, ex, ey, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    inside = (ex >= 0) & (ex < size[0] - 1) & (ey >= 0) & (ey < size[1] - 1)
    consistent = np.linalg.norm(forward + sampled, axis=-1) <= 1.5
    # Exclude textureless pixels even if both estimators happen to predict zero.
    intensity = gray[0].astype(np.float32)
    variance = cv2.blur(intensity**2, (7, 7)) - cv2.blur(intensity, (7, 7))**2
    valid = inside & consistent & (variance >= 4)
    support = cv2.resize(valid.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    numerator = cv2.resize(forward * valid[..., None], (w, h), interpolation=cv2.INTER_AREA)
    flow = numerator / np.maximum(support[..., None], 1e-8) / 8
    return flow, support >= .5


def comparison(amf, measured, selected):
    a, b = amf[selected], measured[selected]
    if not len(a):
        return dict(patches=0, moving_patches=0, cosine=None, zero_amf_fraction=None, amf_dx=None, amf_dy=None,
                    image_dx=None, image_dy=None)
    an, bn = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    moving = bn >= .125  # At least one pixel at the optical-flow working resolution.
    cosine = np.zeros(len(a), dtype=float)  # Zero AMF is a failure to follow measured movement.
    nonzero = (an > 1e-8) & moving
    cosine[nonzero] = (a[nonzero] * b[nonzero]).sum(-1) / (an[nonzero] * bn[nonzero])
    return dict(patches=len(a), moving_patches=int(moving.sum()), cosine=float(cosine[moving].mean()) if moving.any() else None,
                zero_amf_fraction=float((an < 1e-8).mean()),
                amf_dx=float(np.median(a[:, 0])), amf_dy=float(np.median(a[:, 1])),
                image_dx=float(np.median(b[:, 0])), image_dy=float(np.median(b[:, 1])))


def make_reference_report(run, video_path, mask_dir, output_dir):
    import matplotlib.pyplot as plt
    from probe_report import load_trace, _inline_figure
    path, meta, events = load_trace(run)
    f, h, w = meta['grid']
    frames = reference_images(video_path, 4 * (f - 1) + 1, (meta['config']['width'], meta['config']['height']))
    mask_paths = sorted(Path(mask_dir).glob('*.png'))
    if len(mask_paths) != len(frames):
        raise ValueError(f'Expected {len(frames)} aligned masks in {mask_dir}, got {len(mask_paths)}')
    fg = [np.asarray(Image.fromarray((np.asarray(Image.open(p)) > 0).astype(np.float32)).resize((w, h), Image.Resampling.BOX)) >= .1 for p in mask_paths]
    captures = []
    for event in events:
        if event['kind'] == 'training_reference':
            with np.load(path / event['file']) as z:
                fields = z['flow'][np.arange(f - 1) * (f + 1) + 1].reshape(f - 1, h, w, 2)
            captures.append((event['block'], 'actual_native_target', fields))
        elif event['kind'] == 'rope_attention' and event['stage'] == 'reference':
            with np.load(path / event['file']) as z:
                fields = z['hard'].reshape(f - 1, h, w, 2)
            captures.append((event['block'], event['variant'], fields))
    if not captures:
        raise ValueError('No saved reference AMF captures')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    page = ['<!doctype html><meta charset="utf-8"><title>Reference AMF audit</title>',
            '<style>body{font:16px system-ui;max-width:1500px;margin:24px auto}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:5px}</style>',
            '<h1>Reference subject/background AMF audit</h1>',
            '<p>Image motion is a Farneback estimate, not ground truth. Only in-bounds, textured pixels passing a 1.5-pixel forward/backward check contribute; patches require 50% valid support. '
            'Values are patch units. Zero AMF counts as zero cosine when measured motion exceeds 0.125 patch. '
            'Masks are DAVIS reference annotations, used only in this report. No masks enter Wan. '
            'The raw native target includes zero/excluded entries; these comparisons are not the training loss.</p>',
            '<p>Latent i is approximately anchored at decoded frame 4*i. Offsets -2, 0, +2 test sensitivity; causal VAE support is broader. '
            'Background-median subtraction measures motion relative to dominant background translation, not full camera pose, rotation, or depth. '
            'RoPE variants alter only detached Q/K at the observed block; previous layers retain native RoPE.</p>']
    rows = []
    for offset in (-2, 0, 2):
        pairs = []
        for i in range(f - 1):
            start, end = np.clip([4*i+offset, 4*(i+1)+offset], 0, len(frames)-1)
            flow, valid = image_motion(frames[start], frames[end], (h, w))
            # A conservative background excludes foreground at both endpoints.
            subject, background = fg[start] & valid, ~(fg[start] | fg[end]) & valid
            pairs.append((int(start), int(end), flow, valid, subject, background))
        for block, variant, fields in captures:
            for i, (start, end, flow, valid, subject, background) in enumerate(pairs):
                for label, selection in [('subject', subject), ('background', background)]:
                    rows.append(dict(block=block, variant=variant, pair=i, anchor_offset=offset,
                                     video_start=start, video_end=end, region=label,
                                     **comparison(fields[i], flow, selection)))
                if background.any():
                    amf_bg, flow_bg = np.median(fields[i][background], axis=0), np.median(flow[background], axis=0)
                    rows.append(dict(block=block, variant=variant, pair=i, anchor_offset=offset,
                                     video_start=start, video_end=end, region='subject_relative_to_background',
                                     **comparison(fields[i]-amf_bg, flow-flow_bg, subject)))
            if offset == 0 and variant == 'actual_native_target':
                fig, axes = plt.subplots(3, f-1, figsize=(4*(f-1), 8), squeeze=False)
                yy, xx = np.meshgrid(np.arange(h)+.5, np.arange(w)+.5, indexing='ij')
                for i, (start, end, flow, valid, subject, background) in enumerate(pairs):
                    for r, vectors in enumerate((fields[i], flow, fields[i]-flow)):
                        ax = axes[r, i]
                        ax.imshow(frames[start], extent=(0,w,h,0))
                        ax.contour(xx, yy, fg[start], levels=[.5], colors=['yellow'])
                        selected = valid & ((np.floor(xx).astype(int) % 2) == 0) & ((np.floor(yy).astype(int) % 2) == 0)
                        ax.quiver(xx[selected], yy[selected], vectors[...,0][selected], vectors[...,1][selected], angles='xy', scale_units='xy', scale=1, color='cyan')
                        ax.set(xlim=(0,w), ylim=(h,0), xticks=[], yticks=[], title=f'{["Native AMF", "Image-motion estimate", "AMF minus image motion"][r]} {start}->{end}')
                fig.suptitle(block + ' | yellow: reference subject; arrows: valid image-motion patches')
                fig.tight_layout()
                page.append(_inline_figure(fig))
    with (output / 'reference_motion_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    page.append('<h2>Anchor offset 0; equal-weight mean across available pairs</h2><table><tr><th>Block</th><th>Variant</th><th>Region</th><th>Mean cosine</th><th>Selected patches</th></tr>')
    for block, variant, _ in captures:
        for region in ('subject', 'background', 'subject_relative_to_background'):
            subset = [r for r in rows if r['block']==block and r['variant']==variant and r['region']==region and r['anchor_offset']==0]
            values = [r['cosine'] for r in subset if r['cosine'] is not None]
            score = f'{np.mean(values):+.3f}' if values else 'insufficient support'
            page.append('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in (block,variant,region,score,sum(r['patches'] for r in subset))) + '</tr>')
    page.append('</table><p>Interpret supported patterns across offsets and known-motion controls; do not infer correctness from one aggregate score. Inspect subject direction, camera-relative movement, and valid-support coverage. Raw per-pair counts and vectors are in reference_motion_metrics.csv.</p>')
    report = output / 'reference_motion.html'
    report.write_text('\n'.join(page), encoding='utf-8')
    return report
