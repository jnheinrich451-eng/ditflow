"""Known pixel-motion controls and compact reporting for Wan RoPE probes."""

import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image


def make_control_frames(first, kind, count=21, pixels_per_frame=4):
    """Pixel-space controls. RoPE variants never change the generated input."""
    first = np.asarray(first, dtype=np.uint8)
    h, w = first.shape[:2]
    boxes = None
    if kind == 'static':
        frames = [first.copy() for _ in range(count)]
        expected_dx = 0
    elif kind in ('pan_right', 'pan_left'):
        expected_dx = pixels_per_frame * (1 if kind == 'pan_right' else -1)
        frames = [np.roll(first, expected_dx * i, axis=1) for i in range(count)]
    elif kind == 'patch_right':
        expected_dx = pixels_per_frame
        ph, pw, y0, x0 = int(h * .2), int(w * .2), int(h * .45), int(w * .3)
        if x0 + pw + expected_dx * (count - 1) > w:
            raise ValueError('Moving patch leaves the image; reduce pixels_per_frame or frame count')
        # A unique fixed texture; no resampling/interpolation or new RNG per frame.
        texture = np.random.default_rng(17).integers(0, 256, (ph, pw, 3), dtype=np.uint8)
        frames, boxes = [], []
        for i in range(count):
            frame = first.copy()
            x = x0 + expected_dx * i
            frame[y0:y0 + ph, x:x + pw] = texture
            frames.append(frame)
            boxes.append([x, y0, x + pw, y0 + ph])
    else:
        raise ValueError(kind)
    return frames, dict(kind=kind, expected_dx_per_video_frame=expected_dx, frame_boxes=boxes,
                        wrap_exclusion_pixels=abs(expected_dx) * (count - 1) if kind.startswith('pan_') else 0,
                        width=w, height=h, num_frames=count)


def write_control(first, kind, directory, count=21):
    frames, info = make_control_frames(first, kind, count=count)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(directory / f'f{i}.png')
    (directory / 'control.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return info


def evaluation_mask(info, pair, h, w):
    """Conservative known-motion region; approximate causal-VAE alignment."""
    y, x = np.meshgrid((np.arange(h) + .5) / h, (np.arange(w) + .5) / w, indexing='ij')
    if info.get('frame_boxes'):
        # Intersect source boxes over a broad temporal neighborhood and erode
        # one patch. Not a claim of exact single-frame causal-VAE alignment.
        start, end = max(0, 4 * pair - 4), min(info['num_frames'], 4 * pair + 5)
        b = np.asarray(info['frame_boxes'][start:end])
        x0, y0 = b[:, :2].max(0) / [info['width'], info['height']]
        x1, y1 = b[:, 2:].min(0) / [info['width'], info['height']]
        return ((x > x0 + 1 / w) & (x < x1 - 1 / w) & (y > y0 + 1 / h) & (y < y1 - 1 / h)).ravel()
    margin = info.get('wrap_exclusion_pixels', 0) / info['width'] + 1 / w
    return ((x > margin) & (x < 1 - margin) & (y > 1 / h) & (y < 1 - 1 / h)).ravel()


def make_rope_report(runs, output_dir):
    import matplotlib.pyplot as plt
    from probe_report import load_trace, _inline_figure
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    page = ['<!doctype html><meta charset="utf-8"><title>Wan RoPE probe</title>',
            '<style>body{font:16px system-ui;max-width:1600px;margin:24px auto}img{max-width:100%}td,th{padding:6px;border:1px solid #ccc}table{border-collapse:collapse}</style>',
            '<h1>Wan RoPE and known-motion controls</h1><p>All variants use native Q/K before KV injection. '
            'Only the observed block rotation is ablated in detached diagnostics; previous blocks still used full RoPE. '
            'The actual transformer, guidance loss and generation are unchanged. Pre-RoPE Q/K still contain '
            'position effects from earlier layers. Arrow fields are attention correspondences, not optical flow.</p>']
    rows = []
    for run in runs:
        path, meta, events = load_trace(run)
        info = meta['config'].get('rope_control') or dict(kind='real', width=meta['config']['width'], height=meta['config']['height'])
        page.append('<h2>' + html.escape(str(run)) + '</h2><pre>' + html.escape(json.dumps({
            'model': meta['config']['model_key'], 'packages': meta['packages'], 'control': info,
            'guidance_mode': meta['config'].get('guidance_mode'), 'source_prompt': meta['config'].get('source_prompt', '')}, indent=2)) + '</pre>')
        for e in events:
            if e['kind'] == 'reference_latents':
                page.append('<p>VAE adjacent latent RMS changes: ' + html.escape(str(e['adjacent_change_rms'])) +
                            '. A causal VAE can encode identical input frames differently across latent time; nonzero values alone are not a bug.</p>')
        captures = [e for e in events if e['kind'] == 'rope_attention' and e['stage'] == 'reference']
        f, h, w = meta['grid']
        yy, xx = np.meshgrid((np.arange(h) + .5) / h, (np.arange(w) + .5) / w, indexing='ij')
        stride = max(1, int(np.ceil(max(h, w) / 20)))
        for block in dict.fromkeys(e['block'] for e in captures):
            block_events = [e for e in captures if e['block'] == block]
            fig, axes = plt.subplots(len(block_events), f - 1, figsize=(4 * (f - 1), 3 * len(block_events)), squeeze=False)
            for r, e in enumerate(block_events):
                with np.load(path / e['file']) as data:
                    fields = data['hard']
                for i, field in enumerate(fields):
                    selected = evaluation_mask(info, i, h, w)
                    a = field[selected]
                    moved = np.linalg.norm(a, axis=-1) > 0
                    expected = info.get('expected_dx_per_video_frame')
                    row = dict(run=str(run), control=info['kind'], block=block, variant=e['variant'], pair=i,
                               selected_patches=int(selected.sum()), zero_fraction=float((~moved).mean()) if len(a) else None,
                               median_dx=float(np.median(a[:, 0])) if len(a) else None,
                               median_dy=float(np.median(a[:, 1])) if len(a) else None,
                               expected_direction_fraction=(float((a[:, 0] * expected > 0).mean()) if len(a) and expected else None))
                    rows.append(row)
                    ax = axes[r, i]
                    v = field.reshape(h, w, 2) / [w, h]
                    sl = (slice(None, None, stride), slice(None, None, stride))
                    ax.imshow(selected.reshape(h, w), extent=(0, 1, 1, 0), cmap='Greys', alpha=.15, vmin=0, vmax=1)
                    ax.quiver(xx[sl], yy[sl], v[..., 0][sl], v[..., 1][sl], angles='xy', scale_units='xy', scale=1, color='tab:blue')
                    ax.set(xlim=(0, 1), ylim=(1, 0), title=f"{e['variant']} | {i} to {i+1}")
                    ax.set_aspect('equal')
            fig.suptitle(block + ' | ' + info['kind'] + ' | shaded region = evaluation region')
            fig.tight_layout()
            page.append(_inline_figure(fig))
        if not captures:
            page.append('<p>No reference RoPE captures; rerun with --probe_rope.</p>')
    if rows:
        with (output / 'rope_metrics.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        page.append('<h2>Known-motion region metrics</h2><p>For translation controls, expected-direction fraction '
                    'counts zero matches as failures. Patch-control regions are conservative intersections, not '
                    'exact causal-VAE support. See rope_metrics.csv for all pairs.</p><table><tr>'
                    '<th>Control</th><th>Block</th><th>Variant</th><th>Mean zero %</th><th>Mean expected-direction %</th></tr>')
        for key in dict.fromkeys((r['run'], r['control'], r['block'], r['variant']) for r in rows):
            subset = [r for r in rows if (r['run'], r['control'], r['block'], r['variant']) == key]
            def percent(field):
                values = [r[field] for r in subset if r[field] is not None]
                return f'{100 * np.mean(values):.1f}' if values else 'n/a'
            page.append('<tr>' + ''.join('<td>' + html.escape(v) + '</td>' for v in
                        [key[1], key[2], key[3], percent('zero_fraction'), percent('expected_direction_fraction')]) + '</tr>')
        page.append('</table>')
    report = output / 'rope_comparison.html'
    report.write_text('\n'.join(page), encoding='utf-8')
    return report
